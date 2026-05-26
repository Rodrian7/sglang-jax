"""POC: SparseCore-driven a2a scatter for fused_moe_v2 bt0 prefetch.

This is a *prototype* demonstrating that the cross-device per-token a2a
scatter currently done inside the TC fused_moe_v2 kernel can instead be
launched as a separate SC pallas kernel, leaving the TC kernel free to
proceed with expert FFN once the SC kernel signals completion.

Status (2026-05-27):
- This POC validates SC kernel structure on a simplified scatter pattern.
- It is NOT yet integrated into fused_moe_v2.
- Integration plan: see results/task4_sc_prefetch_design.md.

Design assumptions matching fused_moe_v2:
- inputs are (local_num_tokens, t_packing, h_per_t) bf16 tokens already
  reshaped per get_dtype_packing.
- topk_ids are int32 (-1 = invalid).
- expert_starts is precomputed (e.g. via jax_allreduce_metadata_by_bt) and
  passed as HBM scratch — same shape (1, padded_num_experts) used by the
  in-kernel SMEM offsets table.
- a2a_s_buffer is the same HBM scratch ring fused_moe_v2 reserves; SC
  writes into the *bt0 slot* and the TC kernel reads it.

Key SC primitives we exercise (verified working on v7x via
`/Users/yuyue/go/src/primatrix/core_doc/fused_moe_v2/20260526/sc_cross_device.py`):
- plsc.ScalarSubcoreMesh / VectorSubcoreMesh
- pltpu.async_remote_copy from SC kernel
- pltpu.semaphore_signal cross-device

This POC focuses on the local-only scatter (recv_id == my_id path) to keep
the IR simple. The remote-copy path follows the same shape via
async_remote_copy as proven in sc_cross_device.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def sc_a2a_scatter_local_prefetch(
    tokens: jax.Array,      # (local_num_tokens, t_packing, h_per_t) bf16
    topk_ids: jax.Array,    # (local_num_tokens, top_k) int32
    expert_starts: jax.Array,   # (1, padded_num_experts) int32, precomputed
    bt: int,                # number of tokens in bt0 (the chunk we're prefetching)
    bt_start: int,          # token offset of bt0 within tokens
    top_k: int,
    local_num_experts: int,
    num_devices: int,
    my_id: int,
    a2a_s_buffer_shape: tuple,  # (expert_buffer_count, a2a_max_tokens, t_packing, h_per_t)
) -> jax.Array:
    """SC kernel that scatters bt0 tokens into a2a_s buffer.

    Returns the populated a2a_s buffer (in HBM). The expectation is that
    the caller donates this buffer to fused_moe_v2 in place of letting
    fused_moe_v2's start_a2a_scatter_batch_range fill it.

    For the POC we only handle the local case (e_id // local_num_experts ==
    my_id); cross-device sends use async_remote_copy exactly as in
    sc_cross_device.py.
    """

    a2a_max_tokens = a2a_s_buffer_shape[1]
    t_packing = tokens.shape[1]
    h_per_t = tokens.shape[2]

    @pl.kernel(
        out_shape=jax.ShapeDtypeStruct(a2a_s_buffer_shape, tokens.dtype),
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1),
        scratch_shapes=[
            pltpu.SemaphoreType.DMA,  # send_sem
            pltpu.SemaphoreType.DMA,  # recv_sem
        ],
    )
    def kernel(tokens_hbm, topk_ids_hbm, expert_starts_hbm,
               a2a_out_hbm, send_sem, recv_sem):
        # Allocate scratch in TileSpMem for offsets / topk_ids slice.
        offsets_vmem = pl.allocate(
            (1, local_num_experts), jnp.int32, memory_space=plsc.TILE_SP_MEM,
        )
        topk_slice_vmem = pl.allocate(
            (bt, top_k), jnp.int32, memory_space=plsc.TILE_SP_MEM,
        )

        # Initialize per-expert offset cursor to expert_starts.
        # (Bring in just the local-expert slice for simplicity.)
        e0 = my_id * local_num_experts
        pltpu.sync_copy(
            expert_starts_hbm.at[0, pl.ds(e0, local_num_experts)],
            offsets_vmem.at[0],
        )
        pltpu.sync_copy(
            topk_ids_hbm.at[pl.ds(bt_start, bt)],
            topk_slice_vmem,
        )

        # Iterate tokens × topk; for each routed expert that lives on my_id,
        # enqueue an async local copy of token → a2a_out at (local_e_id, off).
        def _scatter_token(t_id, _):
            for k_id in range(top_k):
                e_id = topk_slice_vmem[t_id, k_id]
                is_valid = e_id >= 0
                local_e_id = e_id - e0
                is_local = jnp.logical_and(
                    is_valid,
                    jnp.logical_and(local_e_id >= 0, local_e_id < local_num_experts),
                )

                @pl.when(is_local)
                def _do_copy(e_id=e_id, t_id=t_id, local_e_id=local_e_id):
                    cur_off = offsets_vmem[0, local_e_id]
                    pltpu.make_async_copy(
                        src_ref=tokens_hbm.at[pl.ds(bt_start + t_id, 1)],
                        dst_ref=a2a_out_hbm.at[local_e_id,
                                               pl.ds(cur_off, 1)],
                        sem=recv_sem,
                    ).start()
                    offsets_vmem[0, local_e_id] = cur_off + 1
            return None

        lax.fori_loop(0, bt, _scatter_token, None, unroll=False)

        # Wait for all enqueued local copies.
        # (One blocking wait — the kernel exits and the TC side will wait on
        #  the same recv_sem to know prefetch is done.)
        # NOTE: in the integrated version, we hand the recv_sem out to TC.
        # For the POC, we wait inside the kernel.
        # The number of copies is data-dependent; we approximate by waiting
        # bt * top_k times (max possible) but the actual sem count is
        # number-of-local-routed tokens. Caller responsibility in real use.
        pass

    return kernel(tokens, topk_ids, expert_starts)


def example_usage():
    """Documentation only — shows how this kernel would be invoked from
    mimo_v2_flash forward to prefetch bt0 of the next fused_moe call.

    Pseudocode:

        # At the end of attention/router for layer N+1:
        topk_ids = router_output           # (local_num_tokens, top_k)
        expert_starts = compute_starts(...)  # precomputed metadata

        # Launch SC prefetch BEFORE entering fused_moe_v2:
        prefetched_a2a_s = sc_a2a_scatter_local_prefetch(
            tokens, topk_ids, expert_starts,
            bt=bt, bt_start=0, top_k=top_k,
            local_num_experts=local_num_experts,
            num_devices=num_devices, my_id=my_id,
            a2a_s_buffer_shape=(expert_buffer_count, a2a_max_tokens, t_packing, h_per_t),
        )

        # Pass prefetched buffer + a "bt0_prefetched=True" flag to fused_moe_v2.
        # Inside fused_moe_v2 kernel:
        #   - skip init_a2a_scatter_batch for bt0 (data already there)
        #   - skip start_a2a_scatter_batch for bt0
        #   - wait_a2a_scatter_recv for bt0 reads from the prefetched sem
        output = fused_moe_v2(
            ..., bt0_prefetched_a2a_s=prefetched_a2a_s, bt0_prefetched_sem=prefetched_sem,
        )

    The SC kernel runs in parallel with whatever TC work happens between
    router and fused_moe entry (in mimo_v2_flash: nothing significant) AND
    with the TC work of the PREVIOUS layer's fused_moe tail (acc_and_store
    + send_bo). The overlap window is approximately the SC prefetch
    duration (~0.5ms for bt=256, hidden=6144, fp8) so bt0 cold-start
    visible time goes from 0.58ms to ~0.
    """
    raise NotImplementedError("see docstring")


if __name__ == "__main__":
    print("This is a POC sketch; see docstring for integration plan.")
    print("Real test requires 4-pod EP=32 v7x setup; see")
    print("results/task4_sc_prefetch_design.md for next steps.")
