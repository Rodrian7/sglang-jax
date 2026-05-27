"""SparseCore-driven bt0 a2a scatter for fused_moe_v2 prefetch (v7 — ring shape).

v7 changes:
- Output shape now matches fused_moe_v2 ring buffer: (num_bt_banks, local_num_experts,
  a2a_max_tokens, t_packing, h_per_t).
- SC kernel only writes bank 0 (bt0). Banks 1+ are left as zeros (TC writes them).
- This allows direct buffer donation to fused_moe_v2.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.sharding import PartitionSpec as P


def _make_kernel(
    *,
    bt: int,
    bt_start: int,
    top_k: int,
    local_num_experts: int,
    num_devices: int,
    padded_num_experts: int,
    dp_axis_name: str,
    tp_axis_name: str,
    tp_size: int,
    bt0_bank: int = 0,
):
    def body(
        tokens_hbm,
        topk_ids_hbm,
        expert_starts_hbm,
        a2a_out_hbm,           # (num_bt_banks, local_e, a2a_max_t, t_pack, h_per_t)
        topk_vmem,
        es_vmem,
        cursor_vmem,
        load_sem,
        send_sem,
        recv_sem,
    ):
        dp_rank = lax.axis_index(dp_axis_name)
        tp_rank = lax.axis_index(tp_axis_name)
        my_id = dp_rank * tp_size + tp_rank

        for e in range(padded_num_experts):
            cursor_vmem[e] = jnp.int32(0)

        copy_topk = pltpu.make_async_copy(
            topk_ids_hbm.at[pl.ds(bt_start, bt)], topk_vmem, load_sem,
        )
        copy_es = pltpu.make_async_copy(
            expert_starts_hbm, es_vmem, load_sem,
        )
        copy_topk.start()
        copy_es.start()
        copy_topk.wait()
        copy_es.wait()

        def _scatter_one_token(t_id, _):
            for k_id in range(top_k):
                e_id = topk_vmem[t_id, k_id]
                is_valid = e_id >= 0
                e_id_safe = lax.select(is_valid, e_id, jnp.int32(0))
                target_dev = e_id_safe // local_num_experts
                target_local_e = e_id_safe % local_num_experts

                cur_off = cursor_vmem[e_id_safe]
                inc = lax.select(is_valid, jnp.int32(1), jnp.int32(0))
                cursor_vmem[e_id_safe] = cur_off + inc

                dst_pos = es_vmem[e_id_safe] + cur_off

                @pl.when(jnp.logical_and(is_valid, target_dev == my_id))
                def _local(t_id=t_id, target_local_e=target_local_e, dst_pos=dst_pos):
                    pltpu.make_async_copy(
                        tokens_hbm.at[pl.ds(bt_start + t_id, 1)],
                        a2a_out_hbm.at[bt0_bank, target_local_e, pl.ds(dst_pos, 1)],
                        recv_sem,
                    ).start()

                @pl.when(jnp.logical_and(is_valid, target_dev != my_id))
                def _remote(
                    t_id=t_id, target_local_e=target_local_e, dst_pos=dst_pos,
                    target_dev=target_dev,
                ):
                    dp_idx = target_dev // tp_size
                    tp_idx = target_dev % tp_size
                    pltpu.make_async_remote_copy(
                        tokens_hbm.at[pl.ds(bt_start + t_id, 1)],
                        a2a_out_hbm.at[bt0_bank, target_local_e, pl.ds(dst_pos, 1)],
                        send_sem, recv_sem,
                        device_id=(dp_idx, tp_idx),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()
            return None

        lax.fori_loop(0, bt, _scatter_one_token, None, unroll=False)
        # NOTE: rely on XLA implicit fence at kernel exit to ensure all
        # async DMAs are complete before the buffer is consumed by the next
        # kernel. Explicit drain wait was attempted (v4-v6) but hung due to
        # SC sem semantics differences from TC. v3 (this) PASS confirmed.

    return body


def sc_bt0_scatter(
    tokens: jax.Array,
    topk_ids: jax.Array,
    expert_starts: jax.Array,
    *,
    bt: int,
    bt_start: int,
    top_k: int,
    local_num_experts: int,
    num_devices: int,
    a2a_max_tokens: int,
    padded_num_experts: int,
    num_bt_banks: int = 2,
    bt0_bank: int = 0,
    dp_axis_name: str = "data",
    tp_axis_name: str = "tensor",
    mesh: jax.sharding.Mesh,
) -> jax.Array:
    t_packing = tokens.shape[1]
    h_per_t = tokens.shape[2]
    tp_size = mesh.shape[tp_axis_name]

    # Ring buffer shape matching fused_moe_v2 a2a_s layout.
    out_shape = jax.ShapeDtypeStruct(
        (num_bt_banks, local_num_experts, a2a_max_tokens, t_packing, h_per_t),
        tokens.dtype,
    )

    body = _make_kernel(
        bt=bt, bt_start=bt_start, top_k=top_k,
        local_num_experts=local_num_experts, num_devices=num_devices,
        padded_num_experts=padded_num_experts,
        dp_axis_name=dp_axis_name, tp_axis_name=tp_axis_name, tp_size=tp_size,
        bt0_bank=bt0_bank,
    )

    @pl.kernel(
        out_shape=out_shape,
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1),
        scratch_shapes=[
            plsc.MemoryRef((bt, top_k), jnp.int32,
                           memory_space=pltpu.MemorySpace.SMEM),
            plsc.MemoryRef((padded_num_experts,), jnp.int32,
                           memory_space=pltpu.MemorySpace.SMEM),
            plsc.MemoryRef((padded_num_experts,), jnp.int32,
                           memory_space=pltpu.MemorySpace.SMEM),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
    )
    def kernel(
        tokens_hbm, topk_ids_hbm, expert_starts_hbm,
        a2a_out_hbm,
        topk_vmem, es_vmem, cursor_vmem,
        load_sem, send_sem, recv_sem,
    ):
        body(
            tokens_hbm, topk_ids_hbm, expert_starts_hbm,
            a2a_out_hbm,
            topk_vmem, es_vmem, cursor_vmem,
            load_sem, send_sem, recv_sem,
        )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            P((dp_axis_name, tp_axis_name)),
            P((dp_axis_name, tp_axis_name)),
            P(),
        ),
        out_specs=P(),
        check_vma=False,
    )
    def sharded(tokens_sh, topk_ids_sh, expert_starts_sh):
        return kernel(tokens_sh, topk_ids_sh, expert_starts_sh)

    return sharded(tokens, topk_ids, expert_starts)

