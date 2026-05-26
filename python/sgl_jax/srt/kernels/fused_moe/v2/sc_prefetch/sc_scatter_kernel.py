"""SparseCore-driven bt0 a2a scatter for fused_moe_v2 prefetch (v3 — correct cursor).

v3 key changes from v2:
- Uses SMEM cursor (one int32 per expert) to correctly compute dst offset
  matching fused_moe_v2 internal layout (expert_starts + per-expert running count).
- Output a2a_s buffer is consumable by fused_moe_v2 with skip_bt0_scatter=True.
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
):
    def body(
        tokens_hbm,
        topk_ids_hbm,
        expert_starts_hbm,
        a2a_out_hbm,
        topk_vmem,
        es_vmem,
        cursor_vmem,
        valid_count_vmem,
        load_sem,
        send_sem,
        recv_sem,
    ):
        dp_rank = lax.axis_index(dp_axis_name)
        tp_rank = lax.axis_index(tp_axis_name)
        my_id = dp_rank * tp_size + tp_rank

        # Init cursor and valid_count.
        for e in range(padded_num_experts):
            cursor_vmem[e] = jnp.int32(0)
        valid_count_vmem[0] = jnp.int32(0)

        # Load topk_ids + expert_starts to SMEM.
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

        # Scan tokens × topk, issue DMAs with correct dst offsets.
        # Use synchronous DMA (start + wait per iteration) to guarantee all
        # writes are complete by kernel exit. This serializes DMAs but is
        # correct; concurrent versions need a precise valid-count drain loop
        # (next-step optimization, requires SMEM-int dynamic loop bound).
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
                valid_count_vmem[0] = valid_count_vmem[0] + inc

                dst_pos = es_vmem[e_id_safe] + cur_off

                @pl.when(jnp.logical_and(is_valid, target_dev == my_id))
                def _local(t_id=t_id, target_local_e=target_local_e, dst_pos=dst_pos):
                    dma = pltpu.make_async_copy(
                        tokens_hbm.at[pl.ds(bt_start + t_id, 1)],
                        a2a_out_hbm.at[target_local_e, pl.ds(dst_pos, 1)],
                        recv_sem,
                    )
                    dma.start()
                    dma.wait()

                @pl.when(jnp.logical_and(is_valid, target_dev != my_id))
                def _remote(
                    t_id=t_id, target_local_e=target_local_e, dst_pos=dst_pos,
                    target_dev=target_dev,
                ):
                    dp_idx = target_dev // tp_size
                    tp_idx = target_dev % tp_size
                    dma = pltpu.make_async_remote_copy(
                        tokens_hbm.at[pl.ds(bt_start + t_id, 1)],
                        a2a_out_hbm.at[target_local_e, pl.ds(dst_pos, 1)],
                        send_sem, recv_sem,
                        device_id=(dp_idx, tp_idx),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    )
                    dma.start()
                    dma.wait()
            return None

        lax.fori_loop(0, bt, _scatter_one_token, None, unroll=False)

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
    dp_axis_name: str = "data",
    tp_axis_name: str = "tensor",
    mesh: jax.sharding.Mesh,
) -> jax.Array:
    t_packing = tokens.shape[1]
    h_per_t = tokens.shape[2]
    tp_size = mesh.shape[tp_axis_name]

    out_shape = jax.ShapeDtypeStruct(
        (local_num_experts, a2a_max_tokens, t_packing, h_per_t),
        tokens.dtype,
    )

    body = _make_kernel(
        bt=bt, bt_start=bt_start, top_k=top_k,
        local_num_experts=local_num_experts, num_devices=num_devices,
        padded_num_experts=padded_num_experts,
        dp_axis_name=dp_axis_name, tp_axis_name=tp_axis_name, tp_size=tp_size,
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
            plsc.MemoryRef((1,), jnp.int32,
                           memory_space=pltpu.MemorySpace.SMEM),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
    )
    def kernel(
        tokens_hbm, topk_ids_hbm, expert_starts_hbm,
        a2a_out_hbm,
        topk_vmem, es_vmem, cursor_vmem, valid_count_vmem,
        load_sem, send_sem, recv_sem,
    ):
        body(
            tokens_hbm, topk_ids_hbm, expert_starts_hbm,
            a2a_out_hbm,
            topk_vmem, es_vmem, cursor_vmem, valid_count_vmem,
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
