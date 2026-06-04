"""L1 microbenchmarks for fused_moe_v2 core stages.

This benchmark intentionally does not report derived metrics. Its contract is
to produce profiler traces plus HLO/LLO dumps with stable kernel names; Falcon
analysis plugins consume those artifacts.
"""

from __future__ import annotations

import argparse
import functools
import os
import statistics

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from benchmark.moe.utils import build_mesh
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.kernels.fused_moe.v2.kernel import get_dtype_packing

TRACE_TASK_PREFIX = "fused-moe-v2-l1-weight-dma"


def _weight_dma_l1_kernel(
    w1_hbm,
    w2_hbm,
    w3_hbm,
    w1_scale_hbm,
    w2_scale_hbm,
    w3_scale_hbm,
    out_ref,
    w1_vmem,
    w3_vmem,
    w2_vmem,
    w1_scale_vmem,
    w3_scale_vmem,
    w2_scale_vmem,
    sems,
    *,
    path: str,
    bf: int,
    payload_bf: int,
    num_bf_tiles: int,
    num_expert_iters: int,
    quant_block_k: int,
    issue_together: bool,
    co_drain: bool,
    w2_fetch_order: str,
    w2_fetch_priority: int,
    drain_policy: str,
):
    local_e_id = jnp.int32(0)
    t_packing = w1_vmem.shape[1]
    h_per_t = w1_vmem.shape[2]

    def start_w1(slot, bf_id, priority=1):
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[
                    local_e_id,
                    pl.ds(p * h_per_t, h_per_t),
                    pl.ds(bf_id * bf, payload_bf),
                ],
                dst_ref=w1_vmem.at[slot, p, pl.ds(0, h_per_t), pl.ds(0, payload_bf)],
                sem=sems.at[slot, 0],
            ).start(priority=priority)
            pltpu.make_async_copy(
                src_ref=w1_scale_hbm.at[
                    local_e_id,
                    pl.ds(p * h_per_t // quant_block_k, h_per_t // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(bf_id * bf, payload_bf),
                ],
                dst_ref=w1_scale_vmem.at[
                    slot,
                    p,
                    pl.ds(0, h_per_t // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(0, payload_bf),
                ],
                sem=sems.at[slot, 0],
            ).start(priority=priority)

    def wait_w1(slot):
        pltpu.make_async_copy(
            src_ref=w1_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, h_per_t),
                pl.ds(0, payload_bf),
            ],
            dst_ref=w1_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, h_per_t),
                pl.ds(0, payload_bf),
            ],
            sem=sems.at[slot, 0],
        ).wait()
        pltpu.make_async_copy(
            src_ref=w1_scale_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, h_per_t // quant_block_k),
                pl.ds(0, 1),
                pl.ds(0, payload_bf),
            ],
            dst_ref=w1_scale_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, h_per_t // quant_block_k),
                pl.ds(0, 1),
                pl.ds(0, payload_bf),
            ],
            sem=sems.at[slot, 0],
        ).wait()

    def start_w3(slot, bf_id, priority=1):
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w3_hbm.at[
                    local_e_id,
                    pl.ds(p * h_per_t, h_per_t),
                    pl.ds(bf_id * bf, payload_bf),
                ],
                dst_ref=w3_vmem.at[slot, p, pl.ds(0, h_per_t), pl.ds(0, payload_bf)],
                sem=sems.at[slot, 1],
            ).start(priority=priority)
            pltpu.make_async_copy(
                src_ref=w3_scale_hbm.at[
                    local_e_id,
                    pl.ds(p * h_per_t // quant_block_k, h_per_t // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(bf_id * bf, payload_bf),
                ],
                dst_ref=w3_scale_vmem.at[
                    slot,
                    p,
                    pl.ds(0, h_per_t // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(0, payload_bf),
                ],
                sem=sems.at[slot, 1],
            ).start(priority=priority)

    def wait_w3(slot):
        pltpu.make_async_copy(
            src_ref=w3_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, h_per_t),
                pl.ds(0, payload_bf),
            ],
            dst_ref=w3_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, h_per_t),
                pl.ds(0, payload_bf),
            ],
            sem=sems.at[slot, 1],
        ).wait()
        pltpu.make_async_copy(
            src_ref=w3_scale_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, h_per_t // quant_block_k),
                pl.ds(0, 1),
                pl.ds(0, payload_bf),
            ],
            dst_ref=w3_scale_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, h_per_t // quant_block_k),
                pl.ds(0, 1),
                pl.ds(0, payload_bf),
            ],
            sem=sems.at[slot, 1],
        ).wait()

    def start_w2(slot, bf_id, priority=0):
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[
                    local_e_id,
                    pl.ds(bf_id * bf, payload_bf),
                    pl.ds(p * h_per_t, h_per_t),
                ],
                dst_ref=w2_vmem.at[slot, p, pl.ds(0, payload_bf), pl.ds(0, h_per_t)],
                sem=sems.at[slot, 2],
            ).start(priority=priority)
            pltpu.make_async_copy(
                src_ref=w2_scale_hbm.at[
                    local_e_id,
                    pl.ds(bf_id * bf // quant_block_k, payload_bf // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(p * h_per_t, h_per_t),
                ],
                dst_ref=w2_scale_vmem.at[
                    slot,
                    p,
                    pl.ds(0, payload_bf // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(0, h_per_t),
                ],
                sem=sems.at[slot, 2],
            ).start(priority=priority)

    def wait_w2(slot):
        pltpu.make_async_copy(
            src_ref=w2_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, payload_bf),
                pl.ds(0, h_per_t),
            ],
            dst_ref=w2_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, payload_bf),
                pl.ds(0, h_per_t),
            ],
            sem=sems.at[slot, 2],
        ).wait()
        pltpu.make_async_copy(
            src_ref=w2_scale_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, payload_bf // quant_block_k),
                pl.ds(0, 1),
                pl.ds(0, h_per_t),
            ],
            dst_ref=w2_scale_vmem.at[
                slot,
                pl.ds(0, t_packing),
                pl.ds(0, payload_bf // quant_block_k),
                pl.ds(0, 1),
                pl.ds(0, h_per_t),
            ],
            sem=sems.at[slot, 2],
        ).wait()

    def start_w13_w2(slot, bf_id):
        if w2_fetch_order == "before_w13":
            start_w2(slot, bf_id, priority=w2_fetch_priority)
            start_w1(slot, bf_id, priority=1)
            start_w3(slot, bf_id, priority=1)
        else:
            start_w1(slot, bf_id, priority=1)
            start_w3(slot, bf_id, priority=1)
            start_w2(slot, bf_id, priority=w2_fetch_priority)

    def wait_w13_w2(slot):
        wait_w1(slot)
        wait_w3(slot)
        wait_w2(slot)

    def start_sem_self_w1(slot, priority=1):
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w1_vmem.at[slot, p, pl.ds(0, h_per_t), pl.ds(0, payload_bf)],
                dst_ref=w1_vmem.at[slot, p, pl.ds(0, h_per_t), pl.ds(0, payload_bf)],
                sem=sems.at[slot, 0],
            ).start(priority=priority)
            pltpu.make_async_copy(
                src_ref=w1_scale_vmem.at[
                    slot,
                    p,
                    pl.ds(0, h_per_t // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(0, payload_bf),
                ],
                dst_ref=w1_scale_vmem.at[
                    slot,
                    p,
                    pl.ds(0, h_per_t // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(0, payload_bf),
                ],
                sem=sems.at[slot, 0],
            ).start(priority=priority)

    def start_sem_self_w3(slot, priority=1):
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w3_vmem.at[slot, p, pl.ds(0, h_per_t), pl.ds(0, payload_bf)],
                dst_ref=w3_vmem.at[slot, p, pl.ds(0, h_per_t), pl.ds(0, payload_bf)],
                sem=sems.at[slot, 1],
            ).start(priority=priority)
            pltpu.make_async_copy(
                src_ref=w3_scale_vmem.at[
                    slot,
                    p,
                    pl.ds(0, h_per_t // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(0, payload_bf),
                ],
                dst_ref=w3_scale_vmem.at[
                    slot,
                    p,
                    pl.ds(0, h_per_t // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(0, payload_bf),
                ],
                sem=sems.at[slot, 1],
            ).start(priority=priority)

    def start_sem_self_w2(slot, priority=1):
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w2_vmem.at[slot, p, pl.ds(0, payload_bf), pl.ds(0, h_per_t)],
                dst_ref=w2_vmem.at[slot, p, pl.ds(0, payload_bf), pl.ds(0, h_per_t)],
                sem=sems.at[slot, 2],
            ).start(priority=priority)
            pltpu.make_async_copy(
                src_ref=w2_scale_vmem.at[
                    slot,
                    p,
                    pl.ds(0, payload_bf // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(0, h_per_t),
                ],
                dst_ref=w2_scale_vmem.at[
                    slot,
                    p,
                    pl.ds(0, payload_bf // quant_block_k),
                    pl.ds(0, 1),
                    pl.ds(0, h_per_t),
                ],
                sem=sems.at[slot, 2],
            ).start(priority=priority)

    def start_sem_self_w13_w2(slot):
        if w2_fetch_order == "before_w13":
            start_sem_self_w2(slot, priority=w2_fetch_priority)
            start_sem_self_w1(slot, priority=1)
            start_sem_self_w3(slot, priority=1)
        else:
            start_sem_self_w1(slot, priority=1)
            start_sem_self_w3(slot, priority=1)
            start_sem_self_w2(slot, priority=w2_fetch_priority)

    def run_simple_issue_wait():
        for bf_id in range(num_bf_tiles):
            slot = bf_id % 2
            if path == "w1":
                start_w1(slot, bf_id)
                wait_w1(slot)
            elif path == "w3":
                start_w3(slot, bf_id)
                wait_w3(slot)
            elif path == "w2":
                start_w2(slot, bf_id)
                wait_w2(slot)
            elif path == "w13":
                if issue_together:
                    start_w1(slot, bf_id)
                    start_w3(slot, bf_id)
                    wait_w1(slot)
                    wait_w3(slot)
                else:
                    start_w1(slot, bf_id)
                    wait_w1(slot)
                    start_w3(slot, bf_id)
                    wait_w3(slot)
            elif path == "w13_w2":
                start_w13_w2(slot, bf_id)
                wait_w13_w2(slot)
            else:
                raise ValueError(f"Unsupported L1 weight DMA path: {path}")

    if path in ("w1", "w3", "w2", "w13", "w13_w2"):
        for _expert_i in range(num_expert_iters):
            run_simple_issue_wait()
    elif path == "pipeline_w13_w2":
        for _expert_i in range(num_expert_iters):
            if drain_policy == "end":
                for bf_id in range(num_bf_tiles):
                    start_w13_w2(bf_id % 2, bf_id)
                for bf_id in range(num_bf_tiles):
                    wait_w13_w2(bf_id % 2)
            else:
                if num_bf_tiles >= 1:
                    start_w13_w2(0, 0)
                if num_bf_tiles >= 2:
                    start_w13_w2(1, 1)
                for bf_id in range(num_bf_tiles):
                    slot = bf_id % 2
                    wait_w1(slot)
                    wait_w3(slot)
                    wait_w2(slot)
                    next_bf_id = bf_id + 2
                    if next_bf_id < num_bf_tiles:
                        start_w13_w2(slot, next_bf_id)
    elif path == "empty_loop":
        for _expert_i in range(num_expert_iters):
            for bf_id in range(num_bf_tiles):
                _ = bf_id
    elif path == "sem_self_wait":
        for _expert_i in range(num_expert_iters):
            if drain_policy == "end":
                for bf_id in range(num_bf_tiles):
                    start_sem_self_w13_w2(bf_id % 2)
                for bf_id in range(num_bf_tiles):
                    wait_w13_w2(bf_id % 2)
            else:
                if num_bf_tiles >= 1:
                    start_sem_self_w13_w2(0)
                if num_bf_tiles >= 2:
                    start_sem_self_w13_w2(1)
                for bf_id in range(num_bf_tiles):
                    slot = bf_id % 2
                    wait_w1(slot)
                    wait_w3(slot)
                    wait_w2(slot)
                    next_bf_id = bf_id + 2
                    if next_bf_id < num_bf_tiles:
                        start_sem_self_w13_w2(slot)
    else:
        raise ValueError(f"Unsupported L1 weight DMA path: {path}")


@functools.partial(
    jax.jit,
    static_argnames=[
        "mesh",
        "path",
        "hidden_size",
        "intermediate_size",
        "bf",
        "payload_bf",
        "num_bf_tiles",
        "num_expert_iters",
        "quant_block_k",
        "scope_name",
        "issue_together",
        "co_drain",
        "w2_fetch_order",
        "w2_fetch_priority",
        "drain_policy",
    ],
)
def weight_dma_l1(
    mesh: jax.sharding.Mesh,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    w3_scale: jax.Array,
    *,
    path: str,
    hidden_size: int,
    intermediate_size: int,
    bf: int,
    payload_bf: int,
    num_bf_tiles: int,
    num_expert_iters: int,
    quant_block_k: int,
    scope_name: str,
    issue_together: bool,
    co_drain: bool,
    w2_fetch_order: str,
    w2_fetch_priority: int,
    drain_policy: str,
):
    t_packing = get_dtype_packing(jnp.bfloat16)
    h_per_t = hidden_size // t_packing
    wb_slots = 2

    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    scratch_shapes = (
        pltpu.VMEM((wb_slots, t_packing, h_per_t, bf), w1.dtype),
        pltpu.VMEM((wb_slots, t_packing, h_per_t, bf), w3.dtype),
        pltpu.VMEM((wb_slots, t_packing, bf, h_per_t), w2.dtype),
        pltpu.VMEM((wb_slots, t_packing, h_per_t // quant_block_k, 1, bf), jnp.float32),
        pltpu.VMEM((wb_slots, t_packing, h_per_t // quant_block_k, 1, bf), jnp.float32),
        pltpu.VMEM((wb_slots, t_packing, bf // quant_block_k, 1, h_per_t), jnp.float32),
        pltpu.SemaphoreType.DMA((wb_slots, 3)),
    )

    call = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _weight_dma_l1_kernel,
                path=path,
                bf=bf,
                payload_bf=payload_bf,
                num_bf_tiles=num_bf_tiles,
                num_expert_iters=num_expert_iters,
                quant_block_k=quant_block_k,
                issue_together=issue_together,
                co_drain=co_drain,
                w2_fetch_order=w2_fetch_order,
                w2_fetch_priority=w2_fetch_priority,
                drain_policy=drain_policy,
            ),
            out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    hbm_spec,
                    hbm_spec,
                    hbm_spec,
                    hbm_spec,
                    hbm_spec,
                    hbm_spec,
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                has_side_effects=True,
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
            name=scope_name,
        )
    )

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            P("tensor", None, None),
            P("tensor", None, None),
            P("tensor", None, None),
            P("tensor", None, None, None),
            P("tensor", None, None, None),
            P("tensor", None, None, None),
        ),
        out_specs=P(),
        check_vma=False,
    )
    def kernel(w1_arg, w2_arg, w3_arg, w1_scale_arg, w2_scale_arg, w3_scale_arg):
        return call(
            pltpu.with_memory_space_constraint(w1_arg, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2_arg, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3_arg, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1_scale_arg, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2_scale_arg, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3_scale_arg, pltpu.HBM),
        )

    return kernel(w1, w2, w3, w1_scale, w2_scale, w3_scale)


def _device_zeros(shape, dtype, sharding):
    return jax.jit(lambda: jnp.zeros(shape, dtype=dtype), out_shardings=sharding)()


def _make_inputs(
    *,
    mesh: jax.sharding.Mesh,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    quant_block_k: int,
    weight_dtype: jnp.dtype,
):
    weight_sharding = NamedSharding(mesh, P("tensor", None, None))
    scale_sharding = NamedSharding(mesh, P("tensor", None, None, None))
    w1 = _device_zeros((num_experts, hidden_size, intermediate_size), weight_dtype, weight_sharding)
    w3 = _device_zeros((num_experts, hidden_size, intermediate_size), weight_dtype, weight_sharding)
    w2 = _device_zeros((num_experts, intermediate_size, hidden_size), weight_dtype, weight_sharding)
    w1_scale = _device_zeros(
        (num_experts, hidden_size // quant_block_k, 1, intermediate_size),
        jnp.float32,
        scale_sharding,
    )
    w3_scale = _device_zeros(
        (num_experts, hidden_size // quant_block_k, 1, intermediate_size),
        jnp.float32,
        scale_sharding,
    )
    w2_scale = _device_zeros(
        (num_experts, intermediate_size // quant_block_k, 1, hidden_size),
        jnp.float32,
        scale_sharding,
    )
    return w1, w2, w3, w1_scale, w2_scale, w3_scale


def _scope_name(
    path: str,
    bf: int,
    payload_bf: int,
    num_bf_tiles: int,
    num_expert_iters: int,
    quant_block_k: int,
    w2_fetch_order: str,
    w2_fetch_priority: int,
    drain_policy: str,
) -> str:
    return (
        f"{TRACE_TASK_PREFIX}-{path}-bf_{bf}-payload_{payload_bf}"
        f"-nbf_{num_bf_tiles}-ne_{num_expert_iters}-qbk_{quant_block_k}"
        f"-w2_{w2_fetch_order}_p{w2_fetch_priority}-drain_{drain_policy}"
    )


def run(args: argparse.Namespace) -> None:
    mesh = build_mesh(ep_size=args.ep_size, tp_size=1)
    if args.num_experts % args.ep_size != 0:
        raise ValueError(f"{args.num_experts=} must be divisible by {args.ep_size=}.")
    if args.hidden_size % get_dtype_packing(jnp.bfloat16) != 0:
        raise ValueError(f"{args.hidden_size=} must be divisible by bf16 packing.")
    if args.hidden_size % args.quant_block_k != 0:
        raise ValueError(f"{args.hidden_size=} must be divisible by {args.quant_block_k=}.")
    if args.intermediate_size % args.quant_block_k != 0:
        raise ValueError(f"{args.intermediate_size=} must be divisible by {args.quant_block_k=}.")
    if args.bf % args.quant_block_k != 0:
        raise ValueError(f"{args.bf=} must be divisible by {args.quant_block_k=}.")
    if args.payload_bf is None:
        args.payload_bf = args.bf
    if args.payload_bf <= 0 or args.payload_bf > args.bf:
        raise ValueError(f"Expected 0 < {args.payload_bf=} <= {args.bf=}.")
    if args.payload_bf % args.quant_block_k != 0:
        raise ValueError(f"{args.payload_bf=} must be divisible by {args.quant_block_k=}.")
    if args.bf * args.num_bf_tiles > args.intermediate_size:
        raise ValueError("bf * num_bf_tiles must not exceed intermediate_size.")
    if args.num_expert_iters <= 0:
        raise ValueError(f"{args.num_expert_iters=} must be positive.")

    weight_dtype = jnp.float8_e4m3fn if args.fp8 else jnp.bfloat16
    w1, w2, w3, w1_scale, w2_scale, w3_scale = _make_inputs(
        mesh=mesh,
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        quant_block_k=args.quant_block_k,
        weight_dtype=weight_dtype,
    )

    for path in args.paths:
        scope_name = _scope_name(
            path,
            args.bf,
            args.payload_bf,
            args.num_bf_tiles,
            args.num_expert_iters,
            args.quant_block_k,
            args.w2_fetch_order,
            args.w2_fetch_priority,
            args.drain_policy,
        )
        trace_root = os.path.join(args.trace_root, scope_name)
        print(
            "L1_WEIGHT_DMA "
            f"name={scope_name} path={path} bf={args.bf} "
            f"payload_bf={args.payload_bf} num_bf_tiles={args.num_bf_tiles} "
            f"num_expert_iters={args.num_expert_iters} qbk={args.quant_block_k} "
            f"w2_fetch_order={args.w2_fetch_order} "
            f"w2_fetch_priority={args.w2_fetch_priority} drain_policy={args.drain_policy} "
            f"experts={args.num_experts} hidden={args.hidden_size} "
            f"intermediate={args.intermediate_size} ep={args.ep_size} "
            f"weight_dtype={jnp.dtype(weight_dtype).name}",
            flush=True,
        )

        def compute():
            return weight_dma_l1(
                mesh,
                w1,
                w2,
                w3,
                w1_scale,
                w2_scale,
                w3_scale,
                path=path,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                bf=args.bf,
                payload_bf=args.payload_bf,
                num_bf_tiles=args.num_bf_tiles,
                num_expert_iters=args.num_expert_iters,
                quant_block_k=args.quant_block_k,
                scope_name=scope_name,
                issue_together=args.issue_together,
                co_drain=args.co_drain,
                w2_fetch_order=args.w2_fetch_order,
                w2_fetch_priority=args.w2_fetch_priority,
                drain_policy=args.drain_policy,
            )

        times = multiple_iteration_timeit_from_trace(
            compute_func=compute,
            data_generator=lambda: (),
            task=TRACE_TASK_PREFIX + ".*",
            tries=args.iters,
            warmup=args.warmup_iters,
            trace_root=trace_root,
        )
        mean_ms = statistics.mean(times) if times else float("nan")
        p50_ms = statistics.median(times) if times else float("nan")
        print(
            "L1_WEIGHT_DMA_DONE "
            f"name={scope_name} mean_ms={mean_ms:.6f} p50_ms={p50_ms:.6f} "
            f"samples={times}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="w1,w3,w13,w2", help="Comma-separated L1 paths.")
    parser.add_argument("--bf", type=int, default=512)
    parser.add_argument(
        "--payload-bf",
        type=int,
        default=None,
        help="Actual copied bf width. Defaults to --bf; use a small value for tiny-payload issue/wait probes.",
    )
    parser.add_argument("--num-bf-tiles", type=int, default=1)
    parser.add_argument(
        "--num-expert-iters",
        type=int,
        default=1,
        help="Number of repeated expert-shaped iterations for L1 paths.",
    )
    parser.add_argument("--quant-block-k", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--trace-root", default="/tmp/tpu_logs/v2_l1_trace")
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--issue-together", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--co-drain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--w2-fetch-order",
        choices=["after_w13", "before_w13"],
        default="after_w13",
        help="W2 issue order in w13_w2 and pipeline_w13_w2 paths.",
    )
    parser.add_argument(
        "--w2-fetch-priority",
        type=int,
        choices=[0, 1],
        default=1,
        help="W2 DMA priority in w13_w2 and pipeline_w13_w2 paths.",
    )
    parser.add_argument(
        "--drain-policy",
        choices=["production", "end"],
        default="production",
        help="production waits W1/W3/W2 per bf tile; end issues all tiles before draining waits.",
    )
    args = parser.parse_args()
    args.paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    return args


if __name__ == "__main__":
    run(parse_args())
