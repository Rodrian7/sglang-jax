"""L0 output-store microbenchmarks for fused_moe_v2.

This isolates the final output write pattern from the production kernel:

* ``direct_hbm_store`` computes a tile and attempts to write it directly to
  HBM. This is mostly a negative-control probe: current Pallas TPU kernels only
  allow normal load/store on VMEM/SMEM refs, so this mode is not in the default
  run set.
* ``staged_sync_dma`` computes a tile, stages it in VMEM, synchronizes, then DMA
  copies VMEM to HBM.
* ``staged_no_sync_dma`` computes a tile, stages it in VMEM, then immediately
  starts the VMEM-to-HBM DMA.

The benchmark intentionally reports raw trace timings only. Falcon analysis
plugins consume the traces and compiler dumps for deeper interpretation.
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

TRACE_TASK_PREFIX = "fused-moe-v2-l0-output-store"


def _output_store_l0_kernel(
    x_hbm,
    out_hbm,
    stage_vmem,
    dma_sem,
    barrier_sem,
    *,
    mode: str,
    rows: int,
    hidden_size: int,
    repeats: int,
    num_devices: int,
    tp_size: int,
):
    def get_mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

    def sync_barrier():
        for i in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=get_mesh_device_id(i),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

    def wait_dma():
        ref = out_hbm.at[pl.ds(0, rows), pl.ds(0, hidden_size)]
        pltpu.make_async_copy(src_ref=ref, dst_ref=ref, sem=dma_sem).wait()

    def body(i, _):
        tile = x_hbm.at[pl.ds(0, rows), pl.ds(0, hidden_size)][...]
        tile = tile + jnp.asarray(i + 1, dtype=tile.dtype)

        if mode == "direct_hbm_store":
            out_hbm.at[pl.ds(0, rows), pl.ds(0, hidden_size)] = tile
        else:
            stage_vmem.at[pl.ds(0, rows), pl.ds(0, hidden_size)] = tile
            if mode == "staged_sync_dma":
                sync_barrier()
            pltpu.make_async_copy(
                src_ref=stage_vmem.at[pl.ds(0, rows), pl.ds(0, hidden_size)],
                dst_ref=out_hbm.at[pl.ds(0, rows), pl.ds(0, hidden_size)],
                sem=dma_sem,
            ).start()
            wait_dma()
        return None

    jax.lax.fori_loop(0, repeats, body, None, unroll=False)


def output_store_l0(
    mesh: jax.sharding.Mesh,
    x: jax.Array,
    *,
    mode: str,
    rows: int,
    hidden_size: int,
    repeats: int,
    scope_name: str,
    ep_size: int,
) -> jax.Array:
    if mode not in {"direct_hbm_store", "staged_sync_dma", "staged_no_sync_dma"}:
        raise ValueError(f"Unsupported mode: {mode}")

    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _output_store_l0_kernel,
                mode=mode,
                rows=rows,
                hidden_size=hidden_size,
                repeats=repeats,
                num_devices=ep_size,
                tp_size=1,
            ),
            out_shape=jax.ShapeDtypeStruct((rows, hidden_size), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[hbm_spec],
                out_specs=hbm_spec,
                scratch_shapes=(
                    pltpu.VMEM((rows, hidden_size), x.dtype),
                    pltpu.SemaphoreType.DMA((1,)),
                    pltpu.SemaphoreType.BARRIER,
                ),
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
        in_specs=P("tensor", None),
        out_specs=P("tensor", None),
        check_vma=False,
    )
    def kernel(x_arg):
        return call(pltpu.with_memory_space_constraint(x_arg, pltpu.HBM))

    return kernel(x)


def _make_input(*, mesh: jax.sharding.Mesh, num_tokens: int, hidden_size: int, dtype):
    sharding = NamedSharding(mesh, P("tensor", None))
    return jax.jit(
        lambda: jnp.arange(num_tokens * hidden_size, dtype=dtype).reshape(num_tokens, hidden_size),
        out_shardings=sharding,
    )()


def run(args: argparse.Namespace) -> None:
    if args.num_tokens % args.ep_size != 0:
        raise ValueError(f"{args.num_tokens=} must be divisible by {args.ep_size=}.")
    rows = args.num_tokens // args.ep_size
    mesh = build_mesh(ep_size=args.ep_size, tp_size=1)
    x = _make_input(
        mesh=mesh,
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        dtype=jnp.bfloat16,
    )

    for mode in args.modes:
        scope_name = (
            f"{TRACE_TASK_PREFIX}-{mode}-nt_{args.num_tokens}"
            f"-rows_{rows}-h_{args.hidden_size}-rep_{args.repeats}"
        )
        trace_root = os.path.join(args.trace_root, scope_name)
        print(
            "L0_OUTPUT_STORE "
            f"name={scope_name} mode={mode} rows={rows} hidden={args.hidden_size} "
            f"repeats={args.repeats} ep={args.ep_size}",
            flush=True,
        )

        def compute():
            return output_store_l0(
                mesh,
                x,
                mode=mode,
                rows=rows,
                hidden_size=args.hidden_size,
                repeats=args.repeats,
                scope_name=scope_name,
                ep_size=args.ep_size,
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
            "L0_OUTPUT_STORE_DONE "
            f"name={scope_name} mean_ms={mean_ms:.6f} p50_ms={p50_ms:.6f} "
            f"samples={times}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        default="staged_sync_dma,staged_no_sync_dma",
        help="Comma-separated modes.",
    )
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trace-root", default="/tmp/tpu_logs/v2_l0_output_store_trace")
    args = parser.parse_args()
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    return args


if __name__ == "__main__":
    run(parse_args())
