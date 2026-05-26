#!/usr/bin/env python3
"""poc_batch_dma_scatter.py — Validate batched DMA scatter vs per-token DMA.

Compares scatter dispatch strategies (local DMAs only, single device):
  1) dma_single:       1 DMA × 2048 rows (24MB contiguous, theoretical best)
  2) dma_fori_2048:    fori×2048, 1-row DMA each (pure loop+DMA overhead)
  3) per_token:        fori×256, inner×8, SMEM routing (current kernel approach)
  4) batched_fori:     fori×384, multi-row DMA, SMEM routing (proposed approach)
  5) batched_static:   static×384, unrolled version of 4

Row = (2, 3072) bf16 = 12KB.  Total = 2048 rows = 24MB.

Routing data is passed via scalar_prefetch (auto-placed in SMEM).
"""

import os, sys, time, functools, traceback
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HIDDEN = 6144
PACKING = 2
H_PER_T = HIDDEN // PACKING
BT = 256
TOP_K = 8
NUM_ENTRIES = BT * TOP_K
PADDED_E = 384


def make_routing_data():
    key = jax.random.PRNGKey(42)
    topk_ids = jax.random.randint(key, (NUM_ENTRIES,), 0, PADDED_E, dtype=jnp.int32)
    counts = jnp.zeros(PADDED_E, jnp.int32).at[topk_ids].add(1)
    starts = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(counts)[:-1]])
    return topk_ids, counts, starts


# ─── Raw DMA baselines (no SMEM routing) ───


def _make_dma_single():
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.ANY),
                pl.BlockSpec(memory_space=pltpu.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
    )
    def f(src, dst, out, sem):
        pltpu.make_async_copy(
            src_ref=src.at[pl.ds(0, NUM_ENTRIES)],
            dst_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            sem=sem,
        ).start()
        pltpu.make_async_copy(
            src_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            dst_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            sem=sem,
        ).wait()
    return f


def _make_dma_fori_per_token():
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.ANY),
                pl.BlockSpec(memory_space=pltpu.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
    )
    def f(src, dst, out, sem):
        def _b(i, _):
            pltpu.make_async_copy(
                src_ref=src.at[pl.ds(i, 1)],
                dst_ref=dst.at[pl.ds(i, 1)],
                sem=sem,
            ).start()
            return None
        lax.fori_loop(0, jnp.int32(NUM_ENTRIES), _b, None, unroll=False)
        pltpu.make_async_copy(
            src_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            dst_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            sem=sem,
        ).wait()
    return f


# ─── Per-token scatter: fori×256, inner×8, SMEM routing ───


def _make_per_token():
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2, grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.ANY),
                pl.BlockSpec(memory_space=pltpu.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA,
                pltpu.SMEM((PADDED_E,), jnp.int32),
            ],
        ),
    )
    def f(topk_ref, starts_ref, src, dst, out, sem, offsets_smem):
        def _init(i, _):
            offsets_smem[i] = jnp.int32(0)
            return None
        lax.fori_loop(0, jnp.int32(PADDED_E), _init, None, unroll=False)

        def _scatter(t_id, _):
            for k_id in range(TOP_K):
                e_id = topk_ref[0, t_id * TOP_K + k_id]
                offset = offsets_smem[e_id]
                start = starts_ref[0, e_id] + offset
                offsets_smem[e_id] = offset + jnp.int32(1)
                pltpu.make_async_copy(
                    src_ref=src.at[pl.ds(t_id, 1)],
                    dst_ref=dst.at[pl.ds(start, 1)],
                    sem=sem,
                ).start()
            return None
        lax.fori_loop(0, jnp.int32(BT), _scatter, None, unroll=False)

        pltpu.make_async_copy(
            src_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            dst_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            sem=sem,
        ).wait()
    return f


# ─── Batched scatter: fori×384, multi-row DMAs ───


def _make_batched_fori():
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2, grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.ANY),
                pl.BlockSpec(memory_space=pltpu.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
    )
    def f(counts_ref, starts_ref, src, dst, out, sem):
        def _scatter(e_id, cursor):
            count = counts_ref[0, e_id]
            start = starts_ref[0, e_id]
            @pl.when(count > 0)
            def _(cursor=cursor, count=count, start=start):
                pltpu.make_async_copy(
                    src_ref=src.at[pl.ds(cursor, count)],
                    dst_ref=dst.at[pl.ds(start, count)],
                    sem=sem,
                ).start()
            return cursor + count
        lax.fori_loop(0, jnp.int32(PADDED_E), _scatter, jnp.int32(0), unroll=False)

        pltpu.make_async_copy(
            src_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            dst_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            sem=sem,
        ).wait()
    return f


# ─── Batched scatter: static×384, unrolled ───


def _make_batched_static():
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2, grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.ANY),
                pl.BlockSpec(memory_space=pltpu.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
    )
    def f(counts_ref, starts_ref, src, dst, out, sem):
        cursor = jnp.int32(0)
        for e_id in range(PADDED_E):
            count = counts_ref[0, e_id]
            start = starts_ref[0, e_id]
            @pl.when(count > 0)
            def _(cursor=cursor, count=count, start=start):
                pltpu.make_async_copy(
                    src_ref=src.at[pl.ds(cursor, count)],
                    dst_ref=dst.at[pl.ds(start, count)],
                    sem=sem,
                ).start()
            cursor = cursor + count

        pltpu.make_async_copy(
            src_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            dst_ref=dst.at[pl.ds(0, NUM_ENTRIES)],
            sem=sem,
        ).wait()
    return f


# ─── benchmark ───


def bench_fn(fn, args, warmup=5, iters=30):
    for _ in range(warmup):
        fn(*args).block_until_ready()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2], times[:5]


def main():
    jax.distributed.initialize()
    pid = jax.process_index()
    print(f"[p{pid}] devices={jax.device_count()}", flush=True)
    if pid != 0:
        return

    topk_ids, counts, starts = make_routing_data()
    nonzero = int(jnp.sum(counts > 0))
    print(f"\nRouting: {NUM_ENTRIES} entries across {PADDED_E} experts, "
          f"{nonzero} non-zero (avg {NUM_ENTRIES/nonzero:.1f} entries/expert)", flush=True)
    print(f"Row size: {PACKING}×{H_PER_T}×2B = {PACKING*H_PER_T*2//1024}KB", flush=True)

    src = jnp.zeros((NUM_ENTRIES, PACKING, H_PER_T), jnp.bfloat16)
    dst = jnp.zeros((NUM_ENTRIES, PACKING, H_PER_T), jnp.bfloat16)

    # scalar_prefetch needs leading grid dim: (1, ...)
    topk_ids_sp = topk_ids[None, :]
    counts_sp = counts[None, :]
    starts_sp = starts[None, :]

    # 1) Raw baselines
    print("\n--- Raw DMA baselines (no SMEM routing) ---", flush=True)
    try:
        fn = _make_dma_single()
        med, samp = bench_fn(fn, (src, dst))
        s = ", ".join(f"{x:.0f}" for x in samp)
        print(f"  1 DMA × {NUM_ENTRIES} rows:     {med:>7.0f} μs  [{s}]", flush=True)
    except Exception as e:
        print(f"  1 DMA: ERR {e}", flush=True)
        traceback.print_exc()

    try:
        fn = _make_dma_fori_per_token()
        med, samp = bench_fn(fn, (src, dst))
        s = ", ".join(f"{x:.0f}" for x in samp)
        print(f"  fori×{NUM_ENTRIES}, 1-row each: {med:>7.0f} μs  [{s}]", flush=True)
    except Exception as e:
        print(f"  fori×{NUM_ENTRIES}: ERR {e}", flush=True)
        traceback.print_exc()

    # 2) Per-token scatter with SMEM routing
    print("\n--- Per-token scatter (fori×256, inner×8, SMEM routing) ---", flush=True)
    try:
        fn = _make_per_token()
        med, samp = bench_fn(fn, (topk_ids_sp, starts_sp, src, dst))
        s = ", ".join(f"{x:.0f}" for x in samp)
        print(f"  per_token:           {med:>7.0f} μs  [{s}]", flush=True)
    except Exception as e:
        print(f"  per_token: ERR {e}", flush=True)
        traceback.print_exc()

    # 3) Batched scatter with SMEM routing
    print("\n--- Batched scatter (multi-row DMAs, SMEM routing) ---", flush=True)
    try:
        fn = _make_batched_fori()
        med, samp = bench_fn(fn, (counts_sp, starts_sp, src, dst))
        s = ", ".join(f"{x:.0f}" for x in samp)
        print(f"  batched_fori×{PADDED_E}:    {med:>7.0f} μs  [{s}]", flush=True)
    except Exception as e:
        print(f"  batched_fori: ERR {e}", flush=True)
        traceback.print_exc()

    try:
        fn = _make_batched_static()
        med, samp = bench_fn(fn, (counts_sp, starts_sp, src, dst))
        s = ", ".join(f"{x:.0f}" for x in samp)
        print(f"  batched_static×{PADDED_E}: {med:>7.0f} μs  [{s}]", flush=True)
    except Exception as e:
        print(f"  batched_static: ERR {e}", flush=True)
        traceback.print_exc()

    print(f"\n[p0] done", flush=True)


if __name__ == "__main__":
    main()
