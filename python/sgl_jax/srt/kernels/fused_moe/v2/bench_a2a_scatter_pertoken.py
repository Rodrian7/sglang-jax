"""Per-token streaming scatter microbench (Pallas remote DMA) — the way the kernel scatters.

The fused-MoE v2 kernel does NOT scatter via a bulk collective; it issues **per-token
point-to-point async DMA** (`pltpu.make_async_remote_copy`, 1 row at a time) in batches,
overlapped with compute (`kernel.py:866-924`). This standalone microbench reproduces that
flat per-token scatter and times it, to compare against the bulk `jax.lax.all_to_all`
flat/hierarchical numbers (bench_a2a_flat_vs_hier.py / exp-7pi390q5vb).

("Hierarchical per-token" is not built — hierarchical is a bulk batching optimization; per
token you'd just send each row directly = flat. So flat IS the per-token case.)

Each device sends N rows (= 512 tok × top-8 = 4096) of H=8192 to the other devices, balanced
`rows_per_dest = N/num_devices` to each. Row r → dest `r//rpd`, landing on the dest at slot
`my_id*rpd + r%rpd`. Same 67 MB bf16 / 33.5 MB fp8 per-device payload as the bulk bench.

Mirrors kernel.py: `make_async_remote_copy(...).start()` per row, fired in CHUNK-sized
batches each drained via `make_async_copy(ref,ref,sem).wait()` (bounds outstanding DMAs),
`sync_barrier()` (kernel.py:555-564), `CompilerParams(collective_id=0)`.

Correctness check runs single-host only (pull recv to host); multi-host just times — same as
bench_v2. Run:
  single-host gate:  BENCH_SINGLE_HOST=1 ... (ep=8, v7x-8)
  real number:       4-host v7x-32 (ep=32, 2x2x4)
Env: BENCH_TOTAL_ROWS=4096 BENCH_H=8192 BENCH_CHUNK=128 BENCH_DTYPE={bf16,fp8,both}
     BENCH_WARMUP=3 BENCH_ITERS=20 BENCH_CHECK=1
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:.1f}s][p{jax.process_index()}] {msg}", flush=True)


if os.environ.get("BENCH_SINGLE_HOST", "0") != "1":
    jax.distributed.initialize()

P = jax.sharding.PartitionSpec
DP, TP = "data", "tensor"

N = int(os.environ.get("BENCH_TOTAL_ROWS", "4096"))  # rows each device sends
H = int(os.environ.get("BENCH_H", "8192"))
CHUNK = int(os.environ.get("BENCH_CHUNK", "128"))  # rows fired per batch before draining sends
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
ITERS = int(os.environ.get("BENCH_ITERS", "20"))
DTYPE = os.environ.get("BENCH_DTYPE", "both").lower()
CHECK = os.environ.get("BENCH_CHECK", "1") == "1"
MULT = 1_000_000  # content-check encoding

NDEV = jax.device_count()
if N % NDEV != 0:
    raise SystemExit(f"BENCH_TOTAL_ROWS={N} not divisible by device_count {NDEV}")
if N % CHUNK != 0:
    raise SystemExit(f"BENCH_TOTAL_ROWS={N} not divisible by BENCH_CHUNK {CHUNK}")
RPD = N // NDEV  # rows per destination

mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, NDEV), (DP, TP))
row_spec = P((DP, TP))  # global [NDEV*N, H] sharded on axis 0 -> [N, H] per device


def _make_kernel(h, dtype):
    def _kernel(x_ref, recv_ref, send_sem, recv_sem, barrier_sem):
        tp_size = lax.axis_size(TP)
        dp_size = lax.axis_size(DP)
        num_devices = tp_size * dp_size  # static
        my_id = lax.axis_index(DP) * tp_size + lax.axis_index(TP)  # traced

        def mesh_id(rank):
            return (rank // tp_size, rank % tp_size)

        # ---- barrier: every device ready before any remote write ----
        for i in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem, device_id=mesh_id(i), device_id_type=pltpu.DeviceIdType.MESH
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

        # ---- per-token scatter, fired in CHUNK batches, sends drained per batch ----
        def fire_row(r, _):
            dest = r // RPD
            slot = my_id * RPD + (r - dest * RPD)
            pltpu.make_async_remote_copy(
                src_ref=x_ref.at[pl.ds(r, 1)],
                dst_ref=recv_ref.at[pl.ds(slot, 1)],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=mesh_id(dest),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            return None

        for c in range(0, N, CHUNK):
            lax.fori_loop(c, c + CHUNK, fire_row, None, unroll=False)
            # drain this batch's local sends (bounds outstanding DMA to ~CHUNK rows)
            sref = x_ref.at[pl.ds(0, CHUNK)]
            pltpu.make_async_copy(src_ref=sref, dst_ref=sref, sem=send_sem).wait()

        # ---- drain all N incoming rows (recv_sem signalled by remote senders) ----
        rref = recv_ref.at[pl.ds(0, N)]
        pltpu.make_async_copy(src_ref=rref, dst_ref=rref, sem=recv_sem).wait()

        # ---- barrier: nobody exits (freeing recv) until all peers done sending ----
        for i in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem, device_id=mesh_id(i), device_id_type=pltpu.DeviceIdType.MESH
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((N, h), dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=[
            pltpu.SemaphoreType.DMA,  # send_sem
            pltpu.SemaphoreType.DMA,  # recv_sem
            pltpu.SemaphoreType.BARRIER,  # barrier_sem
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
        ),
    )


def _runner(h, dtype):
    kfn = _make_kernel(h, dtype)

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=(row_spec,), out_specs=row_spec, check_vma=False)
    def run(x):
        return kfn(x)

    return run


def _check():
    """Single-host only: deterministic int32 payload, pull recv, verify routing."""
    h = 8
    run = _runner(h, jnp.int32)
    # global x[g*N + r] = g*MULT + r  (g = global device index)
    rows = np.arange(N, dtype=np.int32)
    x_host = np.concatenate([(g * MULT + rows)[:, None].repeat(h, 1) for g in range(NDEV)], axis=0)
    x = jax.device_put(jnp.asarray(x_host), jax.sharding.NamedSharding(mesh, row_spec))
    recv = np.asarray(jax.block_until_ready(run(x)))  # [NDEV*N, h]
    # expected recv[e*N + s] = (s//RPD)*MULT + (e*RPD + s%RPD)
    s = np.arange(N)
    bad = 0
    for e in range(NDEV):
        exp = (s // RPD) * MULT + (e * RPD + s % RPD)
        got = recv[e * N : (e + 1) * N, 0]
        bad += int(np.sum(exp != got))
    return bad


def _bench(dtype_name):
    dt = jnp.bfloat16 if dtype_name == "bf16" else jnp.float8_e4m3fn
    run = _runner(H, dt)
    key = jax.random.key(0)
    shards = []
    for i, dev in enumerate(jax.local_devices()):
        sk = jax.random.fold_in(key, jax.process_index() * len(jax.local_devices()) + i)
        shards.append(jax.device_put(jax.random.normal(sk, (N, H), jnp.float32).astype(dt), dev))
    x = jax.make_array_from_single_device_arrays(
        (NDEV * N, H), jax.sharding.NamedSharding(mesh, row_spec), shards
    )

    for _ in range(WARMUP):
        jax.block_until_ready(run(x))
    disp, wait = [], []
    for _ in range(ITERS):
        a = time.monotonic()
        out = run(x)
        b = time.monotonic()
        jax.block_until_ready(out)
        c = time.monotonic()
        disp.append((b - a) * 1e3)
        wait.append((c - b) * 1e3)

    bpe = 2 if dtype_name == "bf16" else 1
    payload_mb = N * H * bpe / 1e6
    moved_mb = (NDEV - 1) / NDEV * payload_mb  # off-diagonal leaves the device
    wall = float(np.mean(np.array(disp) + np.array(wait)))
    bw = moved_mb / 1e3 / (wall * 1e-3)
    return wall, float(np.mean(wait)), payload_mb, moved_mb, bw


def main():
    dev = jax.devices()[0]
    log(
        f"device={dev.device_kind} ndev={NDEV} procs={jax.process_count()} | "
        f"N={N} rows/dest={RPD} H={H} CHUNK={CHUNK} | per-token make_async_remote_copy scatter"
    )

    if CHECK and jax.process_count() == 1:
        bad = _check()
        log(f"  correctness (single-host): {'OK' if bad == 0 else f'FAIL ({bad} mismatched rows)'}")
        if bad:
            raise SystemExit("correctness gate failed")
    elif CHECK:
        log("  correctness: SKIP (multi-host; verify on single-host ep=8)")

    dtypes = ["bf16", "fp8"] if DTYPE == "both" else [DTYPE]
    for dn in dtypes:
        wall, waitms, pmb, mmb, bw = _bench(dn)
        if jax.process_index() == 0:
            log(
                f"  [{dn}] per-token scatter: wall={wall:.3f}ms (wait={waitms:.3f}) | "
                f"payload={pmb:.0f}MB/dev moved≈{mmb:.0f}MB/dev | BW≈{bw:.0f} GB/s"
            )
    log("done")


if __name__ == "__main__":
    main()
