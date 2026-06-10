"""Flat vs hierarchical all-to-all microbenchmark (MoE scatter cost, ep=32 v7x torus).

Backs the blog claim "we tried hierarchical all-to-all (exchange dimension by dimension);
total bytes ~multiply; flat all-to-all still faster". The fused-MoE kernel itself does a
flat point-to-point DMA scatter; this is a standalone collective microbench comparing two
strategies on the SAME payload:

  • flat (direct): ONE jax.lax.all_to_all over all `num_devices` — every device sends each
    block directly to its final destination in one collective. Ships (D-1)/D of the payload once.
  • hierarchical (hybrid): a FACTORED all-to-all — factor the device axis into
    FACTORS (default 4×4×2 = 32, torus-aware via create_device_mesh) and exchange one factor
    per stage. Stage g relocates the full local payload along that factor, shipping (g-1)/g
    each → summed over factors ≈ 2× the flat bytes for a 3-factor split. (Measured TIME ratio
    is typically larger than the byte ratio: extra stages add sync barriers and worse link
    overlap — that gap is the point of the experiment.)

Both are functionally-correct all-to-alls (verified per-device by a content check: device
`d` must receive from source `s` exactly the block `s` addressed to `d`). The hierarchical
version is the standard "transpose one factor at a time": stage k runs all_to_all over mesh
axis k with split_axis=concat_axis=k, replacing the k-th DESTINATION factor of the local
tensor with the k-th SOURCE factor. After all stages every dest-factor has become a
source-factor → same logical result as flat.

Payload models the Ling-2.6-1T prefill-16384 scatter: per device `[D, R, H]` with D=32
destinations, R=128 rows/dest (512 local tok × top-8 / 32), H=8192 hidden →
67 MB bf16 / 33.5 MB fp8 per device.

Run on a 4-host v7x slice (ep=32, 2×2×4) via Falcon (multi-process; bare
jax.distributed.initialize() auto-detects the slice). Env:
  BENCH_FACTORS=4,2,2  BENCH_ROWS=128  BENCH_H=8192  BENCH_DTYPE={bf16,fp8,both}
  BENCH_MODE={flat,hier,both}  BENCH_WARMUP=3  BENCH_ITERS=20  BENCH_CHECK=1
  BENCH_SINGLE_HOST=1 (skip distributed init for a single-host correctness run)
"""

from __future__ import annotations

import os
import time
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import mesh_utils

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:.1f}s][p{jax.process_index()}] {msg}", flush=True)


if os.environ.get("BENCH_SINGLE_HOST", "0") != "1":
    jax.distributed.initialize()

P = jax.sharding.PartitionSpec

FACTORS = tuple(int(x) for x in os.environ.get("BENCH_FACTORS", "4,4,2").split(","))
ROWS = int(os.environ.get("BENCH_ROWS", "128"))
H = int(os.environ.get("BENCH_H", "8192"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
ITERS = int(os.environ.get("BENCH_ITERS", "20"))
DTYPE = os.environ.get("BENCH_DTYPE", "both").lower()
MODE = os.environ.get("BENCH_MODE", "both").lower()
CHECK = os.environ.get("BENCH_CHECK", "1") == "1"

D = reduce(lambda a, b: a * b, FACTORS, 1)
NDEV = jax.device_count()
AXES = tuple(f"s{i}" for i in range(len(FACTORS)))

if D != NDEV:
    raise SystemExit(f"BENCH_FACTORS={FACTORS} product {D} != device_count {NDEV}")

# Mesh over the device axis, factored into len(FACTORS) named axes. Prefer a
# topology-aware layout (better ICI locality); fall back to a plain reshape if
# create_device_mesh can't tile this factorization onto the physical slice.
try:
    _devs = mesh_utils.create_device_mesh(FACTORS)
except Exception:  # noqa: BLE001
    _devs = np.array(jax.devices()).reshape(FACTORS)
mesh = jax.sharding.Mesh(_devs, AXES)
full_spec = P(AXES)  # shard the leading device axis across all mesh axes

# strides to flatten factored coords -> linear device index (row-major, s0 most significant)
STRIDES = [int(np.prod(FACTORS[i + 1 :])) for i in range(len(FACTORS))]


def _my_flat_index():
    return sum(lax.axis_index(ax) * STRIDES[i] for i, ax in enumerate(AXES))


def _flat_a2a(x):
    # x: [D, R, H] -> one collective over the combined device axis.
    return lax.all_to_all(x, AXES, split_axis=0, concat_axis=0, tiled=True)


def _hier_a2a(x):
    # x: [f0, f1, ..., R, H] -> exchange one factor per stage.
    for k, ax in enumerate(AXES):
        x = lax.all_to_all(x, ax, split_axis=k, concat_axis=k, tiled=True)
    return x


# ---------------- correctness (int32 content check, self-contained per device) ----------
def _check(mode):
    """Each device builds block[my][dest]=my*1000+dest, runs the a2a, and asserts it now
    holds block[src][my]=src*1000+my for every source. Returns global max abs error."""

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=(full_spec,), out_specs=P(), check_vma=False)
    def fn(_dummy):
        my = _my_flat_index()
        dest = jnp.arange(D, dtype=jnp.int32)  # linear dest index
        # x_flat[dest] = my*1000 + dest, broadcast to [D, 1, 1] (R,H collapsed to 1 for check)
        x = (my * 1000 + dest)[:, None, None].astype(jnp.int32)
        if mode == "flat":
            y = _flat_a2a(x)  # [D,1,1]
        else:
            y = _hier_a2a(x.reshape(*FACTORS, 1, 1)).reshape(D, 1, 1)
        src = jnp.arange(D, dtype=jnp.int32)
        expected = (src * 1000 + my)[:, None, None].astype(jnp.int32)
        err = jnp.max(jnp.abs(y - expected)).astype(jnp.int32)
        return lax.pmax(err, AXES)  # global max, replicated across all devices

    dummy = jax.device_put(
        jnp.zeros((D, 1, 1), jnp.int32), jax.sharding.NamedSharding(mesh, full_spec)
    )
    out = jax.block_until_ready(fn(dummy))
    # `out` is replicated (P()); read a process-local addressable shard (multi-host safe).
    return int(np.asarray(out.addressable_shards[0].data))


# ---------------- timing ----------------------------------------------------------------
def _dtype_of(name):
    return jnp.bfloat16 if name == "bf16" else jnp.float8_e4m3fn


def _bench(mode, dtype_name):
    dt = _dtype_of(dtype_name)
    key = jax.random.key(0)

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=(full_spec,), out_specs=full_spec, check_vma=False)
    def fn(x):
        if mode == "flat":
            y = _flat_a2a(x)
        else:
            y = _hier_a2a(x.reshape(*FACTORS, ROWS, H)).reshape(D, ROWS, H)
        return y

    # per-device payload [D, ROWS, H]; build as a globally-sharded array
    sharding = jax.sharding.NamedSharding(mesh, full_spec)
    shards = []
    for i, dev in enumerate(jax.local_devices()):
        sk = jax.random.fold_in(key, jax.process_index() * len(jax.local_devices()) + i)
        shards.append(
            jax.device_put(jax.random.normal(sk, (D, ROWS, H), jnp.float32).astype(dt), dev)
        )
    x = jax.make_array_from_single_device_arrays((NDEV * D, ROWS, H), sharding, shards)

    for _ in range(WARMUP):
        jax.block_until_ready(fn(x))
    disp, wait = [], []
    for _ in range(ITERS):
        a = time.monotonic()
        out = fn(x)
        b = time.monotonic()
        jax.block_until_ready(out)
        c = time.monotonic()
        disp.append((b - a) * 1e3)
        wait.append((c - b) * 1e3)

    bytes_per_elem = 2 if dtype_name == "bf16" else 1
    payload_mb = D * ROWS * H * bytes_per_elem / 1e6  # per-device local tensor
    stages = 1 if mode == "flat" else len(FACTORS)
    # Bytes leaving each device. A flat a2a ships (D-1)/D of the payload ONCE; a
    # factored a2a ships (g-1)/g per stage g, summed over factors (each stage
    # relocates the full local payload along one factor). So 3 factors ~= 2x bytes.
    if mode == "flat":
        moved_mb = (D - 1) / D * payload_mb
    else:
        moved_mb = sum((g - 1) / g for g in FACTORS) * payload_mb
    wall = float(np.mean(np.array(disp) + np.array(wait)))
    bw = moved_mb / 1e3 / (wall * 1e-3)  # GB/s aggregate per device
    return {
        "mode": mode,
        "dtype": dtype_name,
        "stages": stages,
        "payload_mb": payload_mb,
        "moved_mb": moved_mb,
        "wall_ms": wall,
        "wait_ms": float(np.mean(wait)),
        "bw_gbs": bw,
    }


def main():
    dev = jax.devices()[0]
    log(
        f"device={dev.device_kind} ndev={NDEV} procs={jax.process_count()} | "
        f"FACTORS={FACTORS} axes={AXES} D={D} ROWS={ROWS} H={H} | warmup={WARMUP} iters={ITERS}"
    )

    if CHECK:
        for mode in ["flat", "hier"] if MODE == "both" else [MODE]:
            err = _check(mode)
            status = "OK" if err == 0 else f"FAIL(err={err})"
            log(f"  correctness [{mode}]: {status}")
            if err != 0:
                raise SystemExit(f"correctness gate failed for {mode}")

    modes = ["flat", "hier"] if MODE == "both" else [MODE]
    dtypes = ["bf16", "fp8"] if DTYPE == "both" else [DTYPE]
    results = {}
    for dn in dtypes:
        for mode in modes:
            r = _bench(mode, dn)
            results[(mode, dn)] = r
            if jax.process_index() == 0:
                log(
                    f"  [{dn}] {mode:4s}: wall={r['wall_ms']:.3f}ms (wait={r['wait_ms']:.3f}) "
                    f"| stages={r['stages']} payload={r['payload_mb']:.0f}MB/dev "
                    f"moved≈{r['moved_mb']:.0f}MB/dev | BW≈{r['bw_gbs']:.0f} GB/s"
                )
    if jax.process_index() == 0 and MODE == "both":
        for dn in dtypes:
            f, h = results[("flat", dn)], results[("hier", dn)]
            log(
                f"  [{dn}] hier/flat: time {h['wall_ms'] / f['wall_ms']:.2f}× | "
                f"bytes {h['moved_mb'] / f['moved_mb']:.2f}×"
            )
    log("done")


if __name__ == "__main__":
    main()
