"""Dump LLO/HLO + an xprof TC-counter trace for `grouped_topk_pallas`, one unroll mode per run.

Compares the final-select **full-unroll** vs **rolled** forms of the v1 grouped-topk kernel
(`kernels/grouped_topk/v1/kernel.py`, `unroll=True/False`). Full-unroll overlaps all `topk` picks
but keeps O(topk) live [BT,E]; rolled keeps one. This script captures the compiler artifacts and a
per-instruction xprof trace so you can see *where* the time goes in each form (not just wall time —
use sweep_bt_unroll.py for the BT x unroll timing table).

Run ONE mode per process so the LLO/HLO/xprof dumps stay clean (the mosaic dump dir is fixed at
import time). Pass `--mode full|roll`; each mode writes under its own subdir:

    python -m benchmark.kernels.grouped_topk.llo --mode full --T 4096 --config 256/8/4/8 --bt 512
    python -m benchmark.kernels.grouped_topk.llo --mode roll --T 4096 --config 256/8/4/8 --bt 512

then compare /tmp/pallas-profile/<mode>/{llo,hlo,xprof}. The kernel/gate layout matches
bench_grouped_topk.py: router_logits is gate-fed (realistic GOOD layout), gate matmul outside the
timed named_scope.
"""

# --- argv must be parsed BEFORE `import jax`: the mosaic/HLO dump dirs are baked into env at import.
import argparse
import os
import sys


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "roll"], default="full",
                    help="final-select unroll form: full (unroll=True) or roll (unroll=False)")
    ap.add_argument("--T", type=int, default=4096, help="number of tokens (rows)")
    ap.add_argument("--config", default="256/8/4/8", help="E/G/Gtop/k")
    ap.add_argument("--bt", type=int, default=512, help="block_tokens")
    ap.add_argument("--out", default="/tmp/pallas-profile", help="dump root")
    return ap.parse_args()


ARGS = _parse_args()
DUMP_ROOT = os.path.join(ARGS.out, ARGS.mode)

os.environ["LIBTPU_INIT_ARGS"] = " ".join(
    [
        os.environ.get("LIBTPU_INIT_ARGS", ""),
        "--xla_enable_custom_call_region_trace=true",
        "--xla_xprof_register_llo_debug_info=true",
        f"--xla_mosaic_dump_to={DUMP_ROOT}/llo",
    ]
).strip()
os.environ["XLA_FLAGS"] = " ".join(
    [
        os.environ.get("XLA_FLAGS", ""),
        f"--xla_dump_to={DUMP_ROOT}/hlo",
        "--xla_dump_hlo_as_text",
        "--xla_dump_hlo_as_proto",
    ]
).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from benchmark.kernels.grouped_topk.bench_grouped_topk import H, SCOPE_FUSED, _gate  # noqa: E402
from python.sgl_jax.srt.kernels.grouped_topk.v1.kernel2 import grouped_topk_pallas  # noqa: E402

E, G, Gtop, k = (int(v) for v in ARGS.config.split("/"))
UNROLL = ARGS.mode == "full"  # full -> unroll=True, roll -> unroll=False

w_gate = jax.device_put(jax.random.normal(jax.random.PRNGKey(3), (H, E), dtype=jnp.float32))
bias = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (E,), dtype=jnp.float32) * 0.1)
hidden = jax.device_put(jax.random.normal(jax.random.PRNGKey(5), (ARGS.T, H), dtype=jnp.float32))


@jax.jit
def run_kernel():
    logits = _gate(hidden, w_gate)  # gate matmul OUTSIDE the timed scope (realistic GOOD layout)
    with jax.named_scope(SCOPE_FUSED):
        return grouped_topk_pallas(
            logits,
            bias,
            num_expert_group=G,
            topk_group=Gtop,
            topk=k,
            block_tokens=ARGS.bt,
            unroll=UNROLL,
            interpret=False,
        )


opts = jax.profiler.ProfileOptions()
opts.advanced_configuration = {
    "tpu_enable_periodic_counter_sampling": True,
    "tpu_tc_perf_counter_sampling_options": (
        "interval_us:1 scaling:0 counter_size_bits:1 "
        "indices:1 indices:3 indices:4 indices:10 indices:11 "
        "indices:31 indices:32 indices:33 indices:34 indices:35 "
        "indices:37 indices:38 indices:56 indices:57 indices:58 "
        "indices:73 indices:74 indices:75 indices:105"
    ),
    "num_tensor_cores_to_trace_per_device": 1,
}

print(
    f"JAX {jax.__version__} | {jax.devices()[0].platform} | mode={ARGS.mode} "
    f"unroll={UNROLL} | T={ARGS.T} E={E} G={G} Gtop={Gtop} k={k} BT={ARGS.bt}",
    file=sys.stderr,
)

# warmup so compile/cache happens OUTSIDE the trace (also triggers the LLO/HLO dumps).
for _ in range(3):
    jax.block_until_ready(run_kernel())

logdir = f"{DUMP_ROOT}/xprof"
with jax.profiler.trace(logdir, profiler_options=opts):
    result = run_kernel()
    jax.block_until_ready(result)

print(f"done -> {DUMP_ROOT}/{{llo,hlo,xprof}}", file=sys.stderr)
