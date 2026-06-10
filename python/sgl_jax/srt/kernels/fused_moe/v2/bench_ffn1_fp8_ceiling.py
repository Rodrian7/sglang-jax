"""FFN1 dense FP8 GEMM ceiling — per-device compute lower bound (Ling-2.6-1T shapes).

Validates the "0.12 ms dense fp8 GEMM lower bound" used in the v2 sharing doc.
The workload is Ling-2.6-1T's routed-expert FFN1 (gate_up = w1+w3, fused to N=2*inter)
for the local experts owned by ONE v7x device at prefill scale:

    EP=32  -> 256/32 = 8 local experts/device ; ~512 routed tokens/expert.

Derivation a reader can recompute (per JAX device = one v7x chiplet):

    FLOPs(8 local experts) = NE * 2 * TOK * D * N
                           = 8 * 2 * 512 * 8192 * 4096            = 274.9 GFLOP
    W8A8  (fp8 peak 2307 TFLOPS/device): 274.9e9 / 2307e12 = 0.119 ms  (~0.12 ms)
    W8A16 (bf16 peak 1153 TFLOPS/device): 274.9e9 / 1153e12 = 0.238 ms (~0.24 ms)
    HBM floor (weights, fp8=1B): 8*2*8192*2048 B / 3690 GB/s = 0.073 ms
      -> arithmetic intensity ~1024 FLOP/B > v7x ridge ~625 -> COMPUTE-bound.

We build the layer Ling actually uses (`QuantizedLinear`, per-channel fp8,
`weight_block_size=None`) and time it on a single chiplet. Two activation modes
(W8A8 fp8-act, W8A16 bf16-act) x two shapes (per-expert 8x[512,*], and the
batched-ideal single GEMM [8*512,*]) bound the gap to the compute ceiling.

Run (single host, no model needed):

    BENCH_SINGLE_HOST=1 BENCH_ACT=both python -m \
        sgl_jax.srt.kernels.fused_moe.v2.bench_ffn1_fp8_ceiling

Env knobs (defaults = Ling-2.6-1T): BENCH_D=8192 BENCH_INTER=2048 BENCH_TOKENS=512
BENCH_NEXPERTS=8 BENCH_REPS=50 BENCH_WARMUP=3 BENCH_ITERS=10 BENCH_ACT={fp8,bf16,both}.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.layers.linear import QuantizedLinear
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:.1f}s] {m}", flush=True)


if os.environ.get("BENCH_SINGLE_HOST", "0") != "1":
    jax.distributed.initialize()

# Per JAX device (one v7x chiplet) compute peaks — see module docstring.
TPU_V7X_FP8_PEAK_TFLOPS = 2307.0
TPU_V7X_BF16_PEAK_TFLOPS = 1153.0

# Ling-2.6-1T defaults.
D = int(os.environ.get("BENCH_D", "8192"))  # hidden_size (K)
INTER = int(os.environ.get("BENCH_INTER", "2048"))  # moe_intermediate_size
N = 2 * INTER  # gate_up merged output (w1 + w3)
TOK = int(os.environ.get("BENCH_TOKENS", "512"))  # routed tokens / expert
NE = int(os.environ.get("BENCH_NEXPERTS", "8"))  # local experts / device (EP=32)
REPS = int(os.environ.get("BENCH_REPS", "50"))  # reps inside one jit (amortize dispatch)
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
ITERS = int(os.environ.get("BENCH_ITERS", "10"))
ACT = os.environ.get("BENCH_ACT", "both").lower()  # fp8 | bf16 | both

# Single-device mesh: isolates one chiplet so timing matches the per-device peak.
# kernel_axes=(None, None) -> pure local dense GEMM, no TP all-reduce.
mesh = Mesh(
    np.array(jax.devices()[:1]).reshape(1, 1),
    axis_names=("data", "tensor"),
    axis_types=(AxisType.Explicit, AxisType.Explicit),
)
KEY = jax.random.key(0)


def _make_quant_linear(seed: int, act_dtype):
    """One Ling-style FFN1 gate_up GEMM as QuantizedLinear: [K=D] -> [N], per-channel fp8.

    Distinct random weights per `seed` so XLA can't CSE the experts together.
    """
    w_fp = jax.random.normal(jax.random.fold_in(KEY, seed), (N, D), jnp.float32) * 0.01
    # Per-channel quantize along the input axis -> w_q [N, D] fp8, w_scale [N] f32.
    w_q, w_scale = quantize_tensor(dtype=jnp.float8_e4m3fn, tensor=w_fp, axis=1)
    return QuantizedLinear(
        weight_q=w_q,
        weight_scale=w_scale,
        bias=None,
        activation_dtype=act_dtype,  # fp8 -> W8A8 ; None -> W8A16 (bf16 act)
        mesh=mesh,
        kernel_axes=(None, None),
        params_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,  # fp8,fp8 -> f32 -> bf16 fast path
        weight_block_size=None,  # per-channel (Ling)
        scope_name=f"ffn1_{seed}",
    )


class _Stack(nnx.Module):
    """Holds the per-expert QuantizedLinears so nnx.split sees them as state."""

    def __init__(self, experts):
        self.experts = nnx.List(experts)  # list[QuantizedLinear]


def _bench(experts, x0, sweep_flops, peak_tflops, ceiling_label, tag):
    """Time `experts` applied to x0, chained REPS times to defeat CSE and amortize dispatch.

    sweep_flops = FLOPs for one application of the whole expert list.
    """
    stack = _Stack(experts)
    graphdef, state = nnx.split(stack)

    @jax.jit
    def body(state, x):
        stack = nnx.merge(graphdef, state)
        for _ in range(REPS):
            s = jnp.float32(0.0)
            for ql in stack.experts:
                out, _ = ql(x)  # [M, N]
                s = s + jnp.mean(out.astype(jnp.float32))
            # Cheap scalar dependency: perturbs x so reps can't be CSE'd, ~0 FLOPs.
            x = x + (s * jnp.float32(1e-9)).astype(x.dtype)
        return x

    run = lambda: body(state, x0)  # noqa: E731
    for _ in range(WARMUP):
        jax.block_until_ready(run())
    ts = []
    for _ in range(ITERS):
        a = time.monotonic()
        jax.block_until_ready(run())
        ts.append((time.monotonic() - a) * 1e3)

    per_sweep_ms = float(np.mean(ts)) / REPS
    tflops = sweep_flops / (per_sweep_ms * 1e-3) / 1e12
    ceiling_ms = sweep_flops / (peak_tflops * 1e12) * 1e3
    log(
        f"  {tag}: {per_sweep_ms * 1e3:7.1f} us/sweep | {tflops:6.0f} TFLOPS "
        f"| {100 * tflops / peak_tflops:5.1f}% of {peak_tflops:.0f} "
        f"| ceiling({ceiling_label})={ceiling_ms * 1e3:.1f}us -> {per_sweep_ms / ceiling_ms:.2f}x"
    )


def _run_mode(act_dtype, mode_label, peak_tflops, ceiling_label):
    log(f"[{mode_label}] act_dtype={act_dtype} peak={peak_tflops:.0f} TFLOPS/device")
    sweep_flops = NE * 2 * TOK * D * N  # 8-expert FFN1 sweep

    # (1) per-expert: NE independent GEMMs, M=TOK each (production-faithful).
    experts = [_make_quant_linear(e, act_dtype) for e in range(NE)]
    x_pe = jax.random.normal(jax.random.fold_in(KEY, 1000), (TOK, D), jnp.bfloat16)
    _bench(experts, x_pe, sweep_flops, peak_tflops, ceiling_label, f"per-expert(NE={NE},M={TOK})")

    # (2) batched-ideal: one GEMM, M=NE*TOK (tightest achievable dense fp8; same FLOPs).
    big = [_make_quant_linear(9999, act_dtype)]
    x_b = jax.random.normal(jax.random.fold_in(KEY, 2000), (NE * TOK, D), jnp.bfloat16)
    _bench(big, x_b, sweep_flops, peak_tflops, ceiling_label, f"batched-ideal(M={NE * TOK})")


def main():
    dev = jax.devices()[0]
    log(
        f"device={dev.device_kind} ndev_visible={jax.device_count()} (using 1) | "
        f"shape: tokens/expert={TOK} experts={NE} K(hidden)={D} N(gate_up=2*{INTER})={N} | "
        f"sweep FLOPs={NE * 2 * TOK * D * N / 1e9:.1f} GFLOP | REPS={REPS} warmup={WARMUP} iters={ITERS}"
    )
    if ACT in ("fp8", "both"):
        _run_mode(jnp.float8_e4m3fn, "W8A8 (fp8 act)", TPU_V7X_FP8_PEAK_TFLOPS, "0.12ms")
    if ACT in ("bf16", "both"):
        _run_mode(None, "W8A16 (bf16 act)", TPU_V7X_BF16_PEAK_TFLOPS, "0.24ms")
    log("done")


if __name__ == "__main__":
    main()
