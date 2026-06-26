"""Sweep the BT x final-select-unroll tradeoff for `grouped_topk_pallas` on a TPU host.

At a fixed VMEM budget there is a tradeoff: full-unroll keeps O(topk) live [BT,E] (forces a smaller
BT -> more grid steps), while rolling keeps one [BT,E] (allows a much larger BT -> fewer grid
steps). Which wins depends on T / k / E, so measure it: for each (config, T) this times every
(block_tokens, unroll in {full, rolled}) combination that fits VMEM and reports the best.

Methodology matches bench_grouped_topk.py: gate-matmul-fed router_logits (realistic layout), the
gate kept outside the timed `named_scope`, device time summed from a profiler trace. Run on TPU:

    python -m benchmark.kernels.grouped_topk.sweep_bt_unroll \
        --T 2048,4096,8192 --configs k8:256/8/4/8,k32:256/8/4/32
"""

import argparse
import functools

import jax
import jax.numpy as jnp

from benchmark.kernels.grouped_topk.bench_grouped_topk import (
    H,
    SCOPE_FUSED,
    _gate,
    _trace_scope_us,
)
from sgl_jax.srt.kernels.grouped_topk.v1.kernel import grouped_topk_pallas


def _bt_candidates(T, cap):
    """Power-of-2 block sizes that divide T, from 128 up to min(T, cap)."""
    out, b = [], 128
    while b <= min(T, cap):
        if T % b == 0:
            out.append(b)
        b *= 2
    return out or [T]


def make_fused(w_gate, bias, G, Gtop, k, bt, unroll):
    def fn(hidden):
        logits = _gate(hidden, w_gate)  # gate matmul OUTSIDE the timed scope
        with jax.named_scope(SCOPE_FUSED):
            return grouped_topk_pallas(
                logits,
                bias,
                num_expert_group=G,
                topk_group=Gtop,
                topk=k,
                block_tokens=bt,
                unroll=unroll,
                interpret=False,
            )

    return fn


def sweep_config(name, E, G, Gtop, k, Ts, bt_cap):
    w_gate = jax.device_put(jax.random.normal(jax.random.PRNGKey(3), (H, E), dtype=jnp.float32))
    bias = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (E,), dtype=jnp.float32) * 0.1)
    print(f"\n=== {name} (E={E}, G={G}, Gtop={Gtop}, k={k}) ===")
    for T in Ts:
        h = jax.device_put(jax.random.normal(jax.random.PRNGKey(5), (T, H), dtype=jnp.float32))
        print(f"\n  T={T}")
        print(f"  {'BT':>6} {'full_us':>9} {'roll_us':>9}")
        best = None  # (us, bt, mode)
        for bt in _bt_candidates(T, bt_cap):
            row = {}
            for unroll, key in ((True, "full"), (False, "roll")):
                fn = jax.jit(make_fused(w_gate, bias, G, Gtop, k, bt, unroll))
                try:
                    jax.block_until_ready(fn(h))
                except Exception as ex:  # noqa: BLE001  (VMEM OOM etc. -> skip this combo)
                    row[key] = float("nan")
                    if "RESOURCE_EXHAUSTED" not in str(ex) and "vmem" not in str(ex).lower():
                        print(f"  {bt:>6} {key}: unexpected error: {str(ex)[:80]}")
                    continue
                us, _ = _trace_scope_us(
                    functools.partial(fn, h), SCOPE_FUSED, f"{name}_{T}_{bt}_{key}"
                )
                row[key] = us
                if us == us and (best is None or us < best[0]):  # us==us filters NaN
                    best = (us, bt, key)
            f = row.get("full", float("nan"))
            r = row.get("roll", float("nan"))
            print(f"  {bt:>6} {f:>9.2f} {r:>9.2f}")
        if best is not None:
            print(f"  -> best: {best[0]:.2f}us at BT={best[1]} ({best[2]}-unroll)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="2048,4096,8192")
    ap.add_argument("--configs", default="k8:256/8/4/8,k32:256/8/4/32", help="name:E/G/Gtop/k list")
    ap.add_argument("--bt_cap", type=int, default=8192, help="max block_tokens to try")
    a = ap.parse_args()
    print(f"JAX {jax.__version__} | {jax.devices()[0].platform} | n_dev {len(jax.devices())}")
    print("fused device time (gate-fed, gate outside scope); sweeping BT x {full,rolled} unroll.")
    Ts = [int(x) for x in a.T.split(",")]
    for spec in a.configs.split(","):
        name, cfg = spec.split(":")
        E, G, Gtop, k = (int(v) for v in cfg.split("/"))
        sweep_config(name, E, G, Gtop, k, Ts, a.bt_cap)


if __name__ == "__main__":
    main()
