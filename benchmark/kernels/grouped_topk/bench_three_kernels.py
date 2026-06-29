"""Head-to-head benchmark of the three grouped-topk Pallas implementations (+ the sort reference).

Three interchangeable kernels live under `kernels/grouped_topk/v1/` — they are id-for-id identical
(proven by `test/kernels/grouped_topk_test.py`, which runs every case against all three) but lay the
selection out differently, so their device time differs:

  * kernel   — Python-unrolled final-select; keeps O(topk) live [BT,E] temporaries.
  * kernel1  — narrows to a [BT,C] candidate array (C = G * min(topk,S)) before the global top-k.
  * kernel2  — fori_loop carrying a single [BT,E]; full-unroll when it fits VMEM, else rolled.

Methodology matches bench_grouped_topk.py: gate-matmul-fed router_logits (the realistic GOOD layout
— an isolated jit of the routing picks a ~15x-slower layout), the gate kept OUTSIDE the timed
`named_scope`, device time summed from a profiler trace (no subtraction). Each kernel runs under its
own scope so the trace attributes ops correctly. Run on a TPU host:

    python -m benchmark.kernels.grouped_topk.bench_three_kernels \
        --T 512,1024,2048,4096,8192 --configs A_E256:256/8/4/8,B_E512:512/8/4/8

`--bt auto` (default) uses the tuned table; pass an int to force a block size for all kernels.
`--unroll auto|full|rolled` is forwarded to kernel1/kernel2 (kernel has no unroll knob).
"""

import argparse
import functools
import importlib.util
import os

import jax
import jax.numpy as jnp

from benchmark.kernels.grouped_topk.bench_grouped_topk import (
    H,
    SCOPE_SORT,
    _gate,
    _trace_scope_us,
    ref_biased_grouped_topk,
)

# The three implementations, in report order. `unroll`=False -> kernel has no `unroll` kwarg.
KERNELS = [
    ("kernel", False),
    ("kernel1", True),
    ("kernel2", True),
]

# Directory holding the three kernel files (repo_root/python/sgl_jax/srt/kernels/grouped_topk/v1).
_V1_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "python", "sgl_jax", "srt", "kernels", "grouped_topk", "v1",
    )
)


def _import_kernel(basename):
    """Load grouped_topk_pallas straight from v1/<basename>.py by path.

    Bypasses `v1/__init__.py` (which has a circular sgl_jax/python.sgl_jax import) so the benchmark
    pins exactly the three files on disk regardless of how the package happens to be installed.
    """
    path = os.path.join(_V1_DIR, f"{basename}.py")
    spec = importlib.util.spec_from_file_location(f"_gt_bench_{basename}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grouped_topk_pallas


def _scope(basename):
    return f"FUSEDTOPK_{basename.upper()}"


def make_fused(fn, w_gate, bias, G, Gtop, k, bt, scope, extra_kwargs):
    def run(hidden):
        logits = _gate(hidden, w_gate)  # gate matmul OUTSIDE the timed scope
        with jax.named_scope(scope):
            return fn(
                logits,
                bias,
                num_expert_group=G,
                topk_group=Gtop,
                topk=k,
                block_tokens=bt,
                interpret=False,
                **extra_kwargs,
            )

    return run


def make_sort(w_gate, bias, G, Gtop, k):
    def run(hidden):
        logits = _gate(hidden, w_gate)  # gate OUTSIDE the scope
        with jax.named_scope(SCOPE_SORT):
            return ref_biased_grouped_topk(
                logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
            )

    return run


def _unroll_kwargs(has_unroll, unroll_mode):
    """kernel1/kernel2 take `unroll` (None|True|False); kernel takes none."""
    if not has_unroll or unroll_mode == "auto":
        return {}
    return {"unroll": unroll_mode == "full"}


def bench_config(name, E, G, Gtop, k, Ts, bt_arg, unroll_mode):
    w_gate = jax.device_put(jax.random.normal(jax.random.PRNGKey(3), (H, E), dtype=jnp.float32))
    bias = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (E,), dtype=jnp.float32) * 0.1)

    sfn = jax.jit(make_sort(w_gate, bias, G, Gtop, k))
    kernels = []  # (basename, jitted_fn, scope)
    for basename, has_unroll in KERNELS:
        raw = _import_kernel(basename)
        bt = "auto" if bt_arg == "auto" else int(bt_arg)
        run = make_fused(
            raw, w_gate, bias, G, Gtop, k, bt, _scope(basename),
            _unroll_kwargs(has_unroll, unroll_mode),
        )
        kernels.append((basename, jax.jit(run), _scope(basename)))

    cols = " ".join(f"{b + '_us':>12}" for b, _, _ in kernels)
    print(f"\n=== {name} (E={E}, G={G}, Gtop={Gtop}, k={k}) bt={bt_arg} unroll={unroll_mode} ===")
    print(f"{'T':>7} {'sort_us':>10} {cols}   best")
    for T in Ts:
        h = jax.device_put(jax.random.normal(jax.random.PRNGKey(5), (T, H), dtype=jnp.float32))

        jax.block_until_ready(sfn(h))
        sort_us, _ = _trace_scope_us(functools.partial(sfn, h), SCOPE_SORT, f"sort_{name}_{T}")

        times = {}
        for basename, fn, scope in kernels:
            try:
                jax.block_until_ready(fn(h))
            except Exception as ex:  # noqa: BLE001  (VMEM OOM etc. -> mark NaN, keep going)
                times[basename] = float("nan")
                if "RESOURCE_EXHAUSTED" not in str(ex) and "vmem" not in str(ex).lower():
                    print(f"  {basename} T={T}: unexpected error: {str(ex)[:80]}")
                continue
            us, _ = _trace_scope_us(functools.partial(fn, h), scope, f"{basename}_{name}_{T}")
            times[basename] = us

        valid = {b: t for b, t in times.items() if t == t and t > 0}  # t==t drops NaN
        best = min(valid, key=valid.get) if valid else None
        row = " ".join(f"{times[b]:>12.2f}" for b, _, _ in kernels)
        tag = f"{best} ({sort_us / valid[best]:.2f}x vs sort)" if best else "-"
        print(f"{T:>7} {sort_us:>10.2f} {row}   {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="512,1024,2048,4096,8192")
    ap.add_argument(
        "--configs", default="A_E256:256/8/4/8,B_E512:512/8/4/8", help="name:E/G/Gtop/k comma list"
    )
    ap.add_argument("--bt", default="auto", help="block_tokens for all kernels ('auto' or an int)")
    ap.add_argument("--unroll", choices=["auto", "full", "rolled"], default="auto",
                    help="final-select unroll for kernel1/kernel2 (kernel has no such knob)")
    a = ap.parse_args()
    print(f"JAX {jax.__version__} | {jax.devices()[0].platform} | n_dev {len(jax.devices())}")
    try:
        import libtpu

        print("libtpu", libtpu.__version__)
    except Exception:  # noqa: BLE001
        pass
    print(
        "routing device time = sum of ops under each kernel's named_scope (gate matmul outside the "
        "scope; router_logits gate-fed for the realistic good layout). No subtraction."
    )
    Ts = [int(x) for x in a.T.split(",")]
    for spec in a.configs.split(","):
        name, cfg = spec.split(":")
        E, G, Gtop, k = (int(v) for v in cfg.split("/"))
        bench_config(name, E, G, Gtop, k, Ts, a.bt, a.unroll)


if __name__ == "__main__":
    main()
