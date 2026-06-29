"""Benchmark `grouped_topk_pallas` (argmax-selection) vs the 3-`sort` `_biased_grouped_topk`.

Measures the **routing** device time of each path on a TPU host (e.g. falcon v7x): the whole routing
is wrapped in a `jax.named_scope`, the gate matmul that produces `router_logits` is kept OUTSIDE the
scope, and we sum the device time of every XLA op under the scope from a profiler trace. This reads
the end-to-end routing time directly (no subtraction) and — crucially — feeds `router_logits` from a
real gate matmul so the `sort` path gets the realistic GOOD layout (an isolated jit of the routing
picks a non-representative layout that is ~15x slower; do not benchmark it standalone).

The fused kernel runs at `block_tokens="auto"` (the tuned table) by default; `--bt <int>` forces a
block size and the `bt` column shows the size actually used. `--kernel` selects which of the three
variants to bench (`kernel`/`kernel1`/`kernel2`, or `all` to compare them side by side). Run on a TPU
host:

    python -m benchmark.kernels.grouped_topk.bench_grouped_topk \
        --T 64,128,256,512,1024,2048,4096,8192,16384,32768 --kernel all --bt auto
"""

import argparse
import functools
import glob
import gzip
import json
import os
import re
import time

import jax
import jax.numpy as jnp

try:
    # Real path on a TPU host with sgl_jax installed.
    from python.sgl_jax.srt.kernels.grouped_topk.v1.kernel import grouped_topk_pallas
    from python.sgl_jax.srt.kernels.grouped_topk.v1.kernel1 import grouped_topk_pallas as grouped_topk_pallas1
    from python.sgl_jax.srt.kernels.grouped_topk.v1.kernel2 import grouped_topk_pallas as grouped_topk_pallas2
    from python.sgl_jax.srt.kernels.grouped_topk.v1.kernel3 import grouped_topk_pallas as grouped_topk_pallas3

except Exception:  # noqa: BLE001
    # The falcon-embedded variant prepends the kernel source; `grouped_topk_pallas` is then already
    # defined at module scope, nothing to import.
    pass

TRACE_ROOT = os.environ.get("TOPK_TRACE_ROOT", "/tmp/tpu_logs/grouped_topk_bench")
H = 7168  # hidden size of the gate matmul that feeds router_logits
SCOPE_SORT = "SORTTOPK"
SCOPE_FUSED = "FUSEDTOPK"


def kernel_registry():
    """name -> grouped_topk_pallas impl, for whichever of the three variants imported.

    `kernel` = python-unroll, `kernel1` = narrow-candidate [BT,C], `kernel2` = fori_loop [BT,E].
    In the falcon-embedded variant only `grouped_topk_pallas` exists, so the registry is just that.
    """
    reg = {}
    for key, gname in (
        ("kernel", "grouped_topk_pallas"),
        ("kernel1", "grouped_topk_pallas1"),
        ("kernel2", "grouped_topk_pallas2"),
        ("kernel3", "grouped_topk_pallas3"),
    ):
        fn = globals().get(gname)
        if fn is not None:
            reg[key] = fn
    return reg


def ref_biased_grouped_topk(router_logits, correction_bias, *, num_expert_group, topk_group, topk):
    """Verbatim `gate.py:TopK._biased_grouped_topk` (the 3-`sort` reference)."""
    router_logits = router_logits.astype(jnp.float32)
    nt = router_logits.shape[0]
    s = router_logits.reshape(nt, -1) + jnp.expand_dims(correction_bias, 0)
    sg = s.reshape(nt, num_expert_group, -1)
    gscore = jnp.sum(jax.lax.top_k(sg, k=2)[0], axis=-1)
    gi = jax.lax.top_k(gscore, k=topk_group)[1]
    gm = jnp.clip(jax.nn.one_hot(gi, num_expert_group).sum(axis=1), 0, 1)
    epg = router_logits.shape[-1] // num_expert_group
    sm = jnp.broadcast_to(jnp.expand_dims(gm, -1), (nt, num_expert_group, epg)).reshape(nt, -1)
    tmp = jnp.where(sm, s, float("-inf"))
    ids = jax.lax.top_k(tmp, k=topk)[1]
    w = jnp.take_along_axis(router_logits, ids, axis=1)
    return w, ids


def _gate(hidden, w_gate):
    return jax.nn.sigmoid(jnp.dot(hidden, w_gate, precision=jax.lax.Precision.HIGHEST))


def make_sort(w_gate, bias, G, Gtop, k):
    def fn(hidden):
        logits = _gate(hidden, w_gate)  # gate OUTSIDE the scope
        with jax.named_scope(SCOPE_SORT):
            return ref_biased_grouped_topk(
                logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
            )

    return fn


def make_fused(fn, w_gate, bias, G, Gtop, k, bt):
    def run(hidden):
        logits = _gate(hidden, w_gate)  # gate OUTSIDE the scope
        with jax.named_scope(SCOPE_FUSED):
            return fn(
                logits,
                bias,
                num_expert_group=G,
                topk_group=Gtop,
                topk=k,
                block_tokens=bt,
                interpret=False,
            )

    return run


def resolve_bt(bt, bs, e, G, Gtop, k):
    """The block size grouped_topk_pallas will actually use, for DISPLAY only.

    Mirrors the block_tokens resolution inside grouped_topk_pallas (keep in sync): an explicit int
    is clamped to bs; "auto" consults the tuned table, then a 512-divisor default, then the largest
    VMEM-safe divisor. Returns the raw value if the kernel module can't be introspected.
    """
    if bt != "auto":
        return min(int(bt), bs)
    try:
        from python.sgl_jax.srt.kernels.grouped_topk.v1 import kernel2 as _k
    except Exception:  # noqa: BLE001  (falcon-embedded: no module to introspect)
        return bs
    tuned = _k.get_tuned_bt(bs, e, G, Gtop, k)
    if tuned is not None and bs % tuned == 0:
        return tuned
    if bs % 512 == 0:
        return min(512, bs)
    return _k._largest_safe_divisor(bs) or bs


def _trace_scope_us(run_fn, scope, tag, warmup=3, iters=20):
    """Per-iter device time (us) summed over all XLA ops whose name/args contain `scope`, plus the
    count of distinct matched op names (a sanity check that the scope was found)."""
    tag = re.sub(r"[^A-Za-z0-9]", "_", tag)
    for _ in range(warmup):
        jax.block_until_ready(run_fn())
    troot = os.path.join(TRACE_ROOT, f"{tag}_{os.getpid()}_{int(time.time() * 1000)}")
    os.makedirs(troot, exist_ok=True)
    with jax.profiler.trace(troot):
        for i in range(iters):
            with jax.profiler.StepTraceAnnotation("s", step_num=i):
                jax.block_until_ready(run_fn())
    dirs = glob.glob(os.path.join(troot, "plugins", "profile", "*"))
    if not dirs:
        return float("nan"), 0
    latest = max(dirs, key=os.path.getmtime)
    evs = []
    for tf in sorted(glob.glob(os.path.join(latest, "*.trace.json.gz"))):
        evs += json.load(gzip.open(tf)).get("traceEvents", [])
    pn, tn = {}, {}
    for e in evs:
        if e.get("ph") == "M":
            a = e.get("args", {})
            if e["name"] == "process_name":
                pn[e["pid"]] = a.get("name", "")
            if e["name"] == "thread_name":
                tn[(e["pid"], e["tid"])] = a.get("name", "")
    nmod = 0
    scope_tot = 0.0
    names = set()
    for e in evs:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if pn.get(e["pid"], "") != "/device:TPU:0":
            continue
        t = tn.get((e["pid"], e["tid"]), "")
        dur = e["dur"]
        ddp = e.get("args", {}).get("device_duration_ps")
        if ddp:
            dur = float(ddp) / 1e6
        if t == "XLA Modules":
            nmod += 1
            continue
        if t != "XLA Ops":
            continue
        blob = e["name"] + " " + json.dumps(e.get("args", {}))
        if scope in blob:
            scope_tot += dur
            names.add(re.sub(r"\(\d+\)$", "", e["name"]))
    return scope_tot / max(nmod, 1), len(names)


def bench_config(name, E, G, Gtop, k, Ts, kernels, bt):
    """`kernels`: ordered dict {kernel_name: grouped_topk_pallas_fn} to benchmark vs the sort ref.
    `bt`: block_tokens forwarded to every kernel ("auto" or an int)."""
    w_gate = jax.device_put(jax.random.normal(jax.random.PRNGKey(3), (H, E), dtype=jnp.float32))
    bias = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (E,), dtype=jnp.float32) * 0.1)
    sfn = jax.jit(make_sort(w_gate, bias, G, Gtop, k))
    ffns = {kn: jax.jit(make_fused(fn, w_gate, bias, G, Gtop, k, bt)) for kn, fn in kernels.items()}

    cols = " ".join(f"{kn + '_us':>11}" for kn in ffns)
    print(f"\n=== {name} (E={E}, G={G}, Gtop={Gtop}, k={k}) bt={bt} ===")
    print(f"{'T':>7} {'bt':>6} {'sort_us':>10} {cols}   {'best':>16}")
    for T in Ts:
        h = jax.device_put(jax.random.normal(jax.random.PRNGKey(5), (T, H), dtype=jnp.float32))
        bt_used = resolve_bt(bt, T, E, G, Gtop, k)
        jax.block_until_ready(sfn(h))
        sort_us, _ = _trace_scope_us(functools.partial(sfn, h), SCOPE_SORT, f"sort_{name}_{T}")

        times = {}
        for kn, ffn in ffns.items():
            try:
                jax.block_until_ready(ffn(h))
            except Exception as ex:  # noqa: BLE001  (VMEM OOM etc. -> mark NaN, keep going)
                times[kn] = float("nan")
                if "RESOURCE_EXHAUSTED" not in str(ex) and "vmem" not in str(ex).lower():
                    print(f"  {kn} T={T}: unexpected error: {str(ex)[:80]}")
                continue
            us, _ = _trace_scope_us(functools.partial(ffn, h), SCOPE_FUSED, f"{kn}_{name}_{T}")
            times[kn] = us

        valid = {kn: t for kn, t in times.items() if t == t and t > 0}  # t==t drops NaN
        row = " ".join(f"{times[kn]:>11.2f}" for kn in ffns)
        best = min(valid, key=valid.get) if valid else None
        tag = f"{best} {sort_us / valid[best]:.2f}x" if best else "-"
        print(f"{T:>7} {bt_used:>6} {sort_us:>10.2f} {row}   {tag:>16}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="64,128,256,512,1024,2048,4096,8192,16384,32768")
    ap.add_argument(
        "--configs", default="A_E256:256/8/4/8,B_E512:512/8/4/8", help="name:E/G/Gtop/k comma list"
    )
    ap.add_argument(
        "--kernel",
        default="all",
        help="which kernel(s) to bench: 'all' or a comma list of kernel,kernel1,kernel2",
    )
    ap.add_argument(
        "--bt",
        default="auto",
        help="block_tokens for every kernel: 'auto' (tuned table) or an int (forced block size)",
    )
    a = ap.parse_args()

    bt = a.bt if a.bt == "auto" else int(a.bt)

    registry = kernel_registry()
    if not registry:
        raise SystemExit("no grouped_topk_pallas kernel imported (check the imports at top of file)")
    if a.kernel == "all":
        kernels = registry
    else:
        kernels = {}
        for kn in a.kernel.split(","):
            kn = kn.strip()
            if kn not in registry:
                raise SystemExit(f"unknown/unavailable kernel {kn!r}; have {sorted(registry)}")
            kernels[kn] = registry[kn]

    print(f"JAX {jax.__version__} | {jax.devices()[0].platform} | n_dev {len(jax.devices())}")
    try:
        import libtpu

        print("libtpu", libtpu.__version__)
    except Exception:  # noqa: BLE001
        pass
    print(
        "routing device time = sum of ops under named_scope (gate matmul outside scope; "
        "router_logits gate-fed for the realistic good layout). No subtraction."
    )
    print(f"kernels: {', '.join(kernels)}  ('best' = fastest kernel & its speedup vs sort)")
    Ts = [int(x) for x in a.T.split(",")]
    for spec in a.configs.split(","):
        name, cfg = spec.split(":")
        E, G, Gtop, k = (int(v) for v in cfg.split("/"))
        bench_config(name, E, G, Gtop, k, Ts, kernels, bt)


if __name__ == "__main__":
    main()
