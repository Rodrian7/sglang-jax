"""Per-named-scope device-time profiler for grouped_topk_pallas_v3 (kernel3).

Runs v3 under a JAX profiler trace with custom-call region tracing enabled, so the in-kernel
`jax.named_scope`s (bias_add / group_top2 / group_select / expert_mask / final_select) surface as
trace regions. Parses the trace and prints a per-scope device-time breakdown to STDOUT (so results
come back via `falcon exp logs`, no GCS access needed). Used to locate the current bottleneck.

Run on a TPU host. LIBTPU region-trace flags are set below BEFORE importing jax.
"""

from __future__ import annotations

import os

# Must be set before jax/libtpu init: expose custom-call (Pallas) internal regions in the trace.
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "")
    + " --xla_enable_custom_call_region_trace=true"
    + " --xla_xprof_register_llo_debug_info=true"
).strip()

import argparse  # noqa: E402
import collections  # noqa: E402
import glob  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from sgl_jax.srt.kernels.grouped_topk.v2.kernel3 import grouped_topk_pallas_v3  # noqa: E402

SCOPES = ["bias_add", "group_top2", "group_select", "expert_mask", "final_select"]


def _logits(T, E, seed):
    return jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(seed), (T, E), dtype=jnp.float32))


def _latest_events(troot):
    dirs = glob.glob(os.path.join(troot, "plugins", "profile", "*"))
    if not dirs:
        return []
    latest = max(dirs, key=os.path.getmtime)
    evs = []
    for tf in sorted(glob.glob(os.path.join(latest, "*.trace.json.gz"))):
        evs += json.load(gzip.open(tf)).get("traceEvents", [])
    return evs


def profile_one(T, E, G, Gtop, k, warmup, iters, troot):
    lg = jax.device_put(_logits(T, E, T + E))
    b = jax.device_put(jax.random.normal(jax.random.PRNGKey(E), (E,), dtype=jnp.float32) * 0.1)
    fn = jax.jit(
        lambda l, bb: grouped_topk_pallas_v3(
            l, bb, num_expert_group=G, topk_group=Gtop, topk=k
        )
    )
    for _ in range(warmup):
        jax.block_until_ready(fn(lg, b))
    rt = os.path.join(troot, f"v3_{T}_{int(time.time() * 1000)}")
    os.makedirs(rt, exist_ok=True)
    with jax.profiler.trace(rt):
        for i in range(iters):
            with jax.profiler.StepTraceAnnotation("s", step_num=i):
                jax.block_until_ready(fn(lg, b))
    evs = _latest_events(rt)

    pn, tn = {}, {}
    for e in evs:
        if e.get("ph") == "M":
            a = e.get("args", {})
            if e["name"] == "process_name":
                pn[e["pid"]] = a.get("name", "")
            if e["name"] == "thread_name":
                tn[(e["pid"], e["tid"])] = a.get("name", "")

    # sum device-side (TPU:0) X-event durations by (thread, name)
    by_name = collections.Counter()
    by_thread = collections.Counter()
    cnt = collections.Counter()
    for e in evs:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if pn.get(e.get("pid")) != "/device:TPU:0":
            continue
        thr = tn.get((e.get("pid"), e.get("tid")), "")
        nm = e.get("name", "")
        by_name[nm] += e["dur"]
        by_thread[thr] += e["dur"]
        cnt[nm] += 1

    # kernel total = the custom-call event
    kernel_names = [n for n in by_name if "grouped-topk-v3" in n or "grouped_topk" in n]
    kernel_us = sum(by_name[n] for n in kernel_names) / iters

    print(f"\n===== T={T} E={E} G={G} Gtop={Gtop} k={k}  (iters={iters}) =====")
    print(f"kernel (custom-call) total: {kernel_us:.2f} us/iter")
    print("\n-- device time by thread (us/iter) --")
    for thr, d in by_thread.most_common(10):
        print(f"  {d/iters:9.2f}  {thr}")
    print("\n-- per named-scope (us/iter, % of kernel) --")
    for sc in SCOPES:
        tot = sum(by_name[n] for n in by_name if sc in n)
        c = sum(cnt[n] for n in by_name if sc in n)
        pct = (100 * (tot / iters) / kernel_us) if kernel_us else float("nan")
        print(f"  {sc:14s} {tot/iters:9.3f} us  {pct:6.1f}%   (x{c})")
    print("\n-- top-20 device events by name (us/iter) --")
    for nm, d in by_name.most_common(20):
        print(f"  {d/iters:9.3f}  x{cnt[nm]:<4d}  {nm[:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="8192,16384")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--Gtop", type=int, default=4)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--trace-root", default=os.path.join(os.environ.get("OUT", "/tmp/v3prof"), "xprof"))
    a = ap.parse_args()
    print(f"JAX {jax.__version__} | {jax.devices()[0].device_kind} | LIBTPU_INIT_ARGS set")
    for T in [int(x) for x in a.T.split(",")]:
        profile_one(T, a.E, a.G, a.Gtop, a.topk, a.warmup, a.iters, a.trace_root)


if __name__ == "__main__":
    main()
