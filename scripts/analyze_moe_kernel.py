#!/usr/bin/env python3
"""Bound analysis for fused_ep_moe_v2 (or any Pallas kernel) from xprof data.

Inputs:
  --exp-id <falcon_exp_id>           pull xplane.pb files from GCS, run analysis
  --logdir <path>                    use a local logdir of xplane.pb files
  --run <run_name>                   when multiple runs in logdir, pick this one
  --kernel-pattern <regex>           kernel name regex (default: fused-moe-v2-k_)

Output:
  Stdout: structured bound-analysis report with the SAME bound categories the
  team standardized on (MXU sublane / VPU / HBM / VMEM / ICI / launch).

Caveats:
  - xprof exposes HLO-level timings. Pallas kernel internals are OPAQUE
    (one event per call). For inside-kernel bound analysis you also need
    xplane.pb proto parsing (TODO: separate tool).
  - The script identifies WHICH bound is suspected based on:
      * kernel call duration distribution
      * surrounding HLO ops (collectives, transposes, etc.)
      * memory profile peak vs capacity
      * xprof overview MXU utilization estimate (when available)

Usage:
  python3 scripts/analyze_moe_kernel.py --exp-id exp-uknm7tbazn
  python3 scripts/analyze_moe_kernel.py --logdir /tmp/xprof_logdir --run uknm_direct_64

Output is JSON by default; pass --human for narrative.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field

# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------


@dataclass
class KernelCallStats:
    name: str
    count: int
    total_us: float
    mean_us: float
    p10_us: float
    median_us: float
    p90_us: float
    p99_us: float


@dataclass
class HloCategoryStat:
    category: str
    sum_us: float
    count: int
    avg_us: float
    pct_of_window: float


@dataclass
class BoundReport:
    run: str
    device_type: str = ""
    kernel: KernelCallStats | None = None
    surrounding_hlo: list[HloCategoryStat] = field(default_factory=list)
    inter_call_window_us: float = 0.0
    peak_hbm_bytes_per_dev: int = 0
    hbm_capacity_bytes: int = 0
    hbm_utilization_pct: float = 0.0
    notes: list[str] = field(default_factory=list)
    suspected_bounds: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Local xprof launcher
# -----------------------------------------------------------------------------


def _free_port(start: int = 8090) -> int:
    import socket

    for p in range(start, start + 100):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port in range")


def _ensure_local_xprof(logdir: str) -> tuple[str, subprocess.Popen | None]:
    """Launch a local xprof server on a free port and return (base_url, proc).

    proc is None when an existing xprof on $XPROF_URL is reused.
    """
    if env_url := os.environ.get("XPROF_URL"):
        return env_url.rstrip("/"), None

    port = _free_port(8090)
    proc = subprocess.Popen(
        ["xprof", "-l", logdir, "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait until /runs responds
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/data/plugin/profile/runs", timeout=2) as r:
                _ = r.read()
            return base, proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"local xprof on port {port} did not become ready within 30s")


def _xprof_get_json(base: str, run: str, tag: str, **extra: str) -> object:
    qs = {"run": run, "tag": tag, **extra}
    url = f"{base}/data/plugin/profile/data?" + urllib.parse.urlencode(qs)
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _xprof_runs(base: str) -> list[str]:
    url = f"{base}/data/plugin/profile/runs"
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return json.loads(raw)


# -----------------------------------------------------------------------------
# Trace extraction (chrome trace JSON — kernel call times + surrounding HLO)
# -----------------------------------------------------------------------------


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(round((len(s) - 1) * p))
    return s[k]


def _device_pid_with_most_kernel_events(events: list[dict], kernel_re: re.Pattern) -> int | None:
    counts: dict[int, int] = defaultdict(int)
    for e in events:
        n = e.get("name", "")
        pid = e.get("pid")
        if pid is None or not isinstance(n, str) or not kernel_re.search(n):
            continue
        counts[pid] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _event_dur_us(e: dict) -> float:
    args = e.get("args", {}) or {}
    dp = args.get("device_duration_ps")
    if dp:
        try:
            return float(dp) / 1e6
        except Exception:
            pass
    d = e.get("dur")
    if d is not None:
        try:
            return float(d)
        except Exception:
            return 0.0
    return 0.0


def _event_ts_us(e: dict) -> float:
    try:
        return float(e.get("ts", 0))
    except Exception:
        return 0.0


def _hlo_category(e: dict) -> str:
    args = e.get("args", {}) or {}
    cat = args.get("hlo_category")
    if cat:
        return str(cat)
    n = e.get("name", "")
    if not isinstance(n, str):
        return "?"
    nl = n.lower()
    for needle, label in [
        ("all-gather", "all_gather"),
        ("all-reduce", "all_reduce"),
        ("reduce-scatter", "reduce_scatter"),
        ("barrier", "barrier"),
        ("copy-start", "copy_start"),
        ("copy-done", "copy_done"),
        ("fusion", "fusion"),
        ("reshape", "reshape"),
        ("convert", "convert"),
        ("transpose", "transpose"),
        ("pad", "pad"),
        ("rpa", "rpa"),
        ("topk", "topk"),
    ]:
        if needle in nl:
            return label
    return "other"


def extract_kernel_stats_from_trace(
    trace: dict,
    kernel_pattern: str,
) -> tuple[KernelCallStats, list[HloCategoryStat], float]:
    kernel_re = re.compile(kernel_pattern)
    events = trace.get("traceEvents", [])
    pid = _device_pid_with_most_kernel_events(events, kernel_re)
    if pid is None:
        return KernelCallStats("(none)", 0, 0, 0, 0, 0, 0, 0), [], 0.0

    dev_events = [e for e in events if e.get("pid") == pid]
    kernel_events = [
        e for e in dev_events if isinstance(e.get("name"), str) and kernel_re.search(e["name"])
    ]
    kernel_events.sort(key=_event_ts_us)

    durs = [_event_dur_us(e) for e in kernel_events]
    durs = [d for d in durs if d > 0]

    if not durs:
        return KernelCallStats("(zero-dur)", 0, 0, 0, 0, 0, 0, 0), [], 0.0

    # Strip suffix beyond first space or 80 chars to get the kernel-base name
    name0 = kernel_events[0].get("name", "")
    name = name0[:80]

    stats = KernelCallStats(
        name=name,
        count=len(durs),
        total_us=sum(durs),
        mean_us=sum(durs) / len(durs),
        p10_us=_percentile(durs, 0.10),
        median_us=_percentile(durs, 0.50),
        p90_us=_percentile(durs, 0.90),
        p99_us=_percentile(durs, 0.99),
    )

    # Inter-call windows = time between consecutive kernel events
    # WARNING: this is "next layer's worth of work", NOT kernel-internal.
    # Drop ramp-up/down and step-boundary outliers (use median 50%).
    gaps = []
    for i in range(len(kernel_events) - 1):
        s = _event_ts_us(kernel_events[i]) + _event_dur_us(kernel_events[i])
        e = _event_ts_us(kernel_events[i + 1])
        if e > s:
            gaps.append((s, e, e - s))
    gaps.sort(key=lambda x: x[2])
    if not gaps:
        return stats, [], 0.0
    n = len(gaps)
    mid = gaps[n // 4 : (3 * n) // 4]
    if not mid:
        return stats, [], 0.0

    # Aggregate events per HLO category that fall fully within a mid gap
    timed = [(_event_ts_us(e), _event_dur_us(e), _hlo_category(e)) for e in dev_events]
    timed = [t for t in timed if t[1] > 0]
    timed.sort(key=lambda t: t[0])
    ts_arr = [t[0] for t in timed]
    import bisect

    cat_us: dict[str, float] = defaultdict(float)
    cat_cnt: dict[str, int] = defaultdict(int)
    total_window_us = 0.0
    for g_start, g_end, _g_dur in mid:
        total_window_us += g_end - g_start
        lo = bisect.bisect_left(ts_arr, g_start)
        hi = bisect.bisect_right(ts_arr, g_end)
        for ts, dur, cat in timed[lo:hi]:
            if ts + dur > g_end:
                continue
            cat_us[cat] += dur
            cat_cnt[cat] += 1

    cats = sorted(
        (
            HloCategoryStat(
                category=c,
                sum_us=cat_us[c],
                count=cat_cnt[c],
                avg_us=cat_us[c] / max(1, cat_cnt[c]),
                pct_of_window=100.0 * cat_us[c] / max(1.0, total_window_us),
            )
            for c in cat_us
        ),
        key=lambda x: -x.sum_us,
    )

    return stats, cats, total_window_us / max(1, len(mid))


# -----------------------------------------------------------------------------
# Memory profile
# -----------------------------------------------------------------------------


def extract_hbm_from_memory_profile(mp: dict) -> tuple[int, int]:
    """Return (peak_per_dev_bytes, capacity_per_dev_bytes)."""
    peak = 0
    cap = 0
    allocs = mp.get("memoryProfilePerAllocator", {})
    for _name, payload in allocs.items():
        snap = payload.get("memoryProfileSnapshots", [])
        if not snap:
            continue
        # Last snapshot — find peak via maxStat.peakBytesInUse
        summary = payload.get("profileSummary") or {}
        peak_bytes = summary.get("peakBytesUsage") or summary.get("peakBytesInUse") or 0
        cap_bytes = summary.get("memoryCapacity") or 0
        try:
            peak = max(peak, int(peak_bytes))
            cap = max(cap, int(cap_bytes))
        except Exception:
            pass
    return peak, cap


# -----------------------------------------------------------------------------
# Bound classification
# -----------------------------------------------------------------------------


def classify_bound(report: BoundReport) -> list[str]:
    """Heuristic bound classification from the data we have.

    Honest about what xprof CAN'T tell us:
      - In-kernel sublane/MXU/HBM saturation requires xplane.pb proto-level data
      - We can only diagnose what's around the kernel and gross resource use.
    """
    notes = []
    bounds: list[str] = []

    if report.kernel is None or report.kernel.count == 0:
        return ["no kernel data"]

    # 1) Kernel duration spread = how repeatable is the workload?
    spread = (report.kernel.p90_us - report.kernel.p10_us) / max(1.0, report.kernel.mean_us)
    if spread > 0.20:
        notes.append(
            f"kernel duration spread p10-p90={spread*100:.0f}% — wide; suggests data-dependent path or contention"
        )

    # 2) Inter-call window dominated by collectives?
    coll_us = sum(
        c.sum_us
        for c in report.surrounding_hlo
        if c.category in ("all_gather", "all_reduce", "reduce_scatter")
    )
    if report.inter_call_window_us > 0:
        coll_pct = 100.0 * coll_us / report.inter_call_window_us
        if coll_pct > 25:
            bounds.append(f"ICI (inter-call window {coll_pct:.1f}% in collectives)")
            notes.append("  → consider: reduce SP all-gather count / merge AR / batch collectives")

    fusion_us = sum(c.sum_us for c in report.surrounding_hlo if c.category == "fusion")
    if report.inter_call_window_us > 0:
        fusion_pct = 100.0 * fusion_us / report.inter_call_window_us
        if fusion_pct > 35:
            bounds.append(
                f"surrounding-layer GEMMs (fusion {fusion_pct:.1f}% — likely QKV/Out proj / RMSNorm @ small M)"
            )
            notes.append(
                "  → these fusions are likely M=N/TP MXU-sublane-saturated GEMMs; "
                "increase effective M (mixed_chunk, larger BSZ, smaller TP shard)."
            )

    # 3) HBM utilization vs capacity
    if report.hbm_capacity_bytes > 0:
        util = 100.0 * report.peak_hbm_bytes_per_dev / report.hbm_capacity_bytes
        report.hbm_utilization_pct = util
        if util > 90:
            bounds.append(f"HBM capacity ({util:.1f}% used)")
        elif util < 50:
            notes.append(f"HBM only {util:.1f}% used — capacity headroom exists")

    # 4) Inter-call window magnitude vs kernel time = fixing kernel may be moot
    if report.inter_call_window_us > 0 and report.kernel.mean_us > 0:
        ratio = report.inter_call_window_us / report.kernel.mean_us
        notes.append(
            f"inter-call window / kernel = {ratio:.2f}× — "
            f"{'kernel dominates' if ratio < 0.5 else 'surrounding ops dominate' if ratio > 1.5 else 'roughly balanced'}"
        )

    notes.append(
        "in-kernel sublane/MXU/HBM bound is OPAQUE to xprof; needs xplane.pb proto parser (TODO)"
    )
    if not bounds:
        bounds.append("inconclusive at xprof granularity (need xplane.pb proto for inside-kernel)")

    report.notes = notes
    return bounds


# -----------------------------------------------------------------------------
# Falcon helpers (pull artifacts from GCS)
# -----------------------------------------------------------------------------


def _falcon_artifact_uri(exp_id: str) -> str:
    out = subprocess.check_output(
        ["falcon", "workflow", "artifact", "get", exp_id, "--output", "json"],
        text=True,
    )
    return json.loads(out)["uris"]["artifact_uri"]


def _gsutil_ls(uri: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["gsutil", "ls", "-r", uri], text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError:
        return []
    return [ln for ln in out.splitlines() if ln.strip().startswith("gs://")]


def _gsutil_cp(src: str, dst: str) -> None:
    subprocess.check_call(
        ["gsutil", "cp", src, dst], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _pull_falcon_into_logdir(exp_id: str, logdir: str) -> str:
    """Download the most-recent xplane.pb folder into logdir/<exp_id>/...

    Returns the run name that xprof will see.
    """
    art = _falcon_artifact_uri(exp_id)
    # Look for plugin/profile dirs (server profiling) or trace/v2_trace (bench)
    listing = _gsutil_ls(art)
    candidates = [
        ln for ln in listing if ln.endswith(".xplane.pb") or ln.endswith(".trace.json.gz")
    ]
    # Group by parent (profile timestamp dir)
    parents = defaultdict(list)
    for ln in candidates:
        parents["/".join(ln.split("/")[:-1])].append(ln)
    # Pick the parent dir with most files (most data — usually the real bench, not the warmup)
    if not parents:
        raise RuntimeError(f"no xplane.pb files in {art}")
    parent = max(parents.items(), key=lambda kv: len(kv[1]))[0]
    # parent looks like: gs://.../plugins/profile/<TS>
    # We need the structure: <run>/plugins/profile/<TS>/files
    # Use exp_id as the run name; place under logdir/<exp_id>/plugins/profile/<TS>/
    ts = parent.split("/")[-1]
    target = os.path.join(logdir, exp_id, "plugins", "profile", ts)
    os.makedirs(target, exist_ok=True)
    for src in parents[parent]:
        dst = os.path.join(target, src.split("/")[-1])
        if not os.path.exists(dst):
            _gsutil_cp(src, dst)
    return f"{exp_id}/{ts}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def analyze(
    *,
    logdir: str,
    run: str,
    kernel_pattern: str,
) -> BoundReport:
    base, proc = _ensure_local_xprof(logdir)
    try:
        runs = _xprof_runs(base)
        if run not in runs:
            # Try suffix match
            matches = [r for r in runs if run in r]
            if matches:
                run = matches[0]
            else:
                raise RuntimeError(f"run {run!r} not found; have: {runs[:5]}")

        # Get device type
        op_profile = _xprof_get_json(base, run, "op_profile")
        device_type = (
            (op_profile or {}).get("deviceType", "?") if isinstance(op_profile, dict) else "?"
        )

        # Trace for kernel-event extraction — use trace_viewer endpoint (returns full chrome JSON)
        trace = _xprof_get_json(base, run, "trace_viewer")
        if not isinstance(trace, dict):
            raise RuntimeError(
                "trace_viewer did not return JSON; xprof may not support this endpoint"
            )

        kernel_stats, hlo_cats, gap_us = extract_kernel_stats_from_trace(trace, kernel_pattern)

        # Memory profile
        mp = _xprof_get_json(base, run, "memory_profile")
        peak, cap = (0, 0)
        if isinstance(mp, dict):
            peak, cap = extract_hbm_from_memory_profile(mp)

        report = BoundReport(
            run=run,
            device_type=device_type,
            kernel=kernel_stats,
            surrounding_hlo=hlo_cats,
            inter_call_window_us=gap_us,
            peak_hbm_bytes_per_dev=peak,
            hbm_capacity_bytes=cap,
        )
        report.suspected_bounds = classify_bound(report)
        return report
    finally:
        if proc is not None:
            proc.kill()


def render_human(r: BoundReport) -> str:
    lines = []
    lines.append(f"=== Bound analysis — run={r.run} device={r.device_type} ===")
    if r.kernel:
        k = r.kernel
        lines.append(f"\nKernel: {k.name}")
        lines.append(f"  count={k.count}  total={k.total_us/1e3:.1f} ms  mean={k.mean_us:.2f} µs")
        lines.append(
            f"  p10={k.p10_us:.2f}  median={k.median_us:.2f}  p90={k.p90_us:.2f}  p99={k.p99_us:.2f} µs"
        )
    lines.append(
        f"\nInter-call window (= surrounding non-kernel work): mean={r.inter_call_window_us:.2f} µs"
    )
    if r.surrounding_hlo:
        lines.append("  HLO-category breakdown of inter-call window:")
        lines.append(
            f"    {'category':<18s} {'sum_us/win':>10s} {'cnt/win':>8s} {'avg_us':>8s} {'%win':>6s}"
        )
        for c in r.surrounding_hlo[:12]:
            lines.append(
                f"    {c.category:<18s} {c.sum_us:>10.2f} {c.count:>8d} {c.avg_us:>8.2f} {c.pct_of_window:>5.1f}%"
            )
    if r.hbm_capacity_bytes:
        lines.append(
            f"\nHBM: peak_per_dev={r.peak_hbm_bytes_per_dev/1e9:.2f} GB / "
            f"cap={r.hbm_capacity_bytes/1e9:.2f} GB ({r.hbm_utilization_pct:.1f}%)"
        )
    lines.append("\n=== Bound verdict ===")
    for b in r.suspected_bounds:
        lines.append(f"  - {b}")
    if r.notes:
        lines.append("\nNotes:")
        for n in r.notes:
            lines.append(f"  · {n}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", help="Falcon experiment id; pulls xplane.pb files from GCS")
    ap.add_argument(
        "--logdir", help="Local logdir (xprof layout: <run>/plugins/profile/<ts>/*.xplane.pb)"
    )
    ap.add_argument("--run", help="Run name within logdir (or substring)")
    ap.add_argument(
        "--kernel-pattern",
        default=r"fused-moe-v2-k_",
        help="Regex matching the kernel-event name (default: fused-moe-v2-k_)",
    )
    ap.add_argument("--human", action="store_true", help="Render narrative instead of JSON")
    args = ap.parse_args(argv)

    if not args.exp_id and not args.logdir:
        ap.error("must pass --exp-id or --logdir")

    if args.exp_id:
        # Pull artifact into a tmp logdir
        logdir = tempfile.mkdtemp(prefix=f"xprof_{args.exp_id}_")
        run = _pull_falcon_into_logdir(args.exp_id, logdir)
    else:
        logdir = os.path.abspath(args.logdir)
        if args.run is None:
            ap.error("--run required when --logdir is given")
        run = args.run

    report = analyze(logdir=logdir, run=run, kernel_pattern=args.kernel_pattern)
    if args.human:
        print(render_human(report))
    else:
        print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
