"""Parse a JAX/XProf Chrome trace.json.gz and summarize device-side op time by
category, plus per-stream busy / wall-span / idle (to expose overlap regressions).
Usage: python parse_trace.py <trace.json.gz> [<trace.json.gz> ...]
"""

import gzip
import json
import re
import sys
from collections import defaultdict


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def categorize(nm):
    n = nm.lower()
    if "fused-moe" in n or "moe" in n:
        return "moe"
    if "ragged_paged" in n or "ragged-paged" in n or "flash" in n or "attention" in n:
        return "attention"
    if re.search(r"all-to-all|all-gather|reduce-scatter|all-reduce|collective|permute", n):
        return "collective"
    if re.search(r"rope|rotary", n):
        return "rope"
    if re.search(r"sin|cos|iota", n):
        return "rope-trig"
    if re.search(r"copy|transpose|bitcast|reshape|dynamic-slice|dynamic-update", n):
        return "copy/reshape"
    if "custom-call" in n or "custom_call" in n:
        return "custom-call"
    if "fusion" in n:
        return "fusion"
    if "matmul" in n or "dot" in n or "convolution" in n:
        return "matmul"
    return "other"


def summarize(path):
    data = load(path)
    events = data["traceEvents"]
    pid_name, tid_name = {}, {}
    for e in events:
        if e.get("ph") == "M":
            if e.get("name") == "process_name":
                pid_name[e["pid"]] = e.get("args", {}).get("name", "")
            elif e.get("name") == "thread_name":
                tid_name[(e["pid"], e["tid"])] = e.get("args", {}).get("name", "")

    dev_pids = {p for p, n in pid_name.items() if re.search(r"TPU|TensorCore|/device:", n or "")}

    by_cat = defaultdict(lambda: [0.0, 0])
    # per (pid,tid) stream: busy sum, min start, max end
    stream = defaultdict(lambda: [0.0, float("inf"), 0.0])
    for e in events:
        if e.get("ph") != "X" or "dur" not in e or e.get("pid") not in dev_pids:
            continue
        lane = tid_name.get((e["pid"], e.get("tid")), "")
        if "XLA Ops" not in lane and "TensorCore" not in lane and "XLA Modules" not in lane:
            continue
        if "XLA Modules" in lane:  # module-level lane spans the whole program; skip for op cat
            continue
        cat = categorize(e.get("name", "?"))
        by_cat[cat][0] += e["dur"]
        by_cat[cat][1] += 1
        st, dur = e["ts"], e["dur"]
        s = stream[(e["pid"], e.get("tid"), lane)]
        s[0] += dur
        s[1] = min(s[1], st)
        s[2] = max(s[2], st + dur)

    print(f"\n==================== {path.split('/')[2]}")
    busy_total = sum(v[0] for v in by_cat.values())
    print(f"total device-op busy: {busy_total/1000:.1f} ms")
    for cat, (dur, cnt) in sorted(by_cat.items(), key=lambda x: -x[1][0]):
        print(f"  {cat:14s} {dur/1000:9.1f} ms  ({dur/max(busy_total,1)*100:5.1f}%)  x{cnt}")
    print("  per-stream busy / wall-span / idle:")
    for (pid, tid, lane), (busy, mn, mx) in sorted(stream.items(), key=lambda x: -x[1][0])[:6]:
        wall = mx - mn
        idle = wall - busy
        print(
            f"    [{lane[:24]:24s}] busy={busy/1000:8.1f}  wall={wall/1000:8.1f}  "
            f"idle={idle/1000:8.1f} ({idle/max(wall,1)*100:4.1f}%)"
        )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
