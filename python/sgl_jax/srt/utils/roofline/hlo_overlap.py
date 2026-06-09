"""Parse optimized, scheduled XLA HLO for the Overlap analysis.

Given the HLO text of the compiled forward (from ``compile_forward_hlo`` /
``tools/dump_forward_hlo.py``), extract what the COMPILER actually did about
communication overlap -- not a theoretical envelope:

  * which collectives (all-reduce / all-to-all / reduce-scatter / all-gather /
    collective-permute) were emitted, and how many bytes,
  * which are ASYNC (split into ``*-start`` / ``*-done`` -- i.e. XLA intends to
    overlap them), vs synchronous, and
  * for each async collective, the COMPUTE scheduled in its shadow (the fusion /
    dot / Pallas custom-call instructions between its ``-start`` and ``-done`` in
    schedule order) -- the real overlap window.

This is pure-Python text parsing (no jax). Schedule order = textual order of the
entry computation in a compiled (scheduled) HLO module.
"""

from __future__ import annotations

import re

_DT = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "f8e4m3fn": 1,
    "f8e5m2": 1,
    "bf16": 2,
    "f16": 2,
    "s16": 2,
    "u16": 2,
    "f32": 4,
    "s32": 4,
    "u32": 4,
    "f64": 8,
    "s64": 8,
    "u64": 8,
}
_COLLECTIVES = (
    "all-reduce",
    "all-gather",
    "reduce-scatter",
    "all-to-all",
    "collective-permute",
    "ragged-all-to-all",
)
_SHAPE = re.compile(r"\b(" + "|".join(_DT) + r")\[([\d,]*)\]")
_INSTR = re.compile(r"^\s*(%[\w.\-]+)\s*=\s*(.*)$")


def _first_shape_bytes(rhs: str) -> int:
    """Bytes of the first array shape on the RHS (the instruction's result)."""
    total = 0
    for m in _SHAPE.finditer(rhs):
        dt, dims = m.group(1), m.group(2)
        n = 1
        for d in dims.split(","):
            if d.strip():
                n *= int(d)
        total = n * _DT[dt]
        break
    return total


def _opcode(rhs: str) -> str | None:
    """The HLO opcode = the identifier immediately before the operand '('.
    Robust to tuple shapes by scanning for known opcodes as call heads."""
    # async/collective heads first (longest match), then compute
    for kw in (
        "all-reduce-start",
        "all-reduce-done",
        "all-gather-start",
        "all-gather-done",
        "reduce-scatter",
        "collective-permute-start",
        "collective-permute-done",
        "all-to-all",
        "ragged-all-to-all",
        "all-reduce",
        "all-gather",
        "async-start",
        "async-done",
        "async-update",
        "fusion",
        "convolution",
        "custom-call",
        "dot",
    ):
        if re.search(r"(^|[^\w-])" + re.escape(kw) + r"\(", rhs):
            return kw
    return None


def _coll_type(opcode: str) -> str | None:
    for c in _COLLECTIVES:
        if opcode.startswith(c):
            return c
    return None


def _entry_body(text: str) -> list[str]:
    """Lines of the ENTRY computation body, in schedule order."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("ENTRY ") or " ENTRY " in ln:
            start = i
            break
    if start is None:
        return lines  # fall back to whole module
    depth, body, began = 0, [], False
    for ln in lines[start:]:
        depth += ln.count("{") - ln.count("}")
        if "{" in ln and not began:
            began = True
            continue
        if began:
            if depth <= 0:
                break
            body.append(ln)
    return body


def parse_hlo_overlap(text: str) -> dict:
    body = _entry_body(text)
    instrs = []  # (name, opcode, bytes, is_compute, coll)
    for ln in body:
        m = _INSTR.match(ln)
        if not m:
            continue
        name, rhs = m.group(1), m.group(2)
        op = _opcode(rhs)
        if op is None:
            continue
        # a Pallas/Mosaic kernel is a custom-call but NOT a collective custom-call
        is_compute = op in ("fusion", "dot", "convolution") or (
            op == "custom-call" and "all-" not in rhs and "collective" not in rhs
        )
        instrs.append(
            dict(
                name=name,
                op=op,
                bytes=_first_shape_bytes(rhs),
                compute=is_compute,
                coll=_coll_type(op),
                line=ln.strip(),
            )
        )

    # match async start->done windows; collect shadow compute between them
    starts = {}  # base-name -> index
    asyncs = []
    by_type = {}
    n_sync = 0
    for idx, it in enumerate(instrs):
        op = it["op"]
        if op.endswith("-start") or op == "async-start":
            starts[idx] = it
        elif (op.endswith("-done") or op == "async-done") and starts:
            # match the nearest open start before this done
            sidx = max(starts)
            sit = starts.pop(sidx)
            # shadow = compute instrs strictly between sidx and idx
            shadow = [instrs[j] for j in range(sidx + 1, idx) if instrs[j]["compute"]]
            ct = sit["coll"] or "async"
            rec = dict(
                type=ct,
                bytes=sit["bytes"],
                shadow_ops=len(shadow),
                shadow_bytes=sum(s["bytes"] for s in shadow),
                window=idx - sidx - 1,
            )
            asyncs.append(rec)
            t = by_type.setdefault(
                ct, dict(count=0, async_=0, bytes=0, shadow_ops=0, with_shadow=0)
            )
            t["count"] += 1
            t["async_"] += 1
            t["bytes"] += sit["bytes"]
            t["shadow_ops"] += len(shadow)
            t["with_shadow"] += 1 if shadow else 0
    # sync collectives (no start/done)
    for it in instrs:
        ct = it["coll"]
        if ct and not (it["op"].endswith("-start") or it["op"].endswith("-done")):
            n_sync += 1
            t = by_type.setdefault(
                ct, dict(count=0, async_=0, bytes=0, shadow_ops=0, with_shadow=0)
            )
            t["count"] += 1
            t["bytes"] += it["bytes"]

    n_async = len(asyncs)
    n_overlapped = sum(1 for a in asyncs if a["shadow_ops"] > 0)
    return {
        "n_entry_instrs": len(instrs),
        "by_type": by_type,
        "n_async": n_async,
        "n_overlapped": n_overlapped,  # async collectives with compute in their shadow
        "n_sync_collectives": n_sync,
        "async_detail": sorted(asyncs, key=lambda a: -a["bytes"])[:40],
    }


if __name__ == "__main__":
    import json
    import sys

    with open(sys.argv[1]) as f:
        out = parse_hlo_overlap(f.read())
    print(json.dumps(out, indent=2))
