"""Parse optimized, scheduled XLA HLO for the Overlap analysis.

From the HLO text of the compiled forward (``tools/dump_forward_hlo.py``), report
what the COMPILER actually did about overlap -- ground truth, not a theoretical
envelope. The key, non-obvious findings this surfaces for MiMo-class MoE models:

  * Network collectives that ARE XLA ops: the tensor-parallel ``all-reduce`` /
    ``reduce-scatter`` / ``all-gather`` -- with their ``replica_groups`` (which
    mesh axis) and whether XLA made them async (``-start``/``-done``) or left
    them SYNC (an exposed barrier).
  * The MoE all-to-all is usually NOT an XLA collective -- it is fused inside the
    MoE Pallas kernel (``tpu_custom_call``), so XLA-level overlap does not apply;
    its dispatch/combine run as SparseCore async ops, and whether they actually
    hide behind TensorCore compute is a kernel/device-trace question.
  * XLA issues many async HBM<->VMEM prefetch copies (latency hiding) -- distinct
    from network comm.

Pure-Python text parsing (no jax). Schedule order = textual order of the ENTRY
computation in a compiled (scheduled) HLO module.
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
_NET = (
    "all-reduce",
    "all-gather",
    "reduce-scatter",
    "all-to-all",
    "collective-permute",
    "ragged-all-to-all",
)
_SHAPE = re.compile(r"\b(" + "|".join(_DT) + r")\[([\d,]*)\]")
_INSTR = re.compile(r"^\s*(%[\w.\-]+)\s*=\s*(.*)$")
_REPL = re.compile(r"replica_groups=(?:\[[\d,]*\])?(\[[\d,]*\]|\{[^}]*\})")


def _first_shape_bytes(rhs: str) -> int:
    for m in _SHAPE.finditer(rhs):
        n = 1
        for d in m.group(2).split(","):
            if d.strip():
                n *= int(d)
        return n * _DT[m.group(1)]
    return 0


def _net_opcode(rhs: str):
    """Return (collective_type, is_start, is_done) if the instr is a network
    collective, else None."""
    for c in _NET:
        for suf, st, dn in (
            (c + "-start", True, False),
            (c + "-done", False, True),
            (c, False, False),
        ):
            if re.search(r"(^|[^\w-])" + re.escape(suf) + r"\(", rhs):
                return c, st, dn
    return None


def _is_compute(rhs: str) -> bool:
    for kw in ("fusion(", "dot(", "convolution("):
        if kw in rhs:
            return True
    return "custom-call(" in rhs and "tpu_custom_call" in rhs


def _entry_body(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.lstrip().startswith("ENTRY ") or " ENTRY " in ln),
        None,
    )
    if start is None:
        return lines
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
    # classify entry instrs in schedule order
    instrs = []
    for ln in body:
        m = _INSTR.match(ln)
        if not m:
            continue
        rhs = m.group(2)
        net = _net_opcode(rhs)
        kind = None
        if net:
            kind = "net"
        elif 'async_execution_thread="sparsecore"' in rhs or "sparsecore" in rhs:
            kind = "sc" if ("-start(" in rhs or "-done(" in rhs or "async-" in rhs) else None
        elif re.search(r"(copy-start|copy-done|slice-start|slice-done)\(", rhs):
            kind = "copy"
        instrs.append(
            dict(
                rhs=rhs, net=net, kind=kind, bytes=_first_shape_bytes(rhs), compute=_is_compute(rhs)
            )
        )

    # network collectives
    net_by = {}
    n_net_sync = n_net_async = 0
    for idx, it in enumerate(instrs):
        if not it["net"]:
            continue
        c, st, dn = it["net"]
        rep = _REPL.search(it["rhs"])
        d = net_by.setdefault(
            c, dict(count=0, sync=0, async_=0, bytes=0, groups=rep.group(1) if rep else "")
        )
        if st:
            d["async_"] += 1
            d["count"] += 1
            d["bytes"] += it["bytes"]
            n_net_async += 1
        elif dn:
            pass  # counted at start
        else:
            d["sync"] += 1
            d["count"] += 1
            d["bytes"] += it["bytes"]
            n_net_sync += 1
        if rep and not d["groups"]:
            d["groups"] = rep.group(1)

    # sparsecore async (MoE dispatch/combine) + their TC shadow
    sc_starts = []
    sc = dict(count=0, with_shadow=0, shadow_ops=0)
    for idx, it in enumerate(instrs):
        if it["kind"] != "sc":
            continue
        if "-start(" in it["rhs"] or "async-start(" in it["rhs"] or "call-start" in it["rhs"]:
            sc_starts.append(idx)
        elif (
            "-done(" in it["rhs"] or "async-done(" in it["rhs"] or "call-done" in it["rhs"]
        ) and sc_starts:
            sidx = sc_starts.pop()
            shadow = [j for j in range(sidx + 1, idx) if instrs[j]["compute"]]
            sc["count"] += 1
            sc["shadow_ops"] += len(shadow)
            sc["with_shadow"] += 1 if shadow else 0

    n_copy = sum(1 for it in instrs if it["kind"] == "copy" and "-start(" in it["rhs"])
    n_pallas = sum(1 for it in instrs if it["compute"] and "tpu_custom_call" in it["rhs"])

    return {
        "n_entry_instrs": len(instrs),
        "network": {
            "by_type": net_by,
            "n_sync_barrier": n_net_sync,
            "n_async": n_net_async,
        },
        "sparsecore_async": sc,  # MoE dispatch/combine on SparseCore
        "memory_prefetch_async": n_copy,  # HBM<->VMEM async copies (latency hiding)
        "pallas_kernels": n_pallas,  # tpu_custom_call (attention + fused MoE; a2a is in here)
    }


if __name__ == "__main__":
    import json
    import sys

    with open(sys.argv[1]) as f:
        print(json.dumps(parse_hlo_overlap(f.read()), indent=2))
