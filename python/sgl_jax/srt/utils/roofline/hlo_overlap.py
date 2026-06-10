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


def parse_hlo_overlap(text: str) -> dict:
    # Scan the WHOLE module (collectives can live in nested computations / async
    # wrappers, not just ENTRY) for an accurate inventory. Counts are #instructions
    # in the module (static), not dynamic execution counts.
    lines = text.splitlines()
    net_by = {}
    n_net_sync = n_net_async = 0
    n_sc = n_copy = n_pallas = 0
    # fusion ground truth: XLA's fused regions + whether the theory-candidate
    # elementwise/norm/rope ops were folded into fusions (fused) or left standalone.
    n_fusion = 0
    fusion_kind = {}
    # candidate elementwise/norm/rope categories -> how many of their ops landed
    # INSIDE a fusion region (= XLA fused them). Case-insensitive op_name match.
    cand = {k: 0 for k in ("rms_norm", "rotary", "silu", "softmax")}
    in_fused = False
    for ln in lines:
        # computation boundaries (headers start at column 0 and end with '{')
        if ln and ln[0] not in " \t":
            if ln.rstrip().endswith("{"):
                in_fused = "fused_computation" in ln.split("(")[0]
            elif ln.startswith("}"):
                in_fused = False
        m = _INSTR.match(ln)
        if not m:
            continue
        name, rhs = m.group(1), m.group(2)
        if re.search(r"(^|[^\w-])fusion\(", rhs):
            n_fusion += 1
            km = re.search(r"kind=(k\w+)", rhs)
            if km:
                fusion_kind[km.group(1)] = fusion_kind.get(km.group(1), 0) + 1
        if in_fused and 'op_name="' in rhs:
            low = rhs.lower()
            for k in cand:
                if k in low:
                    cand[k] += 1
                    break
        # A collective shows up either as an opcode (all-reduce(...)) or, when XLA
        # wraps it async, as an instruction NAMED for it (e.g.
        # %all-gather.14.cloned.1.call-start = ... async-start(...)). Detect both;
        # the name carries -start/-done/.call-start/.call-done for sync-vs-async.
        net = _net_opcode(rhs)
        ct = st = dn = None
        if net:
            ct, st, dn = net
        else:
            base = name.lstrip("%")
            for c in _NET:
                if base.startswith(c) or ".cloned" in base and c in base:
                    ct = c
                    st = base.endswith("-start") or base.endswith("call-start")
                    dn = base.endswith("-done") or base.endswith("call-done")
                    break
        if ct:
            rep = _REPL.search(rhs)
            d = net_by.setdefault(
                ct, dict(count=0, sync=0, async_=0, bytes=0, groups=rep.group(1) if rep else "")
            )
            if dn:
                continue  # the -done half is the same collective as its -start
            d["count"] += 1
            d["bytes"] += _first_shape_bytes(rhs)
            if st:
                d["async_"] += 1
                n_net_async += 1
            else:
                d["sync"] += 1
                n_net_sync += 1
            if rep and not d["groups"]:
                d["groups"] = rep.group(1)
        elif 'async_execution_thread="sparsecore"' in rhs and (
            "-start(" in rhs or "call-start" in rhs
        ):
            n_sc += 1
        elif re.search(r"(copy-start|slice-start)\(", rhs):
            n_copy += 1
        elif "custom-call(" in rhs and "tpu_custom_call" in rhs:
            n_pallas += 1

    sp_active = ("reduce-scatter" in net_by) or ("all-gather" in net_by)
    return {
        "n_module_lines": len(lines),
        "network": {
            "by_type": net_by,
            "n_sync_barrier": n_net_sync,
            "n_async": n_net_async,
            "sp_active": sp_active,  # reduce-scatter / all-gather present => SP took effect
        },
        "sparsecore_async": {"count": n_sc},  # MoE dispatch/combine on SparseCore
        "memory_prefetch_async": n_copy,  # HBM<->VMEM async copies (latency hiding)
        "pallas_kernels": n_pallas,  # tpu_custom_call (attention + fused MoE; a2a is in here)
        "fusion": {
            "n_fusions": n_fusion,
            "by_kind": fusion_kind,  # kLoop / kOutput(epilogue) / kInput / kCustom
            "candidates": cand,  # rms_norm/rotary/silu/softmax: #ops folded into fusions
        },
    }


if __name__ == "__main__":
    import json
    import sys

    with open(sys.argv[1]) as f:
        print(json.dumps(parse_hlo_overlap(f.read()), indent=2))
