"""Fusion-opportunity detection on the layer dataflow graph.

A fusion candidate is an intermediate tensor with a single producer and single
consumer where the pair is structurally fusable. Fusing removes that tensor's
HBM round-trip (~2x its bytes: producer write + consumer read) and shortens the
critical path. Candidates are ranked by HBM bytes saved.

NOTE: this is a *theoretical* tool -- it lists structural opportunities only. It
does NOT (and cannot) know whether XLA already fuses a given pair; verify with a
profile/HLO before assuming a candidate is real work or already free.
"""

from __future__ import annotations

from .graph import DataflowGraph
from .report import HardwarePeaks

_EW = {"elementwise", "norm"}


def _reason(p, c) -> str:
    """Structural fusability of producer p -> consumer c ('' if not fusable)."""
    pf, cf = p.fusable, c.fusable
    if pf in _EW and cf in _EW:
        return "elementwise chain"
    if pf == "matmul" and cf in _EW:
        return "matmul epilogue"
    if pf in _EW and cf == "matmul":
        return "matmul prologue"
    if pf in _EW and cf == "pallas":
        return "fold into kernel prologue"
    if pf == "pallas" and cf in _EW:
        return "fold into kernel epilogue"
    return ""


def candidates(graph: DataflowGraph, peaks: HardwarePeaks) -> list[dict]:
    prod = {op.output: op for op in graph.ops}
    out: list[dict] = []
    for tid, nbytes in graph.tensors.items():
        p = prod.get(tid)
        cons = graph.consumers(tid)
        if p is None or len(cons) != 1:
            continue  # need single producer + single consumer
        reason = _reason(p, cons[0])
        if not reason:
            continue
        saved = 2 * nbytes  # remove producer write + consumer read of the intermediate
        out.append(
            {
                "producer": p.label,
                "consumer": cons[0].label,
                "tensor": tid,
                "reason": reason,
                "saved_hbm_bytes": saved,
                "saved_ms": saved / peaks.hbm_bytes_per_s() * 1e3,
            }
        )
    return sorted(out, key=lambda r: -r["saved_hbm_bytes"])
