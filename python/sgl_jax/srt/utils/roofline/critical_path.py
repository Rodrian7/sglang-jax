"""Critical-path (CPM) analysis on the layer dataflow graph.

Edge weight = op ideal_ms (= max of its compute/HBM/ICI time). The critical
path is the longest-weighted dependency chain; for a (mostly serial) transformer
it runs through the layer's main chain. We also report the perfect-overlap lower
bound (per-resource totals) so the gap = the overlap/pipelining headroom that
fusion/scheduling could recover.

This mirrors how xprof identifies the bottleneck chain, but on a theoretical DAG
(a CPM lower bound) rather than a measured trace.
"""

from __future__ import annotations

from .graph import DataflowGraph
from .report import HardwarePeaks


def analyze(graph: DataflowGraph, peaks: HardwarePeaks) -> dict:
    ms = {op.id: op.roofline().ideal_ms(peaks) for op in graph.ops}
    prod = {op.output: op for op in graph.ops}  # tensor -> producing op
    by_id = {op.id: op for op in graph.ops}

    # ops are appended in dependency order, so id order is a topo order
    finish: dict[int, float] = {}
    pred: dict[int, list[int]] = {}
    for op in graph.ops:
        preds = [prod[t].id for t in op.inputs if t in prod]
        pred[op.id] = preds
        finish[op.id] = max([finish[p] for p in preds], default=0.0) + ms[op.id]

    t_critical = max(finish.values(), default=0.0)

    # backtrack the critical path
    path = []
    cur = max(graph.ops, key=lambda o: finish[o.id]) if graph.ops else None
    while cur is not None:
        path.append(cur)
        preds = pred[cur.id]
        cur = (
            max((by_id[p] for p in preds), key=lambda o: finish[o.id], default=None)
            if preds
            else None
        )
    path.reverse()
    crit_ids = {o.id for o in path}

    # perfect-overlap lower bound: each resource's total time
    sum_compute = sum(o.roofline().compute_ms(peaks) for o in graph.ops)
    sum_hbm = sum(o.roofline().hbm_ms(peaks) for o in graph.ops)
    sum_ici = sum(o.roofline().ici_ms(peaks) for o in graph.ops)
    t_resource = max(sum_compute, sum_hbm, sum_ici)

    return {
        "t_critical_ms": t_critical,
        "t_resource_ms": t_resource,
        "gap_ms": max(0.0, t_critical - t_resource),
        "sum_compute_ms": sum_compute,
        "sum_hbm_ms": sum_hbm,
        "sum_ici_ms": sum_ici,
        "path": [(o.label, ms[o.id]) for o in path],
        "off_path": [
            (o.label, ms[o.id]) for o in graph.ops if o.id not in crit_ids and ms[o.id] > 0
        ],
    }
