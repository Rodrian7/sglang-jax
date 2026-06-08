"""Structure view (View A) via ``jax._src.jaxpr_util``.

Traces a model's representative reference forward into a jaxpr and reports the
op histogram and per-source-line attribution (counts), plus an optional pprof
profile. This is the "structure" half of the fused report; the cost half lives
in ``descriptors``/``report``.
"""

from __future__ import annotations

import jax
from jax._src import jaxpr_util as ju

from . import critical_path, fusion
from . import graph as G
from .report import HardwarePeaks


def structure_view(reference) -> dict | None:
    """``reference`` = (fn, abstract_args) from ``descriptors.reference_forward``.

    Returns dicts of {primitive: count} and {source: count}, the equation total,
    and the jaxpr (for pprof). None if no reference is available.
    """
    if reference is None:
        return None
    fn, args = reference
    jaxpr = jax.make_jaxpr(fn)(*args).jaxpr
    return {
        "num_eqns": len(jaxpr.eqns),
        "by_primitive": dict(ju.primitives(jaxpr)),
        "by_source": dict(ju.primitives_by_source(jaxpr)),
        "_jaxpr": jaxpr,
    }


def _hist(d: dict, top: int) -> list[tuple[str, int]]:
    return sorted(d.items(), key=lambda kv: -kv[1])[:top]


def render_structure(view: dict | None, *, top: int = 20) -> str:
    if view is None:
        return "===== View A: structure (jaxpr_util) =====\n(no reference forward registered for this arch)"
    lines = [
        "===== View A: structure (jaxpr_util, one representative layer) =====",
        f"jaxpr has {view['num_eqns']} equations",
        "-- by primitive --",
    ]
    for name, n in _hist(view["by_primitive"], top):
        lines.append(f"  {n:>4} {name}")
    lines.append("-- by source (which code yields the ops) --")
    for src, n in _hist(view["by_source"], top):
        lines.append(f"  {n:>4} {src}")
    return "\n".join(lines)


def write_pprof(view: dict | None, path: str) -> int:
    """Write a pprof equation profile; returns bytes written (0 if unavailable)."""
    if view is None:
        return 0
    data = ju.pprof_equation_profile(view["_jaxpr"])
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def graph_analysis(config: dict, phase: str, par: dict, peaks: HardwarePeaks) -> dict:
    """Whole-model critical-path (CPM) + fusion summary, built from per-layer-type
    dataflow graphs scaled by the hybrid layer pattern."""
    from collections import Counter

    def _cfg(*names, default=None):
        for n in names:
            if config.get(n) is not None:
                return config[n]
        return default

    L = _cfg("num_hidden_layers")
    hlp = _cfg("hybrid_layer_pattern", default=[0] * L)
    mlf = _cfg("moe_layer_freq", default=[1] * L)
    combo = Counter(
        (bool(hlp[i]) if i < len(hlp) else False, bool(mlf[i]) if i < len(mlf) else True)
        for i in range(L)
    )

    t_crit = sc = sh = si = 0.0
    per_type = []
    fus: dict = {}
    sample_paths = {}
    for (swa, moe), cnt in combo.items():
        if cnt == 0:
            continue
        g = G.build_layer_graph(config, phase, par, swa=swa, moe=moe)
        cp = critical_path.analyze(g, peaks)
        t_crit += cnt * cp["t_critical_ms"]
        sc += cnt * cp["sum_compute_ms"]
        sh += cnt * cp["sum_hbm_ms"]
        si += cnt * cp["sum_ici_ms"]
        tag = ("SWA" if swa else "full") + "+" + ("MoE" if moe else "dense")
        per_type.append(
            {
                "type": tag,
                "count": cnt,
                "layer_t_critical_ms": cp["t_critical_ms"],
                "path": cp["path"],
            }
        )
        sample_paths[tag] = cp["path"]
        for f in fusion.candidates(g, peaks):
            key = (f["producer"], f["consumer"])
            e = fus.setdefault(key, {**f, "layers": 0, "total_saved_mb": 0.0})
            e["layers"] += cnt
            e["total_saved_mb"] += cnt * f["saved_hbm_bytes"] / 1e6
    t_resource = max(sc, sh, si)
    fusions = sorted(fus.values(), key=lambda r: -r["total_saved_mb"])
    return {
        "phase": phase,
        "t_critical_ms": t_crit,
        "t_resource_ms": t_resource,
        "gap_ms": max(0.0, t_crit - t_resource),
        "sum_compute_ms": sc,
        "sum_hbm_ms": sh,
        "sum_ici_ms": si,
        "per_type": sorted(per_type, key=lambda r: -r["layer_t_critical_ms"] * r["count"]),
        "fusions": fusions,
    }
