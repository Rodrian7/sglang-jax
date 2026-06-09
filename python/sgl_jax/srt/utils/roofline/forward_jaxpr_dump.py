"""Dump the real model forward as a jaxpr (op + source-line + Pallas cost) — the
generic, per-model-code-free basis for the roofline tool.

Called from a debug hook in ModelRunner.run_model_wrapper (env
SGLJAX_DUMP_FORWARD_JAXPR=<path>): on the first forward it traces the real
``model(forward_batch, memory_pools, logits_metadata)`` to a jaxpr and writes a
JSON with, per equation: primitive, source line (real models/*.py via
source_info), output shape, and -- for pallas_call/custom_call -- the kernel's
declared ``cost_estimate`` (flops/bytes) if present. This is the input the
generic graph_from_jaxpr path consumes (no per-model descriptor needed).
"""

from __future__ import annotations

import json


def dump_forward_jaxpr(make_jaxpr_fn, args, out_path: str) -> str:
    """make_jaxpr_fn(*args) -> a ClosedJaxpr/Jaxpr; extract + write JSON."""
    import jax
    from jax._src import source_info_util as si

    jaxpr = jax.make_jaxpr(make_jaxpr_fn)(*args)
    jp = getattr(jaxpr, "jaxpr", jaxpr)

    def aval_str(v):
        a = getattr(v, "aval", None)
        return f"{a.dtype}{list(a.shape)}" if a is not None else str(v)

    eqns = []
    for e in jp.eqns:
        name = e.primitive.name
        rec = {
            "prim": name,
            "source": si.summarize(e.source_info),
            "out": [aval_str(v) for v in e.outvars][:1],
            "ins": [aval_str(v) for v in e.invars if hasattr(v, "aval")][:4],
        }
        if name in ("pallas_call", "custom_call"):
            ce = e.params.get("cost_estimate")
            if ce is not None:
                rec["cost_estimate"] = {
                    "flops": int(getattr(ce, "flops", 0) or 0),
                    "bytes_accessed": int(getattr(ce, "bytes_accessed", 0) or 0),
                    "transcendentals": int(getattr(ce, "transcendentals", 0) or 0),
                }
            # kernel name if available (for ref-based fallback registry)
            for k in ("name", "kernel_name"):
                if k in e.params:
                    rec["kernel_name"] = str(e.params[k])
                    break
        eqns.append(rec)

    from collections import Counter

    by_prim = Counter(r["prim"] for r in eqns)
    out = {
        "num_eqns": len(eqns),
        "by_primitive": by_prim.most_common(),
        "pallas": [r for r in eqns if r["prim"] in ("pallas_call", "custom_call")],
        "eqns": eqns,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out_path
