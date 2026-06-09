"""Dump the real model forward as a jaxpr (op + source-line + Pallas cost) -- the
generic, per-model-code-free basis for the roofline tool.

Called from a debug hook in ModelRunner.run_model_wrapper (env
SGLJAX_DUMP_FORWARD_JAXPR=<path>): on the first forward it traces the real
``model(forward_batch, memory_pools, logits_metadata)`` to a jaxpr and writes a
JSON with, per equation: primitive, source line (real models/*.py via
source_info), output shape, and -- for pallas_call/custom_call -- the kernel's
declared ``cost_estimate`` (flops/bytes) if present, plus its input/output avals
so a ``ref``-based bytes/flops fallback can price kernels that do not declare one.

It walks the jaxpr RECURSIVELY: the load-bearing kernels (attention, fused MoE)
live inside ``shard_map`` / ``pjit`` sub-jaxprs, so a top-level-only walk misses
every Pallas call. Each nested eqn records the ``ctx`` (enclosing shard_map/pjit
source) it sits under, so an attention kernel is distinguishable from an experts
kernel without any per-model code. This is the input the generic
graph_from_jaxpr path consumes (no per-model descriptor needed).
"""

from __future__ import annotations

import json


def _aval_str(v):
    a = getattr(v, "aval", None)
    return f"{a.dtype}{list(a.shape)}" if a is not None else str(v)


def _aval_obj(v):
    a = getattr(v, "aval", None)
    if a is None:
        return None
    try:
        return {"dtype": str(a.dtype), "shape": list(a.shape)}
    except Exception:
        return None


def _subjaxprs(eqn):
    """Yield (label, jaxpr) for every sub-jaxpr referenced by an eqn's params.

    Covers shard_map (``jaxpr``), pjit/closed_call (``jaxpr``), pallas_call
    (``jaxpr`` -- the kernel body), scan/while (``jaxpr``/``cond_jaxpr``/
    ``body_jaxpr``), cond (``branches``), and any ``call_jaxpr``. Returns the
    inner ``Jaxpr`` (unwrapping ClosedJaxpr) so the caller can recurse uniformly.
    """
    import jax

    Jaxpr = jax.extend.core.Jaxpr
    ClosedJaxpr = jax.extend.core.ClosedJaxpr

    def _unwrap(x):
        if isinstance(x, ClosedJaxpr):
            return x.jaxpr
        if isinstance(x, Jaxpr):
            return x
        return None

    for k, val in eqn.params.items():
        if k == "cost_estimate":
            continue
        j = _unwrap(val)
        if j is not None:
            yield k, j
        elif isinstance(val, (tuple, list)):
            for i, item in enumerate(val):
                j = _unwrap(item)
                if j is not None:
                    yield f"{k}[{i}]", j


def dump_forward_jaxpr(make_jaxpr_fn, args, out_path: str) -> str:
    """make_jaxpr_fn(*args) -> a ClosedJaxpr/Jaxpr; extract + write JSON.

    Recursively walks sub-jaxprs so nested Pallas/custom-call kernels (attention,
    fused MoE) are captured with their cost_estimate and avals.
    """
    import jax
    from jax._src import source_info_util as si

    jaxpr = jax.make_jaxpr(make_jaxpr_fn)(*args)
    jp = getattr(jaxpr, "jaxpr", jaxpr)

    top_eqns: list[dict] = []  # depth-0 eqns (structure + comm at the outer level)
    pallas: list[dict] = []  # every pallas_call/custom_call at ANY depth
    from collections import Counter

    by_prim_all: Counter = Counter()

    def walk(eqns, depth, ctx):
        for e in eqns:
            name = e.primitive.name
            by_prim_all[name] += 1
            src = si.summarize(e.source_info)
            rec = {
                "prim": name,
                "source": src,
                "depth": depth,
                "out": [_aval_str(v) for v in e.outvars][:2],
                "ins": [_aval_str(v) for v in e.invars if hasattr(v, "aval")][:6],
            }
            if depth == 0:
                top_eqns.append(rec)
            if name in ("pallas_call", "custom_call"):
                prec = dict(rec)
                prec["ctx"] = ctx  # enclosing shard_map/pjit source (kernel role)
                ce = e.params.get("cost_estimate")
                if ce is not None:
                    prec["cost_estimate"] = {
                        "flops": int(getattr(ce, "flops", 0) or 0),
                        "bytes_accessed": int(getattr(ce, "bytes_accessed", 0) or 0),
                        "transcendentals": int(getattr(ce, "transcendentals", 0) or 0),
                    }
                prec["in_avals"] = [_aval_obj(v) for v in e.invars if hasattr(v, "aval")]
                prec["out_avals"] = [_aval_obj(v) for v in e.outvars]
                for k in ("name", "kernel_name"):
                    if k in e.params:
                        prec["kernel_name"] = str(e.params[k])
                        break
                pallas.append(prec)
            # recurse into any sub-jaxpr (shard_map / pjit / scan / cond / pallas body)
            for label, subj in _subjaxprs(e):
                sub_ctx = ctx
                if name in ("shard_map", "pjit", "closed_call", "custom_jvp_call"):
                    sub_ctx = f"{name}@{src}"  # this eqn becomes the context for its body
                walk(subj.eqns, depth + 1, sub_ctx)

    walk(jp.eqns, 0, "<root>")

    out = {
        "num_eqns_top": len(top_eqns),
        "num_eqns_all": sum(by_prim_all.values()),
        "by_primitive_all": by_prim_all.most_common(),
        "pallas": pallas,  # nested kernels with cost_estimate / avals / ctx
        "top_eqns": top_eqns,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out_path
