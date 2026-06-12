"""Auto-derive a layer DataflowGraph from a traced jaxpr.

This is the automatic counterpart to the hand-written ``graph.build_layer_graph``:
instead of transcribing the transformer by hand, we trace the descriptor's
*reference forward* into a jaxpr and turn it into a ``DataflowGraph`` directly --
nodes = jaxpr vars (tensors), edges = equations (ops). The win over the
hand-written graph:

  * faithful by construction (it IS the traced program, not a transcription),
  * generalises to any model with a registered reference forward (no per-model
    graph code), and
  * real source attribution per op via ``source_info`` (no hand-typed strings
    that rot).

Cost per equation is a theoretical closed form from the eqn's avals/params
(``_eqn_flops``). Two HBM models are produced:

  * *unfused*: every op reads its inputs + writes its output from/to HBM
    (worst-case, the raw jaxpr has many tiny intermediates), and
  * *fused*: a light XLA-like fusion model where only *materialised* tensors
    (graph i/o, anchor in/outputs, or fan-out > 1) cost an HBM round-trip;
    elementwise/movement chains between matmul/Pallas anchors are fused away.

Pallas/``custom_call`` kernels are opaque to a static walk, so their cost is
supplied by a pluggable ``pallas_coster`` (the descriptor knows the real RPA /
GMM costs).  Limitation: multi-output eqns (e.g. ``top_k``) expose only their
first output in the CPM edge model; cheap ops, off the critical path.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax._src import source_info_util as _si

from .graph import DataflowGraph, GOp

try:  # jax >= 0.6 moved core IR types under jax.extend.core
    from jax.extend.core import Literal as _Literal
except ImportError:  # pragma: no cover
    from jax.core import Literal as _Literal

# ---- primitive taxonomy ---------------------------------------------------
# Anchors: real compute that materialises its output and reads materialised
# inputs (XLA does not fuse two of these together for free).
_ANCHORS = {"dot_general", "conv_general_dilated", "pallas_call", "custom_call"}
# Movement / layout: no FLOPs, fuse into a neighbour.
_MOVEMENT = {
    "reshape",
    "transpose",
    "broadcast_in_dim",
    "convert_element_type",
    "squeeze",
    "expand_dims",
    "slice",
    "concatenate",
    "pad",
    "rev",
    "copy",
    "copy_p",
    "dynamic_slice",
    "dynamic_update_slice",
    "gather",
    "stop_gradient",
    "split",
    "bitcast_convert_type",
    "real",
    "imag",
    "device_put",
}
# Reductions: FLOPs ~ input size, fusible.
_REDUCE = {
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_and",
    "reduce_or",
    "argmax",
    "argmin",
    "cumsum",
    "cumprod",
    "cummax",
}
# Everything else with FLOPs ~ output size is treated as fusible elementwise.
_CATEGORY = {
    "dot_general": "linear",
    "conv_general_dilated": "linear",
    "pallas_call": "pallas",
    "custom_call": "pallas",
}
for _p in _MOVEMENT:
    _CATEGORY[_p] = "movement"
for _p in _REDUCE:
    _CATEGORY[_p] = "reduce"


def _dtype_bytes(aval) -> int:
    try:
        return jnp.dtype(aval.dtype).itemsize
    except Exception:
        return 2


def _elems(aval) -> int:
    n = 1
    for d in getattr(aval, "shape", ()) or ():
        n *= int(d)
    return n


def _aval_bytes(aval) -> int:
    return _elems(aval) * _dtype_bytes(aval)


def _dot_flops(eqn) -> int:
    """2 * output_elems * contracting_elems for a dot_general."""
    (lhs_c, _rhs_c), _ = eqn.params["dimension_numbers"]
    lhs = eqn.invars[0].aval
    out = eqn.outvars[0].aval
    contract = 1
    for ax in lhs_c:
        contract *= int(lhs.shape[ax])
    return 2 * _elems(out) * contract


def _eqn_flops(eqn) -> int:
    name = eqn.primitive.name
    if name == "dot_general":
        return _dot_flops(eqn)
    if name in _MOVEMENT:
        return 0
    if name in _REDUCE:
        return sum(_elems(v.aval) for v in eqn.invars if hasattr(v, "aval"))
    if name in ("top_k", "sort"):
        return sum(_elems(v.aval) for v in eqn.invars if hasattr(v, "aval"))
    # elementwise (and unknown): ~ output element count
    return sum(_elems(v.aval) for v in eqn.outvars)


def _source(eqn) -> str:
    try:
        return _si.summarize(eqn.source_info)
    except Exception:
        return ""


@dataclass
class _T:
    """Tracking record for one tensor (jaxpr var)."""

    tid: str
    bytes: int
    producer: int | None = None  # op id
    consumers: int = 0
    is_input: bool = False
    is_output: bool = False


def build_graph_from_jaxpr(jaxpr, *, pallas_coster=None, label_hint="") -> DataflowGraph:
    """Trace -> DataflowGraph. ``pallas_coster(eqn, occurrence) -> dict`` returns
    {flops, hbm_bytes, ici_bytes, category, peak_kind, label} for pallas/custom
    eqns; if None they are marked opaque (output bytes only)."""
    g = DataflowGraph()
    tinfo: dict[int, _T] = {}  # var-id -> _T
    counter = [0]

    def tid(v):
        if isinstance(v, _Literal):
            return None
        k = id(v)
        if k not in tinfo:
            name = f"t{counter[0]}"
            counter[0] += 1
            tinfo[k] = _T(name, _aval_bytes(v.aval))
            g.tensors[name] = tinfo[k].bytes
        return tinfo[k].tid

    for v in jaxpr.invars + jaxpr.constvars:
        t = tid(v)
        if t is not None:
            tinfo[id(v)].is_input = True
            g.inputs.append(t)
    for v in jaxpr.outvars:
        t = tid(v)  # register if produced later
        if t is not None:
            tinfo[id(v)].is_output = True
            g.outputs.append(t)

    pallas_occ = 0
    for i, eqn in enumerate(jaxpr.eqns):
        name = eqn.primitive.name
        ins = [tid(v) for v in eqn.invars]
        ins = [t for t in ins if t is not None]
        outs = [tid(v) for v in eqn.outvars]
        outs = [t for t in outs if t is not None]
        out = outs[0] if outs else f"_void{i}"
        for t in ins:
            for k, rec in tinfo.items():
                if rec.tid == t:
                    rec.consumers += 1
                    break
        for v in eqn.outvars:
            if not isinstance(v, _Literal):
                tinfo[id(v)].producer = i

        cat = _CATEGORY.get(name, "elementwise")
        peak = "bf16"
        ici = 0
        flops = _eqn_flops(eqn)
        label = name
        if name in ("pallas_call", "custom_call"):
            if pallas_coster is not None:
                c = pallas_coster(eqn, pallas_occ) or {}
                flops = int(c.get("flops", 0))
                ici = int(c.get("ici_bytes", 0))
                cat = c.get("category", "pallas")
                peak = c.get("peak_kind", "bf16")
                label = c.get("label", name)
            pallas_occ += 1
        # unfused per-op HBM: read inputs + write outputs
        in_bytes = sum(g.tensors.get(t, 0) for t in ins)
        out_bytes = sum(g.tensors.get(t, 0) for t in outs)
        op = GOp(
            id=i,
            label=label,
            category=cat,
            inputs=ins,
            output=out,
            flops=int(flops),
            hbm_bytes=int(in_bytes + out_bytes),
            ici_bytes=int(ici),
            peak_kind=peak,
            fusable=("movement" if cat == "movement" else ("matmul" if cat == "linear" else "")),
            source=_source(eqn),
        )
        # pallas coster may override hbm explicitly
        if name in ("pallas_call", "custom_call") and pallas_coster is not None:
            c = pallas_coster(eqn, pallas_occ - 1) or {}
            if "hbm_bytes" in c:
                op.hbm_bytes = int(c["hbm_bytes"])
        g.ops.append(op)

    g.meta = {"source": "jaxpr", "n_eqns": len(jaxpr.eqns), "label": label_hint}
    g._tinfo = tinfo  # for the fusion pass
    return g


def fuse(graph: DataflowGraph) -> DataflowGraph:
    """Light XLA-like fusion: an op's output is *materialised* (costs an HBM
    round-trip) only if it is a graph output, consumed by an anchor op, or has
    fan-out > 1. Elementwise/movement intermediates feeding a single non-anchor
    consumer are fused away (their output is not written to HBM). FLOPs are
    unchanged. Returns a NEW graph with adjusted per-op ``hbm_bytes``."""
    by_out = {op.output: op for op in graph.ops}
    consumers: dict[str, list[GOp]] = {}
    for op in graph.ops:
        for t in op.inputs:
            consumers.setdefault(t, []).append(op)

    def is_anchor(op):
        return op.category in ("linear", "pallas", "moe", "attention")

    materialised: set[str] = set(graph.outputs) | set(graph.inputs)
    for t in graph.tensors:
        cons = consumers.get(t, [])
        prod = by_out.get(t)
        if len(cons) > 1:
            materialised.add(t)
        if any(is_anchor(c) for c in cons):
            materialised.add(t)
        if prod is not None and is_anchor(prod):
            materialised.add(t)

    new_ops = []
    for op in graph.ops:
        # Pallas/MoE/attention kernels read weights that are NOT jaxpr tensors
        # (the reference forward passes only activations to the placeholder); the
        # coster supplied the authoritative HBM, so keep it as-is.
        if op.category in ("pallas", "moe", "attention"):
            new_ops.append(op)
            continue
        # read only materialised inputs; write output only if materialised
        in_b = sum(graph.tensors.get(t, 0) for t in op.inputs if t in materialised)
        out_b = graph.tensors.get(op.output, 0) if op.output in materialised else 0
        import dataclasses

        new_ops.append(dataclasses.replace(op, hbm_bytes=int(in_b + out_b)))
    fg = DataflowGraph(
        ops=new_ops,
        tensors=dict(graph.tensors),
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        meta={**graph.meta, "fused": True, "n_materialised": len(materialised)},
    )
    return fg


# ==========================================================================
# Reference-forward driver: trace -> auto graph -> critical path, with a
# cross-check against the hand-written build_layer_graph.
# ==========================================================================
def _cfg(config, *names, default=None):
    for n in names:
        if config.get(n) is not None:
            return config[n]
    return default


def mimo_pallas_coster(config, phase, par):
    """Coster for the MiMo reference forward (occ 0 = attention, 1 = experts),
    reusing the same reference costs the descriptor uses. Unsharded (the traced
    reference is full-size); apply tp/ep externally if needed."""
    from . import references

    H = _cfg(config, "hidden_size")
    nh = _cfg(config, "num_attention_heads")
    nkv = _cfg(config, "num_key_value_heads")
    hd = _cfg(config, "head_dim")
    vhd = _cfg(config, "v_head_dim", default=hd)
    NEXP = _cfg(config, "n_routed_experts", "num_experts", default=8)
    TOPK = _cfg(config, "num_experts_per_tok", default=2)
    MOEF = _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size"))
    tokens = par["batch"] if phase == "decode" else par["chunk"]
    ctx = par["seq_len"] if phase == "decode" else par["chunk"]
    inter = int(tokens * (ctx if phase == "decode" else ctx / 2))

    def coster(eqn, occ):
        if occ == 0:
            c = references.attention_cost(
                num_q_heads=nh,
                num_kv_heads=nkv,
                head_dim=hd,
                v_head_dim=vhd,
                q_tokens=tokens,
                kv_tokens=ctx,
                total_interactions=inter,
            )
            return {
                "flops": c["flops"],
                "hbm_bytes": c["hbm_bytes"],
                "category": "attention",
                "label": "attention[PALLAS]",
            }
        c = references.moe_experts_cost(
            tokens_per_device=tokens * TOPK, local_experts=NEXP, d=H, f=MOEF
        )
        return {
            "flops": c["flops"],
            "hbm_bytes": c["hbm_bytes"],
            "category": "moe",
            "label": "experts[PALLAS]",
        }

    return coster


# coster factory per architecture (parallels descriptors.REFERENCES)
COSTERS = {
    "MiMoV2FlashForCausalLM": mimo_pallas_coster,
    "MiMoV2ProForCausalLM": mimo_pallas_coster,
    "MiMoV2ForCausalLM": mimo_pallas_coster,
}


def analyze_reference(arch, config, phase, par, peaks) -> dict | None:
    """Trace the registered reference forward into a jaxpr, build the auto graph,
    fuse, run critical-path, and cross-check totals against build_layer_graph.
    None if no reference/coster for ``arch``."""

    from . import critical_path, descriptors
    from . import graph as G

    ref = descriptors.reference_forward(arch, config, phase, par)
    coster_factory = COSTERS.get(arch)
    if ref is None or coster_factory is None:
        return None
    fn, args = ref
    jaxpr = jax.make_jaxpr(fn)(*args).jaxpr
    coster = coster_factory(config, phase, par)
    raw = build_graph_from_jaxpr(jaxpr, pallas_coster=coster, label_hint=f"{arch}/{phase}")
    fg = fuse(raw)
    cp = critical_path.analyze(fg, peaks)

    def _tot(g):
        return (
            sum(o.flops for o in g.ops),
            sum(o.hbm_bytes for o in g.ops),
            sum(o.roofline().compute_ms(peaks) for o in g.ops),
            sum(o.roofline().hbm_ms(peaks) for o in g.ops),
        )

    af, ah, acm, ahm = _tot(fg)
    # hand-written reference (unsharded full+MoE layer) for cross-validation
    hw = G.build_layer_graph(
        config,
        phase,
        {**par, "tp": 1, "dp": 1, "ep": 1, "devices": 1},
        swa=False,
        moe=True,
    )
    hf, hh, hcm, hhm = _tot(hw)
    return {
        "phase": phase,
        "n_ops": len(fg.ops),
        "n_tensors": len(fg.tensors),
        "n_eqns": raw.meta["n_eqns"],
        "flops": af,
        "hbm_bytes": ah,
        "unfused_hbm": sum(o.hbm_bytes for o in raw.ops),
        "t_critical_ms": cp["t_critical_ms"],
        "path": cp["path"],
        "top_ops": sorted(
            [(o.label, o.category, o.flops, o.hbm_bytes, o.source) for o in fg.ops],
            key=lambda r: -(r[2] + r[3]),
        )[:12],
        "xcheck": {
            "auto_flops": af,
            "hw_flops": hf,
            "auto_hbm": ah,
            "hw_hbm": hh,
            "flops_ratio": (af / hf) if hf else 0.0,
            "hbm_ratio": (ah / hh) if hh else 0.0,
        },
    }


def render_auto_graph(res: dict | None) -> str:
    if res is None:
        return (
            "===== View F: auto dataflow graph (jaxpr) =====\n(no reference/coster for this arch)"
        )
    x = res["xcheck"]
    lines = [
        "===== View F: auto-derived dataflow graph (traced from jaxpr) =====",
        f"phase={res['phase']}  {res['n_eqns']} jaxpr eqns -> {res['n_ops']} graph ops, "
        f"{res['n_tensors']} tensors (UNSHARDED, one full+MoE layer)",
        f"FLOPs={res['flops']/1e9:.3f} GFLOP   HBM(fused)={res['hbm_bytes']/1e6:.1f} MB   "
        f"HBM(unfused)={res['unfused_hbm']/1e6:.1f} MB   t_critical={res['t_critical_ms']:.4f} ms",
        f"cross-check vs hand-written build_layer_graph:  "
        f"FLOPs x{x['flops_ratio']:.4f}  HBM x{x['hbm_ratio']:.4f}  (1.0 = exact match)",
        "-- top ops (flops+hbm) with auto source attribution --",
    ]
    for label, cat, fl, hb, src in res["top_ops"]:
        s = src.split("/")[-1] if src else ""
        lines.append(f"  {fl/1e9:7.3f} GFLOP  {hb/1e6:8.1f} MB  {cat:10s} {label:18s} <- {s}")
    return "\n".join(lines)
