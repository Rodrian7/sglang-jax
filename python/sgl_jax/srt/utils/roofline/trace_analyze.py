"""Generic trace -> roofline analyzer.

Turns the per-device jaxpr captured by ``standalone_trace`` (or the on-device
dump hook) into a ``ModelRoofline`` for ANY registered model, with **zero
per-model code**:

  * the op inventory + per-type LAYER COUNTS are read from the trace (e.g. 70
    qkv GEMMs, 69 MoE-expert kernels, 10 full + 60 SWA attention kernels) -- no
    hand-written hybrid-layer pattern,
  * each op is attributed to its REAL ``models/*.py`` call site via the captured
    ``source_stack`` (the innermost frame is the op kind -- linear.py / gate.py /
    logits_processor.py -- and the caller frame is the role: qkv vs o_proj vs
    down_proj, all of which share ``LinearBase``),
  * COSTS reuse the validated roofline primitives (``ops``/``references``/
    ``quant``/``parallelism``), so the result matches the hand-written
    ``descriptors`` while being trace-driven, and
  * Pallas kernels (attention, fused MoE) are priced by a kernel-name registry
    (RPAd/RPAm -> attention, fused-moe-v2 -> experts), dims from per-device avals
    + config, the real context length supplied as a workload input.

Quant + real context length are applied here analytically (the trace abstracts
weights to bf16 and uses a tiny KV pool); the trace supplies structure, not
magnitudes.
"""

from __future__ import annotations

import re

from . import ops, parallelism, references
from .quant import quant_specs_from_config
from .report import HardwarePeaks, ModelRoofline, OpRoofline


def _cfg(config, *names, default=None):
    for n in names:
        if config.get(n) is not None:
            return config[n]
    return default


_AVAL = re.compile(r"([a-z0-9_]+)\[([0-9,\s]*)\]")


def _parse(aval_str):
    """'bfloat16[512, 24576]' -> ('bfloat16', [512, 24576])."""
    m = _AVAL.match(aval_str or "")
    if not m:
        return None, []
    dims = [int(x) for x in m.group(2).split(",") if x.strip()]
    return m.group(1), dims


def _last_model_frame(stack):
    """The caller frame inside models/* (the role-bearing site), else the
    innermost user frame."""
    for fr in stack:
        if "/models/" in fr or fr.startswith("srt/models/"):
            return fr
    return stack[0] if stack else ""


# ---------------------------------------------------------------------------
# GEMM role classification (innermost-file kind + shape + caller frame)
# ---------------------------------------------------------------------------
def _model_frames(stack):
    return [f for f in (stack or []) if "/models/" in f or f.startswith("srt/models/")]


def _gemm_role(e, dims, attn_only=frozenset(), mlp_only=frozenset()):
    """Classify a top-level dot_general into (role, category, quant_role, m, k, n).

    Innermost source file gives the kind (gate.py->router, logits_processor->
    lm_head); shape vs config dims gives qkv / gate_up / lm_head; for the
    ambiguous n==H projections (o_proj and dense down share k when attn_out ==
    dense_inter) the model-file call frame disambiguates: o_proj's stack carries
    the attention-block frame, down's the MLP-block frame (both derived from the
    qkv / gate_up stacks, no hard-coded line numbers)."""
    stack = e.get("source_stack", []) or []
    inner = stack[0] if stack else (e.get("source", "") or "")
    _, od = _parse(e["out"][0]) if e["out"] else (None, [])
    ins = [_parse(s) for s in e.get("ins", [])]
    if len(od) < 2 or not ins or len(ins[0][1]) < 2:
        return None
    m, n = od[0], od[1]
    k = ins[0][1][-1]
    H = dims["H"]
    if "gate.py" in inner:
        return ("router", "router", "router", m, k, n)
    if "logits_processor" in inner or n == dims["vocab"]:
        return ("lm_head", "lm_head", "lm_head", m, k, n)
    qkv_sizes = {dims["q_size"], dims["k_size"], dims["v_size"]}
    if k == H and n in qkv_sizes:
        return ("qkv", "linear", "qkv", m, k, n)
    if k == H and n == dims.get("dense_inter"):
        return ("gate_up", "linear", "mlp", m, k, n)
    if n == H:
        if k == dims.get("attn_out") and k != dims.get("dense_inter"):
            return ("o_proj", "o_proj", "o_proj", m, k, n)
        if k == dims.get("dense_inter") and k != dims.get("attn_out"):
            return ("down", "linear", "mlp", m, k, n)
        # ambiguous (attn_out == dense_inter): split by which block frame the
        # call stack carries.
        frames = set(_model_frames(stack))
        if frames & mlp_only and not (frames & attn_only):
            return ("down", "linear", "mlp", m, k, n)
        return ("o_proj", "o_proj", "o_proj", m, k, n)
    return ("linear", "linear", "mlp", m, k, n)


# ---------------------------------------------------------------------------
# Pallas kernel pricing registry
# ---------------------------------------------------------------------------
def _kernel_kind(name):
    n = (name or "").lower()
    if n.startswith("rpa") or "ragged_paged" in n:
        return "attention"
    if "moe" in n or n.startswith("gmm"):
        return "moe"
    return "other"


def analyze_trace(
    records: dict,
    config: dict,
    layout: parallelism.ParallelLayout,
    peaks: HardwarePeaks,
    *,
    phase: str,
    seq_len: int,
    arch: str = "traced",
) -> ModelRoofline:
    """records = forward_jaxpr_dump.extract_jaxpr_records output."""
    H = _cfg(config, "hidden_size")
    nh = _cfg(config, "num_attention_heads")
    nkv = _cfg(config, "num_key_value_heads")
    hd = _cfg(config, "head_dim")
    vhd = _cfg(config, "v_head_dim", default=hd)
    VOCAB = _cfg(config, "vocab_size")
    NEXP = _cfg(config, "n_routed_experts", "num_experts", default=8)
    TOPK = _cfg(config, "num_experts_per_tok", default=2)
    MOEF = _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size"))
    DENSE_F = _cfg(config, "intermediate_size")
    qs = quant_specs_from_config(config)
    t, dp, ep, devices = layout.t, layout.dp, layout.ep, layout.devices

    dims = dict(
        H=H,
        vocab=VOCAB,
        q_size=nh * hd,
        k_size=nkv * hd,
        v_size=nkv * vhd,
        attn_out=nh * vhd,
        dense_inter=DENSE_F,
    )

    rows: list[OpRoofline] = []
    top = records["top_eqns"]

    # ---- global token count: from the qkv GEMM's m (== global extend/decode tokens)
    gemms = [e for e in top if e["prim"] == "dot_general"]
    global_m = 0
    for e in gemms:
        _, od = _parse(e["out"][0]) if e["out"] else (None, [])
        if len(od) == 2 and od[1] in (dims["q_size"], dims["q_size"] + dims["k_size"]):
            global_m = od[0]
            break
    if not global_m and gemms:
        global_m = _parse(gemms[0]["out"][0])[1][0]
    tokens_pd = max(1, global_m // dp)  # per-DP-group tokens (descriptor basis)

    def add(rr, label, category, *, count, shard=1, peak_kind="bf16", source="", ici=0):
        op = ops.to_op(rr, label, category, peak_kind=peak_kind, source=source)
        op.count = int(count)
        f = count / shard
        op.flops = int(op.flops * f)
        op.hbm_bytes = int(op.hbm_bytes * f)
        op.ici_bytes = int(op.ici_bytes * f) + int(ici * count)
        rows.append(op)

    # ---- GEMMs grouped by (role, shape) -> one row per role with summed count
    from collections import defaultdict

    # pre-pass: block-frame sets to disambiguate o_proj (attention block) from
    # dense down (MLP block) when attn_out == dense_inter. Derived from the
    # unambiguous qkv (attention) and gate_up (MLP) call stacks -- no hard-coded
    # lines, so it generalises to any model.
    attn_frames, mlp_frames = set(), set()
    for e in gemms:
        _, od = _parse(e["out"][0]) if e["out"] else (None, [])
        ins = [_parse(s) for s in e.get("ins", [])]
        if len(od) < 2 or not ins or len(ins[0][1]) < 2:
            continue
        n, k = od[1], ins[0][1][-1]
        fr = set(_model_frames(e.get("source_stack", [])))
        if k == H and n in {dims["q_size"], dims["k_size"], dims["v_size"]}:
            attn_frames |= fr
        elif k == H and n == dims["dense_inter"]:
            mlp_frames |= fr
    attn_only = attn_frames - mlp_frames
    mlp_only = mlp_frames - attn_frames

    gemm_groups = defaultdict(lambda: {"count": 0, "src": "", "m": 0, "k": 0, "n": 0})
    router_count = 0
    router_src = "gate.py"
    for e in gemms:
        r = _gemm_role(e, dims, attn_only, mlp_only)
        if r is None:
            continue
        role, category, qrole, m, k, n = r
        if role == "router":
            router_count += 1
            router_src = _last_model_frame(e.get("source_stack", [])) or e.get("source", "")
            continue
        g = gemm_groups[(role, k, n)]
        g["count"] += 1
        g["m"], g["k"], g["n"] = m, k, n
        g["category"], g["qrole"] = category, qrole
        g["src"] = _last_model_frame(e.get("source_stack", [])) or e.get("source", "")

    row_parallel_roles = {"o_proj", "down"}
    for (role, k, n), g in sorted(gemm_groups.items()):
        qspec = qs.get(g["qrole"], qs.get("mlp"))
        rr = ops.gemm(tokens_pd, k, n, qspec)
        ici = 0
        if role in row_parallel_roles:
            ici = parallelism.row_parallel_reduce_bytes(tokens_pd, H, layout)
        # lm_head + projections are sharded over the tensor axis t
        add(
            rr,
            f"{role}[{qspec.tag()}]x{g['count']}",
            g["category"],
            count=g["count"],
            shard=t,
            peak_kind=qspec.peak_kind(),
            source=g["src"],
            ici=ici,
        )

    # router gate (matmul + softmax + top_k); replicated (not tensor-sharded)
    if router_count:
        add(
            ops.router_gate(tokens_pd, H, NEXP, TOPK),
            f"routerx{router_count}",
            "router",
            count=router_count,
            shard=1,
            source=router_src,
        )

    # ---- Pallas: attention (full/SWA) + MoE experts, by kernel-name registry
    pallas = records["pallas"]
    # dedupe RPAd vs RPAm variants of the SAME layers: count distinct layers as
    # the MAX single-variant occurrence within each (is_swa) group.
    attn_full = attn_swa = 0
    moe_kernels = []
    attn_src = "ragged_paged_attention_v3"
    moe_src = "fused_ep_moe_v2"
    from collections import Counter

    variant_counts = Counter()
    for p in pallas:
        kind = _kernel_kind(p.get("kernel_name"))
        kn = p.get("kernel_name", "")
        if kind == "attention":
            is_swa = "-sw_" in kn or "sw_" in kn
            variant_counts[("swa" if is_swa else "full", kn)] += 1
            attn_src = p.get("ctx", attn_src)
        elif kind == "moe":
            moe_kernels.append(p)
            moe_src = p.get("ctx", moe_src)
    # distinct layers per group = max single-variant count
    full_variants = [c for (grp, _), c in variant_counts.items() if grp == "full"]
    swa_variants = [c for (grp, _), c in variant_counts.items() if grp == "swa"]
    attn_full = max(full_variants) if full_variants else 0
    attn_swa = max(swa_variants) if swa_variants else 0

    def attn_dims(swa):
        if swa:
            return dict(
                nh=_cfg(config, "swa_num_attention_heads", default=nh),
                nkv=_cfg(config, "swa_num_key_value_heads", default=nkv),
                hd=_cfg(config, "swa_head_dim", default=hd),
                vhd=_cfg(config, "swa_v_head_dim", default=vhd),
                window=_cfg(config, "sliding_window_size", default=4096),
            )
        return dict(nh=nh, nkv=nkv, hd=hd, vhd=vhd, window=0)

    for swa, count in (("full", attn_full), ("swa", attn_swa)):
        is_swa = swa == "swa"
        if not count:
            continue
        d = attn_dims(is_swa)
        eff_ctx = min(seq_len, d["window"]) if d["window"] else seq_len
        if phase == "decode":
            inter = tokens_pd * eff_ctx
        else:
            inter = int(tokens_pd * (min(tokens_pd, eff_ctx) / 2))
            eff_ctx = min(tokens_pd, eff_ctx) if not d["window"] else eff_ctx
        rr = references.attention_cost(
            num_q_heads=max(1, d["nh"] // t),
            num_kv_heads=parallelism.kv_heads_per_device(d["nkv"], t),
            head_dim=d["hd"],
            v_head_dim=d["vhd"],
            q_tokens=tokens_pd,
            kv_tokens=eff_ctx,
            total_interactions=int(inter),
        )
        rows.append(
            OpRoofline(
                label=f"attention[{'SWA' if is_swa else 'full'},PALLAS]x{count}",
                category="attention",
                source=attn_src,
                count=count,
                flops=int(rr["flops"]) * count,
                hbm_bytes=int(rr["hbm_bytes"]) * count,
                ici_bytes=0,
                peak_kind="bf16",
            )
        )

    # ---- MoE experts: local_experts from aval, tokens_per_device from layout
    if moe_kernels:
        n_moe = len(moe_kernels)
        local_experts = NEXP / ep
        for p in moe_kernels[:1]:  # read #local experts from a weight aval [E, d, f]
            for a in p.get("in_avals", []):
                if not a or len(a["shape"]) != 3:
                    continue
                e0, d1, d2 = a["shape"]
                if e0 <= NEXP and {d1, d2} <= {H, MOEF} and d1 >= MOEF:
                    local_experts = e0
                    break
        moe_tokens = tokens_pd * dp  # full-mesh EP pools all DP groups
        tokens_per_dev = max(1, moe_tokens * TOPK // ep)
        remote = (ep - 1) / ep if ep > 1 else 0.0
        a2a = int(2 * (moe_tokens * TOPK / ep) * H * 2 * remote)
        rr = references.moe_experts_cost(
            tokens_per_device=tokens_per_dev,
            local_experts=local_experts,
            d=H,
            f=MOEF,
            qspec=qs["experts"],
        )
        op = OpRoofline(
            label=f"experts[PALLAS,{qs['experts'].tag()}]x{n_moe}",
            category="moe",
            source=moe_src,
            count=n_moe,
            flops=int(rr["flops"]) * n_moe,
            hbm_bytes=int(rr["hbm_bytes"]) * n_moe,
            ici_bytes=(a2a + parallelism.row_parallel_reduce_bytes(tokens_pd, H, layout)) * n_moe,
            peak_kind=qs["experts"].peak_kind(),
        )
        rows.append(op)

    # ---- norm / rope / elementwise residuals: counts from the trace
    # norms: count rsqrt occurrences = number of RMSNorms
    n_norm = sum(1 for e in top if e["prim"] == "rsqrt" and "layernorm" in (e.get("source") or ""))
    if n_norm:
        rows.append(
            ops.to_op(
                ops.rms_norm(tokens_pd, H), f"rms_normx{n_norm}", "norm", source="layernorm.py"
            )
        )
        rows[-1].count = n_norm
        rows[-1].flops *= n_norm
        rows[-1].hbm_bytes *= n_norm
    n_rope = sum(1 for e in top if e["prim"] == "cos" and "embeddings" in (e.get("source") or ""))
    if n_rope:
        # rope acts on head-sharded q/k -> per-device work is /t
        rr_rope = ops.rope(tokens_pd, dims["q_size"], dims["k_size"])
        op = ops.to_op(rr_rope, f"ropex{n_rope}", "rope", source="embeddings.py")
        op.count = n_rope
        op.flops = int(op.flops * n_rope / t)
        op.hbm_bytes = int(op.hbm_bytes * n_rope / t)
        rows.append(op)

    # residual adds (2 per decoder layer) + dense SiLU -> "other"
    n_layers = attn_full + attn_swa
    n_dense = max(0, n_layers - len(moe_kernels))
    if n_layers:
        op = ops.to_op(
            ops.elementwise(tokens_pd, H), f"residualx{2*n_layers}", "other", source="residual add"
        )
        op.count = 2 * n_layers
        op.flops *= 2 * n_layers
        op.hbm_bytes *= 2 * n_layers
        rows.append(op)
    if n_dense:
        op = ops.to_op(
            ops.elementwise(tokens_pd, DENSE_F), f"silux{n_dense}", "other", source="MLP silu"
        )
        op.count = n_dense
        op.flops *= n_dense
        op.hbm_bytes *= n_dense
        rows.append(op)

    meta = dict(
        arch=arch,
        phase=phase,
        global_tokens=global_m,
        tokens_per_dp=tokens_pd,
        seq_len=seq_len,
        tp_total=layout.tp_total,
        dp=dp,
        attention_tp=t,
        ep_effective=ep,
        devices=devices,
        n_attn_full=attn_full,
        n_attn_swa=attn_swa,
        n_moe=len(moe_kernels),
        source="trace",
    )
    return ModelRoofline(arch=arch, phase=phase, peaks=peaks, rows=rows, meta=meta)
