"""Per-architecture roofline descriptors.

A descriptor turns a model ``config`` (the checkpoint ``config.json`` as a dict),
a ``phase`` ('decode'|'prefill') and a ``parallelism`` dict into a
``ModelRoofline`` (per-device, whole-model) by composing ``ops`` primitives over
the model's layer structure, scaled by the hybrid layer pattern and parallel
degrees.

It also exposes a *traceable reference forward* (one representative layer) used
by ``interp`` for the jaxpr_util structure view.

v1 implements ``MiMoV2FlashForCausalLM``. The registry is keyed by architecture
name so other models can be added later.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import ops, parallelism, quant
from .report import HardwarePeaks, ModelRoofline, OpRoofline

DESCRIPTORS = {}  # arch_name -> build(config, phase, par) -> ModelRoofline
REFERENCES = {}  # arch_name -> reference_forward(config, phase, par) -> (fn, abstract_args)


def register(arch_name):
    def deco(fn):
        DESCRIPTORS[arch_name] = fn
        return fn

    return deco


def register_reference(arch_name):
    def deco(fn):
        REFERENCES[arch_name] = fn
        return fn

    return deco


def reference_forward(arch_name, config, phase, par):
    """Return (traceable_fn, abstract_args) for one representative decoder layer,
    used by ``interp`` for the jaxpr_util structure view. None if unavailable."""
    fn = REFERENCES.get(arch_name)
    return fn(config, phase, par) if fn else None


def build(arch_name, config, phase, par, peaks: HardwarePeaks | None = None) -> ModelRoofline:
    if arch_name not in DESCRIPTORS:
        raise ValueError(
            f"No roofline descriptor for architecture '{arch_name}'. "
            f"Available: {sorted(DESCRIPTORS)}"
        )
    return DESCRIPTORS[arch_name](config, phase, par, peaks or HardwarePeaks(), arch_name)


def _cfg(config, *names, default=None):
    for n in names:
        v = config.get(n)
        if v is not None:
            return v
    return default


def _op(rr, label, category, *, layers=1, shard=1, peak_kind="bf16", source=""):
    """RooflineResult -> OpRoofline. ``count`` reflects #ops (layers); flops/bytes
    are the per-device totals = rr * layers / shard."""
    op = ops.to_op(rr, label, category, peak_kind=peak_kind, source=source)
    op.count = int(layers)
    f = layers / shard
    op.flops = int(op.flops * f)
    op.hbm_bytes = int(op.hbm_bytes * f)
    op.ici_bytes = int(op.ici_bytes * f)
    return op


# ==========================================================================
# MiMo-V2 family (Flash / Pro) -- same forward structure, config-driven dims
# ==========================================================================
@register("MiMoV2FlashForCausalLM")
@register("MiMoV2ProForCausalLM")
@register("MiMoV2ForCausalLM")
def _mimo_v2_family(config, phase, par, peaks, arch_name="MiMoV2") -> ModelRoofline:
    H = _cfg(config, "hidden_size")
    L = _cfg(config, "num_hidden_layers")
    VOCAB = _cfg(config, "vocab_size")
    NEXP = _cfg(config, "n_routed_experts", "num_experts", default=8)
    TOPK = _cfg(config, "num_experts_per_tok", default=2)
    MOEF = _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size"))
    DENSE_F = _cfg(config, "intermediate_size")

    # full-attention dims
    full = dict(
        nh=_cfg(config, "num_attention_heads"),
        nkv=_cfg(config, "num_key_value_heads"),
        hd=_cfg(config, "head_dim"),
        vhd=_cfg(config, "v_head_dim", default=_cfg(config, "head_dim")),
        window=0,
    )
    # sliding-window dims
    swa = dict(
        nh=_cfg(config, "swa_num_attention_heads", default=full["nh"]),
        nkv=_cfg(config, "swa_num_key_value_heads", default=full["nkv"]),
        hd=_cfg(config, "swa_head_dim", default=full["hd"]),
        vhd=_cfg(config, "swa_v_head_dim", default=full["vhd"]),
        window=_cfg(config, "sliding_window_size", default=4096),
    )

    hlp = _cfg(config, "hybrid_layer_pattern", default=[0] * L)
    mlf = _cfg(config, "moe_layer_freq", default=[1] * L)

    # Resolve + validate the parallelism against the runtime 2D mesh
    # [data=dp, tensor=tp//dp]: tensor-parallel degree for attention/linears is
    # t = tp//dp (NOT tp), and the fused-MoE EP group is the full mesh = devices.
    lp, _warns = parallelism.resolve(
        config,
        par,
        moe_backend=par.get("moe_backend", "fused_v2"),
        enable_sp=par.get("enable_sp", False),
    )
    tp = lp.t  # tensor-parallel degree (= tp_size // dp_size)
    ep = lp.ep  # effective expert-parallel group (= full mesh = devices)
    qs = quant.quant_specs_from_config(config)  # role -> QuantSpec (from quantization_config)

    # phase -> token counts and attention interaction counts
    if phase == "decode":
        tokens = par["batch"]
        ctx_full = par["seq_len"]
        logits_tokens = par["batch"]
    else:  # prefill / extend
        tokens = par["chunk"]
        ctx_full = par["chunk"]  # causal within the chunk
        logits_tokens = par["batch"]  # only last token per seq emits logits

    rows: list[OpRoofline] = []

    def attn_block(d, count, tag):
        if count <= 0:
            return
        q_size, k_size, v_size = (
            d["nh"] * d["hd"],
            d["nkv"] * d["hd"],
            d["nkv"] * d["vhd"],
        )
        ao = d["nh"] * d["vhd"]
        eff_ctx = min(ctx_full, d["window"]) if d["window"] else ctx_full
        inter = tokens * (eff_ctx / 2 if phase != "decode" else eff_ctx)
        # qkv projection (heads sharded by tp), quant per config
        qkv_q = qs["qkv"]
        rr = ops.gemm(tokens, H, q_size + k_size + v_size, qkv_q)
        rows.append(
            _op(
                rr,
                f"{tag}.qkv_proj[{qkv_q.tag()}]",
                "linear",
                layers=count,
                shard=tp,
                peak_kind=qkv_q.peak_kind(),
                source="mimo_v2_flash.py:465 qkv",
            )
        )
        rows.append(
            _op(
                ops.rope(tokens, q_size, k_size),
                f"{tag}.rope",
                "rope",
                layers=count,
                shard=tp,
                source="embeddings.py rotary",
            )
        )
        # attention (per-device q heads = nh/t; kv heads replicated to 1/device
        # when t > num_kv_heads -- not divided below 1)
        rr = ops.attention(
            num_q_heads=max(1, d["nh"] // tp),
            num_kv_heads=parallelism.kv_heads_per_device(d["nkv"], tp),
            head_dim=d["hd"],
            v_head_dim=d["vhd"],
            q_tokens=tokens,
            total_interactions=int(inter),
        )
        rows.append(
            _op(
                rr,
                f"{tag}.attention[PALLAS]",
                "attention",
                layers=count,
                source="ragged_paged_attention_v3",
            )
        )
        # o_proj (row-parallel reduce); quant per config (often bf16/ignored).
        # Reduce is over the tensor axis t (all-reduce), or full-mesh reduce-
        # scatter + residual all-gather under sequence parallelism above the
        # scatter threshold.
        op_q = qs["o_proj"]
        rr = ops.gemm(tokens, ao, H, op_q)
        o = _op(
            rr,
            f"{tag}.o_proj[{op_q.tag()}]",
            "o_proj",
            layers=count,
            shard=tp,
            peak_kind=op_q.peak_kind(),
            source="mimo_v2_flash.py:515 o_proj",
        )
        o.ici_bytes += parallelism.row_parallel_reduce_bytes(tokens, H, lp) * count
        rows.append(o)
        # 2 norms + residual adds
        rows.append(
            _op(
                ops.rms_norm(tokens, H),
                f"{tag}.norms",
                "norm",
                layers=2 * count,
                source="layernorm.py RMSNorm",
            )
        )
        rows.append(
            _op(
                ops.elementwise(tokens, H),
                f"{tag}.residual",
                "other",
                layers=2 * count,
                source="mimo_v2_flash.py residual",
            )
        )

    def moe_block(count, tag):
        if count <= 0:
            return
        rows.append(
            _op(
                ops.router_gate(tokens, H, NEXP, TOPK),
                f"{tag}.router",
                "router",
                layers=count,
                source="gate.py:50 gate",
            )
        )
        # Tokens entering the MoE = per-DP-group tokens * dp groups: the fused MoE
        # is expert-parallel over the FULL mesh (data*tensor=devices), so all dp
        # groups' tokens are pooled across the experts. Per-device load then =
        # moe_tokens*topk/devices = tokens*topk/t.
        moe_tokens = tokens * lp.dp
        tokens_per_dev = max(1, moe_tokens * TOPK // ep)
        # EP all-to-all per device (dispatch + combine), balanced routing. ep here
        # = lp.ep = devices (--ep-size is ignored by the fused kernel). (ep-1)/ep
        # goes remote. Balanced theoretical floor; real a2a is imbalance-bound.
        remote = (ep - 1) / ep if ep > 1 else 0.0
        a2a = int(2 * (moe_tokens * TOPK / ep) * H * 2 * remote)
        exp_q = qs["experts"]
        rr = ops.moe_experts(
            tokens_per_device=tokens_per_dev,
            local_experts=NEXP / ep,
            d=H,
            f=MOEF,
            qspec=exp_q,
            ici_bytes=a2a,
        )
        eo = _op(
            rr,
            f"{tag}.experts[PALLAS,{exp_q.tag()}]",
            "moe",
            peak_kind=exp_q.peak_kind(),
            layers=count,
            source="fused_ep_moe_v2",
        )
        # MoE output reshard back to the reduce sharding (same SP/DP rule as o_proj)
        eo.ici_bytes += parallelism.row_parallel_reduce_bytes(tokens, H, lp) * count
        rows.append(eo)

    def dense_block(count, tag):
        if count <= 0:
            return
        mq = qs["mlp"]
        rows.append(
            _op(
                ops.gemm(tokens, H, 2 * DENSE_F, mq),
                f"{tag}.gate_up",
                "linear",
                layers=count,
                shard=tp,
                peak_kind=mq.peak_kind(),
                source="mimo_v2_flash.py:102 MLP",
            )
        )
        rows.append(
            _op(
                ops.gemm(tokens, DENSE_F, H, mq),
                f"{tag}.down",
                "linear",
                layers=count,
                shard=tp,
                peak_kind=mq.peak_kind(),
                source="mimo_v2_flash.py:105 MLP",
            )
        )
        rows.append(
            _op(
                ops.elementwise(tokens, DENSE_F),
                f"{tag}.silu",
                "other",
                layers=count,
                source="mimo_v2_flash.py:104 silu",
            )
        )

    # count layers by (attn_type, mlp_type)
    def is_swa(i):
        return bool(hlp[i]) if i < len(hlp) else False

    def is_moe(i):
        return bool(mlf[i]) if i < len(mlf) else True

    from collections import Counter

    combo = Counter((is_swa(i), is_moe(i)) for i in range(L))
    attn_block(full, combo[(False, True)] + combo[(False, False)], "full")
    attn_block(swa, combo[(True, True)] + combo[(True, False)], "swa")
    moe_block(combo[(False, True)] + combo[(True, True)], "moe")
    dense_block(combo[(False, False)] + combo[(True, False)], "dense")

    # model-level: embedding + final norm + lm_head
    rows.append(
        _op(
            ops.elementwise(tokens, H, n_inputs=0),
            "embedding",
            "embedding",
            layers=1,
            source="embeddings.py embed",
        )
    )
    rows.append(
        _op(
            ops.rms_norm(tokens, H),
            "final_norm",
            "norm",
            layers=1,
            source="mimo_v2_flash.py:714 norm",
        )
    )
    rows.append(
        _op(
            ops.gemm(logits_tokens, H, VOCAB, qs["lm_head"]),
            "lm_head",
            "lm_head",
            layers=1,
            shard=tp,
            peak_kind=qs["lm_head"].peak_kind(),
            source="logits_processor.py:495",
        )
    )

    meta = dict(
        batch=par.get("batch"),
        seq_len=par.get("seq_len"),
        chunk=par.get("chunk"),
        num_layers=L,
        tp_total=lp.tp_total,
        dp=lp.dp,
        attention_tp=lp.t,
        ep_effective=lp.ep,
        devices=lp.devices,
        enable_sp=lp.enable_sp,
        quant={r: qs[r].tag() for r in qs},
        n_full=combo[(False, True)] + combo[(False, False)],
        n_swa=combo[(True, True)] + combo[(True, False)],
        n_moe=combo[(False, True)] + combo[(True, True)],
        n_dense=combo[(False, False)] + combo[(True, False)],
    )
    return ModelRoofline(arch=arch_name, phase=phase, peaks=peaks, rows=rows, meta=meta)


@register_reference("MiMoV2FlashForCausalLM")
@register_reference("MiMoV2ProForCausalLM")
@register_reference("MiMoV2ForCausalLM")
def _mimo_v2_flash_reference(config, phase, par):
    """One representative full-attention + MoE decoder layer as a traceable JAX
    fn (real dims, Pallas as labeled placeholders) for the jaxpr_util structure
    view. Costs are NOT taken from here -- only the op graph / source attribution.
    """
    from jax.experimental import pallas as pl

    H = _cfg(config, "hidden_size")
    nh, nkv, hd = (
        _cfg(config, "num_attention_heads"),
        _cfg(config, "num_key_value_heads"),
        _cfg(config, "head_dim"),
    )
    vhd = _cfg(config, "v_head_dim", default=hd)
    NEXP = _cfg(config, "n_routed_experts", "num_experts", default=8)
    TOPK = _cfg(config, "num_experts_per_tok", default=2)
    qs, ks, vs, ao = nh * hd, nkv * hd, nkv * vhd, nh * vhd
    m = par["batch"] if phase == "decode" else par["chunk"]
    bf16 = jnp.bfloat16

    def _pallas_id(x):  # opaque kernel placeholder (attention / experts)
        def k(xr, orf):
            orf[...] = xr[...]

        return pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), interpret=True)(
            x
        )

    def linear(x, w):
        return jnp.dot(x, w, preferred_element_type=jnp.float32).astype(bf16)

    def rms_norm(x, w):
        v = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        return (x * jax.lax.rsqrt(v + 1e-6)).astype(x.dtype) * w

    def layer(hidden, residual, wn1, wq, wk, wv, wo, wn2, wg):
        hidden = hidden + residual
        x = rms_norm(hidden, wn1)
        q, k, v = linear(x, wq), linear(x, wk), linear(x, wv)
        attn = _pallas_id(
            jnp.concatenate([q, k, v], axis=-1)[:, :ao]
        )  # [PALLAS] attention (reads q,k,v)
        attn_out = linear(attn, wo)
        h2 = attn_out + hidden
        y = rms_norm(h2, wn2)
        logits = jnp.dot(y.astype(jnp.float32), wg)
        gate = jax.nn.softmax(logits, axis=-1)
        gw, _ = jax.lax.top_k(gate, TOPK)
        experts = _pallas_id(y)  # [PALLAS] MoE experts
        return experts + h2, gw

    A = lambda *s, d=bf16: jax.ShapeDtypeStruct(s, d)
    args = (
        A(m, H),
        A(m, H),
        A(H),
        A(H, qs),
        A(H, ks),
        A(H, vs),
        A(ao, H),
        A(H),
        A(H, NEXP, d=jnp.float32),
    )
    return layer, args
