"""Dataflow graph of one decoder layer: nodes = activation tensors, edges = ops.

Built explicitly from the (known) transformer structure rather than reverse-
engineered from a traced jaxpr -- this keeps Pallas kernels atomic, reuses the
v1 quant/parallelism cost logic, and puts fusion analysis at an actionable
(kernel/module) granularity. Costs come from ``references`` (Pallas, flops via
estimate_cost) and ``ops``/``quant`` (XLA GEMMs, norms, elementwise).

The graph feeds ``critical_path`` (CPM) and ``fusion`` (candidate detection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import parallelism, references
from .report import OpRoofline


@dataclass
class GOp:
    """An edge = one op: consumes input tensors, produces one output tensor."""

    id: int
    label: str
    category: str
    inputs: list[str]
    output: str
    flops: int = 0
    hbm_bytes: int = 0
    ici_bytes: int = 0
    peak_kind: str = "bf16"
    fusable: str = ""  # '', 'elementwise', 'epilogue', 'matmul', 'norm', 'pallas'
    source: str = ""

    def roofline(self) -> OpRoofline:
        return OpRoofline(
            self.label,
            self.category,
            self.source,
            1,
            self.flops,
            self.hbm_bytes,
            self.ici_bytes,
            self.peak_kind,
        )


@dataclass
class DataflowGraph:
    ops: list[GOp] = field(default_factory=list)
    tensors: dict[str, int] = field(default_factory=dict)  # tensor id -> bytes
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def consumers(self, tid: str) -> list[GOp]:
        return [o for o in self.ops if tid in o.inputs]

    def producer(self, tid: str):
        for o in self.ops:
            if o.output == tid:
                return o
        return None


def _cfg(c, *names, default=None):
    for n in names:
        if c.get(n) is not None:
            return c[n]
    return default


def build_layer_graph(config, phase, par, *, swa: bool, moe: bool) -> DataflowGraph:
    """Dataflow graph for ONE decoder layer (per-device costs)."""
    H = _cfg(config, "hidden_size")
    lp, _warns = parallelism.resolve(
        config,
        par,
        moe_backend=par.get("moe_backend", "fused_v2"),
        enable_sp=par.get("enable_sp", False),
    )
    tp, ep = lp.t, lp.ep  # tensor-parallel degree (tp//dp) and fused-MoE EP (=devices)
    tokens = par["batch"] if phase == "decode" else par["chunk"]
    bf = 2  # bf16 activation bytes
    from . import quant as _q

    qs = _q.quant_specs_from_config(config)

    if swa:
        nh, nkv, hd = (
            _cfg(
                config,
                "swa_num_attention_heads",
                default=_cfg(config, "num_attention_heads"),
            ),
            _cfg(
                config,
                "swa_num_key_value_heads",
                default=_cfg(config, "num_key_value_heads"),
            ),
            _cfg(config, "swa_head_dim", default=_cfg(config, "head_dim")),
        )
        vhd = _cfg(config, "swa_v_head_dim", default=_cfg(config, "v_head_dim", default=hd))
        window = _cfg(config, "sliding_window_size", default=4096)
    else:
        nh, nkv, hd = (
            _cfg(config, "num_attention_heads"),
            _cfg(config, "num_key_value_heads"),
            _cfg(config, "head_dim"),
        )
        vhd = _cfg(config, "v_head_dim", default=hd)
        window = 0
    qsz, ksz, vsz, ao = nh * hd, nkv * hd, nkv * vhd, nh * vhd
    eff_ctx = (
        min(par["seq_len"] if phase == "decode" else tokens, window)
        if window
        else (par["seq_len"] if phase == "decode" else tokens)
    )
    inter = int(tokens * (eff_ctx / 2 if phase != "decode" else eff_ctx))

    g = DataflowGraph()
    g.meta = dict(swa=swa, moe=moe, tokens=tokens, H=H)
    _ctr = [0]

    def T(tid, bytes_):  # register a tensor node
        g.tensors[tid] = int(bytes_)
        return tid

    def E(label, category, inputs, output, cost, *, fusable="", source="", shard=1, ici=0):
        _ctr[0] += 1
        flops = int(cost.get("flops", 0) / shard)
        hbm = int(cost.get("hbm_bytes", 0) / shard)
        g.ops.append(
            GOp(
                _ctr[0],
                label,
                category,
                inputs,
                output,
                flops,
                hbm,
                ici,
                cost.get("peak_kind", "bf16"),
                fusable,
                source,
            )
        )

    act = tokens * H * bf
    hin = T("hidden_in", act)
    res_in = T("residual_in", act)
    g.inputs = [hin, res_in]

    # residual add + input norm
    t0 = T("resid0", act)
    E(
        "residual_in_add",
        "other",
        [hin, res_in],
        t0,
        {"flops": tokens * H, "hbm_bytes": 3 * act},
        fusable="elementwise",
        source="mimo_v2_flash residual",
    )
    t1 = T("normed1", act)
    E(
        "input_layernorm",
        "norm",
        [t0],
        t1,
        {"flops": 4 * tokens * H, "hbm_bytes": 2 * act + H * bf},
        fusable="norm",
        source="layernorm.py RMSNorm",
    )
    # qkv proj
    qkv = T("qkv", tokens * (qsz + ksz + vsz) * bf)
    E(
        f"qkv_proj[{qs['qkv'].tag()}]",
        "linear",
        [t1],
        qkv,
        references.gemm_cost(m=tokens, k=H, n=qsz + ksz + vsz, qspec=qs["qkv"]),
        fusable="matmul",
        source="mimo_v2_flash.py:465 qkv",
        shard=tp,
    )
    # rope
    qkr = T("qk_roped", tokens * (qsz + ksz) * bf)
    E(
        "rope",
        "rope",
        [qkv],
        qkr,
        {"flops": 6 * (qsz + ksz) * tokens, "hbm_bytes": 2 * (qsz + ksz) * tokens * bf},
        fusable="elementwise",
        source="embeddings.py rotary",
        shard=tp,
    )
    # attention (PALLAS, atomic)
    attn = T("attn_out", tokens * ao * bf)
    acost = references.attention_cost(
        num_q_heads=max(1, nh // tp),
        num_kv_heads=parallelism.kv_heads_per_device(nkv, tp),
        head_dim=hd,
        v_head_dim=vhd,
        q_tokens=tokens,
        kv_tokens=eff_ctx,
        total_interactions=inter,
    )
    E(
        "attention[PALLAS]",
        "attention",
        [qkr],
        attn,
        acost,
        fusable="pallas",
        source="ragged_paged_attention_v3",
    )
    # o_proj (row-parallel reduce: all-reduce over tensor axis t, or SP reduce-
    # scatter + residual all-gather over the full mesh above the scatter threshold)
    t2 = T("attn_proj", act)
    oc = references.gemm_cost(m=tokens, k=ao, n=H, qspec=qs["o_proj"])
    E(
        f"o_proj[{qs['o_proj'].tag()}]",
        "linear",
        [attn],
        t2,
        oc,
        fusable="matmul",
        source="mimo_v2_flash.py:515 o_proj",
        shard=tp,
        ici=parallelism.row_parallel_reduce_bytes(tokens, H, lp),
    )
    # residual + post-attn norm
    t3 = T("resid1", act)
    E(
        "attn_residual_add",
        "other",
        [t2, t0],
        t3,
        {"flops": tokens * H, "hbm_bytes": 3 * act},
        fusable="elementwise",
        source="mimo_v2_flash residual",
    )
    t4 = T("normed2", act)
    E(
        "post_attention_layernorm",
        "norm",
        [t3],
        t4,
        {"flops": 4 * tokens * H, "hbm_bytes": 2 * act + H * bf},
        fusable="norm",
        source="layernorm.py RMSNorm",
    )

    if moe:
        NEXP = _cfg(config, "n_routed_experts", "num_experts", default=8)
        TOPK = _cfg(config, "num_experts_per_tok", default=2)
        MOEF = _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size"))
        gatew = T("gate_w", tokens * TOPK * 4)
        E(
            "router_gate",
            "router",
            [t4],
            gatew,
            references.gemm_cost(m=tokens, k=H, n=NEXP),
            fusable="matmul",
            source="gate.py:50 gate",
        )
        tpd = max(1, tokens * TOPK // ep)
        # fused-MoE EP group = full mesh = devices (ep = lp.ep). (ep-1)/ep remote;
        # plus a MoE-output reshard (same SP/DP reduce rule as o_proj).
        remote = (ep - 1) / ep if ep > 1 else 0.0
        a2a = int(2 * (tokens * TOPK / ep) * H * bf * remote)
        a2a += parallelism.row_parallel_reduce_bytes(tokens, H, lp)
        eo = T("expert_out", act)
        ec = references.moe_experts_cost(
            tokens_per_device=tpd,
            local_experts=NEXP / ep,
            d=H,
            f=MOEF,
            qspec=qs["experts"],
        )
        E(
            "experts[PALLAS]",
            "moe",
            [t4, gatew],
            eo,
            ec,
            fusable="pallas",
            source="fused_ep_moe_v2",
            ici=a2a,
        )
        out = T("hidden_out", act)
        E(
            "moe_residual_add",
            "other",
            [eo, t3],
            out,
            {"flops": tokens * H, "hbm_bytes": 3 * act},
            fusable="elementwise",
            source="mimo_v2_flash residual",
        )
    else:
        DF = _cfg(config, "intermediate_size")
        gu = T("gate_up", tokens * 2 * DF * bf)
        E(
            "gate_up_proj",
            "linear",
            [t4],
            gu,
            references.gemm_cost(m=tokens, k=H, n=2 * DF, qspec=qs["mlp"]),
            fusable="matmul",
            source="mimo_v2_flash.py:102 MLP",
            shard=tp,
        )
        si = T("silu", tokens * DF * bf)
        E(
            "silu_mul",
            "other",
            [gu],
            si,
            {"flops": tokens * DF, "hbm_bytes": tokens * 3 * DF * bf},
            fusable="elementwise",
            source="mimo_v2_flash.py:104 silu",
        )
        dn = T("mlp_down", act)
        E(
            "down_proj",
            "linear",
            [si],
            dn,
            references.gemm_cost(m=tokens, k=DF, n=H, qspec=qs["mlp"]),
            fusable="matmul",
            source="mimo_v2_flash.py:105 MLP",
            shard=tp,
        )
        out = T("hidden_out", act)
        E(
            "mlp_residual_add",
            "other",
            [dn, t3],
            out,
            {"flops": tokens * H, "hbm_bytes": 3 * act},
            fusable="elementwise",
            source="mimo_v2_flash residual",
        )

    g.outputs = [out]
    return g
