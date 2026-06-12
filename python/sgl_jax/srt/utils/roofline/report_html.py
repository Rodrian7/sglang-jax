"""Interactive HTML roofline report.

Bakes a model's config-derived constants + hardware peaks (+ a one-layer jaxpr
op/source index) into a single self-contained HTML page that re-runs the
(closed-form) cost model **in the browser**: pick the parallelism layout (tp/dp
constrained to valid mesh combos), the quantization scheme (per-tensor /
per-channel / block-wise + block size + W8A16/W8A8), and drag the workload knobs;
the roofline, a per-op dataflow view, fusion opportunities, the per-category cost
table and the bottleneck summary update live. A jaxpr tab shows the traced op
histogram + source lines. No server, no external JS libs (vanilla + high-DPI
responsive canvas), works offline.

The JS cost model mirrors ``descriptors._mimo_v2_family`` + ``parallelism`` +
``ops`` (tensor axis t = tp//dp, fused-MoE EP = devices, MoE global tokens =
per-DP tokens * dp; row-parallel outputs complete with a SYNC all-reduce on the
'tensor' axis, and under SP an ASYNC all-gather re-collects the sequence shards
before the next linear — HLO-verified, no standalone reduce-scatter).
Quant: fp8 weight = 1 byte + scale (per-tensor ~0 / per-channel 4/k / block
4/B^2 per elem); block-wise stays bf16 MXU rate, per-tensor/per-channel + fp8
acts reach the fp8 MXU rate. Closed-form roofline only; the jaxpr View F costs
need a real trace and stay Python-side.
"""

from __future__ import annotations

import contextlib
import json
from collections import Counter

from .report import HardwarePeaks


def _cfg(config, *names, default=None):
    for n in names:
        if config.get(n) is not None:
            return config[n]
    return default


def _quant_default(config) -> dict:
    """Derive the default quant-knob state from config.json's quantization_config.
    fp8 + weight_block_size -> block-wise (block size from the config); fp8 without
    a block -> per-tensor; activation_scheme present -> W8A8 else W8A16; no quant
    config -> bf16."""
    qc = config.get("quantization_config") or {}
    qm = str(qc.get("quant_method") or "").lower()
    if "fp8" not in qm:
        return {"wq": "bf16", "blk": 128, "aq": "bf16"}
    wbs = qc.get("weight_block_size")
    if wbs:
        wq, blk = "block", int(wbs[0])
    else:
        wq, blk = "per_tensor", 128
    aq = "fp8" if qc.get("activation_scheme") in ("dynamic", "static") else "bf16"
    return {"wq": wq, "blk": blk, "aq": aq}


def _bake_moe_block_table(config) -> dict:
    """The fused-MoE-v2 block config the kernel WOULD pick, per (ep, num_tokens),
    by calling the kernel's OWN lookup ``get_tuned_fused_moe_v2_block_config`` —
    not by parsing the table. So if the tuned table (or its key schema, or the
    lookup logic) changes, the report tracks it on the next regenerate with **no
    roofline code change**. num_tokens = global tokens entering MoE (= per-DP
    chunk x dp). Returns {ep: [{n, bt, bf}, ... sorted]}."""
    try:
        import jax.numpy as jnp

        from sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs import (
            get_tuned_fused_moe_v2_block_config,
        )
    except Exception:
        return {}
    NEXP = _cfg(config, "n_routed_experts", "num_experts", default=0)
    TOPK = _cfg(config, "num_experts_per_tok", default=0)
    H = _cfg(config, "hidden_size")
    MOEF = _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size"))
    if not (NEXP and TOPK and H and MOEF):
        return {}
    qc = config.get("quantization_config") or {}
    is_fp8 = "fp8" in str(qc.get("quant_method") or "").lower()
    wdt = jnp.float8_e4m3fn if is_fp8 else jnp.bfloat16
    wbs = qc.get("weight_block_size")
    qbk = int(wbs[0]) if wbs else 128
    # MoE VMEM estimator: AST-extract the PURE estimator from bench_v2 WITHOUT importing
    # the module (bench_v2 calls jax.distributed.initialize() + jax.devices() at top
    # level). Reads the live source each regenerate -> tracks formula changes, no copy
    # that can drift. None if unavailable -> the JS card falls back to its rough formula.
    vmem_est = None
    try:
        import ast as _ast
        import math as _math
        import os as _os

        import sgl_jax.srt.kernels.fused_moe.v2 as _v2pkg

        _p = _os.path.join(_os.path.dirname(_v2pkg.__file__), "bench_v2.py")
        with open(_p) as _f:
            _tree = _ast.parse(_f.read())
        _ns = {"math": _math}
        for _node in _tree.body:
            if isinstance(_node, _ast.FunctionDef) and _node.name in (
                "_align_to",
                "_estimate_vmem_bytes_v2",
            ):
                exec(compile(_ast.Module([_node], []), _p, "exec"), _ns)
        vmem_est = _ns.get("_estimate_vmem_bytes_v2")
    except Exception:
        vmem_est = None
    buckets = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    out: dict[int, list] = {}
    for ep in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if NEXP % ep != 0:
            continue
        row = []
        for n in buckets:
            if n % ep != 0:
                continue
            try:
                cfg = get_tuned_fused_moe_v2_block_config(
                    num_tokens=n,
                    num_experts=NEXP,
                    top_k=TOPK,
                    hidden_size=H,
                    intermediate_size=MOEF,
                    dtype=jnp.bfloat16,
                    weight_dtype=wdt,
                    ep_size=ep,
                )
                e = {"n": n, "bt": int(cfg.bt), "bf": int(cfg.bf)}
                if vmem_est is not None:
                    # kernel's own VMEM working-set estimate (double-buffered W1/W3/W2 + staging)
                    with contextlib.suppress(Exception):
                        bc = cfg.effective_for(num_tokens=n, ep_size=ep)
                        e["vmem"] = int(
                            vmem_est(
                                bt=bc.bt,
                                bf=bc.bf,
                                btc=bc.btc,
                                bse=bc.bse,
                                bts=bc.bts,
                                hidden_size=H,
                                intermediate_size=MOEF,
                                num_experts=NEXP,
                                top_k=TOPK,
                                ep_size=ep,
                                num_tokens=n,
                                use_fp8=is_fp8,
                                quant_block_k=qbk,
                                direct_scaled_dot=True,
                            )
                        )
                row.append(e)
            except Exception:
                pass
        if row:
            out[ep] = row
    return out


def _bake_rpa_block_table(config) -> dict:
    """The RPA-v3 (bq, bkv) block config the kernel WOULD pick, per (tensor-axis t,
    stage, sliding_window, max_num_tokens), by calling the kernel's OWN
    ``get_tuned_block_sizes_v3`` — same change-resilience as the MoE table: tune
    table / key schema / lookup changes are tracked on the next regenerate with no
    roofline code change. ``max_num_tokens`` = per-device query tokens (= the token
    knob, since ``max_num_tokens = queries.shape[0]`` in the kernel). Heads are
    per-device (nh/t, kv with replication), mirroring the JS card. Returns
    ``{"t|stage|sw": [{n, bq, bkv}, ...]}``; misses are skipped (JS falls back to the
    traced kernel name)."""
    try:
        import jax.numpy as jnp

        from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
            get_vmem_estimate_bytes,
        )
        from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes_v3 import (
            get_tuned_block_sizes_v3,
        )
    except Exception:
        return {}
    nh = _cfg(config, "num_attention_heads")
    nkv = _cfg(config, "num_key_value_heads", default=nh)
    hd = _cfg(config, "head_dim")
    if not (nh and hd):
        return {}
    swa = _cfg(config, "sliding_window_size", default=None)
    page_size = 256
    buckets = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    windows = [None] + ([int(swa)] if swa else [])
    out: dict[str, list] = {}
    for t in (1, 2, 4, 8, 16, 32, 64, 128):
        nq = max(1, nh // t)
        nkv_dev = 1 if t >= nkv else -(-nkv // t)  # ceil; replicated when t>=nkv
        for stage in ("d", "p", "m"):
            for sw in windows:
                row = []
                for n in buckets:
                    try:
                        hit = get_tuned_block_sizes_v3(
                            stage,
                            jnp.bfloat16,
                            jnp.bfloat16,
                            nq,
                            nkv_dev,
                            hd,
                            page_size,
                            n,
                            sliding_window=sw,
                        )
                    except Exception:
                        hit = None
                    if not hit:
                        continue
                    e = {"n": n, "bq": int(hit[0]), "bkv": int(hit[1])}
                    # kernel's own VMEM working-set estimate (double-buffered KV+mask, f32 scores)
                    with contextlib.suppress(Exception):
                        e["vmem"] = int(
                            get_vmem_estimate_bytes(
                                nkv_dev,
                                max(1, nq // nkv_dev),
                                hd,
                                hit[0],
                                hit[1],
                                jnp.bfloat16,
                                jnp.bfloat16,
                                use_custom_mask=False,
                                bkv_csz=hit[3],
                            )
                        )
                    row.append(e)
                if row:
                    out[f"{t}|{stage}|{'full' if sw is None else sw}"] = row
    return out


def _bake_jaxpr(arch, config) -> dict | None:
    """Trace one reference layer to a jaxpr; bake the primitive histogram + the
    source line that emits each (shortened). None if jax / reference unavailable."""
    try:
        from . import descriptors, interp

        ref = descriptors.reference_forward(arch, config, "decode", {"batch": 1, "chunk": 1})
        sv = interp.structure_view(ref)
        if sv is None:
            return None

        def opname(s):
            # source_info "file.py:line:col (a.b.<locals>.fn)" -> innermost fn name
            if "(" in s:
                return s.split("(", 1)[1].rstrip(")").split(".")[-1]
            return s.rsplit("/", 1)[-1] if "/" in s else s

        def top(d, n=24):
            return [[k, v] for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n]]

        byop: dict[str, int] = {}
        for k, v in sv["by_source"].items():
            byop[opname(k)] = byop.get(opname(k), 0) + v
        return {
            "num_eqns": sv["num_eqns"],
            "by_primitive": top(sv["by_primitive"]),
            "by_source": top(byop),
        }
    except Exception:
        return None


_COMPOSERS = {}  # arch_name -> compose(config) -> {"attn":[...], "ffn":[...]}


def register_composer(*names):
    def deco(fn):
        for n in names:
            _COMPOSERS[n] = fn
        return fn

    return deco


_MOE_BACKENDS = {
    "fused_v2": "fused_moe_v2",  # in-kernel a2a on SparseCore (FusedEPMoE / MiMo)
    "fused": "fused_moe_v2",
    "fused_ep": "fused_moe_v2",
    "epmoe": "ep_moe",  # expert-parallel MoE with explicit XLA all-to-all
    "ep": "ep_moe",
    "ep_moe": "ep_moe",
    "fused_v1": "fused_moe_v1",
    "v1": "fused_moe_v1",
}


def _moe_module(config, defaults) -> str:
    """Resolve which MoE-kernel module to model. Priority: explicit generation-time
    override (``defaults['moe_backend']``, e.g. trace_roofline ``--moe-backend``) >
    checkpoint/runtime (``config.moe_backend``) > ``fused_moe_v2`` (the default we
    model). The roofline is a what-if tool, so the backend is a free knob — pass
    ``--moe-backend epmoe`` to compare EPMoE against the fused-v2 kernel."""
    bk = (defaults or {}).get("moe_backend") or _cfg(config, "moe_backend") or "fused_v2"
    return _MOE_BACKENDS.get(str(bk).lower(), "fused_moe_v2")


def _compose(arch, config, defaults=None) -> dict:
    """Detect which kernel/layer-type modules a model uses, and how many layers of
    each, as ``{"attn":[{m,count,dims}], "ffn":[{m,count,dims}]}``. The JS generator
    iterates this and dispatches to its MODULES registry, so the report is not tied
    to any one model. Per-arch composers register below; the generic fallback covers
    RPA(full/SWA) attention + MoE / dense FFN models. The MoE-kernel module is a
    configurable knob (see ``_moe_module``)."""
    return _COMPOSERS.get(arch, _compose_generic)(config, _moe_module(config, defaults))


@register_composer("MiMoV2FlashForCausalLM", "MiMoV2ProForCausalLM", "MiMoV2ForCausalLM")
def _compose_generic(config, moe_m="fused_moe_v2") -> dict:
    """RPA attention (full + sliding-window) + fused-MoE-v2 / dense FFN, keyed off
    ``hybrid_layer_pattern`` (SWA layers) and ``moe_layer_freq`` (MoE vs dense).
    Covers the MiMo-V2 family and dense / all-MoE RPA models (e.g. Qwen3-MoE,
    whose missing patterns default to all-full-attention + all-MoE)."""
    L = _cfg(config, "num_hidden_layers")
    hlp = _cfg(config, "hybrid_layer_pattern", default=[0] * L)
    mlf = _cfg(config, "moe_layer_freq", default=[1] * L)

    def is_swa(i):
        return bool(hlp[i]) if i < len(hlp) else False

    def is_moe(i):
        return bool(mlf[i]) if i < len(mlf) else True

    combo = Counter((is_swa(i), is_moe(i)) for i in range(L))
    full = dict(
        nh=_cfg(config, "num_attention_heads"),
        nkv=_cfg(config, "num_key_value_heads"),
        hd=_cfg(config, "head_dim"),
        vhd=_cfg(config, "v_head_dim", default=_cfg(config, "head_dim")),
        window=0,
    )
    swa = dict(
        nh=_cfg(config, "swa_num_attention_heads", default=full["nh"]),
        nkv=_cfg(config, "swa_num_key_value_heads", default=full["nkv"]),
        hd=_cfg(config, "swa_head_dim", default=full["hd"]),
        vhd=_cfg(config, "swa_v_head_dim", default=full["vhd"]),
        window=_cfg(config, "sliding_window_size", default=4096),
    )
    NEXP = _cfg(config, "n_routed_experts", "num_experts", default=8)
    TOPK = _cfg(config, "num_experts_per_tok", default=2)
    MOEF = _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size"))
    DENSE_F = _cfg(config, "intermediate_size")
    n_full = combo[(False, True)] + combo[(False, False)]
    n_swa = combo[(True, True)] + combo[(True, False)]
    n_moe = combo[(False, True)] + combo[(True, True)]
    n_dense = combo[(False, False)] + combo[(True, False)]
    return {
        "attn": [
            x
            for x in [
                {"m": "rpa", "count": n_full, "dims": {**full, "variant": "full"}},
                {"m": "rpa", "count": n_swa, "dims": {**swa, "variant": "swa"}},
            ]
            if x["count"] > 0
        ],
        "ffn": [
            x
            for x in [
                {
                    "m": moe_m,
                    "count": n_moe,
                    "dims": {"NEXP": NEXP, "TOPK": TOPK, "MOEF": MOEF},
                },
                {"m": "dense", "count": n_dense, "dims": {"DENSE_F": DENSE_F}},
            ]
            if x["count"] > 0
        ],
    }


@register_composer("Qwen3MoeForCausalLM")
def _compose_qwen3_moe(config, moe_m="fused_moe_v2") -> dict:
    """Qwen3-MoE: GQA full attention (no sliding window) + per-layer MoE/dense.
    A layer is MoE when it is not in ``mlp_only_layers`` and (i+1) is a multiple of
    ``decoder_sparse_step`` (HF Qwen3Moe rule); the rest are dense. Qwen3-MoE has no
    shared experts. Falls back to all-MoE when the sparsity fields are absent
    (e.g. Qwen3-30B-A3B)."""
    L = _cfg(config, "num_hidden_layers")
    NEXP = _cfg(config, "num_experts", "n_routed_experts", default=0)
    TOPK = _cfg(config, "num_experts_per_tok", default=8)
    MOEF = _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size"))
    DENSE_F = _cfg(config, "intermediate_size")
    step = _cfg(config, "decoder_sparse_step", default=1)
    mlp_only = set(_cfg(config, "mlp_only_layers", default=[]) or [])
    nh = _cfg(config, "num_attention_heads")
    full = dict(
        nh=nh,
        nkv=_cfg(config, "num_key_value_heads", default=nh),
        hd=_cfg(config, "head_dim", default=_cfg(config, "hidden_size") // nh),
        vhd=_cfg(config, "head_dim", default=_cfg(config, "hidden_size") // nh),
        window=0,
    )

    def is_moe(i):
        return NEXP > 0 and i not in mlp_only and ((i + 1) % step == 0)

    n_moe = sum(is_moe(i) for i in range(L))
    n_dense = L - n_moe
    return {
        "attn": [{"m": "rpa", "count": L, "dims": {**full, "variant": "full"}}],
        "ffn": [
            x
            for x in [
                {
                    "m": moe_m,
                    "count": n_moe,
                    "dims": {"NEXP": NEXP, "TOPK": TOPK, "MOEF": MOEF},
                },
                {"m": "dense", "count": n_dense, "dims": {"DENSE_F": DENSE_F}},
            ]
            if x["count"] > 0
        ],
    }


@register_composer("BailingMoeV3ForCausalLM")
def _compose_bailing_moe_v3(config, moe_m="fused_moe_v2") -> dict:
    """Ling3 / BailingMoeV3: hybrid KDA (linear/recurrent) + MLA (latent) attention
    + MoE with an always-on shared expert. The LAST layer of every
    ``layer_group_size`` group is MLA (full attention); the rest are KDA. The first
    ``first_k_dense_replace`` layers are dense MLP, the rest MoE (each MoE layer also
    runs the shared expert). MoE-kernel module is configurable (default fused-v2)."""
    L = _cfg(config, "num_hidden_layers")
    gsz = _cfg(config, "layer_group_size", default=1)
    n_mla = sum(1 for i in range(L) if (i + 1) % gsz == 0)
    n_kda = L - n_mla
    nh = _cfg(config, "num_attention_heads")
    mla = {
        "nh": nh,
        "qk_nope": _cfg(config, "qk_nope_head_dim"),
        "qk_rope": _cfg(config, "qk_rope_head_dim"),
        "v_head": _cfg(config, "v_head_dim"),
        "kv_lora": _cfg(config, "kv_lora_rank"),
        "q_lora": _cfg(config, "q_lora_rank") or 0,
    }
    kda = {
        "nh": nh,
        "hd": _cfg(config, "head_dim"),
        "conv": _cfg(config, "short_conv_kernel_size", default=4),
    }
    NEXP = _cfg(config, "num_experts", "n_routed_experts", default=0)
    TOPK = _cfg(config, "num_experts_per_tok", default=8)
    MOEF = _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size"))
    DENSE_F = _cfg(config, "intermediate_size")
    fkd = _cfg(config, "first_k_dense_replace", default=0)
    n_shared = _cfg(config, "num_shared_experts", default=0)
    SHF = _cfg(config, "moe_shared_expert_intermediate_size", default=MOEF) * max(1, n_shared)
    n_moe = max(0, L - fkd)
    n_dense = fkd
    return {
        "attn": [
            x
            for x in [
                {"m": "kda", "count": n_kda, "dims": kda},
                {"m": "mla", "count": n_mla, "dims": mla},
            ]
            if x["count"] > 0
        ],
        "ffn": [
            x
            for x in [
                {"m": moe_m, "count": n_moe, "dims": {"NEXP": NEXP, "TOPK": TOPK, "MOEF": MOEF}},
                {
                    "m": "shared_expert",
                    "count": n_moe if n_shared > 0 else 0,
                    "dims": {"SHF": SHF},
                },
                {"m": "dense", "count": n_dense, "dims": {"DENSE_F": DENSE_F}},
            ]
            if x["count"] > 0
        ],
    }


def _bake(arch, config, peaks: HardwarePeaks, defaults: dict) -> dict:
    H = _cfg(config, "hidden_size")
    L = _cfg(config, "num_hidden_layers")
    return {
        "arch": arch,
        "H": H,
        "L": L,
        "VOCAB": _cfg(config, "vocab_size"),
        "NEXP": _cfg(config, "n_routed_experts", "num_experts", default=8),
        "TOPK": _cfg(config, "num_experts_per_tok", default=2),
        "MOEF": _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size")),
        "DENSE_F": _cfg(config, "intermediate_size"),
        "composition": _compose(arch, config, defaults),
        "jaxpr": _bake_jaxpr(arch, config),
        "moe_blocks": _bake_moe_block_table(config),
        "rpa_blocks": _bake_rpa_block_table(config),
        "peaks": {
            "bf16_tflops": peaks.bf16_tflops,
            "fp8_tflops": peaks.fp8_tflops,
            "hbm_gbps": peaks.hbm_gbps,
            "ici_gbps": peaks.ici_gbps,
            "vmem_mb": peaks.vmem_mb,
        },
        "defaults": {
            "tp": defaults.get("tp", 8),
            "dp": defaults.get("dp", 1),
            "batch": defaults.get("batch", 64),
            "seq_len": defaults.get("seq_len", 4096),
            "chunk": defaults.get("chunk", 16384),
            "enable_sp": bool(defaults.get("enable_sp", False)),
            "scatter_min": 128,
            **_quant_default(config),
        },
    }


def build_html_report(
    arch,
    config,
    peaks: HardwarePeaks,
    defaults: dict,
    out_path: str,
    codepath: dict | None = None,
    hlo: dict | None = None,
) -> str:
    data = _bake(arch, config, peaks, defaults)
    data["codepath"] = codepath  # real per-op code-path index + Pallas kernels from a trace
    data["hlo"] = hlo  # compiler ground-truth overlap (parse_hlo_overlap), optional
    html = _TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>roofline</title>
<style>
 body{font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f8fa;color:#1c2330}
 #wrap{display:flex;flex-wrap:wrap;gap:20px;padding:20px}
 #left{flex:0 0 300px;position:sticky;top:12px;align-self:flex-start;max-height:calc(100vh - 24px);overflow-y:auto} #right{flex:1 1 700px;min-width:560px}
 h1{font-size:17px;margin:0 0 2px} .sub{color:#667;font-size:11px;margin-bottom:14px}
 .ctl{margin:11px 0} .ctl label{display:block;color:#445;font-size:11px;margin-bottom:4px;font-weight:600}
 .ctl .v{color:#0a5;font-weight:700}
 input[type=range]{width:100%} select{background:#fff;color:#1c2330;border:1px solid #bcc;border-radius:5px;padding:4px 8px;font-size:13px}
 select:disabled{background:#eef;color:#aab}
 .seg{display:inline-flex;border:1px solid #bcc;border-radius:6px;overflow:hidden}
 .seg button{background:#fff;color:#556;border:0;padding:5px 14px;cursor:pointer;font-size:13px}
 .seg button.on{background:#2563eb;color:#fff}
 #summary{background:#fff;border:1px solid #dde;border-radius:8px;padding:12px;margin-top:10px;box-shadow:0 1px 3px #0001}
 #summary b{color:#0a5} .pill{display:inline-block;background:#eef;border-radius:5px;padding:2px 7px;margin:3px 4px 0 0;font-size:11px;color:#335}
 canvas{background:#fff;border:1px solid #dde;border-radius:8px;box-shadow:0 1px 4px #0001;display:block}
 .panel{background:#fff;border:1px solid #dde;border-radius:8px;box-shadow:0 1px 4px #0001;padding:14px 16px;box-sizing:border-box}
 table{border-collapse:collapse;width:100%;margin-top:12px;font-size:12px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px #0001}
 th,td{padding:4px 9px;text-align:right;border-bottom:1px solid #eef} th{color:#556;background:#f1f4f8} td.l,th.l{text-align:left}
 .tag{font-size:10px;padding:1px 5px;border-radius:4px}
 .b-HBM{background:#dbeafe;color:#1e40af} .b-ICI{background:#fce7f3;color:#9d174d} .b-compute{background:#dcfce7;color:#166534}
 #tip{position:fixed;pointer-events:none;background:#1c2330;color:#fff;border-radius:5px;padding:5px 8px;font-size:11px;display:none;z-index:9;box-shadow:0 2px 8px #0004}
 .legend{font-size:11px;color:#556;margin-top:6px}
 .dfrow{display:flex;align-items:center;margin:3px 0;font-size:12px}
 .dfrow .nm{flex:0 0 168px;color:#334} .dfrow .barwrap{flex:1 1 auto;background:#f1f4f8;border-radius:4px;height:16px;margin:0 8px}
 .dfrow .bar{height:16px;border-radius:4px} .dfrow .ms{flex:0 0 130px;text-align:right;color:#556}
 .dfarrow{color:#cbd5e1;font-size:11px;margin-left:80px}
 .lh{font-size:14px;font-weight:700;color:#1c2330;margin-bottom:4px}
 .note{font-size:11px;color:#667;margin:4px 0 8px}
 .verdict{margin-top:9px;padding:8px 11px;border-radius:7px;font-size:12px;line-height:1.5}
 .v-warn{background:#fff7ed;border:1px solid #fdba74;color:#9a3412}
 .v-go{background:#ecfdf5;border:1px solid #6ee7b7;color:#065f46}
 .mono{font-family:ui-monospace,Menlo,monospace;font-size:11px}
 .scennav{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
 .scennav button{flex:1 1 auto;padding:9px 8px;border:1px solid #cbd5e1;background:#fff;border-radius:8px;cursor:pointer;font-size:13px;color:#445;font-weight:600}
 .scennav button.on{background:#2563eb;color:#fff;border-color:#2563eb}
 .scenhelp{font-size:12px;color:#667;margin:0 0 12px}
</style></head><body>
<div id="wrap">
 <div id="left">
  <h1>Roofline · <span id="arch"></span></h1>
  <div class="sub">per-device · v7x · adjust layout / quant / workload, recomputed live</div>
  <div class="ctl"><label>phase</label>
    <span class="seg"><button id="ph-decode" class="on">decode</button><button id="ph-prefill">prefill</button></span>
    &nbsp; <label style="display:inline">SP</label> <input type="checkbox" id="sp"></div>
  <div class="ctl"><label>weight quant (qkv/mlp/experts/lm_head; o_proj bf16)</label>
    <select id="wq"><option value="bf16">bf16 (none)</option><option value="per_tensor">fp8 per-tensor</option><option value="per_channel">fp8 per-channel</option><option value="block">fp8 block-wise</option></select></div>
  <div class="ctl"><label>block size (block-wise only)</label>
    <select id="blk"><option value="128">128</option><option value="256">256</option><option value="512">512</option></select></div>
  <div class="ctl"><label>activation</label>
    <select id="aq"><option value="bf16">bf16 (W·A16)</option><option value="fp8">fp8 (W·A8)</option></select></div>
  <div class="ctl"><label>tp_size = devices = mesh total</label><select id="tp"></select></div>
  <div class="ctl"><label>dp_size — tensor axis t = tp/dp = <span class="v" id="tv"></span></label><select id="dp"></select></div>
  <div class="ctl" id="ctl-batch"><label>decode batch (tokens) <span class="v" id="batchv"></span></label>
    <input type="range" id="batch" min="1" max="2048" step="1"></div>
  <div class="ctl" id="ctl-seq"><label>decode KV context <span class="v" id="seqv"></span></label>
    <input type="range" id="seq_len" min="256" max="262144" step="256"></div>
  <div class="ctl" id="ctl-chunk"><label>prefill chunk tokens <span class="v" id="chunkv"></span></label>
    <input type="range" id="chunk" min="256" max="32768" step="256"></div>
  <div id="summary"></div>
 </div>
 <div id="right">
  <div class="scennav" id="scennav">
   <button data-sc="overview" class="on">Overview</button><button data-sc="overlap">Overlap</button><button data-sc="kernel">Kernel</button><button data-sc="fusion">Fusion</button><button data-sc="trace">Trace</button>
  </div>
  <div id="scenhelp" class="scenhelp"></div>
  <div id="body"></div>
 </div>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;
const CAT = {moe:"#dc2626",linear:"#2563eb",o_proj:"#0d9488",attention:"#16a34a",router:"#9333ea",lm_head:"#b45309",norm:"#db2777",rope:"#6b7280",other:"#a16207",embedding:"#0891b2"};
const P = D.peaks;
const flops_per_s = k => (k==="fp8"?P.fp8_tflops:P.bf16_tflops)*1e12;
const HBMBW = P.hbm_gbps*1e9, ICIBW = P.ici_gbps*1e9;

// ---- quantization model (parametric) ----
let Q={wq:"bf16", blk:128, aq:"bf16"};
const WROLES={qkv:1,mlp:1,experts:1,lm_head:1};   // o_proj kept bf16
function wbytes(k,n){ if(Q.wq==="bf16") return 2*k*n;
  let sc; if(Q.wq==="per_tensor") sc=4; else if(Q.wq==="per_channel") sc=4*n;
  else sc=4*Math.ceil(k/Q.blk)*Math.ceil(n/Q.blk);
  return k*n + sc; }
function abytes(m,k){ return (Q.aq==="fp8"?1:2)*m*k; }
function wpeak(){ return (Q.wq!=="bf16" && Q.wq!=="block" && Q.aq==="fp8") ? "fp8" : "bf16"; }

// ---- cost primitives (mirror ops.py / descriptors.py) ----
function gemm(m,k,n,role){const q=WROLES[role]&&Q.wq!=="bf16";
  const wb=q?wbytes(k,n):2*k*n, ab=q?abytes(m,k):2*m*k;
  return {flops:2*m*k*n, hbm:wb+ab+2*m*n, ici:0, peak:q?wpeak():"bf16"};}
// bq = query-block size for the flash KV-read model. Prefill blocks ONE sequence's
// queries (bq=32) so the KV streams ~once per block (~inter/bq). Decode's qtok tokens
// are SEPARATE sequences each reading their own full KV — no shared blocking, bq=1 ->
// inter (matches the tuned decode block bq=1). Hardcoding 32 under-counted decode KV 32x.
function attention(nq,nkv,hd,vhd,qtok,inter,bq,kvShare){bq=bq||32; const kc=kvShare?1:2; const f=4*nq*hd*inter;
  const hbm=qtok*nq*hd*2 + qtok*nq*vhd*2 + Math.floor(inter/bq)*nkv*kc*hd*2 + qtok*nkv*kc*hd*2; return {flops:f,hbm:hbm,ici:0,peak:"bf16"};}
function moe(tpd,le,d,f,role){const q=WROLES[role]&&Q.wq!=="bf16";
  const wbf=q?(2*wbytes(d,f)+wbytes(f,d)):(2*2*d*f+2*f*d); const act=(q?abytes(tpd,d):2*tpd*d)+2*tpd*d;
  return {flops:2*tpd*3*d*f, hbm:le*wbf+act, ici:0, peak:q?wpeak():"bf16"};}
function rope(m,qs,ks){return {flops:6*(qs+ks)*m, hbm:2*(qs+ks)*m*2, ici:0, peak:"bf16"};}
function rms(m,h){return {flops:4*m*h, hbm:2*m*h*2+h*2, ici:0, peak:"bf16"};}
function elt(m,h,ninp){return {flops:m*h, hbm:(ninp+1)*m*h*2, ici:0, peak:"bf16"};}
function router(m,h,ne){return {flops:2*m*h*ne, hbm:m*h*2+h*ne*4+m*ne*4*5, ici:0, peak:"bf16"};}
function allreduce(msg,p){return p<=1?0:2*(p-1)/p*msg;}
function reducescatter(msg,p){return p<=1?0:(p-1)/p*msg;}
function resolve(s){const tp=s.tp, dp=s.dp, devices=tp; const t=Math.max(1,Math.floor(tp/dp)); return {t, ep:devices, devices, dp};}
function kvpd(nkv,t){return t>=nkv?1:Math.ceil(nkv/t);}
// Exposed row-parallel completion = all-reduce on the 'tensor' axis. HLO ground
// truth (SP on, 4096 tok): replica_groups = groups of t, SYNC barrier, no
// reduce-scatter survives. The contracted dim consumed 'tensor' upstream so the
// partial sum must reduce over t (no-op when t=1). Under SP the result is
// sequence-sharded by a FREE local dynamic-slice; re-collecting it is a separate
// ASYNC all-gather (spGather) hidden behind the next linear — XLA lowered the
// scatter as all-reduce + slice, so the reduce itself is a plain TP all-reduce.
function rowReduce(tokens,H,L){return allreduce(tokens*H*2,L.t);}
// SP re-gather (seq-shard -> tensor-replicated) before each input linear; async in
// the HLO (overlaps the linear), so it counts as hidden comm, not an exposed barrier.
function spGather(tokens,H,L){const g=tokens*L.dp;
  const on = L.sp && L.t>1 && g>=L.devices*D.defaults.scatter_min && g%L.devices===0;
  return on ? (L.t-1)/L.t*tokens*H*2 : 0;}

// ---- module registry (model-agnostic): each kernel/layer-type's cost ----
// costRows(ctx,dims) -> [{cat, o:{flops,hbm,ici,peak}, mult?, shard?}] replicating the
// legacy add() calls; ctx={tokens,ctxlen,decode,t,ep,L}. compute() sums MODULES[m].costRows
// x layer-count over D.composition.{attn,ffn}, so the generator is not tied to any model.
const MODULES = {
  rpa: {
    kind:"attn",
    label(dims){return "RPA attention (per device, %PH%) — "+(dims.window?("SWA (window="+dims.window+")"):"full");},
    flow(dims){return dims.window?"SWA-attn":"full-attn";},
    headDiv(dims){return dims.nh;},  // tensor axis t must divide the query heads
    costRows(ctx,dims){
      const nh=dims.nh,nkv=dims.nkv,hd=dims.hd,vhd=dims.vhd,window=dims.window;
      const qs=nh*hd, ks=nkv*hd, vs=nkv*vhd, ao=nh*vhd;
      const eff=window?Math.min(ctx.ctxlen,window):ctx.ctxlen;
      const inter=ctx.tokens*(ctx.decode?eff:eff/2);
      return [
        {cat:"linear", o:gemm(ctx.tokens,D.H,qs+ks+vs,"qkv"), shard:ctx.t},
        {cat:"rope", o:rope(ctx.tokens,qs,ks), shard:ctx.t},
        {cat:"attention", o:attention(Math.max(1,Math.floor(nh/ctx.t)),kvpd(nkv,ctx.t),hd,vhd,ctx.tokens,inter,ctx.decode?1:32)},
        {cat:"o_proj", o:gemm(ctx.tokens,ao,D.H,"o_proj"), shard:ctx.t},
        {cat:"o_proj", o:{flops:0,hbm:0,ici:rowReduce(ctx.tokens,D.H,ctx.L),peak:"bf16"}},
        {cat:"norm", o:rms(ctx.tokens,D.H), mult:2},
        {cat:"other", o:elt(ctx.tokens,D.H,2), mult:2},
      ];
    },
    // one representative layer's attention sub-chain (dataflow lens; bq defaults to
    // 32 to match the legacy single-layer view, distinct from costRows' decode bq=1)
    chain(ctx,dims){
      const nh=dims.nh,nkv=dims.nkv,hd=dims.hd,vhd=dims.vhd,window=dims.window;
      const qs=nh*hd, ks=nkv*hd, vs=nkv*vhd, ao=nh*vhd;
      const eff=window?Math.min(ctx.ctxlen,window):ctx.ctxlen;
      const inter=ctx.tokens*(ctx.decode?eff:eff/2);
      const oproj=gemm(ctx.tokens,ao,D.H,"o_proj"); oproj.ici=rowReduce(ctx.tokens,D.H,ctx.L);
      return [
        {name:"qkv_proj",cat:"linear",o:gemm(ctx.tokens,D.H,qs+ks+vs,"qkv"),shard:ctx.t},
        {name:"rope",cat:"rope",o:rope(ctx.tokens,qs,ks),shard:ctx.t},
        {name:"attention",cat:"attention",o:attention(Math.max(1,Math.floor(nh/ctx.t)),kvpd(nkv,ctx.t),hd,vhd,ctx.tokens,inter)},
        {name:"o_proj +reduce",cat:"o_proj",o:oproj,shard:ctx.t},
      ];
    },
  },
  fused_moe_v2: {
    kind:"ffn",
    label(){return "fused-MoE-v2 experts (per device, per layer)";},
    flow(){return "MoE";},
    costRows(ctx,dims){
      const NEXP=dims.NEXP,TOPK=dims.TOPK,MOEF=dims.MOEF;
      const mt=ctx.tokens*ctx.L.dp, tpd=Math.max(1,Math.floor(mt*TOPK/ctx.ep)), remote=ctx.ep>1?(ctx.ep-1)/ctx.ep:0;
      const e=moe(tpd, NEXP/ctx.ep, D.H, MOEF, "experts");
      e.ici=2*(mt*TOPK/ctx.ep)*D.H*2*remote + rowReduce(ctx.tokens,D.H,ctx.L);
      return [ {cat:"router", o:router(ctx.tokens,D.H,NEXP)}, {cat:"moe", o:e} ];
    },
    chain(ctx,dims){
      const NEXP=dims.NEXP,TOPK=dims.TOPK,MOEF=dims.MOEF;
      const mt=ctx.tokens*ctx.L.dp, tpd=Math.max(1,Math.floor(mt*TOPK/ctx.ep)), remote=ctx.ep>1?(ctx.ep-1)/ctx.ep:0;
      const e=moe(tpd,NEXP/ctx.ep,D.H,MOEF,"experts"); e.ici=2*(mt*TOPK/ctx.ep)*D.H*2*remote+rowReduce(ctx.tokens,D.H,ctx.L);
      return [ {name:"router_gate",cat:"router",o:router(ctx.tokens,D.H,NEXP)}, {name:"experts +a2a",cat:"moe",o:e} ];
    },
  },
  dense: {
    kind:"ffn",
    label(){return "dense MLP";},
    flow(){return "dense";},
    costRows(ctx,dims){
      const F=dims.DENSE_F;
      return [
        {cat:"linear", o:gemm(ctx.tokens,D.H,2*F,"mlp"), shard:ctx.t},
        {cat:"linear", o:gemm(ctx.tokens,F,D.H,"mlp"), shard:ctx.t},
        {cat:"other", o:elt(ctx.tokens,F,1)},
      ];
    },
    chain(ctx,dims){
      const F=dims.DENSE_F;
      return [
        {name:"gate_up_proj",cat:"linear",o:gemm(ctx.tokens,D.H,2*F,"mlp"),shard:ctx.t},
        {name:"silu",cat:"other",o:elt(ctx.tokens,F,1)},
        {name:"down_proj",cat:"linear",o:gemm(ctx.tokens,F,D.H,"mlp"),shard:ctx.t},
      ];
    },
  },
  // ---- MLA (DeepSeek latent attention, absorbed/MQA-over-latent) ----
  mla: {
    kind:"attn",
    label(){return "MLA attention (per device, %PH%) — latent KV";},
    flow(){return "MLA";},
    headDiv(dims){return dims.nh;},
    costRows(ctx,dims){
      const nh=dims.nh, qkn=dims.qk_nope, qkr=dims.qk_rope, qkh=qkn+qkr, vh=dims.v_head, kvl=dims.kv_lora, ql=dims.q_lora||0, t=ctx.t;
      const rows=[];
      // Q projection: flat (q_lora=0) or LoRA down/up + norm
      if(ql>0){ rows.push({cat:"linear", o:gemm(ctx.tokens,D.H,ql,"qkv"), shard:t});
        rows.push({cat:"norm", o:rms(ctx.tokens,ql)});
        rows.push({cat:"linear", o:gemm(ctx.tokens,ql,nh*qkh,"qkv"), shard:t}); }
      else rows.push({cat:"linear", o:gemm(ctx.tokens,D.H,nh*qkh,"qkv"), shard:t});
      // KV-A compress (latent kv_lora + rope), shared across heads (replicated); + norm
      rows.push({cat:"linear", o:gemm(ctx.tokens,D.H,kvl+qkr,"qkv")});
      rows.push({cat:"norm", o:rms(ctx.tokens,kvl)});
      // KV-B up-proj (absorbed W_UK/W_UV), per-head, head-sharded
      rows.push({cat:"linear", o:gemm(ctx.tokens,kvl,nh*(qkn+vh),"qkv"), shard:t});
      // rope on q_rope (nh*qkr) + k_rope (qkr, 1 latent head)
      rows.push({cat:"rope", o:rope(ctx.tokens,nh*qkr,qkr), shard:t});
      // absorbed attention: MQA over the latent (qk dim = kvl+qkr, v dim = kvl, nkv=1).
      // kvShare=true: the absorbed cache stores ONE shared latent per token (k==v), not separate K/V.
      const eff=ctx.ctxlen, inter=ctx.tokens*(ctx.decode?eff:eff/2);
      rows.push({cat:"attention", o:attention(Math.max(1,Math.floor(nh/t)),1,kvl+qkr,kvl,ctx.tokens,inter,ctx.decode?1:32,true)});
      // head-wise output gate g_proj (H->nh), then o_proj + TP reduce
      rows.push({cat:"linear", o:gemm(ctx.tokens,D.H,nh,"qkv"), shard:t});
      rows.push({cat:"o_proj", o:gemm(ctx.tokens,nh*vh,D.H,"o_proj"), shard:t});
      rows.push({cat:"o_proj", o:{flops:0,hbm:0,ici:rowReduce(ctx.tokens,D.H,ctx.L),peak:"bf16"}});
      rows.push({cat:"norm", o:rms(ctx.tokens,D.H), mult:2});
      rows.push({cat:"other", o:elt(ctx.tokens,D.H,2), mult:2});
      return rows;
    },
    chain(ctx,dims){
      const nh=dims.nh, qkn=dims.qk_nope, qkr=dims.qk_rope, qkh=qkn+qkr, vh=dims.v_head, kvl=dims.kv_lora, ql=dims.q_lora||0, t=ctx.t;
      const eff=ctx.ctxlen, inter=ctx.tokens*(ctx.decode?eff:eff/2);
      const oproj=gemm(ctx.tokens,nh*vh,D.H,"o_proj"); oproj.ici=rowReduce(ctx.tokens,D.H,ctx.L);
      return [
        {name:"q_proj",cat:"linear",o:ql>0?gemm(ctx.tokens,ql,nh*qkh,"qkv"):gemm(ctx.tokens,D.H,nh*qkh,"qkv"),shard:t},
        {name:"kv_a_proj (latent)",cat:"linear",o:gemm(ctx.tokens,D.H,kvl+qkr,"qkv")},
        {name:"kv_b_proj (W_UK/W_UV)",cat:"linear",o:gemm(ctx.tokens,kvl,nh*(qkn+vh),"qkv"),shard:t},
        {name:"rope",cat:"rope",o:rope(ctx.tokens,nh*qkr,qkr),shard:t},
        {name:"attention (latent MQA)",cat:"attention",o:attention(Math.max(1,Math.floor(nh/t)),1,kvl+qkr,kvl,ctx.tokens,inter,undefined,true)},
        {name:"o_proj +reduce",cat:"o_proj",o:oproj,shard:t},
      ];
    },
  },
  // ---- KDA (Kimi delta / gated-delta linear attention; recurrent state, no growing KV) ----
  kda: {
    kind:"attn",
    label(){return "KDA linear attention (per device, %PH%) — recurrent state";},
    flow(){return "KDA";},
    headDiv(dims){return dims.nh;},
    costRows(ctx,dims){
      const nh=dims.nh, hd=dims.hd, proj=nh*hd, conv=dims.conv, t=ctx.t, nqh=Math.max(1,Math.floor(nh/t));
      const rows=[];
      // q,k,v,f,g projections (H -> proj), head-sharded; b_proj (H -> nh) tiny gate
      for(let i=0;i<5;i++) rows.push({cat:"linear", o:gemm(ctx.tokens,D.H,proj,"qkv"), shard:t});
      rows.push({cat:"linear", o:gemm(ctx.tokens,D.H,nh,"qkv"), shard:t});
      // short causal depthwise conv on q,k,v (kernel=conv): ~elementwise
      rows.push({cat:"other", o:{flops:3*ctx.tokens*proj*conv*2, hbm:3*ctx.tokens*proj*2*2, ici:0, peak:"bf16"}});
      // gated delta recurrent: per token/head state d_k x d_v (=hd x hd); ~4 ops/tok/head.
      // HBM driver = recurrent state read+write (fp32). Decode reads+writes each request's
      // full [H,d_k,d_v] state once (scales with tokens=batch); prefill keeps it in VMEM
      // across the chunked scan (one init+final per sequence). No growing KV cache.
      const st=ctx.decode?ctx.tokens*nqh*hd*hd*4*2:nqh*hd*hd*4*2;
      rows.push({cat:"attention", o:{flops:ctx.tokens*nqh*4*hd*hd, hbm:ctx.tokens*nqh*hd*2*5 + st, ici:0, peak:"bf16"}});
      // o_norm (gated rms) + o_proj + TP reduce
      rows.push({cat:"norm", o:rms(ctx.tokens,proj)});
      rows.push({cat:"o_proj", o:gemm(ctx.tokens,proj,D.H,"o_proj"), shard:t});
      rows.push({cat:"o_proj", o:{flops:0,hbm:0,ici:rowReduce(ctx.tokens,D.H,ctx.L),peak:"bf16"}});
      rows.push({cat:"norm", o:rms(ctx.tokens,D.H), mult:2});
      rows.push({cat:"other", o:elt(ctx.tokens,D.H,2), mult:2});
      return rows;
    },
    chain(ctx,dims){
      const nh=dims.nh, hd=dims.hd, proj=nh*hd, t=ctx.t, nqh=Math.max(1,Math.floor(nh/t));
      const oproj=gemm(ctx.tokens,proj,D.H,"o_proj"); oproj.ici=rowReduce(ctx.tokens,D.H,ctx.L);
      return [
        {name:"qkv+fg proj (×5)",cat:"linear",o:{flops:5*2*ctx.tokens*D.H*proj, hbm:5*(2*D.H*proj+2*ctx.tokens*D.H+2*ctx.tokens*proj), ici:0, peak:"bf16"},shard:t},
        {name:"short conv (q,k,v)",cat:"other",o:{flops:3*ctx.tokens*proj*dims.conv*2, hbm:3*ctx.tokens*proj*2*2, ici:0, peak:"bf16"}},
        {name:"gated-delta recurrent",cat:"attention",o:{flops:ctx.tokens*nqh*4*hd*hd, hbm:ctx.tokens*nqh*hd*2*5+(ctx.decode?ctx.tokens*nqh*hd*hd*4*2:nqh*hd*hd*4*2), ici:0, peak:"bf16"}},
        {name:"o_norm",cat:"norm",o:rms(ctx.tokens,proj)},
        {name:"o_proj +reduce",cat:"o_proj",o:oproj,shard:t},
      ];
    },
  },
  // ---- EPMoE (expert-parallel MoE with explicit XLA all-to-all; Ling3 default backend) ----
  ep_moe: {
    kind:"ffn",
    label(){return "EPMoE experts (per device, per layer)";},
    flow(){return "MoE";},
    costRows(ctx,dims){
      const NEXP=dims.NEXP,TOPK=dims.TOPK,MOEF=dims.MOEF;
      const mt=ctx.tokens*ctx.L.dp, tpd=Math.max(1,Math.floor(mt*TOPK/ctx.ep));
      const e=moe(tpd, NEXP/ctx.ep, D.H, MOEF, "experts");
      // EPMoE replicates ALL mt tokens to every shard (reshard P(None)) and combines via
      // an expert-axis all-reduce of the FULL [mt,H] output (psum — NOT a routed-token
      // a2a like fused-v2); a separate TP all-reduce (rowReduce) completes the down-proj.
      e.hbm += 2*mt*D.H*2;  // full replicated output materialised for the psum
      e.ici=allreduce(mt*D.H*2,ctx.ep) + rowReduce(ctx.tokens,D.H,ctx.L);
      return [ {cat:"router", o:router(ctx.tokens,D.H,NEXP)}, {cat:"moe", o:e} ];
    },
    chain(ctx,dims){
      const NEXP=dims.NEXP,TOPK=dims.TOPK,MOEF=dims.MOEF;
      const mt=ctx.tokens*ctx.L.dp, tpd=Math.max(1,Math.floor(mt*TOPK/ctx.ep));
      const e=moe(tpd,NEXP/ctx.ep,D.H,MOEF,"experts"); e.hbm+=2*mt*D.H*2; e.ici=allreduce(mt*D.H*2,ctx.ep)+rowReduce(ctx.tokens,D.H,ctx.L);
      return [ {name:"router_gate",cat:"router",o:router(ctx.tokens,D.H,NEXP)}, {name:"experts +EP all-reduce (psum)",cat:"moe",o:e} ];
    },
  },
  // ---- shared expert (always-on dense MLP that runs alongside the routed experts) ----
  shared_expert: {
    kind:"ffn",
    label(){return "shared expert (dense MLP, per MoE layer)";},
    flow(){return "shared";},
    costRows(ctx,dims){
      const F=dims.SHF;
      return [
        {cat:"linear", o:gemm(ctx.tokens,D.H,2*F,"mlp"), shard:ctx.t},
        {cat:"linear", o:gemm(ctx.tokens,F,D.H,"mlp"), shard:ctx.t},
        {cat:"other", o:elt(ctx.tokens,F,1)},
      ];
    },
    chain(ctx,dims){
      const F=dims.SHF;
      return [
        {name:"shared gate_up",cat:"linear",o:gemm(ctx.tokens,D.H,2*F,"mlp"),shard:ctx.t},
        {name:"shared silu",cat:"other",o:elt(ctx.tokens,F,1)},
        {name:"shared down",cat:"linear",o:gemm(ctx.tokens,F,D.H,"mlp"),shard:ctx.t},
      ];
    },
  },
  // ---- fused-MoE-v1 (older fused MoE; same expert math as v2, explicit a2a) ----
  fused_moe_v1: {
    kind:"ffn",
    label(){return "fused-MoE-v1 experts (per device, per layer)";},
    flow(){return "MoE";},
    costRows(ctx,dims){
      const NEXP=dims.NEXP,TOPK=dims.TOPK,MOEF=dims.MOEF;
      const mt=ctx.tokens*ctx.L.dp, tpd=Math.max(1,Math.floor(mt*TOPK/ctx.ep)), remote=ctx.ep>1?(ctx.ep-1)/ctx.ep:0;
      const e=moe(tpd, NEXP/ctx.ep, D.H, MOEF, "experts");
      e.ici=2*(mt*TOPK/ctx.ep)*D.H*2*remote + rowReduce(ctx.tokens,D.H,ctx.L);
      return [ {cat:"router", o:router(ctx.tokens,D.H,NEXP)}, {cat:"moe", o:e} ];
    },
    chain(ctx,dims){
      const NEXP=dims.NEXP,TOPK=dims.TOPK,MOEF=dims.MOEF;
      const mt=ctx.tokens*ctx.L.dp, tpd=Math.max(1,Math.floor(mt*TOPK/ctx.ep)), remote=ctx.ep>1?(ctx.ep-1)/ctx.ep:0;
      const e=moe(tpd,NEXP/ctx.ep,D.H,MOEF,"experts"); e.ici=2*(mt*TOPK/ctx.ep)*D.H*2*remote+rowReduce(ctx.tokens,D.H,ctx.L);
      return [ {name:"router_gate",cat:"router",o:router(ctx.tokens,D.H,NEXP)}, {name:"experts +a2a",cat:"moe",o:e} ];
    },
  },
};

function compute(s){
  const L=resolve(s); L.sp=s.enable_sp;
  const decode = s.phase==="decode";
  const tokens = decode? s.batch : s.chunk;
  const ctxlen = decode? s.seq_len : s.chunk;
  const t=L.t, ep=L.ep;
  const ctx={tokens, ctxlen, decode, t, ep, L};
  const cat={};
  const add=(c,o,cnt,shard)=>{cnt=cnt||1;shard=shard||1; const e=cat[c]||(cat[c]={flops:0,hbm:0,ici:0,cnt:0,peak:o.peak});
    e.flops+=o.flops*cnt/shard; e.hbm+=o.hbm*cnt/shard; e.ici+=(o.ici||0)*cnt/shard; e.cnt+=cnt; if(o.peak==="fp8")e.peak="fp8";};
  const emit=(rows,count)=>{ if(count<=0)return; for(const r of rows) add(r.cat, r.o, (r.mult||1)*count, r.shard||1); };
  const comp=D.composition||{attn:[],ffn:[]};
  for(const a of (comp.attn||[])) emit(MODULES[a.m].costRows(ctx,a.dims), a.count);
  for(const f of (comp.ffn||[])) emit(MODULES[f.m].costRows(ctx,f.dims), f.count);
  // model head: embedding + final norm + lm_head (universal, not per-layer modules)
  add("embedding", elt(tokens,D.H,0), 1);
  add("norm", rms(tokens,D.H), 1);
  add("lm_head", gemm(s.batch,D.H,D.VOCAB,"lm_head"), 1, t);

  const rows=[]; let thbm=0,tici=0;
  for(const c in cat){const e=cat[c];
    const cms=e.flops/flops_per_s(e.peak)*1e3, hms=e.hbm/HBMBW*1e3, ims=e.ici/ICIBW*1e3;
    const ideal=Math.max(cms,hms,ims); const bound= ideal===ims&&ims>0?"ICI":(ideal===cms?"compute":"HBM");
    rows.push({cat:c,cnt:e.cnt,flops:e.flops,hbm:e.hbm,ici:e.ici,peak:e.peak,oi:e.hbm>0?e.flops/e.hbm:0,ideal:ideal,bound:bound});
    thbm+=e.hbm;tici+=e.ici;}
  const Tc=rows.reduce((a,r)=>a+r.flops/flops_per_s(r.peak),0)*1e3, Th=thbm/HBMBW*1e3, Ti=tici/ICIBW*1e3;
  const tot=Math.max(Tc,Th,Ti); const tbound= tot===Ti&&Ti>0?"ICI":(tot===Tc?"compute":"HBM");
  rows.sort((a,b)=>b.ideal-a.ideal); const sumIdeal=rows.reduce((a,x)=>a+x.ideal,0);
  for(const r of rows) r.pct=sumIdeal>0?r.ideal/sumIdeal*100:0;
  return {rows, L, tot, tbound, Tc, Th, Ti, decode, tokens};
}

function buildChain(s){
  const L=resolve(s); L.sp=s.enable_sp; const decode=s.phase==="decode";
  const tokens=decode?s.batch:s.chunk, ctxlen=decode?s.seq_len:s.chunk, t=L.t, ep=L.ep;
  const ctx={tokens, ctxlen, decode, t, ep, L};
  const comp=D.composition||{attn:[],ffn:[]};
  const ra=(comp.attn||[])[0], rf=(comp.ffn||[])[0];   // representative layer = first attn + first ffn
  const ch=[];
  const msof=(o,shard)=>{shard=shard||1; const cms=o.flops/shard/flops_per_s(o.peak)*1e3, hms=o.hbm/shard/HBMBW*1e3, ims=(o.ici||0)/ICIBW*1e3;
    const m=Math.max(cms,hms,ims); return {ms:m, bound:(m===ims&&ims>0)?"ICI":(m===cms?"compute":"HBM")};};
  const add=(name,cat,o,shard)=>{const r=msof(o,shard); ch.push({name,cat,ms:r.ms,bound:r.bound});};
  add("input_layernorm","norm",rms(tokens,D.H));
  if(ra&&MODULES[ra.m].chain) for(const r of MODULES[ra.m].chain(ctx,ra.dims)) add(r.name,r.cat,r.o,r.shard);
  add("+ residual","other",elt(tokens,D.H,2));
  add("post_attn_layernorm","norm",rms(tokens,D.H));
  if(rf&&MODULES[rf.m].chain) for(const r of MODULES[rf.m].chain(ctx,rf.dims)) add(r.name,r.cat,r.o,r.shard);
  add("+ residual","other",elt(tokens,D.H,2));
  return ch;
}

// ---------- roofline canvas (high-DPI, responsive, light) ----------
let LAST=null; const CH=520;
function draw(R){const cv=g("cv"); if(!cv)return; const cx=cv.getContext("2d");
  const dpr=window.devicePixelRatio||1, W=Math.max(480,(g("body").clientWidth||700));
  cv.style.width=W+"px"; cv.style.height=CH+"px"; cv.width=Math.round(W*dpr); cv.height=Math.round(CH*dpr); cx.setTransform(dpr,0,0,dpr,0,0);
  LAST=R; const Hh=CH, ml=82,mr=20,mt=52,mb=50;
  const tlabel=(txt,x,y,col)=>{const w=cx.measureText(txt).width; cx.fillStyle="rgba(255,255,255,0.9)"; cx.fillRect(x-2,y-10,w+4,13); cx.fillStyle=col||"#64748b"; cx.fillText(txt,x,y);};
  cx.clearRect(0,0,W,Hh);
  const rows=R.rows.filter(r=>r.flops>0&&r.hbm>0);
  const ceil=(rows.some(r=>r.peak==="fp8")?P.fp8_tflops:P.bf16_tflops);
  const oiv=rows.map(r=>r.oi), perfs=rows.map(r=>r.flops/(r.ideal/1e3)/1e12);
  let xmin=Math.min(...oiv)/3, xmax=Math.max(...oiv)*3; if(!(xmin>0))xmin=0.01;
  let ymax=ceil*2.2, ymin=Math.min(...perfs.filter(p=>p>0),ceil)/80; if(!isFinite(ymin)||ymin<=0)ymin=ceil/1000;
  const lx=v=>ml+(Math.log10(v)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin))*(W-ml-mr);
  const ly=v=>mt+(Math.log10(ymax)-Math.log10(v))/(Math.log10(ymax)-Math.log10(ymin))*(Hh-mt-mb);
  cx.strokeStyle="#eef1f5";cx.lineWidth=1;cx.font="11px sans-serif";
  for(let e=-3;e<=7;e++){const x=Math.pow(10,e); if(x<xmin||x>xmax)continue; cx.beginPath();cx.moveTo(lx(x),mt);cx.lineTo(lx(x),Hh-mb);cx.stroke(); cx.fillStyle="#889";cx.textAlign="center";cx.fillText("1e"+e,lx(x),Hh-mb+15);}
  cx.textAlign="right";
  for(let e=-3;e<=4;e++){const y=Math.pow(10,e); if(y<ymin||y>ymax)continue; cx.beginPath();cx.moveTo(ml,ly(y));cx.lineTo(W-mr,ly(y));cx.stroke(); cx.fillStyle="#889";cx.fillText("1e"+e,ml-8,ly(y)+3);}
  cx.textAlign="left";
  cx.fillStyle="#445";cx.font="12px sans-serif";cx.textAlign="center";cx.fillText("operational intensity (FLOP / HBM-byte)",(ml+W-mr)/2,Hh-8);cx.textAlign="left";
  cx.save();cx.translate(20,(mt+Hh-mb)/2);cx.rotate(-Math.PI/2);cx.textAlign="center";cx.fillText("attainable TFLOP/s",0,0);cx.restore();cx.textAlign="left";
  cx.strokeStyle="#334155";cx.lineWidth=2.5;cx.beginPath();let first=true;
  for(let i=0;i<=160;i++){const x=xmin*Math.pow(xmax/xmin,i/160); const y=Math.min(x*HBMBW/1e12,ceil); const px=lx(x),py=ly(y); if(first){cx.moveTo(px,py);first=false;}else cx.lineTo(px,py);} cx.stroke();
  cx.setLineDash([6,4]);cx.strokeStyle="#94a3b8";cx.lineWidth=1.2;
  cx.beginPath();cx.moveTo(ml,ly(P.bf16_tflops));cx.lineTo(W-mr,ly(P.bf16_tflops));cx.stroke();
  cx.beginPath();cx.moveTo(ml,ly(P.fp8_tflops));cx.lineTo(W-mr,ly(P.fp8_tflops));cx.stroke();
  cx.setLineDash([]); cx.font="11px sans-serif";
  tlabel("bf16 "+P.bf16_tflops.toFixed(0)+" TF/s", ml+8, ly(P.bf16_tflops)-4);
  tlabel("fp8 "+P.fp8_tflops.toFixed(0)+" TF/s", ml+8, ly(P.fp8_tflops)-4);
  const ridge=ceil/(HBMBW/1e12); if(ridge>xmin&&ridge<xmax){cx.strokeStyle="#cbd5e1";cx.lineWidth=1;cx.beginPath();cx.moveTo(lx(ridge),mt);cx.lineTo(lx(ridge),Hh-mb);cx.stroke(); tlabel("ridge OI="+ridge.toFixed(0), Math.min(lx(ridge)+4, W-mr-86), mt-6, "#94a3b8");}
  const smax=Math.max(...rows.map(r=>r.ideal))||1; R._pts=[];
  for(const r of rows){const x=r.oi, y=r.flops/(r.ideal/1e3)/1e12; const px=lx(x),py=ly(y); const rad=6+15*(r.ideal/smax);
    cx.fillStyle=CAT[r.cat]||"#888";
    if(r.bound==="ICI"){cx.save();cx.translate(px,py);cx.rotate(Math.PI/4);cx.lineWidth=3.5;cx.strokeStyle=CAT[r.cat]||"#888";cx.beginPath();cx.moveTo(-rad,0);cx.lineTo(rad,0);cx.moveTo(0,-rad);cx.lineTo(0,rad);cx.stroke();cx.restore();}
    else{cx.strokeStyle="#fff";cx.lineWidth=1.5;cx.beginPath();cx.arc(px,py,rad,0,7);cx.fill();cx.stroke();}
    R._pts.push({px,py,rad,r});
    if(r.ideal>0.03*smax){cx.fillStyle="#1c2330";cx.font="11px sans-serif";const tx=px+rad+4>W-mr-50?px-rad-4-cx.measureText(r.cat).width:px+rad+4;cx.fillText(r.cat,tx,py+4);}
  }
}
function fmt(x){const a=Math.abs(x); if(a===0)return "0"; if(a>=100)return x.toFixed(0); if(a>=1)return x.toFixed(1); if(a>=0.01)return x.toFixed(2); if(a>=0.0001)return x.toFixed(4); return x.toExponential(1);}
// ---------- scenario lenses (expert task-oriented views) ----------
function rowMs(r){return {c:r.flops/flops_per_s(r.peak)*1e3, h:r.hbm/HBMBW*1e3, i:r.ici/ICIBW*1e3};}
function ridgeOI(peak){return (peak==="fp8"?P.fp8_tflops:P.bf16_tflops)/(HBMBW/1e12);}
function lensOverlap(s,R){
  const L=R.L, ep=L.ep, tokens=R.tokens, mt=tokens*L.dp, remote=ep>1?(ep-1)/ep:0, msi=b=>b/ICIBW*1e3;
  const comp=D.composition||{attn:[],ffn:[]};
  const attnN=(comp.attn||[]).reduce((a,x)=>a+x.count,0);
  const moeMods=(comp.ffn||[]).filter(f=>/moe/.test(f.m)), denseMods=(comp.ffn||[]).filter(f=>f.m==="dense");
  const nMoe=moeMods.reduce((a,x)=>a+x.count,0), nDense=denseMods.reduce((a,x)=>a+x.count,0);
  const items=[];
  if(nMoe>0){
    const md=moeMods[0].dims, NEXP=md.NEXP, TOPK=md.TOPK, MOEF=md.MOEF;
    const a2aB=2*(mt*TOPK/ep)*D.H*2*remote, tpd=Math.max(1,Math.floor(mt*TOPK/ep));
    const e=moe(tpd,NEXP/ep,D.H,MOEF,"experts");
    // the a2a can pipeline behind the experts kernel's execution = its ideal time
    // (compute & HBM overlapped inside the kernel)
    const expMs=Math.max(e.flops/flops_per_s(e.peak), e.hbm/HBMBW)*1e3*nMoe;
    items.push({name:"MoE all-to-all (dispatch + combine)",op:"kernel",ms:msi(a2aB)*nMoe,type:"pipelineable",cap:expMs,behind:"experts kernel "+expMs.toFixed(1)+" ms"});
    items.push({name:"MoE output all-reduce (TP, tensor axis)",op:"all-reduce",ms:msi(rowReduce(tokens,D.H,L))*nMoe,type:"barrier",cap:0,behind:"—"});
  }
  if(attnN>0) items.push({name:"o_proj all-reduce (TP, tensor axis)",op:"all-reduce",ms:msi(rowReduce(tokens,D.H,L))*attnN,type:"barrier",cap:0,behind:"—"});
  if(nDense>0) items.push({name:"down_proj all-reduce (TP, tensor axis)",op:"all-reduce",ms:msi(rowReduce(tokens,D.H,L))*nDense,type:"barrier",cap:0,behind:"—"});
  // SP re-gather: async all-gather (one per block before its input linear) that XLA
  // overlaps with the linear -> hidden comm, not an exposed barrier (HLO ground truth).
  {const agB=spGather(tokens,D.H,L); if(agB>0) items.push({name:"SP all-gather (re-collect seq before linears)",op:"all-gather",ms:msi(agB)*(attnN+nMoe+nDense),type:"pipelineable",cap:1e9,behind:"input linears (XLA async)"});}
  // embedding lookup all-reduce (vocab-sharded embed gather over the tensor axis); once per step
  if(L.t>1) items.push({name:"embedding all-reduce (vocab-sharded)",op:"all-reduce",ms:msi(allreduce(tokens*D.H*2,L.t)),type:"barrier",cap:0,behind:"—"});
  let hidden=0,exposed=0,commTot=0;
  for(const it of items){if(it.type==="pipelineable"){it.hidden=Math.min(it.ms,it.cap);it.exposed=it.ms-it.hidden;}else{it.hidden=0;it.exposed=it.ms;} hidden+=it.hidden;exposed+=it.exposed;commTot+=it.ms;}
  const nonComm=Math.max(R.Tc,R.Th), pipeStep=nonComm+exposed, noOv=nonComm+commTot;
  let h="<div class='lh'>Overlap — comm hidden behind compute, or exposed?</div>";
  h+="<div class='note'>Each collective is classified by whether it can pipeline behind adjacent compute: MoE a2a can hide inside the fused-expert kernel; TP reduces are layer-boundary barriers. step ≈ max(ΣC,ΣH) + <b>exposed</b> comm.</div>";
  // comm budget stacked bar
  const cmx=Math.max(commTot,1e-9);
  h+="<div class='dfrow'><div class='nm'>comm budget ΣICI</div><div class='barwrap' style='display:flex'>"
    +"<div class='bar' style='width:"+(hidden/cmx*100)+"%;background:#22c55e' title='hidden'></div>"
    +"<div class='bar' style='width:"+(exposed/cmx*100)+"%;background:#ec4899' title='exposed'></div></div>"
    +"<div class='ms'>"+commTot.toFixed(3)+" ms</div></div>";
  h+="<div class='note'><span style='color:#16a34a'>■</span> hidden "+hidden.toFixed(3)+" ms &nbsp; <span style='color:#db2777'>■</span> exposed "+exposed.toFixed(3)+" ms</div>";
  // step = compute/HBM wall + exposed comm (single decomposed bar)
  const W=Math.max(pipeStep,1e-9);
  h+="<div class='note' style='margin-top:8px'><b>step ≈ "+pipeStep.toFixed(2)+" ms</b> = compute/HBM wall + exposed comm (ΣC "+R.Tc.toFixed(2)+" / ΣH "+R.Th.toFixed(2)+" / ΣI "+commTot.toFixed(2)+" ms)</div>";
  h+="<div class='dfrow'><div class='nm'>step breakdown</div><div class='barwrap' style='display:flex'>"
    +"<div class='bar' style='width:"+(nonComm/W*100)+"%;background:#3b82f6' title='compute/HBM wall'></div>"
    +"<div class='bar' style='width:"+(exposed/W*100)+"%;background:#ec4899' title='exposed comm'></div></div>"
    +"<div class='ms'>"+pipeStep.toFixed(2)+" ms</div></div>";
  h+="<div class='note'><span style='color:#2563eb'>■</span> compute/HBM wall = max(ΣC,ΣH) = <b>"+nonComm.toFixed(2)+" ms</b> &nbsp; <span style='color:#db2777'>■</span> exposed comm <b>"+exposed.toFixed(2)+" ms</b> &nbsp;·&nbsp; overlap already hides "+hidden.toFixed(2)+" ms of comm.</div>";
  // three reference step estimates (perfect overlap = the lower bound)
  h+="<div class='note' style='background:#f1f5f9;border-radius:6px;padding:6px 9px'>reference step estimates: "
    +"&nbsp;<b>perfect overlap</b> (all engines, lower bound) = max(ΣC,ΣH,ΣI) = <b>"+Math.max(R.Tc,R.Th,commTot).toFixed(1)+" ms</b>"
    +"&nbsp;·&nbsp; pipeline model (this bar) = "+pipeStep.toFixed(1)+" ms"
    +"&nbsp;·&nbsp; no overlap (comm serial) = "+noOv.toFixed(1)+" ms</div>";
  // verdict — lead with the robust ΣI-vs-wall comparison (model-independent)
  const floor=Math.max(R.Tc,R.Th,commTot);  // perfect-overlap lower bound
  if(commTot>nonComm) h+="<div class='verdict v-warn'><b>ICI / comm-bound.</b> ΣI ("+commTot.toFixed(0)+" ms) &gt; compute/HBM wall ("+nonComm.toFixed(0)+" ms): even <b>perfect</b> overlap can't go below the comm time, so step ≥ <b>"+floor.toFixed(0)+" ms</b> regardless of scheduling. Overlap is <b>not</b> the lever — you must <b>reduce comm</b> (the MoE a2a): smaller prefill chunk, EP locality / topology, or fewer cross-host hops.</div>";
  else if(exposed<0.02*Math.max(nonComm,1e-9)) h+="<div class='verdict v-go'>Exposed comm ≈ <b>"+exposed.toFixed(3)+" ms</b> (≪ "+nonComm.toFixed(2)+" ms compute/HBM) → comm is <b>not</b> the bottleneck; step stays "+R.tbound+"-bound. Overlap won't move the needle — cut "+(R.Th>=R.Tc?"HBM bytes":"flops")+".</div>";
  else h+="<div class='verdict v-warn'>Exposed comm ≈ <b>"+exposed.toFixed(2)+" ms</b> on top of the "+nonComm.toFixed(2)+" ms compute/HBM floor ("+(exposed/pipeStep*100).toFixed(0)+"% of step). Lever: hide the a2a (kernel pipelining / async) or cut barriers (SP, topology, EP locality).</div>";
  // per-collective table — MERGED: model classification + XLA ground truth (HLO) per row
  const HV=(D.hlo&&D.hlo.network&&D.hlo.network.by_type)||null;
  function xlaCell(op,type){
    if(op==="kernel") return "<span style='color:#d97706;font-weight:700'>⚠</span> in-kernel (SparseCore), not XLA-scheduled · <b>measured exposed</b>";
    if(!HV) return "<span style='color:#aab'>—</span>";
    const t=HV[op]; if(!t) return "<span style='color:#aab'>absent in HLO</span>";
    const sy=t.sync||0, as=t.async_||0, isAsync=as>0&&sy===0, isSync=sy>0&&as===0;
    const verdict=isAsync?"ASYNC (overlapped)":isSync?"SYNC (barrier)":sy+" sync / "+as+" async";
    const ok=type==="barrier"?isSync:isAsync, c=ok?"#16a34a":"#d97706";
    return "<span style='color:"+c+";font-weight:700'>"+(ok?"✓":"⚠")+"</span> "+verdict;
  }
  h+="<table style='margin-top:10px'><thead><tr><th class='l'>collective</th><th>ICI ms</th><th class='l'>type (model)</th><th>hidden</th><th>exposed</th><th class='l'>hides behind</th>"+(HV?"<th class='l'>XLA actual (HLO)</th>":"")+"</tr></thead><tbody>";
  for(const it of items.sort((a,b)=>b.ms-a.ms)) h+="<tr><td class='l'>"+it.name+"</td><td>"+it.ms.toFixed(3)+"</td><td class='l'><span class='tag "+(it.type==="pipelineable"?"b-compute":"b-ICI")+"'>"+it.type+"</span></td><td>"+it.hidden.toFixed(3)+"</td><td>"+it.exposed.toFixed(3)+"</td><td class='l' style='font-size:11px;color:#667'>"+it.behind+"</td>"+(HV?"<td class='l' style='font-size:11px'>"+xlaCell(it.op,it.type)+"</td>":"")+"</tr>";
  if(!items.length) h+="<tr><td class='l' colspan="+(HV?7:6)+">no collectives at this layout</td></tr>";
  h+="</tbody></table>";
  const cm=D.hlo&&D.hlo.compile, spon=D.hlo&&D.hlo.network&&D.hlo.network.sp_active;
  h+="<div class='note'>type = model prediction (can it pipeline behind compute?); <b>XLA actual</b> = what the compiled HLO actually scheduled (✓ agrees · ⚠ model says hideable but it is exposed / not XLA-scheduled)"+(cm?", from the compiled HLO ("+cm.n_layers_compiled+" representative layers · "+cm.tokens_global+" tokens · SP "+(spon?"on":"off")+")":"")+". Rows are model <b>categories</b> (each spans all its layers) while the HLO counts opcodes — so the all-reduce rows all map to the one all-reduce opcode. The a2a is in-kernel (XLA can't touch it) and is <b>measured exposed</b> at the torus floor (cross-host) / VMEM-blocked; a device trace is the final word on how much compute hides in its shadow.</div>";
  return h;}
function lensKernel(s,R){
  let h="<div class='lh'>Kernel — which to attack, and how</div><div class='note'>Ranked by ideal ms. Bound → lever: HBM → ↓ bytes; compute → ↑ MXU rate / ↓ flops; ICI → overlap / ↓ comm.</div>";
  h+="<table><thead><tr><th class='l'>op</th><th>ideal ms</th><th>%step</th><th>bound</th><th>OI</th><th class='l'>lever</th></tr></thead><tbody>";
  for(const r of R.rows){const m=rowMs(r); let lever;
    if(r.bound==="HBM"){const rg=ridgeOI(r.peak); lever="↓ bytes: quantize (knobs) / layout / fewer materializations · compute "+m.c.toFixed(3)+" ms idle · OI "+r.oi.toFixed(0)+", need ≥ "+rg.toFixed(0)+" to flip compute-bound";}
    else if(r.bound==="compute") lever="↑ MXU rate (non-block W8A8) or ↓ flops · right of ridge";
    else lever="↓ / overlap comm (SP / topology) · "+(m.c+m.h).toFixed(3)+" ms could hide it";
    h+="<tr><td class='l'><span style='color:"+(CAT[r.cat]||'#888')+"'>●</span> "+r.cat+(r.peak==='fp8'?" <span class='tag' style='background:#fef3c7;color:#92400e'>fp8</span>":"")+"</td><td>"+r.ideal.toFixed(3)+"</td><td>"+r.pct.toFixed(0)+"%</td><td><span class='tag b-"+r.bound+"'>"+r.bound+"</span></td><td>"+r.oi.toFixed(1)+"</td><td class='l' style='font-size:11px'>"+lever+"</td></tr>";}
  h+="</tbody></table>";
  h+=kernelTune(s,R);
  return h;}
// ---------- per-kernel tuning deep-dive (Pallas: fused-MoE-v2, RPA attention) ----------
function blockOf(re,keys){const K=(D.codepath&&D.codepath.kernels)||[]; const k=K.find(x=>re.test(x.name||"")); const o={name:k?k.name:""};
  for(const key of keys){const m=k&&(k.name||"").match(new RegExp(key+"_(\\d+)")); o[key]=m?+m[1]:0;} return o;}
// adaptive time format: keep ms (3 dp) for >=1us, drop to us/ns below so per-layer
// per-device decode kernels (sub-us at small batch) read as e.g. "270 ns" not "0.000 ms".
function fms(ms){ if(!(ms>0)) return "0 ms"; if(ms>=0.001) return ms.toFixed(3)+" ms"; const ns=ms*1e6; return ns>=1000?(ns/1000).toFixed(2)+" µs":ns.toFixed(0)+" ns"; }
function kbar(lab,ms,tot,col,sub){return "<div class='dfrow'><div class='nm' style='flex-basis:160px'>"+lab+"</div><div class='barwrap'><div class='bar' style='width:"+Math.max(1,ms/Math.max(tot,1e-9)*100)+"%;background:"+col+"'></div></div><div class='ms'>"+fms(ms)+(sub?" · "+sub:"")+"</div></div>";}
// ---------- per-kernel tuning cards (generic: each module renders its own) ----------
// tuneCard = generic engine-split card (HBM weight+act | compute MXU | comm ICI parts;
// ideal=max; + OI/ridge + VMEM-fit + bound-aware lever). Modules call it from .card().
function tuneCard(title,blk,wB,aB,flops,iciParts,peakKind,vmem,levFn){
  const P=D.peaks, HB=HBMBW, ICI=ICIBW, VMEM=(P.vmem_mb||64)*1e6;
  const peak=(peakKind==="fp8"?P.fp8_tflops:P.bf16_tflops);
  const iciB=iciParts.reduce((a,p)=>a+p[1],0);
  const wms=wB/HB*1e3, ams=aB/HB*1e3, hms=wms+ams, cms=flops/(peak*1e12)*1e3, ims=iciB/ICI*1e3;
  const ideal=Math.max(hms,cms,ims), bound=ideal===ims&&ims>0?"ICI":(ideal===cms?"compute":"HBM");
  const oi=flops/(wB+aB), ridge=peak*1e12/HB;
  const parts=[["weight HBM",wms,"#d62728"],["act HBM",ams,"#f59e0b"],["compute (MXU)",cms,"#22c55e"]];
  for(const p of iciParts) parts.push([p[0],p[1]/ICI*1e3,p[2]]);
  const tot=Math.max(...parts.map(p=>p[1]));
  let c="<div class='panel' style='margin:8px 0;box-shadow:none'><div style='font-weight:700'>"+title+" — <span class='tag b-"+(bound==="HBM"?"HBM":bound==="compute"?"compute":"ICI")+"'>"+bound+"-bound</span> ideal "+fms(ideal)+"</div><div class='note mono'>block: "+blk+"</div>";
  for(const p of parts) c+=kbar(p[0],p[1],tot,p[2]);
  c+="<div class='note' style='margin-top:4px'>HBM = weight+act = "+fms(hms)+" · compute "+fms(cms)+" · comm (ICI) "+fms(ims)+" → ideal = max = <b>"+fms(ideal)+"</b>. OI = <b>"+oi.toFixed(1)+"</b> · ridge "+ridge.toFixed(0)+" → MXU at "+Math.min(100,oi/ridge*100).toFixed(0)+"% "+peakKind+" peak.</div>";
  c+="<div class='note'>VMEM working set ≈ <b>"+(vmem/1e6).toFixed(1)+" MB</b> (kernel's own estimate) / "+(VMEM/1e6).toFixed(0)+" MB "+(vmem>VMEM?"<span class='tag b-ICI'>over budget — would spill</span>":"<span class='tag b-compute'>fits ("+((VMEM-vmem)/1e6).toFixed(0)+" MB headroom)</span>")+"</div>";
  c+="<div class='verdict "+(bound==="compute"?"v-go":"v-warn")+"'>"+levFn(bound,{wB,aB,flops,iciB,oi,ridge,hms,ims})+"</div>";
  return c+"</div>";
}

MODULES.rpa.card = function(ctx,dims,R){
  const t=ctx.t, tokens=ctx.tokens, d=dims;
  const nq=Math.max(1,Math.floor(d.nh/t)), nkv=kvpd(d.nkv,t), hd=d.hd, vhd=d.vhd;
  const cl=R.decode?ctx.seq_len:tokens, eff=d.window?Math.min(cl,d.window):cl, inter=tokens*(R.decode?eff:eff/2);
  const o=attention(nq,nkv,hd,vhd,tokens,inter,R.decode?1:32), flops=o.flops;
  const kvB=Math.floor(inter/(R.decode?1:32))*nkv*2*hd*2, qoB=o.hbm-kvB;
  const stg=R.decode?"d":"m", sw=d.window?d.window:"full", RB=D.rpa_blocks||{};
  const rk=RB[t+"|"+stg+"|"+sw]||(R.decode?null:RB[t+"|p|"+sw]);
  let bq,bkv,blkLbl,vmem;
  if(rk&&rk.length){const rb=rk.find(e=>e.n>=tokens)||rk[rk.length-1]; bq=rb.bq; bkv=rb.bkv;
    vmem=rb.vmem||(bq*hd*2 + bkv*hd*2*2 + bq*bkv*4);
    blkLbl="tuned @ max_num_tokens="+rb.n+" (t="+t+", "+(R.decode?"decode/d":"prefill/m")+(d.window?", SWA "+d.window:"")+") → bq="+bq+" bkv="+bkv;}
  else {const blk=blockOf(/RPA[dm]-/,["bq","bkv","p"]); bq=blk.bq||16; bkv=blk.bkv||512;
    vmem=bq*hd*2 + bkv*hd*2*2 + bq*bkv*4;
    blkLbl=(blk.name||"n/a")+" (traced; no tuned entry for t="+t+"/"+stg+"/"+sw+")";}
  const name=d.window?("SWA (window="+d.window+")"):"full";
  return tuneCard("RPA attention — "+name+" (per device, "+(R.decode?"decode":"prefill")+")", blkLbl, kvB, qoB, flops, [], "bf16", vmem,
    (bound,x)=>{
      if(bound==="HBM") return "<b>KV-read-bound.</b> KV-cache read ≈ "+fmt(x.wB/1e9)+" GB dominates"+(d.window?" (window "+d.window+" caps ctx→"+eff+")":"")+". Levers: ① fp8 KV cache (½ read); ② fewer KV heads / GQA (nkv/dev="+nkv+")"+(d.window?"":"; ③ shorter context / SWA")+". bq/bkv tune VMEM + MXU util, not the KV bytes (workload-fixed).";
      return "<b>compute-bound.</b> Levers: ↑ MXU util (bq/bkv tiling), or it's just cheap (decode attention often is).";
    });
};

MODULES.fused_moe_v2.card = function(ctx,dims,R){
  const t=ctx.t, ep=ctx.ep, L=ctx.L, tokens=ctx.tokens;
  const E=dims.NEXP/ep, d=D.H, f=dims.MOEF, mt=tokens*L.dp, tpd=Math.max(1,Math.floor(mt*dims.TOPK/ep)), q=Q.wq!=="bf16";
  const wB=E*(q?(2*wbytes(d,f)+wbytes(f,d)):(2*2*d*f+2*f*d)), aB=2*tpd*d*2, flops=2*tpd*3*d*f;
  const remote=ep>1?(ep-1)/ep:0, a2aB=2*(mt*dims.TOPK/ep)*d*2*remote;  // dispatch + combine a2a (in-kernel)
  const reshardB=rowReduce(tokens,d,L);  // post-kernel TP all-reduce (tensor axis); SP all-gather is async/hidden
  const iciParts=[["a2a (in-kernel)",a2aB,"#ec4899"],["output all-reduce (TP)",reshardB,"#db2777"]];
  const tbl=(D.moe_blocks&&(D.moe_blocks[ep]||D.moe_blocks[String(ep)]))||null;
  const tb=tbl&&tbl.length?(tbl.find(e=>e.n>=mt)||tbl[tbl.length-1]):null;
  const bt=tb?tb.bt:16, bf=tb?tb.bf:512;
  const blkLbl=tb?("tuned @ num_tokens="+tb.n+" (chunk×dp="+mt+") → bt="+bt+" bf="+bf):("bt="+bt+" bf="+bf+" (default; no tuned entry)");
  const vmem=(tb&&tb.vmem)?tb.vmem:(bf*d*(q?1:2)+bt*d*2+bt*bf*4);  // kernel estimate; fallback rough
  return tuneCard("fused-MoE-v2 experts (per device, per layer)", blkLbl, wB, aB, flops, iciParts, wpeak(), vmem,
    (bound,x)=>{
      if(bound==="ICI") return "<b>comm-bound (ICI).</b> a2a (in-kernel, SparseCore) "+(a2aB/ICIBW*1e3).toFixed(3)+" ms + output all-reduce (TP, tensor axis) "+(reshardB/ICIBW*1e3).toFixed(3)+" ms = "+x.ims.toFixed(3)+" ms &gt; HBM "+x.hms.toFixed(3)+" ms. a2a is measured exposed at the torus floor; the all-reduce is a SYNC barrier over the tensor axis (HLO-verified). The SP all-gather that re-collects the sequence is async/hidden, not counted here. Levers: EP locality / topology / fewer cross-host hops; smaller chunk shrinks a2a; raise t-axis locality for the all-reduce.";
      if(bound==="HBM") return "<b>weight-HBM-bound.</b> weights "+fmt(x.wB/1e9)+" GB of "+fmt((x.wB+x.aB)/1e9)+" GB ("+(x.wB/(x.wB+x.aB)*100).toFixed(0)+"%), read once. Levers: ① fp8 weights (quant knob; block-fp8 caps at bf16 MXU); ② more EP (↓ local experts E="+E.toFixed(0)+"); ③ raise OI ("+x.oi.toFixed(0)+"→ridge "+x.ridge.toFixed(0)+") via more tokens/expert (tpd="+tpd+", bigger batch/chunk). bt/bf (tuned for num_tokens="+(tb?tb.n:"?")+") set VMEM + MXU util, not the byte budget.";
      return "<b>compute-bound.</b> Above the ridge — lever: ↑ MXU rate (non-block W8A8) or ↓ flops.";
    });
};

MODULES.mla.card = function(ctx,dims,R){
  const t=ctx.t, tokens=ctx.tokens, d=dims;
  const nq=Math.max(1,Math.floor(d.nh/t)), lat=d.kv_lora+d.qk_rope;
  const cl=R.decode?ctx.seq_len:tokens, inter=tokens*(R.decode?cl:cl/2);
  const o=attention(nq,1,lat,d.kv_lora,tokens,inter,R.decode?1:32,true), flops=o.flops;
  const kvB=Math.floor(inter/(R.decode?1:32))*1*1*lat*2, qoB=o.hbm-kvB;   // single shared latent (kvShare)
  const bq=R.decode?1:32, bkv=512, vmem=bq*lat*2 + bkv*lat*2 + bq*bkv*4;  // rough; no tuned MLA table
  return tuneCard("MLA attention — latent KV (per device, "+(R.decode?"decode":"prefill")+")",
    "bq="+bq+" bkv="+bkv+" (theory; latent dim "+lat+", v "+d.kv_lora+")", kvB, qoB, flops, [], "bf16", vmem,
    (bound,x)=>{
      if(bound==="HBM") return "<b>latent-KV-read-bound.</b> the absorbed cache stores ONE shared latent ("+lat+" = kv_lora "+d.kv_lora+" + rope "+d.qk_rope+") per token, MQA across all "+nq+" q-heads/device — read ≈ "+fmt(x.wB/1e9)+" GB. This compression IS the MLA win (vs "+d.nh+"-head MHA). Levers: ① fp8 latent cache (½); ② shorter context. Already MQA (nkv=1) + low-rank.";
      return "<b>compute-bound.</b> absorbed Q·W_UK + score·W_UV folds dominate; cheap at decode.";
    });
};

MODULES.kda.card = function(ctx,dims,R){
  const t=ctx.t, tokens=ctx.tokens, nh=dims.nh, hd=dims.hd, nqh=Math.max(1,Math.floor(nh/t));
  const flops=tokens*nqh*4*hd*hd;
  const stB=(R.decode?tokens:1)*nqh*hd*hd*4*2;   // recurrent state read+write (fp32)
  const actB=tokens*nqh*hd*2*5;
  const vmem=nqh*hd*hd*4 + tokens*nqh*hd*2;       // state + activation working set
  return tuneCard("KDA recurrent — gated-delta (per device, "+(R.decode?"decode":"prefill")+")",
    "chunk scan (prefill) / recurrent (decode); state d_k×d_v="+hd+"×"+hd+"/head, "+nqh+" heads/dev", stB, actB, flops, [], "bf16", vmem,
    (bound,x)=>{
      if(bound==="HBM") return "<b>recurrent-state-bound.</b> reads+writes the "+nqh+"×"+hd+"×"+hd+" fp32 state per request ≈ "+fmt(x.wB/1e9)+" GB"+(R.decode?" (×"+tokens+" decode requests)":"")+". FIXED per token — no growing KV cache (the linear-attention win for long context). Levers: ① bf16 state; ② fewer heads; ③ larger chunk (prefill) keeps state in VMEM.";
      return "<b>compute-bound.</b> the O(d_k·d_v) delta-rule update per token; small vs the q/k/v/f/g projections (see the linear rows).";
    });
};

MODULES.ep_moe.card = function(ctx,dims,R){
  const t=ctx.t, ep=ctx.ep, L=ctx.L, tokens=ctx.tokens;
  const E=dims.NEXP/ep, d=D.H, f=dims.MOEF, mt=tokens*L.dp, tpd=Math.max(1,Math.floor(mt*dims.TOPK/ep)), q=Q.wq!=="bf16";
  const wB=E*(q?(2*wbytes(d,f)+wbytes(f,d)):(2*2*d*f+2*f*d)), aB=2*tpd*d*2 + 2*mt*d*2, flops=2*tpd*3*d*f;
  const arB=allreduce(mt*d*2,ep), reshardB=rowReduce(tokens,d,L);
  const iciParts=[["EP all-reduce (psum, full output)",arB,"#ec4899"],["output all-reduce (TP)",reshardB,"#db2777"]];
  const vmem=f*d*(q?1:2)+tpd*d*2;
  return tuneCard("EPMoE — megablox GMM (grouped matmul) + XLA all-reduce (per device, per layer)",
    "megablox GMM ×3 (gate/up/down, grouped); the only Pallas kernel — reshard(P(None)) + sort/permute + psum are XLA, NOT a fused kernel", wB, aB, flops, iciParts, wpeak(), vmem,
    (bound,x)=>{
      if(bound==="ICI") return "<b>comm-bound (ICI).</b> EPMoE is XLA-orchestrated: it replicates ALL "+mt+" tokens (reshard P(None)) and combines via an expert-axis all-reduce of the FULL [mt,H] output ("+(arB/ICIBW*1e3).toFixed(3)+" ms) + the TP all-reduce ("+(reshardB/ICIBW*1e3).toFixed(3)+" ms) = "+x.ims.toFixed(3)+" ms — plain XLA collectives, not in-kernel. ≈ ep/topk = "+(ep/dims.TOPK).toFixed(0)+"× heavier than fused-v2's in-kernel routed a2a. Lever: <b>switch to the fused-v2 backend</b>, or fewer EP / better topology.";
      if(bound==="HBM") return "<b>GMM-weight-HBM-bound.</b> the megablox GMM streams "+fmt(x.wB/1e9)+" GB of expert weights once + the full replicated [mt,H] output materialised for the psum. Levers: ① fp8 weights; ② more EP (↓ local experts E="+E.toFixed(0)+"); ③ fused-v2 avoids the full-token replication.";
      return "<b>compute-bound.</b> the megablox GMM grouped matmul — ↑ MXU rate (non-block W8A8) / GMM tiling, or ↓ flops.";
    });
};

function kernelTune(s,R){
  let h="<div class='lh' style='margin-top:14px'>Pallas kernel tuning — per kernel/layer-type</div>";
  h+="<div class='note'>Each kernel's time split into the three engines (HBM = weight+act, same bandwidth → they sum; compute on MXU; comm on ICI → parallel axes). <b>ideal = max(HBM, compute, comm)</b> and the longest bar is the bound. Theory gives this ceiling + the VMEM-fit; whether the kernel hits it (tiling / MXU util / pipelining) needs a device trace.</div>";
  const comp=D.composition||{attn:[],ffn:[]};
  const ctx={tokens:R.decode?s.batch:s.chunk, decode:R.decode, t:R.L.t, ep:R.L.ep, L:R.L, seq_len:s.seq_len, chunk:s.chunk};
  for(const e of (comp.ffn||[]).concat(comp.attn||[])){
    const mod=MODULES[e.m]; if(!mod||!mod.card) continue;
    h += mod.card(ctx, e.dims, R);
  }
  return h;}
function lensFusion(s,R){
  const decode=s.phase==="decode", tokens=decode?s.batch:s.chunk, H=D.H;
  const comp=D.composition||{attn:[],ffn:[]};
  const attnN=(comp.attn||[]).reduce((a,x)=>a+x.count,0);
  const repAttn=(comp.attn||[])[0], qsz=repAttn?repAttn.dims.nh*repAttn.dims.hd:0;
  const moeMods=(comp.ffn||[]).filter(f=>/moe/.test(f.m)), denseMods=(comp.ffn||[]).filter(f=>f.m==="dense");
  const nMoe=moeMods.reduce((a,x)=>a+x.count,0), nDense=denseMods.reduce((a,x)=>a+x.count,0);
  const DENSE_F=(denseMods[0]||{dims:{}}).dims.DENSE_F||D.DENSE_F;
  const f=(D.hlo&&D.hlo.fusion)||null, ko=(f&&f.by_kind&&f.by_kind.kOutput)||0, ki=(f&&f.by_kind&&f.by_kind.kInput)||0;
  // status of a candidate fusion against the compiled HLO. epilogue = folded into
  // a matmul output (kOutput); prologue = into a matmul input (kInput — TPU MXU
  // does NOT do this, so it stays unfused & the activation materialises); kernel
  // = folded inside a Pallas kernel (kCustom, not a separate XLA fusion).
  function status(kind){
    if(!f) return ["b-none","— (no HLO; theory only)"];
    if(kind==="kernel") return ["b-compute","✓ fused — inside Pallas kernel (kCustom)"];
    if(kind==="epilogue") return ko>0?["b-compute","✓ fused by XLA — matmul epilogue (kOutput×"+ko+")"]:["b-ICI","✗ not fused"];
    return ki>0?["b-compute","✓ fused — matmul prologue (kInput×"+ki+")"]:["b-ICI","✗ not fused — TPU MXU has no prologue fusion; activation materialises"];
  }
  const C=[
    ["input_norm → qkv","prologue",H,attnN],
    ["o_proj → residual_add","epilogue",H,attnN],
    ["post_norm → "+(nMoe>0?"router":"gate_up"),"prologue",H,attnN],
    ["rope → attention","kernel",qsz,attnN],
  ];
  if(nMoe>0) C.push(["experts → residual_add","kernel",H,nMoe]);
  else C.push(["gate_up → silu","epilogue",DENSE_F,nDense],["silu → down_proj","prologue",DENSE_F,nDense]);
  const rows=C.map(c=>{const gb=tokens*c[2]*2*c[3]/1e9,[cls,txt]=status(c[1]);return {name:c[0],kind:c[1],gb,ms:gb*1e9/HBMBW*1e3,cls,txt};}).sort((a,b)=>b.ms-a.ms);
  const totGB=rows.reduce((a,r)=>a+r.gb,0), totMs=rows.reduce((a,r)=>a+r.ms,0), HgB=R.Th*HBMBW/1e3/1e9;
  let h="<div class='lh'>Fusion — which intermediate HBM round-trips are removed</div>";
  h+="<div class='note'>Fold a single producer→consumer activation into the neighbouring matmul/kernel, dropping the intermediate's HBM round-trip. Model is <b>"+R.tbound+"-bound</b>"+(R.tbound==="HBM"?" → bytes saved ≈ step saved.":".")+" The <b>status</b> column is from the compiled HLO"+(f?(" ("+f.n_fusions+" fusions: "+Object.keys(f.by_kind).map(k=>f.by_kind[k]+"× "+k).join(", ")+")"):" — none baked")+".</div>";
  h+="<table><thead><tr><th class='l'>fusion (producer → consumer)</th><th class='l'>type</th><th>HBM GB</th><th>saved ms</th><th class='l'>XLA status (from HLO)</th></tr></thead><tbody>";
  for(const r of rows) h+="<tr><td class='l'>"+r.name+"</td><td class='l' style='color:#667'>"+r.kind+"</td><td>"+fmt(r.gb)+"</td><td>"+r.ms.toFixed(3)+"</td><td class='l'><span class='tag "+r.cls+"'>"+r.txt+"</span></td></tr>";
  h+="</tbody></table>";
  h+="<div class='verdict v-go'>Theory upper bound ≈ <b>"+fmt(totGB)+" GB</b> ≈ "+(R.tot>0?(totMs/R.tot*100).toFixed(0):0)+"% of step. But per the HLO: the <b>epilogue</b> + in-<b>kernel</b> fusions are <b>already done</b> by XLA; the only unrealised ones are <b>matmul prologues</b>, which TPU's MXU does not fuse anyway (the normed activation must materialise before the matmul). → <b>fusion is not a lever here.</b></div>";
  return h;}
function dataflowHTML(s){const ch=buildChain(s); const mx=Math.max(...ch.map(o=>o.ms))||1;
  const BCOL={HBM:"#3b82f6",ICI:"#ec4899",compute:"#22c55e"};
  const comp=D.composition||{attn:[],ffn:[]}, ra=(comp.attn||[])[0], rf=(comp.ffn||[])[0];
  const flow=[ra&&MODULES[ra.m].flow?MODULES[ra.m].flow(ra.dims):"", rf&&MODULES[rf.m].flow?MODULES[rf.m].flow(rf.dims):""].filter(Boolean).join(" + ");
  let h="<div class='note'>one "+flow+" layer · per-device · bar ∝ ideal ms · color = bound</div>";
  for(let i=0;i<ch.length;i++){const o=ch[i];
    h+="<div class='dfrow'><div class='nm'><span style='color:"+(CAT[o.cat]||'#888')+"'>●</span> "+o.name+"</div>"
      +"<div class='barwrap'><div class='bar' style='width:"+Math.max(1.5,o.ms/mx*100)+"%;background:"+(BCOL[o.bound]||'#999')+"'></div></div>"
      +"<div class='ms'>"+o.ms.toFixed(4)+" ms <span class='tag b-"+o.bound+"'>"+o.bound+"</span></div></div>";}
  const tot=ch.reduce((a,o)=>a+o.ms,0);
  return h+"<div style='margin-top:8px;font-size:12px;color:#334'><b>layer Σ ideal ≈ "+tot.toFixed(3)+" ms</b> (serial; cross-op overlap not modelled)</div>";}
function legendHTML(){return "<div class='legend'>"+Object.keys(CAT).map(c=>"<span style='color:"+CAT[c]+"'>●</span> "+c).join(" &nbsp; ")+" &nbsp; ✕ = ICI-bound (below roof)</div>";}
function costTableHTML(R){let h="<table style='margin-top:0'><thead><tr><th class='l'>op category</th><th>cnt</th><th>TFLOP</th><th>HBM GB</th><th>ICI GB</th><th>OI</th><th>ideal ms</th><th>%step</th><th>bound</th></tr></thead><tbody>";
  for(const r of R.rows) h+="<tr><td class='l'><span style='color:"+(CAT[r.cat]||'#888')+"'>●</span> "+r.cat+(r.peak==="fp8"?" <span class='tag' style='background:#fef3c7;color:#92400e'>fp8</span>":"")+"</td><td>"+r.cnt+"</td><td>"+fmt(r.flops/1e12)+"</td><td>"+fmt(r.hbm/1e9)+"</td><td>"+fmt(r.ici/1e9)+"</td><td>"+r.oi.toFixed(1)+"</td><td>"+r.ideal.toFixed(3)+"</td><td>"+r.pct.toFixed(0)+"%</td><td><span class='tag b-"+r.bound+"'>"+r.bound+"</span></td></tr>";
  return h+"</tbody></table>";}
function codepathHTML(){const C=D.codepath; if(!C)return "<div style='color:#a55'>code-path index unavailable (built without a trace).</div>";
  let h="<div class='lh'>Code path — real forward, traced</div><div class='note'>"+(C.num_eqns_all||0).toLocaleString()+" jaxpr equations ("+(C.num_eqns_top||0).toLocaleString()+" top-level). Each op group → its actual <b>models/*.py</b> call chain (innermost = op kind, outer = role). Layer counts are emergent from the trace.</div>";
  h+="<table><thead><tr><th class='l'>role</th><th class='l'>category</th><th>count</th><th class='l'>code path (innermost ← caller)</th></tr></thead><tbody>";
  for(const r of (C.gemms||[])){const chain=(r.stack||[]).slice(0,5).map((f,i)=>i===0?("<b>"+f+"</b>"):f).join(" <span style='color:#94a3b8'>←</span> ");
    h+="<tr><td class='l'>"+r.role+"</td><td class='l'><span style='color:"+(CAT[r.category]||'#888')+"'>●</span> "+r.category+"</td><td>"+r.count+"</td><td class='l mono' style='color:#475569'>"+chain+"</td></tr>";}
  return h+"</tbody></table>";}
function kernelsHTML(){const C=D.codepath; if(!C)return "";
  let h="<div class='lh'>Pallas kernels</div><div class='note'>Real kernel names + per-device in/out avals + the shard_map call site. RPA-v3 / fused-MoE-v2 declare no cost_estimate, so the roofline prices them from their reference math.</div>";
  for(const k of (C.kernels||[])){const av=a=>(a||[]).map(x=>x.dtype+"["+x.shape.join(",")+"]").join(", ");
    const col=k.kind==="attention"?CAT.attention:(k.kind==="moe"?CAT.moe:"#888");
    h+="<div style='margin:4px 0;padding:8px 10px;border:1px solid #e6e9ef;border-radius:8px'><div style='font-weight:600;color:#0f172a'><span style='color:"+col+"'>●</span> "+k.name+" <span class='pill'>×"+k.count+"</span> <span class='pill'>"+k.kind+"</span></div>"
      +"<div class='mono' style='color:#667;margin-top:3px'>in: "+av(k.in_avals)+"<br>out: "+av(k.out_avals)+"<br>@ "+(k.ctx||"")+"</div></div>";}
  return h;}
const HELP={
  overview:"Overall roofline · per-category cost · one-layer dataflow.",
  overlap:"Can comm (ICI) hide behind compute/HBM — drag tp / tokens to see when it gets exposed.",
  kernel:"Ops ranked by cost; each tells you whether to cut bytes or raise compute.",
  fusion:"Which intermediate-activation HBM round-trips to fold away, ranked by step saved.",
  trace:"This model's real forward: code-path + Pallas kernels (from the trace)."};
function card(inner){return "<div class='panel' style='margin-bottom:12px'>"+inner+"</div>";}
function chartHTML(){return "<canvas id='cv' style='margin-bottom:12px'></canvas>";}
function render(){const s=state(); const R=compute(s);
  g("scenhelp").innerHTML=HELP[SCEN]||"";
  let html="";
  try{
    if(SCEN==="overview") html=chartHTML()+card(legendHTML()+costTableHTML(R))+card(dataflowHTML(s));
    else if(SCEN==="overlap") html=chartHTML()+card(lensOverlap(s,R));
    else if(SCEN==="kernel") html=chartHTML()+card(lensKernel(s,R));
    else if(SCEN==="fusion") html=card(lensFusion(s,R));
    else if(SCEN==="trace") html=card(codepathHTML())+card(kernelsHTML());
  }catch(e){html="<div class='panel' style='color:#a33'>render error in '"+SCEN+"': "+e.message+"</div>";}
  g("body").innerHTML=html;
  if(g("cv")){draw(R); attachTip();}
  updateSummary(s,R);
}
function attachTip(){const cv=g("cv"); if(!cv)return; const tip=g("tip");
  cv.onmousemove=e=>{const rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;
    let hit=null; if(LAST&&LAST._pts)for(const p of LAST._pts){if((mx-p.px)**2+(my-p.py)**2<(p.rad+5)**2){hit=p;break;}}
    if(hit){const r=hit.r,ts=r.ideal/1e3,aTF=r.flops/ts/1e12,aBW=r.hbm/ts/1e9,aICI=r.ici/ts/1e9,cpk=(r.peak==="fp8"?P.fp8_tflops:P.bf16_tflops);
      tip.style.display="block";tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+8)+"px";
      tip.innerHTML="<b>"+r.cat+"</b> (×"+r.cnt+") · <b>"+r.bound+"-bound</b><br>"+fmt(r.flops/1e12)+" TFLOP · "+fmt(r.hbm/1e9)+" GB HBM · "+fmt(r.ici/1e9)+" GB ICI · OI="+r.oi.toFixed(1)
        +"<br>ideal "+r.ideal.toFixed(3)+" ms → achieved:<br>&nbsp;&nbsp;"+aTF.toFixed(0)+" TFLOP/s ("+(aTF/cpk*100).toFixed(0)+"% compute)<br>&nbsp;&nbsp;"+aBW.toFixed(0)+" GB/s ("+(aBW/P.hbm_gbps*100).toFixed(0)+"% HBM)"
        +(aICI>0?("<br>&nbsp;&nbsp;"+aICI.toFixed(0)+" GB/s ("+(aICI/P.ici_gbps*100).toFixed(0)+"% ICI)"):"");}
    else tip.style.display="none";};
  cv.onmouseleave=()=>tip.style.display="none";}
function updateSummary(s,R){const L=R.L;
  const qstr=(Q.wq==="bf16"?"bf16":("fp8 "+(Q.wq==="block"?("block-"+Q.blk):Q.wq)+" "+(Q.aq==="fp8"?"W8A8":"W8A16")))+(Q.wq!=="bf16"?(" → "+wpeak()+" MXU"+(Q.wq==="block"?" (capped)":"")):"");
  g("summary").innerHTML =
   "<b>mesh</b> data="+L.dp+" × tensor="+L.t+" = "+L.devices+" dev &nbsp; <b>EP</b>="+L.ep+(s.enable_sp?" &nbsp;<span class='pill'>+SP</span>":"")+" &nbsp;<span class='pill'>"+qstr+"</span>"
   +"<br><b>"+(R.decode?"decode":"prefill")+"</b> · tokens/DP="+R.tokens+" · MoE global="+(R.tokens*L.dp)
   +"<br><b>bound: <span class='tag b-"+R.tbound+"'>"+R.tbound+"</span></b> &nbsp; step ≈ "+R.tot.toFixed(2)+" ms"
   +"<div style='margin-top:6px'><span class='pill'>compute "+R.Tc.toFixed(2)+"ms</span><span class='pill'>HBM "+R.Th.toFixed(2)+"ms</span><span class='pill'>ICI "+R.Ti.toFixed(2)+"ms</span></div>";
}

function divisors(n){const a=[];for(let i=1;i<=n;i++)if(n%i===0)a.push(i);return a;}
function validDp(tp){const comp=D.composition||{attn:[]};
  return divisors(tp).filter(d=>{const t=Math.floor(tp/d);
    return (comp.attn||[]).every(a=>{const hd=MODULES[a.m].headDiv?MODULES[a.m].headDiv(a.dims):null; return hd==null||hd%t===0;});});}
function g(id){return document.getElementById(id);}
let PHASE="decode"; let SCEN="overview";
function setScen(name){SCEN=name; document.querySelectorAll("#scennav button").forEach(b=>b.classList.toggle("on",b.dataset.sc===name)); render();}
function state(){return {tp:+g("tp").value, dp:+g("dp").value, batch:+g("batch").value, seq_len:+g("seq_len").value, chunk:+g("chunk").value, phase:PHASE, enable_sp:g("sp").checked};}
function fillDp(){const tp=+g("tp").value; const cur=+g("dp").value; const opts=validDp(tp);
  g("dp").innerHTML=opts.map(d=>"<option value='"+d+"'>"+d+"</option>").join("");
  g("dp").value = opts.includes(cur)? cur : opts[opts.length-1];}
function syncQuant(){Q.wq=g("wq").value; Q.blk=+g("blk").value; Q.aq=g("aq").value; g("blk").disabled=(Q.wq!=="block");}
function syncLabels(){g("tv").textContent="t="+Math.max(1,Math.floor(g("tp").value/g("dp").value));
  g("batchv").textContent=g("batch").value; g("seqv").textContent=g("seq_len").value; g("chunkv").textContent=g("chunk").value;}
function syncPhaseCtl(){const dec=PHASE==="decode"; g("ctl-batch").style.display=dec?"block":"none"; g("ctl-seq").style.display=dec?"block":"none"; g("ctl-chunk").style.display=dec?"none":"block";}
function init(){
  g("arch").textContent=D.arch;
  const d=D.defaults;
  const tpopts=divisors(D.NEXP).filter(x=>x<=1024 && validDp(x).length>0);
  g("tp").innerHTML=tpopts.map(x=>"<option value='"+x+"'>"+x+"</option>").join("");
  g("tp").value = tpopts.includes(d.tp)? d.tp : tpopts[tpopts.length-1];
  fillDp(); if(validDp(+g("tp").value).includes(d.dp)) g("dp").value=d.dp;
  g("batch").value=d.batch; g("seq_len").value=d.seq_len; g("chunk").value=d.chunk; g("sp").checked=d.enable_sp;
  if(d.wq)g("wq").value=d.wq; if(d.blk)g("blk").value=d.blk; if(d.aq)g("aq").value=d.aq;
  syncQuant();
  g("tp").addEventListener("change",()=>{fillDp();syncLabels();render();});
  g("dp").addEventListener("change",()=>{syncLabels();render();});
  ["wq","blk","aq"].forEach(id=>g(id).addEventListener("change",()=>{syncQuant();render();}));
  ["batch","seq_len","chunk"].forEach(id=>g(id).addEventListener("input",()=>{syncLabels();render();}));
  g("sp").addEventListener("change",render);
  g("ph-decode").onclick=()=>{PHASE="decode";g("ph-decode").className="on";g("ph-prefill").className="";syncPhaseCtl();render();};
  g("ph-prefill").onclick=()=>{PHASE="prefill";g("ph-prefill").className="on";g("ph-decode").className="";syncPhaseCtl();render();};
  document.querySelectorAll("#scennav button").forEach(b=>b.onclick=()=>setScen(b.dataset.sc));
  window.addEventListener("resize",()=>{if(g("cv"))draw(LAST);});
  syncLabels(); syncPhaseCtl(); render();
}
init();
</script></body></html>"""
