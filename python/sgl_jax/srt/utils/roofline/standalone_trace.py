"""Standalone CPU forward tracer — the lightweight, server-free basis for the
roofline tool.

Constructs ANY registered sglang-jax model ABSTRACTLY (``nnx.eval_shape``, zero
weight allocation) on a fake CPU device mesh, builds a dummy ForwardBatch, and
``make_jaxpr``s the REAL forward. The result is the per-device jaxpr (ops + real
``models/*.py`` source + Pallas kernels with per-device avals) with **no TPU, no
weights, no checkpoint load, no server** — seconds on a laptop/CPU pod.

Why it works: ``make_jaxpr`` only TRACES (it never lowers Mosaic), so Pallas
kernels trace fine on CPU and the per-device jaxpr structure is identical to a
real multi-host run (validated: 6808 top eqns, 209 Pallas kernels, identical
kernel names + GEMM shapes + source attribution vs a real 32-device EP32 run).
Weight *values* never affect the jaxpr — only shapes — so abstract weights and a
tiny KV pool give the right structure; quantization and the real context length
are applied analytically by the cost model (from ``config.json`` + the parallel
layout), not from this trace.

Usage (CPU; pretends to be v7x so kernel block-size selection resolves):
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=<devices>
    python -m sgl_jax.srt.utils.roofline.standalone_trace ...
or via ``tools/trace_roofline.py`` which sets the env for you.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax
import numpy as np
from flax import nnx


@dataclass
class TraceResult:
    jaxpr: object  # ClosedJaxpr of the real forward
    model_config: object
    arch: str
    tp: int
    dp: int
    attention_tp: int
    ep: int
    phase: str
    tokens_global: int


def patch_for_cpu(ver: int = 7):
    """On CPU there is no TPU, so kernel host-code (block-size selection) sees
    tpu_version=-1 / a non-TPU device_kind and bails. Since make_jaxpr only
    TRACES, pretend to be v7x so the real per-device block config is selected.
    Patches get_tpu_version / get_device_name wherever imported by-name."""
    import importlib

    def dev_name(num_devices=None):
        return "TPU v7" + (f"-{num_devices}" if num_devices is not None else "")

    def ver_fn(*a, **k):
        return ver

    patches = {
        "sgl_jax.srt.utils.jax_utils": {"get_device_name": dev_name},
        "sgl_jax.srt.kernels.ragged_paged_attention.util": {"get_tpu_version": ver_fn},
        "sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3": {
            "get_tpu_version": ver_fn
        },
        "sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes_v3": {
            "get_tpu_version": ver_fn
        },
        "sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes": {
            "get_device_name": dev_name
        },
        "sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.tuned_block_sizes": {
            "get_tpu_version": ver_fn
        },
        "sgl_jax.srt.kernels.fused_moe.v1.tuned_block_configs": {"get_device_name": dev_name},
        "sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs": {"get_device_name": dev_name},
    }
    for mod_name, attrs in patches.items():
        try:
            mod = importlib.import_module(mod_name)
            for attr, fn in attrs.items():
                if hasattr(mod, attr):
                    setattr(mod, attr, fn)
        except Exception:
            pass


class _Runner:
    """Minimal ModelRunner stand-in: only what ForwardBatch.init_new +
    attn_backend_wrapper actually touch (mesh / attn_backend / model_config, and
    the linear-attn configs that gate the hybrid wrapper)."""

    def __init__(self, mesh, attn_backend, model_config):
        self.mesh = mesh
        self.attn_backend = attn_backend
        self.model_config = model_config
        self.linear_recurrent_config = None
        self.kimi_linear_config = None
        self.lightning_config = None
        self.bailing_moe_v3_config = None
        self.qwen3_5_hybrid_config = None


def _make_dummy_batch(
    bs, num_tokens, mode, max_cache_loc_size, vocab_size, dp_size, recurrent=False
):
    """Pure-numpy ModelWorkerBatch mirroring CompilationManager._make_dummy_batch."""
    from sgl_jax.srt.managers.schedule_batch import (
        ForwardMode,
        ModelWorkerBatch,
        ModelWorkerSamplingInfo,
    )
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

    per_dp = bs // dp_size
    extend = mode == ForwardMode.EXTEND
    # Hybrid recurrent models (KDA/GDN/Mamba) need the linear-attn metadata: one
    # recurrent-state slot per request, and a per-request has-initial-state flag
    # (decode continues prior state; a fresh extend starts empty).
    recurrent_indices = np.arange(bs, dtype=np.int32) if recurrent else None
    has_initial_state = np.array([not extend] * bs, dtype=bool) if recurrent else None
    return ModelWorkerBatch(
        bid=1,
        forward_mode=mode,
        input_ids=np.concat(
            [np.array([1] * bs, np.int32), np.array([0] * (num_tokens - bs), np.int32)]
        ),
        real_input_ids_len=bs,
        real_bs=bs,
        req_pool_indices=np.arange(bs, dtype=np.int32),
        seq_lens=np.array([1] * bs, dtype=np.int32),
        out_cache_loc=np.concat(
            [np.arange(1, bs + 1, dtype=np.int32), np.array([-1] * (num_tokens - bs), np.int32)]
        ),
        return_logprob=False,
        return_output_logprob_only=True,
        sampling_info=ModelWorkerSamplingInfo.generate_for_precompile(bs, vocab_size),
        extend_input_logprob_token_ids=None,
        positions=np.concat(
            [np.array([0] * bs, np.int32), np.array([0] * (num_tokens - bs), np.int32)]
        ),
        cache_loc=np.concat([np.arange(bs), np.array([0] * (max_cache_loc_size - bs), np.int32)]),
        extend_prefix_lens=(np.array([0] * bs) if extend else None),
        extend_seq_lens=(np.array([1] * bs) if extend else None),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        logits_indices=(np.array([0] * bs) if extend else None),
        capture_hidden_mode=CaptureHiddenMode.NULL,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        lora_ids=["0"] * bs,
        dp_size=dp_size,
        per_dp_bs_size=per_dp,
        real_bs_per_dp=[per_dp] * dp_size,
        logits_indices_selector=np.arange(bs, dtype=np.int32),
        recurrent_indices=recurrent_indices,
        has_initial_state=has_initial_state,
    )


def _build_model_config(
    model_path, attention_tp, ep, moe_backend, dtype, layers=None, representative=False
):
    from sgl_jax.srt.configs.model_config import ModelConfig

    # representative mode builds the FULL config then keeps one layer per distinct
    # (is_swa, is_moe) type -- so a truncated compile still covers every collective
    # even when the model is dense-first / MoE-later. Plain `layers` = first-N.
    mc = ModelConfig(
        model_path=model_path,
        trust_remote_code=True,
        dtype=dtype,
        model_layer_nums=None if representative else layers,
    )
    if representative:
        _apply_representative_layers(mc.hf_config)
    # mirror ModelRunner.load_model's hf_config injection (without loading weights)
    mc.configure_for_tensor_parallel(attention_tp)
    hf = mc.hf_config
    hf.ep_size = ep
    hf.ep_num_redundant_experts = 0
    hf.moe_backend = moe_backend
    hf.use_jax_allreduce_metadata = True
    hf.use_absorbed_mla = True
    hf.enable_sequence_parallel = True
    return mc


def representative_layer_types(hf):
    """One layer index per distinct (is_swa, is_moe) combo, in first-occurrence
    order -- covers every collective type regardless of layer ordering."""
    L = hf.num_hidden_layers
    hlp = list(getattr(hf, "hybrid_layer_pattern", None) or [0] * L)
    mlf = getattr(hf, "moe_layer_freq", None)
    mlf = list(mlf) if (mlf is not None and not isinstance(mlf, int)) else [1] * L
    seen, picked = set(), []
    for i in range(L):
        combo = (
            bool(hlp[i]) if i < len(hlp) else False,
            bool(mlf[i]) if i < len(mlf) else True,
        )
        if combo not in seen:
            seen.add(combo)
            picked.append((i, combo))
    return picked


def _apply_representative_layers(hf):
    picked = representative_layer_types(hf)
    hlp = list(getattr(hf, "hybrid_layer_pattern", None) or [0] * hf.num_hidden_layers)
    mlf = getattr(hf, "moe_layer_freq", None)
    mlf = (
        list(mlf) if (mlf is not None and not isinstance(mlf, int)) else [1] * hf.num_hidden_layers
    )
    idxs = [i for i, _ in picked]
    hf.num_hidden_layers = len(idxs)
    if getattr(hf, "hybrid_layer_pattern", None) is not None:
        hf.hybrid_layer_pattern = [hlp[i] if i < len(hlp) else 0 for i in idxs]
    if getattr(hf, "moe_layer_freq", None) is not None and not isinstance(hf.moe_layer_freq, int):
        hf.moe_layer_freq = [mlf[i] if i < len(mlf) else 1 for i in idxs]


def _build_kv_pool(mc, mesh, attention_tp, dp, page_size):
    from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, SWAKVPool

    head_num = mc.get_total_num_kv_heads_with_replication(attention_tp)
    head_dim = (mc.head_dim + 127) // 128 * 128
    dtype = jax.numpy.bfloat16
    # Tracing only needs valid shapes (tiny is fine + fast). But a real TPU
    # *compile* runs memory-space assignment, which can wrongly place a too-small
    # fused-KV buffer in VMEM and crash; a realistic pool keeps it in HBM.
    pages = int(os.environ.get("RL_KV_PAGES", "16"))
    size = pages * page_size * dp
    swa_ids = list(getattr(mc, "swa_attention_layer_ids", []) or [])
    full_ids = list(getattr(mc, "full_attention_layer_ids", []) or [])
    if not (swa_ids or full_ids):
        full_ids = list(range(mc.hf_config.num_hidden_layers))
    if swa_ids:
        swa_num_kv = getattr(mc.hf_config, "swa_num_key_value_heads", None)
        swa_head_num = max(swa_num_kv, attention_tp) if swa_num_kv else None
        return SWAKVPool(
            size=size,
            size_swa=size,
            page_size=page_size,
            swa_attention_layer_ids=swa_ids,
            full_attention_layer_ids=full_ids,
            token_to_kv_pool_class=MHATokenToKVPool,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_head_num=swa_head_num,
            mesh=mesh,
            dp_size=dp,
        )
    return MHATokenToKVPool(
        size=size,
        page_size=page_size,
        layer_num=mc.hf_config.num_hidden_layers,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        mesh=mesh,
        dp_size=dp,
    )


def trace_model_forward(
    model_path: str,
    tp: int,
    dp: int,
    *,
    phase: str = "extend",
    num_tokens: int = 512,
    moe_backend: str = "fused_v2",
    dtype: str = "bfloat16",
    page_size: int = 256,
) -> TraceResult:
    """Trace the REAL forward of a registered model abstractly and return the
    ClosedJaxpr + resolved layout. Call ``patch_for_cpu()`` first on CPU."""
    fwd, args, mesh, mc, arch, attention_tp, ep, tokens_global = _build_forward(
        model_path,
        tp,
        dp,
        phase=phase,
        num_tokens=num_tokens,
        moe_backend=moe_backend,
        dtype=dtype,
        page_size=page_size,
    )
    with jax.set_mesh(mesh):
        jaxpr = jax.make_jaxpr(fwd)(*args)
    return TraceResult(
        jaxpr=jaxpr,
        model_config=mc,
        arch=arch,
        tp=tp,
        dp=dp,
        attention_tp=attention_tp,
        ep=ep,
        phase=phase,
        tokens_global=tokens_global,
    )


def _build_forward(
    model_path,
    tp,
    dp,
    *,
    phase,
    num_tokens,
    moe_backend,
    dtype,
    page_size,
    layers=None,
    representative=False,
):
    """Build the abstract model + dummy batch; return the forward closure, its
    (abstract-weights, fb, mp, lm) args, the mesh, and meta. Shared by the jaxpr
    tracer and the HLO compiler."""
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.managers.schedule_batch import ForwardMode
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
        _build_non_hybrid_memory_pools,
    )
    from sgl_jax.srt.model_loader.arch import get_model_architecture
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    attention_tp = tp // dp
    ep = tp
    mesh = create_device_mesh(ici_parallelism=[dp, attention_tp], dcn_parallelism=[1, 1])
    mc = _build_model_config(
        model_path,
        attention_tp,
        ep,
        moe_backend,
        dtype,
        layers=layers,
        representative=representative,
    )
    model_class, arch = get_model_architecture(mc)

    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: model_class(mc.hf_config, dtype=mc.dtype, mesh=mesh))
        hf = mc.hf_config
        # Hybrid recurrent model (e.g. Ling3 BailingMoeV3: KDA + MLA): build the MLA
        # full-attn backend wrapped in HybridLinearAttnBackend + the recurrent-state
        # + MLA-latent pools, instead of a single MHA/SWA pool + FlashAttention.
        hybrid = bool(getattr(hf, "linear_attn_config", None)) and bool(
            getattr(hf, "full_attention_layer_ids", None)
        )
        runner = _Runner(mesh, None, mc)

        mode = ForwardMode.EXTEND if phase == "extend" else ForwardMode.DECODE
        if mode == ForwardMode.EXTEND:
            bs, ntok = dp, num_tokens  # per_dp_bs=1; chunk carries the load
        else:
            bs = max(ep, dp)  # decode: global tokens(=bs) must align to ep_size
            ntok = bs
        batch = _make_dummy_batch(
            bs, ntok, mode, 4 * page_size * dp, hf.vocab_size, dp, recurrent=hybrid
        )

        if hybrid:
            attn_backend, mp = _build_hybrid_attn_and_pools(
                mc, mesh, attention_tp, dp, page_size, runner, bs
            )
        else:
            from sgl_jax.srt.layers.attention.flashattention_backend import (
                FlashAttention,
            )

            nkv = mc.get_total_num_kv_heads_with_replication(attention_tp)
            attn_backend = FlashAttention(
                mc.num_attention_heads, nkv, mc.head_dim, page_size=page_size, mesh=mesh
            )
            mp = _build_non_hybrid_memory_pools(
                _build_kv_pool(mc, mesh, attention_tp, dp, page_size)
            )
        runner.attn_backend = attn_backend
        attn_backend.forward_metadata = attn_backend.get_forward_metadata(batch)
        fb = ForwardBatch.init_new(batch, runner)
        lm = LogitsMetadata.from_model_worker_batch(batch, mesh)

        gd, state = nnx.split(model)
        leaves, treedef = jax.tree_util.tree_flatten(state)

        def fwd(state_leaves, forward_batch, memory_pools, logits_metadata):
            st = jax.tree_util.tree_unflatten(treedef, state_leaves)
            return nnx.merge(gd, st)(forward_batch, memory_pools, logits_metadata)

    tokens_global = ntok if mode == ForwardMode.EXTEND else bs
    return fwd, (leaves, fb, mp, lm), mesh, mc, arch, attention_tp, ep, tokens_global


def _build_hybrid_attn_and_pools(mc, mesh, attention_tp, dp, page_size, runner, max_num_reqs):
    """Hybrid recurrent attention (KDA/GDN/... linear + MLA/full): build the MLA
    full-attn backend, wrap it via ``attn_backend_wrapper`` (which adds the matching
    linear sub-backend), and build the recurrent-state + MLA-latent memory pools.
    Mirrors ``model_runner._get_attention_backend`` + ``_build_hybrid_pools``."""
    import jax.numpy as jnp

    from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
        attn_backend_wrapper,
    )
    from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
    from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
        _build_hybrid_pools,
    )

    hf = mc.hf_config
    full_attn_backend = MLAAttentionBackend(
        num_attn_heads=mc.num_attention_heads,
        kv_lora_rank=hf.kv_lora_rank,
        qk_nope_head_dim=hf.qk_nope_head_dim,
        qk_rope_head_dim=hf.qk_rope_head_dim,
        v_head_dim=hf.v_head_dim,
        page_size=page_size,
        mesh=mesh,
        attention_data_partition_axis="data",
        # On CPU get_tpu_info() raises ("Unsupported TPU device kind: cpu"); pin the
        # v7x VMEM cap (64 MiB) so __init__ never calls it. Shapes-only trace anyway.
        vmem_limit_bytes=int(64 * 1024 * 1024 * 0.9),
    )
    # Gate the wrapper's linear sub-backend (mirror the runner's config properties).
    from sgl_jax.srt.configs.bailing_moe_v3 import BailingMoeV3Config

    if isinstance(hf, BailingMoeV3Config):
        runner.bailing_moe_v3_config = hf
    runner.linear_recurrent_config = hf  # has full_attention_layer_ids / linear_layer_ids
    attn_backend = attn_backend_wrapper(runner, full_attn_backend)

    pages = int(os.environ.get("RL_KV_PAGES", "16"))
    size = pages * page_size * dp
    mla_pool = MLATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=jnp.bfloat16,
        kv_lora_rank=hf.kv_lora_rank,
        qk_rope_head_dim=hf.qk_rope_head_dim,
        # index by GLOBAL layer_id (the MLA layers pass their absolute id); the
        # recurrent (KDA) layers' slots stay unused. Tracing is shapes-only so the
        # few unused tiny buffers are free.
        layer_num=hf.num_hidden_layers,
        mesh=mesh,
        dp_size=dp,
    )
    _, _, mp = _build_hybrid_pools(
        cfg=hf,
        max_num_reqs=max_num_reqs,
        max_context_len=4 * page_size * dp,
        tp_size=attention_tp,
        token_to_kv_pool=mla_pool,
        mesh=mesh,
        dp_size=dp,
        state_size=max_num_reqs,
    )
    return attn_backend, mp


def compile_forward_hlo(
    model_path,
    tp,
    dp,
    *,
    phase="extend",
    num_tokens=512,
    moe_backend="fused_v2",
    dtype="bfloat16",
    page_size=256,
    layers=None,
    representative=True,
) -> str:
    """Lower + compile the real forward for the (real-device) mesh and return the
    optimized, scheduled HLO text. Run on a TPU host/cluster with
    ``jax.distributed`` initialized (so ``jax.devices()`` spans all chips) -- the
    async-collective schedule is target-specific, so CPU HLO is not representative.
    A full 70-layer compile overflows XLA's memory-space assignment with the SWA
    kernel, so the model is shrunk: ``representative=True`` (default) keeps one
    layer per distinct (swa, moe) type -- covers every collective even for
    dense-first/MoE-later models; ``layers=N`` instead keeps the first N."""
    fwd, args, mesh, mc, arch, attention_tp, ep, tokens_global = _build_forward(
        model_path,
        tp,
        dp,
        phase=phase,
        num_tokens=num_tokens,
        moe_backend=moe_backend,
        layers=layers,
        representative=representative,
        dtype=dtype,
        page_size=page_size,
    )
    with jax.set_mesh(mesh):
        compiled = jax.jit(fwd).lower(*args).compile()
    devices = tp
    sp_triggered = bool(mc.hf_config.enable_sequence_parallel) and (
        tokens_global >= devices * 128 and tokens_global % devices == 0
    )
    types = [
        ("SWA" if sw else "full") + ("+MoE" if moe else "+dense")
        for _, (sw, moe) in representative_layer_types(mc.hf_config)
    ]
    meta = {
        "tp": tp,
        "dp": dp,
        "devices": devices,
        "ep": ep,
        "tokens_global": tokens_global,
        "phase": phase,
        "n_layers_compiled": mc.hf_config.num_hidden_layers,
        "layer_types": types,
        "enable_sp": bool(mc.hf_config.enable_sequence_parallel),
        "sp_triggered": sp_triggered,
        "sp_threshold_tokens": devices * 128,
    }
    return compiled.as_text(), meta


def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Standalone CPU forward tracer (roofline)")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--phase", choices=["extend", "decode"], default="extend")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--moe-backend", default="fused_v2")
    ap.add_argument("--out", default="/tmp/fwd_jaxpr_cpu.json")
    args = ap.parse_args()

    if jax.default_backend() != "tpu":
        patch_for_cpu(7)
    print(
        f"platform={jax.default_backend()} devices={len(jax.devices())} tp={args.tp} dp={args.dp}"
    )
    res = trace_model_forward(
        args.model_path,
        args.tp,
        args.dp,
        phase=args.phase,
        num_tokens=args.tokens,
        moe_backend=args.moe_backend,
    )
    print(f"traced arch={res.arch} phase={res.phase} tokens_global={res.tokens_global}")
    from sgl_jax.srt.utils.roofline.forward_jaxpr_dump import dump_closed_jaxpr

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dump_closed_jaxpr(res.jaxpr, args.out)
    print(f"dumped -> {args.out}")


if __name__ == "__main__":
    _main()
