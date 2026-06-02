"""
Tune fused_ep_moe_v2 block configs (lean, decode-focused v2 adaptation).

This is a v2 adaptation of ``benchmark/moe/bench_fused_moe.py``. It tunes the
``fused_ep_moe_v2`` kernel's block config by timing candidate configs with the
SAME canonical marker-based timer the v1 tuner uses
(``multiple_iteration_timeit_from_trace``), then prints the best config as a
v2 tuned-table entry that can be pasted into
``sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py``.

Candidate enumeration is **v2-native**: a self-contained port of the
``generate_tune_candidates`` / ``_estimate_vmem_bytes_v2`` logic out of
``python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py`` (see
``generate_v2_tune_candidates`` below). It produces v2
``FusedMoEBlockConfig(bt, bf, btc, bse, bts)`` objects directly and filters
them against the real v7x **64 MB** VMEM budget using v2's own VMEM model.
We do NOT import bench_v2.py (it has heavy module-level side effects — it
builds a mesh and reads env at import — so importing it would fail/hang
off-TPU and on a fresh process). The v1 ``select_block_configs`` (whose VMEM
estimate models the v1 kernel's bd1/bd2/bfc blocking) is intentionally NOT
used: it mis-modeled v2 VMEM and emitted btc=1 / large-bts configs that OOM
v2 on v7x.

It still reuses by IMPORTING:
  - ``multiple_iteration_timeit_from_trace`` from benchmark.utils (the
    marker-based timer that produced the real tuned tables; timing does not
    depend on the kernel's own event name).
  - mesh / case / input helpers from benchmark.moe.utils, mirroring how
    bench_fused_moe.py builds the mesh, fp8-quantizes weights, jits the
    forward, and prints the tuned-table line.

The v2 layer (``FusedEPMoEV2``) has an identical constructor to ``FusedEPMoE``;
we only swap the class and pass ``metadata_mode`` (default ``"direct"``, the
production decode mode for MiMo-V2-Pro). Each candidate is passed explicitly
as ``block_config=<v2 config>`` so the layer never auto-looks-up the table.

Multi-host: mesh / sharding / process handling is identical to
bench_fused_moe.py (it already tuned ep=32 tables on v7x multi-host), so this
script works under ``process_count > 1`` the same way.

Intended real invocation (run on a v7x-32 falcon job; NOT run here):

    python -m benchmark.moe.bench_fused_moe_v2 --tune-block-config \\
      --num-experts 384 --top-k 8 --hidden-size 6144 --intermediate-size 2048 \\
      --ep-size 32 --quant-block-k 128 --weight-dtype float8_e4m3fn \\
      --num-tokens 64 --metadata-mode direct --iters 20 --warmup-iters 5

Omitted v1 features (decode tuner is intentionally lean): imbalance
simulation, token-valid-mask sweeps, and shared experts. Add them back via the
shared helpers if needed.
"""

from __future__ import annotations

import argparse
import faulthandler
import math
import sys
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec as P

from benchmark.moe.utils import (
    DEFAULT_NUM_TOKENS,
    MoEBenchmarkCase,
    build_mesh,
    make_moe_cases,
    prepare_fused_moe_inputs,
    select_cases,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.kernels.fused_moe.v2.kernel import FusedMoEBlockConfig as V2BlockConfig
from sgl_jax.srt.kernels.fused_moe.v2.kernel import (
    validate_fused_moe_block_config as v2_validate,
)
from sgl_jax.srt.layers.fused_moe import FusedEPMoEV2
from sgl_jax.srt.layers.moe import TopK

# Real v7x VMEM is 64 MB. The v1 tuner's DEFAULT_TPU_VMEM_BUDGET_MB (96 MB)
# does NOT apply to the v2 kernel; v2 candidates are filtered against 64 MB.
DEFAULT_TPU_VMEM_BUDGET_MB = 64


# ---------------------------------------------------------------------------
# v2-native candidate enumeration (self-contained port of
# generate_tune_candidates / _estimate_vmem_bytes_v2 from
# python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py — no mesh/jax side effects).
# ---------------------------------------------------------------------------


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _pow2_floor(x: float) -> int:
    if x <= 1:
        return 1
    return 1 << int(math.floor(math.log2(x)))


def _pow2_ceil(x: float) -> int:
    if x <= 1:
        return 1
    return 1 << int(math.ceil(math.log2(x)))


def _aligned_divisors(n: int, alignment: int = 8) -> list[int]:
    """All divisors of n that are multiples of `alignment`, descending."""
    if n <= 0:
        return []
    divs: set[int] = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            if i % alignment == 0:
                divs.add(i)
            j = n // i
            if j % alignment == 0:
                divs.add(j)
    return sorted(divs, reverse=True)


def _estimate_vmem_bytes_v2(
    *,
    bt: int,
    bf: int,
    btc: int,
    bse: int,
    bts: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    ep_size: int,
    num_tokens: int,
    use_fp8: bool = False,
    quant_block_k: int = 128,
    direct_scaled_dot: bool = True,
    interleave_bt: bool = True,
    enable_bt_scatter_overlap: bool = True,
) -> int:
    """v2 kernel VMEM+SMEM estimate (bytes). Ported verbatim from bench_v2.py's
    ``_estimate_vmem_bytes_v2`` (sans verbose logging)."""
    t_packing = 1  # bf16 activations
    w_bytes = 1 if use_fp8 else 2
    token_bytes = 2  # bf16
    h_per_t = hidden_size // t_packing
    padded_num_experts = _align_to(num_experts, 128)
    padded_top_k = _align_to(top_k, 128)
    acc_bt = math.gcd(bt, 16)
    local_num_tokens = num_tokens // ep_size
    num_bt = local_num_tokens // bt if bt > 0 else 1
    use_bt_scatter_bank = enable_bt_scatter_overlap and num_bt > 1
    use_gather_bank = interleave_bt and num_bt > 1
    smem_banks = num_bt if use_gather_bank else 2

    b_a2a_g_acc = 2 * top_k * acc_bt * hidden_size * token_bytes
    b_topk_w = smem_banks * bt * padded_top_k * 4
    b_topk_id = smem_banks * bt * padded_top_k * 4
    b_output = smem_banks * bt * hidden_size * token_bytes

    b_w1 = 2 * hidden_size * bf * w_bytes
    b_w3 = 2 * hidden_size * bf * w_bytes
    b_w2 = 2 * bf * hidden_size * w_bytes

    b_w1_scale = 0
    b_w3_scale = 0
    b_w2_scale = 0
    if use_fp8:
        b_w1_scale = 2 * t_packing * (h_per_t // quant_block_k) * bf * 4
        b_w3_scale = b_w1_scale
        b_w2_scale = 2 * t_packing * (bf // quant_block_k) * h_per_t * 4

    b_w1_dq = 0
    b_w3_dq = 0
    b_w2_dq = 0
    if use_fp8 and not direct_scaled_dot:
        b_w1_dq = t_packing * h_per_t * bf * 2  # bf16
        b_w3_dq = b_w1_dq
        b_w2_dq = t_packing * bf * h_per_t * 2  # bf16

    b_gate_acc = bts * bf * 4
    b_up_acc = bts * bf * 4
    b_x = bts * hidden_size * token_bytes
    b_y_acc = bts * hidden_size * 4
    b_y_stage = bts * hidden_size * token_bytes

    local_num_experts = num_experts // ep_size
    b_scoped = (
        bt * padded_top_k * 4
        + ep_size * padded_num_experts * 4
        + 2 * padded_num_experts * 4
        + padded_num_experts * 4
        + padded_num_experts * 4
    )

    num_bt_banks = num_bt if use_gather_bank else (2 if use_bt_scatter_bank else 1)
    b_sems = (
        2 * 4
        + smem_banks * 10 * 4
        + 3
        * (
            num_bt_banks * local_num_experts * 4
            if (use_bt_scatter_bank or use_gather_bank)
            else local_num_experts * 4
        )
        + (num_bt_banks * 4 if use_gather_bank else 4)
        + 3 * 4
    )

    b_smem = (  # noqa: F841  # SMEM-budget sentinel (parity with v2/bench_v2.py)
        smem_banks * bt * padded_top_k * 4
        + smem_banks * ep_size * padded_num_experts * 4
        + smem_banks * 2 * padded_num_experts * 4
        + smem_banks * padded_num_experts * 4
        + smem_banks * padded_num_experts * 4
    )

    total = (
        b_a2a_g_acc
        + b_topk_w
        + b_topk_id
        + b_output
        + b_w1
        + b_w3
        + b_w2
        + b_w1_scale
        + b_w3_scale
        + b_w2_scale
        + b_w1_dq
        + b_w3_dq
        + b_w2_dq
        + b_gate_acc
        + b_up_acc
        + b_x
        + b_y_acc
        + b_y_stage
        + b_scoped
        + b_sems
    )
    return total


def generate_v2_tune_candidates(
    *,
    intermediate_size: int,
    hidden_size: int,
    num_tokens: int,
    ep_size: int,
    num_experts: int,
    top_k: int,
    use_fp8: bool = False,
    quant_block_k: int = 128,
    direct_scaled_dot: bool = True,
    interleave_bt: bool = True,
    enable_bt_scatter_overlap: bool = True,
    vmem_budget_bytes: int = DEFAULT_TPU_VMEM_BUDGET_MB * 1024 * 1024,
    vmem_headroom: float = 0.95,
    max_configs: int = 48,
    bse: int = 256,
    verbose: bool = False,
) -> list[V2BlockConfig]:
    """v2-native candidate enumeration + 64 MB VMEM feasibility filter.

    Ported from ``generate_tune_candidates`` in bench_v2.py. Emits v2 5-field
    ``FusedMoEBlockConfig(bt, bf, btc, bse, bts)`` objects whose effective form
    fits ``vmem_budget_bytes * vmem_headroom`` per the v2 VMEM model. btc comes
    from ``_aligned_divisors(bts, 8)`` so every candidate has btc % 8 == 0
    (hence btc % t_packing(2) == 0). Returns at most ``max_configs`` configs.
    """
    local_num_tokens = num_tokens // ep_size
    effective_budget = int(vmem_budget_bytes * vmem_headroom)

    bf_list = sorted(
        {
            v
            for v in [128, 256, 512, 1024, 2048]
            if v <= intermediate_size and intermediate_size % v == 0
        }
    )

    bt_list: list[int] = []
    for p_val in [2, 4]:
        if local_num_tokens == p_val:
            bt_list.append(p_val)
    p = 8
    while p <= local_num_tokens:
        if local_num_tokens % p == 0:
            bt_list.append(p)
        p *= 2
    if not bt_list:
        bt_list = [local_num_tokens]
    bt_list = sorted(set(bt_list))

    configs: list[V2BlockConfig] = []
    seen: set[tuple] = set()

    for bt in bt_list:
        max_bts = bt * ep_size
        expected = bt * ep_size * top_k / num_experts
        lo = _pow2_floor(expected)
        hi = _pow2_ceil(expected)
        exp_floor8 = (int(expected) // 8) * 8
        exp_ceil8 = _align_to(int(math.ceil(expected)), 8)
        exp_hi8 = _align_to(int(math.ceil(expected * 1.25)), 8)
        bts_cands = sorted(
            {
                v
                for v in [bt, lo, hi, hi * 2, exp_floor8, exp_ceil8, exp_hi8]
                if 0 < v <= max_bts and v % 8 == 0
            }
        )
        if not bts_cands:
            bts_cands = [bt]

        for bts_val in bts_cands:
            btc_cands = _aligned_divisors(bts_val, 8)
            if not btc_cands:
                continue

            for bf in bf_list:
                for btc in btc_cands:
                    bc = V2BlockConfig(bt=bt, bf=bf, btc=btc, bse=bse, bts=bts_val)
                    try:
                        bc_eff = bc.effective_for(num_tokens=num_tokens, ep_size=ep_size)
                    except ValueError:
                        continue

                    key = (bc_eff.bt, bc_eff.bf, bc_eff.btc, bc_eff.bts)
                    if key in seen:
                        continue
                    seen.add(key)

                    est = _estimate_vmem_bytes_v2(
                        bt=bc_eff.bt,
                        bf=bc_eff.bf,
                        btc=bc_eff.btc,
                        bse=bc_eff.bse,
                        bts=bc_eff.bts,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        num_experts=num_experts,
                        top_k=top_k,
                        ep_size=ep_size,
                        num_tokens=num_tokens,
                        use_fp8=use_fp8,
                        quant_block_k=quant_block_k,
                        direct_scaled_dot=direct_scaled_dot,
                        interleave_bt=interleave_bt,
                        enable_bt_scatter_overlap=enable_bt_scatter_overlap,
                    )
                    if est > effective_budget:
                        if verbose:
                            print(
                                f"  VMEM skip bt={bc_eff.bt},bf={bc_eff.bf},"
                                f"btc={bc_eff.btc},bts={bc_eff.bts}: "
                                f"{est / (1024 * 1024):.1f}MB > "
                                f"{effective_budget / (1024 * 1024):.1f}MB"
                            )
                        continue
                    configs.append(bc)

    if len(configs) <= max_configs:
        return configs

    # Round-robin across (bt, bts) buckets, preferring large bf/btc first.
    buckets: dict[tuple, list[V2BlockConfig]] = {}
    for cfg in configs:
        bk = (cfg.bt, cfg.bts or cfg.bt)
        buckets.setdefault(bk, []).append(cfg)
    for bk in buckets:
        buckets[bk].sort(key=lambda c: (c.bf, c.btc), reverse=True)

    selected: list[V2BlockConfig] = []
    selected_keys: set[tuple] = set()
    bucket_keys = sorted(buckets.keys(), reverse=True)
    while len(selected) < max_configs:
        made_progress = False
        for bk in bucket_keys:
            bucket = buckets[bk]
            if not bucket:
                continue
            cfg = bucket.pop(0)
            key = (cfg.bt, cfg.bf, cfg.btc, cfg.bts)
            if key not in selected_keys:
                selected_keys.add(key)
                selected.append(cfg)
                made_progress = True
            if len(selected) >= max_configs:
                break
        if not made_progress:
            break
    return selected


def run_all(
    iters: int,
    *,
    weight_dtype: jnp.dtype = jnp.float8_e4m3fn,
    dtype: jnp.dtype = jnp.bfloat16,
    warmup_iters: int = 1,
    trace_root: str = "/tmp/sglang_jax_moe_trace",
    tune_block_config: bool = False,
    bt_candidates: list[int] | None = None,
    bts_candidates: list[int] | None = None,
    bf_candidates: list[int] | None = None,
    bd_candidates: list[int] | None = None,
    bse_candidates: list[int] | None = None,
    num_tokens: list[int] | None = None,
    num_experts: int = 384,
    top_k: int = 8,
    hidden_size: int = 6144,
    intermediate_size: int = 2048,
    ep_size: int | None = 32,
    activation: str = "silu",
    renormalize_topk_logits: bool = True,
    tpu_vmem_budget_bytes: int = DEFAULT_TPU_VMEM_BUDGET_MB * 1024 * 1024,
    tpu_vmem_headroom_ratio: float = 0.90,
    tpu_vmem_estimate_scale: float = 1.0,
    max_configs: int = 9,
    quant_block_k_override: int | None = None,
    metadata_mode: str = "direct",
    return_results: bool = False,
) -> list[dict[str, object]] | None:
    use_shared_expert = False  # lean decode tuner: omitted
    use_grouped_topk = False  # lean decode tuner: omitted

    token_list = DEFAULT_NUM_TOKENS if num_tokens is None else num_tokens
    raw_cases = make_moe_cases(
        num_tokens=token_list,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        renormalize_topk_logits=renormalize_topk_logits,
        num_expert_group=0,
        topk_group=0,
        name_prefix="fused_moe_v2",
    )
    if num_tokens is not None:
        requested = set(num_tokens)
        raw_cases = [case for case in raw_cases if case.num_tokens in requested]

    cases_all = list(select_cases(raw_cases))
    # If an explicit ep_size is requested, pin it (overriding select_cases'
    # device-count-based choice). This keeps tuned-table keys deterministic
    # regardless of how many devices `jax.devices()` reports.
    cases: list[MoEBenchmarkCase] = []
    for c in cases_all:
        target_ep = ep_size if ep_size is not None else c.ep_size
        if c.num_tokens % target_ep != 0 or c.num_experts % target_ep != 0:
            print(
                f"skip [case={c.name}] tokens={c.num_tokens}/experts={c.num_experts} "
                f"not divisible by ep_size={target_ep}"
            )
            continue
        cases.append(
            MoEBenchmarkCase(
                name=c.name,
                num_tokens=c.num_tokens,
                num_experts=c.num_experts,
                top_k=c.top_k,
                hidden_size=c.hidden_size,
                intermediate_size=c.intermediate_size,
                activation=c.activation,
                renormalize_topk_logits=c.renormalize_topk_logits,
                num_expert_group=c.num_expert_group,
                topk_group=c.topk_group,
                routed_scaling_factor=c.routed_scaling_factor,
                ep_size=target_ep,
                tp_size=1,
            )
        )
    if not cases:
        print("No runnable fused_moe_v2 cases after filtering.")
        return [] if return_results else None

    tuned_results: dict[str, dict[tuple, tuple[int, int, int, int, object]]] = {}
    if tune_block_config:
        from sgl_jax.srt.utils.jax_utils import get_device_name
    results: list[dict[str, object]] = []

    print(f"Running fused_moe_v2 benchmarks with weight_dtype={jnp.dtype(weight_dtype).name}")
    print(
        f"  metadata_mode={metadata_mode}, shared_expert={use_shared_expert}, grouped_topk={use_grouped_topk}"
    )
    print(
        "  shape: "
        f"num_experts={num_experts}, top_k={top_k}, hidden_size={hidden_size}, "
        f"intermediate_size={intermediate_size}, activation={activation}, "
        f"renormalize_topk_logits={renormalize_topk_logits}"
    )
    print(
        "  vmem_filter: "
        f"budget={tpu_vmem_budget_bytes / (1024 * 1024):.0f}MB, "
        f"headroom_ratio={tpu_vmem_headroom_ratio:.2f}, "
        f"estimate_scale={tpu_vmem_estimate_scale:.2f}"
    )

    for case in cases:
        mesh = build_mesh(ep_size=case.ep_size, tp_size=case.tp_size)
        mesh_ep = mesh.shape["tensor"]
        if mesh_ep != case.ep_size:
            print(f"warning [case={case.name}] mesh_ep={mesh_ep} != case.ep_size={case.ep_size}")
        local_num_tokens = case.num_tokens // mesh_ep
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}, "
            f"ep_size={case.ep_size}, local_num_tokens={local_num_tokens}"
        )
        print(
            f"  mesh: ep_size={case.ep_size}, tp_size={case.tp_size}, "
            f"devices_used={case.ep_size * case.tp_size}/{len(jax.devices())}"
        )

        data = prepare_fused_moe_inputs(
            case,
            weight_dtype=weight_dtype,
            mesh=mesh,
            include_weights=False,
            include_shared_expert=use_shared_expert,
        )
        # Balanced placeholder logits (lean tuner: no imbalance simulation).
        data["router_logits"] = jax.device_put(
            data["router_logits"], jax.sharding.NamedSharding(mesh, P("tensor", None))
        )

        # Determine quant_block_k for FP8 quantization (mirror v1 default 256).
        if quant_block_k_override is not None:
            quant_block_k = quant_block_k_override
        else:
            quant_block_k = 256 if weight_dtype == jnp.float8_e4m3fn else None

        if weight_dtype == jnp.float8_e4m3fn:
            quantization_config = QuantizationConfig(
                moe_weight_dtype=weight_dtype,
                moe_activation_dtype=None,  # activation stays bfloat16
            )
        else:
            quantization_config = None

        with jax.set_mesh(mesh):
            fused_layer = FusedEPMoEV2(
                hidden_size=case.hidden_size,
                num_experts=case.num_experts,
                num_experts_per_tok=case.top_k,
                ep_size=case.ep_size,
                mesh=mesh,
                intermediate_dim=case.intermediate_size,
                weight_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
                activation=case.activation,
                layer_id=0,
                renormalize_topk_logits=case.renormalize_topk_logits,
                use_grouped_topk=use_grouped_topk,
                num_groups=1,
                top_k_groups=1,
                num_shared_experts=0,
                moe_shared_expert_intermediate_size=None,
                quantization_config=quantization_config,
                metadata_mode=metadata_mode,
            )
            if quantization_config is not None:
                if quant_block_k is not None:
                    fused_layer.quant_block_k = quant_block_k
                fused_layer.quantize_weights()

            v2_block_cfgs: list[V2BlockConfig | None]
            if tune_block_config:
                use_fp8 = weight_dtype == jnp.float8_e4m3fn
                # v2 VMEM model needs the fp8 quant block size (defaults to 128
                # in the kernel/model). quant_block_k above may be 256 for the
                # weight quantizer, but the v2 VMEM scale buffers use the
                # kernel default; pass the effective quant_block_k (>=128).
                vmem_qbk = quant_block_k if (use_fp8 and quant_block_k) else 128
                effective_budget_mb = (
                    tpu_vmem_budget_bytes / (1024 * 1024)
                ) * tpu_vmem_headroom_ratio
                cand_cfgs = generate_v2_tune_candidates(
                    intermediate_size=case.intermediate_size,
                    hidden_size=case.hidden_size,
                    num_tokens=case.num_tokens,
                    ep_size=mesh_ep,
                    num_experts=case.num_experts,
                    top_k=case.top_k,
                    use_fp8=use_fp8,
                    quant_block_k=vmem_qbk,
                    vmem_budget_bytes=tpu_vmem_budget_bytes,
                    vmem_headroom=tpu_vmem_headroom_ratio,
                    max_configs=max_configs,
                    bse=(bse_candidates[0] if bse_candidates else 256),
                    verbose=True,
                )
                # Drop any config v2 itself would reject (mirrors kernel validate).
                v2_block_cfgs = []
                seen: set[tuple] = set()
                for c in cand_cfgs:
                    try:
                        v2_validate(
                            num_tokens=case.num_tokens,
                            num_experts=case.num_experts,
                            top_k=case.top_k,
                            hidden_size=case.hidden_size,
                            intermediate_size=case.intermediate_size,
                            dtype=dtype,
                            ep_size=mesh_ep,
                            block_config=c,
                        )
                    except ValueError:
                        continue
                    key = (c.bt, c.bf, c.btc, c.bse, c.bts)
                    if key in seen:
                        continue
                    seen.add(key)
                    v2_block_cfgs.append(c)
                print(
                    f"  v2 candidates: {len(cand_cfgs)} enumerated (<= {effective_budget_mb:.0f}MB "
                    f"effective VMEM) -> {len(v2_block_cfgs)} valid"
                )
                for c in v2_block_cfgs:
                    print(f"    cand bt={c.bt}, bf={c.bf}, btc={c.btc}, bse={c.bse}, bts={c.bts}")
                if not v2_block_cfgs:
                    print(
                        "  WARNING: no v2 candidates survived enumeration+validation for "
                        f"case={case.name}; nothing to time."
                    )
            else:
                v2_block_cfgs = [None]

            topk_module = TopK(
                topk=case.top_k,
                renormalize=case.renormalize_topk_logits,
                num_expert_group=0,
                topk_group=0,
                routed_scaling_factor=case.routed_scaling_factor,
                layer_id=0,
            )

            moe_def, moe_state = nnx.split(fused_layer)
            moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)
            topk_def, topk_state = nnx.split(topk_module)
            topk_state_leaves, topk_state_def = jax.tree_util.tree_flatten(topk_state)

            @partial(jax.jit, static_argnames=("moe_state_def", "topk_state_def", "block_config"))
            def run_v2(
                tokens,
                router_logits,
                *,
                moe_state_def,
                moe_state_leaves,
                topk_state_def,
                topk_state_leaves,
                block_config,
            ):
                moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
                moe = nnx.merge(moe_def, moe_state)
                topk_state = jax.tree_util.tree_unflatten(topk_state_def, topk_state_leaves)
                topk = nnx.merge(topk_def, topk_state)
                topk_weights, topk_ids = topk(router_logits)
                return moe(tokens, topk_weights, topk_ids, block_config=block_config)

            best: tuple[float, V2BlockConfig | None] | None = None
            default_ms: float | None = None
            n_succeeded = 0
            n_failed = 0
            for i, block_cfg in enumerate(v2_block_cfgs):
                tag = "default" if block_cfg is None else str(i)
                if block_cfg is None:
                    print("  fused_moe_v2 [default] -> (block_config=None, auto table lookup)")
                else:
                    print(
                        f"  fused_moe_v2 blocks [{i + 1}/{len(v2_block_cfgs)}] -> "
                        f"bt={block_cfg.bt}, bf={block_cfg.bf}, btc={block_cfg.btc}, "
                        f"bse={block_cfg.bse}, bts={block_cfg.bts}"
                    )

                def _compute(block_cfg=block_cfg):
                    return run_v2(
                        data["tokens"],
                        data["router_logits"],
                        moe_state_def=moe_state_def,
                        moe_state_leaves=moe_state_leaves,
                        topk_state_def=topk_state_def,
                        topk_state_leaves=topk_state_leaves,
                        block_config=block_cfg,
                    )

                task = "fused-moe-v2-k_.*"
                try:
                    if block_cfg is not None:
                        # Skip configs whose v2-effective form is invalid (raises).
                        v2_validate(
                            num_tokens=case.num_tokens,
                            num_experts=case.num_experts,
                            top_k=case.top_k,
                            hidden_size=case.hidden_size,
                            intermediate_size=case.intermediate_size,
                            dtype=dtype,
                            ep_size=mesh_ep,
                            block_config=block_cfg,
                        )
                    times = multiple_iteration_timeit_from_trace(
                        compute_func=_compute,
                        data_generator=lambda: (),
                        task=task,
                        tries=iters,
                        warmup=warmup_iters,
                        trace_root=trace_root,
                    )
                except ValueError as e:
                    print(f"SKIP fused_moe_v2 blocks [{i + 1}/{len(v2_block_cfgs)}], reason: {e}")
                    n_failed += 1
                    continue
                except jax.errors.JaxRuntimeError as e:
                    # RESOURCE_EXHAUSTED (VMEM OOM) and other runtime/compile
                    # failures: mark this config FAILED and keep sweeping.
                    msg = str(e)
                    short = msg.splitlines()[0] if msg else type(e).__name__
                    print(
                        f"FAILED fused_moe_v2 blocks [{i + 1}/{len(v2_block_cfgs)}] "
                        f"(bt={block_cfg.bt}, bf={block_cfg.bf}, btc={block_cfg.btc}, "
                        f"bse={block_cfg.bse}, bts={block_cfg.bts}): "
                        f"{type(e).__name__}: {short}",
                        flush=True,
                    )
                    n_failed += 1
                    continue
                except SystemExit as e:
                    print(
                        f"ERROR fused_moe_v2 blocks [{i + 1}/{len(v2_block_cfgs)}]: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)
                    raise
                except Exception as e:
                    print(
                        f"FAILED fused_moe_v2 blocks [{i + 1}/{len(v2_block_cfgs)}]: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)
                    n_failed += 1
                    continue

                if len(times) > 1:
                    times = times[1:]
                mean_ms = float(np.mean(times)) if times else float("nan")
                print(f"     fused_moe_v2[{tag}]: {mean_ms:.3f} ms (trace) | samples={times}")
                if np.isfinite(mean_ms):
                    n_succeeded += 1
                else:
                    n_failed += 1
                if block_cfg is None:
                    default_ms = mean_ms
                if tune_block_config and np.isfinite(mean_ms):
                    if best is None or mean_ms < best[0]:
                        best = (mean_ms, block_cfg)

            if tune_block_config:
                print(
                    f"  [case={case.name}] sweep summary: {n_succeeded} succeeded, "
                    f"{n_failed} failed out of {len(v2_block_cfgs)} candidate(s)."
                )
                if n_succeeded == 0:
                    print(
                        f"  NO CONFIG SUCCEEDED for case={case.name} "
                        f"(tokens={case.num_tokens}, ep={case.ep_size}): "
                        "every candidate failed (VMEM OOM / compile / validate). "
                        "No tuned_v2 entry emitted for this case."
                    )

            if tune_block_config and best is not None:
                best_ms, best_cfg = best
                if best_cfg is None:
                    print(f"  best: default ({best_ms:.3f} ms)")
                else:
                    device_name = get_device_name()
                    # v2 table key (matches get_simplified_key ordering, after device):
                    table_key = (
                        jnp.dtype(dtype).name,
                        jnp.dtype(weight_dtype).name,
                        case.num_tokens,
                        case.num_experts,
                        case.top_k,
                        case.hidden_size,
                        case.intermediate_size,
                        case.ep_size,
                        use_shared_expert,
                        use_grouped_topk,
                    )
                    cfg_tuple = (
                        best_cfg.bt,
                        best_cfg.bf,
                        best_cfg.btc,
                        best_cfg.bse,
                        best_cfg.bts,
                    )
                    print(
                        f"  best: bt={best_cfg.bt}, bf={best_cfg.bf}, btc={best_cfg.btc}, "
                        f"bse={best_cfg.bse}, bts={best_cfg.bts} ({best_ms:.3f} ms)"
                    )
                    print(
                        f"  tuned_v2[{device_name!r}][{table_key}] = {cfg_tuple}   "
                        f"# {best_ms:.3f} ms"
                    )
                    per_device = tuned_results.setdefault(device_name, {})
                    per_device[table_key] = cfg_tuple

            if return_results:
                if tune_block_config:
                    if best is None:
                        rb_ms, rb_cfg = float("nan"), None
                    else:
                        rb_ms, rb_cfg = best
                else:
                    rb_ms, rb_cfg = default_ms, None
                results.append(
                    {
                        "case": case.name,
                        "num_tokens": case.num_tokens,
                        "num_experts": case.num_experts,
                        "top_k": case.top_k,
                        "hidden_size": case.hidden_size,
                        "intermediate_size": case.intermediate_size,
                        "ep_size": case.ep_size,
                        "best_ms": rb_ms,
                        "best_cfg": (
                            (rb_cfg.bt, rb_cfg.bf, rb_cfg.btc, rb_cfg.bse, rb_cfg.bts)
                            if rb_cfg is not None
                            else None
                        ),
                    }
                )

    if tune_block_config and tuned_results:
        print("\n# --- Copy/paste into v2/tuned_block_configs.py ---")
        for device_name in sorted(tuned_results.keys()):
            entries = tuned_results[device_name]
            print(f'TUNED_BLOCK_CONFIGS.setdefault("{device_name}", {{}}).update({{')
            for k in sorted(entries.keys(), key=lambda t: (t[2], t[3], t[4], t[5], t[6], t[7])):
                print(f"    {k}: {entries[k]},")
            print("})\n")

    if return_results:
        return results
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune fused_ep_moe_v2 block configs.")
    parser.add_argument("--iters", type=int, default=20, help="Number of benchmark iterations.")
    parser.add_argument(
        "--trace-root",
        type=str,
        default="/tmp/sglang_jax_moe_trace",
        help="Where jax.profiler.trace writes (point at the artifact dir to persist traces).",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warmup iterations before profiling (per case / block_config).",
    )
    parser.add_argument(
        "--weight-dtype",
        type=str,
        default="float8_e4m3fn",
        choices=["bfloat16", "float8_e4m3fn"],
        help="Weight dtype. fp8 quantization is implied by float8_e4m3fn / --quant-block-k.",
    )
    parser.add_argument(
        "--tune-block-config",
        action="store_true",
        help="Benchmark candidate block_config variants and print the best v2 tuned entry.",
    )
    parser.add_argument("--bt-candidates", type=int, nargs="+", help="Candidate list for bt.")
    parser.add_argument("--bts-candidates", type=int, nargs="+", help="Candidate list for bts.")
    parser.add_argument("--bf-candidates", type=int, nargs="+", help="Candidate list for bf.")
    parser.add_argument(
        "--bd-candidates",
        type=int,
        nargs="+",
        help="(DEPRECATED, ignored) v1-only bd1/bd2; the v2 kernel has no bd blocking.",
    )
    parser.add_argument(
        "--bse-candidates",
        type=int,
        nargs="+",
        help="Candidate list for bse (shared expert tile; lean tuner ignores SE but bse is kept).",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[64],
        help="Token counts to benchmark (e.g. --num-tokens 64 128 256).",
    )
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument(
        "--renormalize-topk-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--quant-block-k",
        type=int,
        default=128,
        help="Sub-channel quantization block size (fp8). Set to 0 / negative to disable.",
    )
    parser.add_argument(
        "--metadata-mode",
        type=str,
        default="direct",
        choices=["recursive", "direct", "jax"],
        help="v2 metadata mode (production decode mode for MiMo-V2-Pro is 'direct').",
    )
    parser.add_argument(
        "--tpu-vmem-budget-mb",
        type=int,
        default=DEFAULT_TPU_VMEM_BUDGET_MB,
        help="VMEM budget used to filter candidate block configs (MiB).",
    )
    parser.add_argument(
        "--tpu-vmem-headroom-ratio",
        type=float,
        default=0.90,
        help="Fraction of the VMEM budget exposed to the estimator after headroom reservation.",
    )
    parser.add_argument(
        "--tpu-vmem-estimate-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the VMEM estimate before candidate filtering.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=9,
        help="Maximum number of block configs to benchmark per case when --tune-block-config is set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        faulthandler.enable(file=sys.stdout, all_threads=True)
    except Exception:
        pass
    args = parse_args()
    DTYPE_MAP = {
        "bfloat16": jnp.bfloat16,
        "float8_e4m3fn": jnp.float8_e4m3fn,
    }
    weight_dtype = DTYPE_MAP[args.weight_dtype]
    quant_block_k = args.quant_block_k if args.quant_block_k and args.quant_block_k > 0 else None
    tpu_vmem_budget_bytes = int(args.tpu_vmem_budget_mb) * 1024 * 1024
    try:
        run_all(
            args.iters,
            weight_dtype=weight_dtype,
            warmup_iters=args.warmup_iters,
            trace_root=args.trace_root,
            tune_block_config=args.tune_block_config,
            bt_candidates=args.bt_candidates,
            bts_candidates=args.bts_candidates,
            bf_candidates=args.bf_candidates,
            bd_candidates=args.bd_candidates,
            bse_candidates=args.bse_candidates,
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            top_k=args.top_k,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            ep_size=args.ep_size,
            activation=args.activation,
            renormalize_topk_logits=args.renormalize_topk_logits,
            tpu_vmem_budget_bytes=tpu_vmem_budget_bytes,
            tpu_vmem_headroom_ratio=args.tpu_vmem_headroom_ratio,
            tpu_vmem_estimate_scale=args.tpu_vmem_estimate_scale,
            max_configs=args.max_configs,
            quant_block_k_override=quant_block_k,
            metadata_mode=args.metadata_mode,
            return_results=True,
        )
    except BaseException as e:
        print(f"FATAL: {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
