"""
Tune fused_ep_moe_v2 block configs (lean, decode-focused v2 adaptation).

This is a v2 adaptation of ``benchmark/moe/bench_fused_moe.py``. It tunes the
``fused_ep_moe_v2`` kernel's block config by timing candidate configs with the
SAME canonical marker-based timer the v1 tuner uses
(``multiple_iteration_timeit_from_trace``), then prints the best config as a
v2 tuned-table entry that can be pasted into
``sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py``.

It deliberately reuses the v1 tuner's infrastructure by IMPORTING it rather
than copy-pasting:
  - ``select_block_configs`` from bench_fused_moe (candidate enumeration with
    VMEM filtering). It returns v1 ``FusedMoEBlockConfig`` objects, which we
    convert to v2 configs using the shared 5 fields (bt, bf, btc, bse, bts).
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
import sys
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec as P

# Reuse the v1 tuner's candidate enumeration (VMEM-filtered) verbatim.
from benchmark.moe.bench_fused_moe import (
    DEFAULT_TPU_VMEM_BUDGET_MB,
    select_block_configs,
)
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


def _to_v2_config(c) -> V2BlockConfig:
    """Convert a v1 FusedMoEBlockConfig (or any object with bt/bf/btc/bse/bts)
    to the v2 5-field FusedMoEBlockConfig."""
    return V2BlockConfig(bt=c.bt, bf=c.bf, btc=c.btc, bse=c.bse, bts=c.bts)


def run_all(
    iters: int,
    *,
    weight_dtype: jnp.dtype = jnp.float8_e4m3fn,
    dtype: jnp.dtype = jnp.bfloat16,
    warmup_iters: int = 1,
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
                v1_cfgs = select_block_configs(
                    case,
                    dtype,
                    weight_dtype=weight_dtype,
                    router_dtype=data["router_logits"].dtype,
                    bt_candidates=bt_candidates or [2, 4, 8, 16, 32, 64, 128, 256, 512],
                    bts_candidates=bts_candidates,
                    bf_candidates=bf_candidates or [128, 256, 512, 1024, 2048],
                    bd_candidates=bd_candidates or [256, 512, 1024, 2048, 4096, 8192],
                    bse_candidates=bse_candidates,
                    tpu_vmem_budget_bytes=tpu_vmem_budget_bytes,
                    tpu_vmem_headroom_ratio=tpu_vmem_headroom_ratio,
                    tpu_vmem_estimate_scale=tpu_vmem_estimate_scale,
                    max_configs=max_configs,
                    use_shared_expert=use_shared_expert,
                    quant_block_k=quant_block_k,
                    excluded_configs=None,
                )
                # Convert v1 -> v2 configs, dedup on the v2 5-tuple.
                v2_block_cfgs = []
                seen: set[tuple] = set()
                for c in v1_cfgs:
                    v2c = _to_v2_config(c)
                    key = (v2c.bt, v2c.bf, v2c.btc, v2c.bse, v2c.bts)
                    if key in seen:
                        continue
                    seen.add(key)
                    v2_block_cfgs.append(v2c)
                print(f"  v2 candidates: {len(v1_cfgs)} v1 -> {len(v2_block_cfgs)} unique v2")
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
                    )
                except ValueError as e:
                    print(f"SKIP fused_moe_v2 blocks [{i + 1}/{len(v2_block_cfgs)}], reason: {e}")
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
                        f"ERROR fused_moe_v2 blocks [{i + 1}/{len(v2_block_cfgs)}]: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)
                    continue

                if len(times) > 1:
                    times = times[1:]
                mean_ms = float(np.mean(times)) if times else float("nan")
                print(f"     fused_moe_v2[{tag}]: {mean_ms:.3f} ms (trace) | samples={times}")
                if block_cfg is None:
                    default_ms = mean_ms
                if tune_block_config and np.isfinite(mean_ms):
                    if best is None or mean_ms < best[0]:
                        best = (mean_ms, block_cfg)

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
    parser.add_argument("--bd-candidates", type=int, nargs="+", help="Candidate list for bd1/bd2.")
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
