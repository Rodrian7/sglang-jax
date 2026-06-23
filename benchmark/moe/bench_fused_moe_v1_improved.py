"""Benchmark explicit block configs for the fused_moe v1-improved path.

This runner is intentionally narrower than ``bench_fused_moe.py``: it only
tests the normal v1 path with explicit block configs and records compile
failures as metrics rows so Falcon can report both fit and no-fit configs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from benchmark.moe.utils import (
    MoEBenchmarkCase,
    MoEImbalanceSimulator,
    build_mesh,
    prepare_fused_moe_inputs,
    select_cases,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    validate_fused_moe_block_config,
)
from sgl_jax.srt.layers.moe import FusedEPMoE, TopK

DEFAULT_CONFIGS: tuple[str, ...] = (
    "128,128,512,128,512,1024,1024",
    "128,160,512,80,512,1024,1024",
    "128,128,512,64,512,2048,1024",
    "128,160,256,80,256,2048,1024",
    "128,128,256,128,256,2048,2048",
    "128,80,512,80,512,1024,2048",
)


def _dtype_packing(dtype: jnp.dtype) -> int:
    bits = jnp.dtype(dtype).itemsize * 8
    if 32 % bits != 0:
        raise ValueError(f"Unsupported dtype packing for {dtype=} ({bits=} bits).")
    return 32 // bits


def _parse_config(spec: str, *, hidden_size: int) -> FusedMoEBlockConfig:
    parts = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) != 7:
        raise ValueError("config must be 'bt,bts,bf,btc,bfc,bd1c,bd2c', " f"got {spec!r}")
    bt, bts, bf, btc, bfc, bd1c, bd2c = parts
    return FusedMoEBlockConfig(
        bt=bt,
        bts=bts,
        bf=bf,
        bd1=hidden_size,
        bd2=hidden_size,
        btc=btc,
        bfc=bfc,
        bd1c=bd1c,
        bd2c=bd2c,
        bse=bf,
    )


def _explicit_vmem_bytes(
    *,
    cfg: FusedMoEBlockConfig,
    hidden_size: int,
    top_k: int,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    quant_block_k: int | None,
) -> tuple[int, dict[str, int]]:
    token_bytes = jnp.dtype(dtype).itemsize
    weight_bytes = jnp.dtype(weight_dtype).itemsize
    t_packing = _dtype_packing(dtype)
    hidden_per_pack = hidden_size // t_packing
    padded_top_k = ((top_k + 127) // 128) * 128

    parts: dict[str, int] = {}
    parts["a2a_g_acc"] = 2 * top_k * math.gcd(cfg.bt, 16) * hidden_size * token_bytes
    parts["topk_weights_ids"] = 2 * cfg.bt * padded_top_k * 4 * 2
    parts["output_x2"] = 2 * cfg.bt * hidden_size * token_bytes
    parts["w1_x2"] = 2 * hidden_size * cfg.bf * weight_bytes
    parts["w3_x2"] = 2 * hidden_size * cfg.bf * weight_bytes
    parts["w2_x2"] = 2 * cfg.bf * hidden_size * weight_bytes
    if quant_block_k is not None:
        parts["w1_scale_x2"] = 2 * t_packing * (hidden_per_pack // quant_block_k) * cfg.bf * 4
        parts["w3_scale_x2"] = 2 * t_packing * (hidden_per_pack // quant_block_k) * cfg.bf * 4
        parts["w2_scale_x2"] = 2 * t_packing * (cfg.bf // quant_block_k) * hidden_per_pack * 4
    parts["b_acc"] = 2 * cfg.bts * cfg.bf * 4
    parts["token_stage_x2"] = 2 * cfg.bts * hidden_size * token_bytes
    parts["acc_stage_x3"] = 3 * cfg.bts * hidden_size * token_bytes
    return sum(parts.values()), parts


def _mb(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _write_metric(path: Path | None, row: dict[str, object]) -> None:
    line = json.dumps(row, sort_keys=True, default=_json_default)
    print("METRIC " + line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _is_leader() -> bool:
    value = os.environ.get("FALCON_OPERATOR_IS_LEADER")
    if value is not None:
        return value == "1"
    return jax.process_index() == 0


def run(args: argparse.Namespace) -> None:
    dtype = jnp.bfloat16
    weight_dtype = jnp.float8_e4m3fn if args.weight_dtype == "float8_e4m3fn" else jnp.bfloat16
    quant_block_k = args.quant_block_k if weight_dtype == jnp.float8_e4m3fn else None

    raw_case = MoEBenchmarkCase(
        name="fused_moe_v1_improved",
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        activation=args.activation,
        renormalize_topk_logits=args.renormalize_topk_logits,
        num_expert_group=args.num_expert_group,
        topk_group=args.topk_group,
    )
    case = next(iter(select_cases([raw_case])))
    if case.tp_size != 1:
        raise ValueError(f"Expected tp_size=1 for fused_moe, got {case.tp_size}")

    mesh = build_mesh(ep_size=case.ep_size, tp_size=case.tp_size)
    mesh_ep = mesh.shape["tensor"]
    if mesh_ep != case.ep_size:
        raise ValueError(f"mesh_ep={mesh_ep} != case.ep_size={case.ep_size}")

    configs = [
        _parse_config(spec, hidden_size=case.hidden_size).effective_for(
            num_tokens=case.num_tokens,
            ep_size=case.ep_size,
            dtype=dtype,
            intermediate_size=case.intermediate_size,
            hidden_size=case.hidden_size,
            quant_block_k=quant_block_k,
        )
        for spec in (args.config or DEFAULT_CONFIGS)
    ]

    metric_path = Path(args.output) if args.output and _is_leader() else None
    if metric_path is not None and metric_path.exists():
        metric_path.unlink()

    print(
        "v1-improved bench: "
        f"tokens={case.num_tokens}, experts={case.num_experts}, top_k={case.top_k}, "
        f"hidden={case.hidden_size}, intermediate={case.intermediate_size}, ep_size={case.ep_size}, "
        f"devices={len(jax.devices())}, process={jax.process_index()}/{jax.process_count()}, "
        f"weight_dtype={jnp.dtype(weight_dtype).name}, quant_block_k={quant_block_k}",
        flush=True,
    )

    data = prepare_fused_moe_inputs(
        case,
        weight_dtype=weight_dtype,
        mesh=mesh,
        include_weights=False,
        include_shared_expert=False,
    )

    if args.num_expert_group and args.topk_group:
        custom_logits, sim_stats = MoEImbalanceSimulator.create_grouped_topk_logits(
            case.num_tokens,
            case.num_experts,
            case.top_k,
            num_groups=args.num_expert_group,
            top_k_groups=args.topk_group,
            mode=args.imbalance_mode,
            hotspot_ratio=args.hotspot_ratio,
            hotspot_count=args.hotspot_count,
            zero_expert_count=args.zero_expert_count,
            non_hotspot_alpha=args.non_hotspot_alpha,
        )
        use_grouped_topk = True
        print(f"imbalance(sim): {sim_stats}", flush=True)
    else:
        target_counts = MoEImbalanceSimulator.generate_counts(
            case.num_tokens,
            case.top_k,
            case.num_experts,
            args.imbalance_mode,
            hotspot_ratio=args.hotspot_ratio,
            hotspot_count=args.hotspot_count,
            zero_expert_count=args.zero_expert_count,
            non_hotspot_alpha=args.non_hotspot_alpha,
        )
        custom_logits = MoEImbalanceSimulator.create_logits_from_counts(
            case.num_tokens, case.num_experts, case.top_k, target_counts
        )
        use_grouped_topk = False

    data["router_logits"] = jax.device_put(
        custom_logits.astype(jnp.bfloat16), NamedSharding(mesh, P("tensor", None))
    )

    with jax.set_mesh(mesh):
        quantization_config = (
            QuantizationConfig(moe_weight_dtype=weight_dtype, moe_activation_dtype=None)
            if weight_dtype == jnp.float8_e4m3fn
            else None
        )
        fused_layer = FusedEPMoE(
            hidden_size=case.hidden_size,
            num_experts=case.num_experts,
            num_experts_per_tok=case.top_k,
            ep_size=case.ep_size,
            mesh=mesh,
            intermediate_dim=case.intermediate_size,
            weight_dtype=jnp.bfloat16,
            dtype=dtype,
            activation=case.activation,
            layer_id=0,
            renormalize_topk_logits=case.renormalize_topk_logits,
            use_grouped_topk=use_grouped_topk,
            num_groups=case.num_expert_group if use_grouped_topk else 1,
            top_k_groups=case.topk_group if use_grouped_topk else 1,
            quantization_config=quantization_config,
        )
        if quantization_config is not None:
            fused_layer.quant_block_k = quant_block_k
            fused_layer.quantize_weights()

        topk_module = TopK(
            topk=case.top_k,
            renormalize=case.renormalize_topk_logits,
            num_expert_group=case.num_expert_group if use_grouped_topk else 0,
            topk_group=case.topk_group if use_grouped_topk else 0,
            layer_id=0,
        )

        moe_def, moe_state = nnx.split(fused_layer)
        moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)
        topk_def, topk_state = nnx.split(topk_module)
        topk_state_leaves, topk_state_def = jax.tree_util.tree_flatten(topk_state)

        @partial(jax.jit, static_argnames=("moe_state_def", "topk_state_def", "block_config"))
        def run_one(
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

        for idx, cfg in enumerate(configs):
            explicit_vmem, vmem_parts = _explicit_vmem_bytes(
                cfg=cfg,
                hidden_size=case.hidden_size,
                top_k=case.top_k,
                dtype=dtype,
                weight_dtype=weight_dtype,
                quant_block_k=quant_block_k,
            )
            cfg_kwargs = cfg.as_kwargs()
            print(
                f"\nconfig[{idx}] {cfg_kwargs} explicit_vmem={_mb(explicit_vmem):.2f} MiB "
                f"parts={{{', '.join(f'{k}:{_mb(v):.2f}' for k, v in vmem_parts.items())}}}",
                flush=True,
            )

            row_base: dict[str, object] = {
                "variant": "v1_improved",
                "config_index": idx,
                "block_config": cfg_kwargs,
                "num_tokens": case.num_tokens,
                "num_experts": case.num_experts,
                "top_k": case.top_k,
                "hidden_size": case.hidden_size,
                "intermediate_size": case.intermediate_size,
                "ep_size": case.ep_size,
                "weight_dtype": jnp.dtype(weight_dtype).name,
                "quant_block_k": quant_block_k,
                "explicit_vmem_mib": _mb(explicit_vmem),
                "explicit_vmem_parts_mib": {k: _mb(v) for k, v in vmem_parts.items()},
            }
            started = time.perf_counter()
            try:
                validate_fused_moe_block_config(
                    num_tokens=case.num_tokens,
                    num_experts=case.num_experts,
                    top_k=case.top_k,
                    hidden_size=case.hidden_size,
                    intermediate_size=case.intermediate_size,
                    dtype=dtype,
                    ep_size=mesh_ep,
                    quant_block_k=quant_block_k,
                    block_config=cfg,
                )

                def _compute(block_cfg=cfg):
                    return run_one(
                        data["tokens"],
                        data["router_logits"],
                        moe_state_def=moe_state_def,
                        moe_state_leaves=moe_state_leaves,
                        topk_state_def=topk_state_def,
                        topk_state_leaves=topk_state_leaves,
                        block_config=block_cfg,
                    )

                times = multiple_iteration_timeit_from_trace(
                    compute_func=_compute,
                    data_generator=lambda: (),
                    task="fused-moe-k_.*",
                    tries=args.iters,
                    warmup=args.warmup_iters,
                    trace_root=args.trace_root,
                )
                if len(times) > 1:
                    times = times[1:]
                mean_ms = float(np.mean(times)) if times else float("nan")
                _write_metric(
                    metric_path,
                    {
                        **row_base,
                        "status": "ok",
                        "latency_ms": mean_ms,
                        "samples_ms": [float(x) for x in times],
                        "elapsed_s": time.perf_counter() - started,
                    },
                )
            except Exception as exc:  # keep testing later configs after no-fit compile failures
                _write_metric(
                    metric_path,
                    {
                        **row_base,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:4000],
                        "elapsed_s": time.perf_counter() - started,
                    },
                )
                print(traceback.format_exc(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trace-root", type=str, default="/tmp/sglang_jax_v1_improved_trace")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--config", action="append", help="bt,bts,bf,btc,bfc,bd1c,bd2c")
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument(
        "--renormalize-topk-logits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-expert-group", type=int, default=8)
    parser.add_argument("--topk-group", type=int, default=4)
    parser.add_argument(
        "--imbalance-mode", choices=["balanced", "hotspot", "sparse_hotspot"], default="balanced"
    )
    parser.add_argument("--hotspot-ratio", type=float, default=1.0)
    parser.add_argument("--hotspot-count", type=int, default=48)
    parser.add_argument("--zero-expert-count", type=int, default=0)
    parser.add_argument("--non-hotspot-alpha", type=float, default=100.0)
    parser.add_argument(
        "--weight-dtype", choices=["bfloat16", "float8_e4m3fn"], default="float8_e4m3fn"
    )
    parser.add_argument("--quant-block-k", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except BaseException as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
