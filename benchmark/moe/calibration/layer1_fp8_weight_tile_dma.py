"""Layer 1 fused-MoE FP8 weight + scale tile DMA calibration.

This module mirrors the v2 production kernel `start_fetch_w1`, `start_fetch_w3`,
and `start_fetch_w2` HBM->VMEM async copies in
`python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py` for the MiMo v2 Pro FP8 path.

Each fetch in v2 issues TWO `pltpu.make_async_copy` ops per t_packing iteration:
the FP8 weight tile and the f32 quant-scale tile (quant_block_k=128). Both
copies share the same DMA semaphore. The `wait_fetch_*` helper drains by issuing
two self-copy waits on that same sem (one for the weight VMEM buffer, one for
the scale VMEM buffer). This calibration kernel reproduces exactly that pattern
and is side-effect-only (no MXU, no A2A, no metadata, no expert traversal).

Companion: docs/performance/mimo_v2_pro_fused_moe_v2_current_state.md (J3).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from benchmark.moe.calibration.common import (
    build_observation_row,
    collect_runtime_identity,
)

SCENARIO_LAYER1_FP8_WEIGHT_TILE_DMA = "layer1_fp8_weight_tile_dma"
SUITE_V7X32_FP8_WEIGHT_TILE_DMA_MIMO = "v7x32_fp8_weight_tile_dma_mimo"
SUPPORTED_SUITES = (SUITE_V7X32_FP8_WEIGHT_TILE_DMA_MIMO,)

DTYPE = "float8_e4m3fn"
WEIGHT_DTYPE = "float8_e4m3fn"
SCALE_DTYPE = "float32"
WEIGHT_BYTES = 1
SCALE_BYTES = 4
T_PACKING = 2
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
H_PER_T_PACKING = HIDDEN_SIZE // T_PACKING  # 3072
NUM_EXPERTS = 384
LOCAL_NUM_EXPERTS = 12
QUANT_BLOCK_K = 128

KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py"
DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 9
DEFAULT_TRACE_DISCARD_RUNS = 1
VMEM_LIMIT_BYTES = 96 * 1024 * 1024
LOCAL_SEMAPHORE_SHAPE = (2, 14)

STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"

WEIGHT_PATHS: tuple[str, ...] = ("w1", "w3", "w2")
PATH_CLASS_MAP: dict[str, str] = {"w1": "w1w3", "w3": "w1w3", "w2": "w2"}
KERNEL_LINE_MAP: dict[str, int] = {"w1": 1099, "w3": 1139, "w2": 1179}
SEM_INDEX_MAP: dict[str, int] = {"w1": 4, "w3": 5, "w2": 6}

TARGET_RUNTIME_V7X32 = {
    "device_type": "v7x",
    "falcon_device_count": 32,
    "falcon_device_topo": "2x2x4",
    "replica": 4,
    "jax_device_count": 32,
    "jax_local_device_count": 8,
    "jax_process_count": 4,
    "chip_count": 16,
    "tensorcore_or_jax_device_count": 32,
}

IMPLEMENTATION_NOTE = (
    "Measured with a Pallas TPU microkernel that issues the v2 fused-MoE "
    "start_fetch_w1/w3/w2 HBM->VMEM async copies for both the FP8 weight tile "
    "and the f32 quant-scale tile (quant_block_k=128) for t_packing=2, then "
    "drains via two self-copy waits on the shared DMA semaphore. Side-effect "
    "only; no MXU compute, no A2A, no metadata, no expert traversal. "
    "bytes_hbm = bytes_weight + bytes_scale (FP8 1B + f32 scale 4B)."
)


@dataclass(frozen=True)
class FP8WeightTilePlan:
    path: str
    path_class: str
    bf: int
    bytes_weight: int
    bytes_scale: int
    bytes_total: int
    dma_count_weight: int
    dma_count_scale: int
    weight_tile_shape: tuple[int, ...]
    scale_tile_shape: tuple[int, ...]


def plan_for(path: str, bf: int) -> FP8WeightTilePlan:
    if path not in WEIGHT_PATHS:
        raise ValueError(f"Unsupported path {path!r}; expected one of {WEIGHT_PATHS}")
    if bf <= 0:
        raise ValueError(f"bf={bf} must be positive")
    if INTERMEDIATE_SIZE % bf != 0:
        raise ValueError(f"bf={bf} must divide intermediate_size={INTERMEDIATE_SIZE}")
    if bf < QUANT_BLOCK_K and path == "w2":
        raise ValueError(
            f"For path=w2, bf={bf} must be >= quant_block_k={QUANT_BLOCK_K} so the scale "
            "tile is non-empty along the bf axis."
        )
    if path == "w2":
        weight_tile = (T_PACKING, bf, H_PER_T_PACKING)
        scale_tile = (T_PACKING, bf // QUANT_BLOCK_K, 1, H_PER_T_PACKING)
        bytes_scale = T_PACKING * (bf // QUANT_BLOCK_K) * 1 * H_PER_T_PACKING * SCALE_BYTES
    else:
        weight_tile = (T_PACKING, H_PER_T_PACKING, bf)
        scale_tile = (T_PACKING, H_PER_T_PACKING // QUANT_BLOCK_K, 1, bf)
        bytes_scale = T_PACKING * (H_PER_T_PACKING // QUANT_BLOCK_K) * 1 * bf * SCALE_BYTES
    bytes_weight = T_PACKING * bf * H_PER_T_PACKING * WEIGHT_BYTES
    return FP8WeightTilePlan(
        path=path,
        path_class=PATH_CLASS_MAP[path],
        bf=bf,
        bytes_weight=bytes_weight,
        bytes_scale=bytes_scale,
        bytes_total=bytes_weight + bytes_scale,
        dma_count_weight=T_PACKING,
        dma_count_scale=T_PACKING,
        weight_tile_shape=weight_tile,
        scale_tile_shape=scale_tile,
    )


def build_rows(
    *,
    suite: str,
    paths: Iterable[str],
    bf_values: Iterable[int],
    execution_mode: str,
    runtime: dict[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if suite not in SUPPORTED_SUITES:
        raise ValueError(f"Unsupported suite: {suite}")
    runtime = collect_runtime_identity() if runtime is None else runtime
    paths_list = list(paths)
    bf_list = list(bf_values)

    if execution_mode != "pallas":
        return _schema_only_rows(
            paths_list,
            bf_list,
            suite=suite,
            execution_mode=execution_mode,
            runtime=runtime,
            source=source,
            metadata=metadata,
            implementation_note=(
                "layer1_fp8_weight_tile_dma emitted schema-only rows; "
                f"execution_mode={execution_mode!r} is not measured."
            ),
        )

    unavailable = _pallas_unavailable_note(runtime)
    if unavailable is not None:
        return _schema_only_rows(
            paths_list,
            bf_list,
            suite=suite,
            execution_mode=execution_mode,
            runtime=runtime,
            source=source,
            metadata=metadata,
            implementation_note=unavailable,
        )

    warmup = _positive_int_env("CALIBRATION_LAYER1_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    samples_n = _positive_int_env("CALIBRATION_LAYER1_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard = _nonnegative_int_env(
        "CALIBRATION_LAYER1_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER1_TRACE_ROOT", "/tmp/sglang_jax_layer1_fp8_dma_trace")

    rows: list[dict[str, Any]] = []
    for path in paths_list:
        for bf in bf_list:
            try:
                plan = plan_for(path, bf)
            except ValueError as exc:
                rows.append(
                    _make_row(
                        plan_or_proxy=_proxy_plan(path=path, bf=bf),
                        samples=[],
                        status=STATUS_NOT_IMPLEMENTED,
                        suite=suite,
                        execution_mode=execution_mode,
                        runtime=runtime,
                        source=source,
                        metadata=metadata,
                        implementation_note=f"layer1_fp8_weight_tile_dma plan rejected: {exc}",
                    )
                )
                continue
            try:
                samples = _measure_dma_ms(
                    plan,
                    warmup_runs=warmup,
                    sample_runs=samples_n,
                    discard_runs=discard,
                    trace_root=trace_root,
                )
                status = STATUS_MEASURED
                impl_note = IMPLEMENTATION_NOTE
            except Exception as exc:  # pragma: no cover — runtime-dependent
                samples = []
                status = STATUS_NOT_IMPLEMENTED
                impl_note = (
                    "layer1_fp8_weight_tile_dma Pallas measurement failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            rows.append(
                _make_row(
                    plan_or_proxy=plan,
                    samples=samples,
                    status=status,
                    suite=suite,
                    execution_mode=execution_mode,
                    runtime=runtime,
                    source=source,
                    metadata=metadata,
                    implementation_note=impl_note,
                )
            )
    return rows


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _schema_only_rows(
    paths: list[str],
    bf_values: list[int],
    *,
    suite: str,
    execution_mode: str,
    runtime: dict[str, Any],
    source: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    implementation_note: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for bf in bf_values:
            try:
                plan = plan_for(path, bf)
            except ValueError:
                plan = _proxy_plan(path=path, bf=bf)
            rows.append(
                _make_row(
                    plan_or_proxy=plan,
                    samples=[],
                    status=STATUS_NOT_IMPLEMENTED,
                    suite=suite,
                    execution_mode=execution_mode,
                    runtime=runtime,
                    source=source,
                    metadata=metadata,
                    implementation_note=implementation_note,
                )
            )
    return rows


def _proxy_plan(path: str, bf: int) -> FP8WeightTilePlan:
    return FP8WeightTilePlan(
        path=path,
        path_class=PATH_CLASS_MAP.get(path, "unknown"),
        bf=bf,
        bytes_weight=0,
        bytes_scale=0,
        bytes_total=0,
        dma_count_weight=0,
        dma_count_scale=0,
        weight_tile_shape=(),
        scale_tile_shape=(),
    )


def _make_row(
    *,
    plan_or_proxy: FP8WeightTilePlan,
    samples: list[float],
    status: str,
    suite: str,
    execution_mode: str,
    runtime: dict[str, Any],
    source: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    implementation_note: str,
) -> dict[str, Any]:
    plan = plan_or_proxy
    fp8_meta: dict[str, Any] = {
        "matrix_kind": "fused_moe_fp8_weight_tile_dma",
        "target_runtime": TARGET_RUNTIME_V7X32,
        "target_family": {
            "model": "MiMo v2 Pro",
            "dtype": DTYPE,
            "weight_dtype": WEIGHT_DTYPE,
            "scale_dtype": SCALE_DTYPE,
            "quant_block_k": QUANT_BLOCK_K,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_experts": NUM_EXPERTS,
            "local_num_experts": LOCAL_NUM_EXPERTS,
            "top_k": 8,
        },
        "kernel_mapping": {
            "kernel_path": KERNEL_PATH,
            "kernel_reference": (
                f"v2 start_fetch_{plan.path}, kernel.py:{KERNEL_LINE_MAP.get(plan.path, 'n/a')}"
            ),
            "p_loop": "for p in range(t_packing), with t_packing=2 for bf16 tokens",
            "primary_copy_only": True,
            "issues_per_p": ["weight_async_copy", "scale_async_copy"],
            "wait_pattern": "two self-copy waits on shared DMA sem (weight then scale)",
            "excluded_from_phase1_row": (
                "dot/MXU compute",
                "A2A",
                "expert traversal",
                "metadata SMEM init",
                "fused-MoE control flow",
            ),
        },
        "fp8_traffic": {
            "bytes_weight": plan.bytes_weight,
            "bytes_scale": plan.bytes_scale,
            "bytes_total": plan.bytes_total,
            "dma_count_weight": plan.dma_count_weight,
            "dma_count_scale": plan.dma_count_scale,
            "weight_tile_shape": list(plan.weight_tile_shape),
            "scale_tile_shape": list(plan.scale_tile_shape),
        },
        "benchmark": {
            "name": "layer1_pallas_fp8_weight_tile_dma",
            "warmup_runs": _positive_int_env("CALIBRATION_LAYER1_WARMUP_RUNS", DEFAULT_WARMUP_RUNS),
            "sample_runs": _positive_int_env("CALIBRATION_LAYER1_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS),
            "trace_discard_runs": _nonnegative_int_env(
                "CALIBRATION_LAYER1_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
            ),
            "timing": "jax_profiler_trace_device_duration_ms",
            "trace_root": os.getenv(
                "CALIBRATION_LAYER1_TRACE_ROOT",
                "/tmp/sglang_jax_layer1_fp8_dma_trace",
            ),
            "vmem_limit_bytes": VMEM_LIMIT_BYTES,
            "local_semaphore_shape": LOCAL_SEMAPHORE_SHAPE,
            "has_side_effects": True,
        },
    }
    if metadata:
        fp8_meta.update(dict(metadata))

    return build_observation_row(
        scenario=SCENARIO_LAYER1_FP8_WEIGHT_TILE_DMA,
        suite=suite,
        layer=1,
        path=plan.path,
        path_class=plan.path_class,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        bf=plan.bf,
        bd=HIDDEN_SIZE,
        tile_shape=plan.weight_tile_shape,
        bytes_hbm=plan.bytes_total,
        bytes_per_fetch=plan.bytes_total,
        dma_count=plan.dma_count_weight + plan.dma_count_scale,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=samples,
        runtime=runtime,
        source=dict(source) if source else _default_source(),
        metadata=fp8_meta,
        implementation_note=implementation_note,
    )


def _default_source() -> dict[str, Any]:
    return {
        "coordination_repo": "jimoosciuc/fused-moe-calibration-lab",
        "kernel_path": KERNEL_PATH,
        "spec_doc": "docs/performance/mimo_v2_pro_fused_moe_v2_current_state.md",
        "spec_section": "J3 FP8 weight path calibration",
    }


def _pallas_unavailable_note(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    if backend != "tpu":
        return (
            "layer1_fp8_weight_tile_dma did not emit synthetic latency samples. "
            "Pallas DMA measurements require JAX default_backend='tpu'; "
            f"observed default_backend={backend!r}."
        )
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        from jax.experimental import pallas as pl  # noqa: F401
        from jax.experimental.pallas import tpu as pltpu  # noqa: F401

        from benchmark.utils import multiple_iteration_timeit_from_trace  # noqa: F401
    except Exception as exc:
        return (
            "layer1_fp8_weight_tile_dma could not import the JAX/Pallas APIs "
            f"needed for measured DMA; {type(exc).__name__}: {exc}."
        )
    return None


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


# ----------------------------------------------------------------------
# Pallas measurement
# ----------------------------------------------------------------------


def _measure_dma_ms(
    plan: FP8WeightTilePlan,
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from benchmark.utils import multiple_iteration_timeit_from_trace

    weight_dtype = jnp.float8_e4m3fn
    scale_dtype = jnp.float32
    sem_index = SEM_INDEX_MAP[plan.path]

    is_w2 = plan.path == "w2"
    if is_w2:
        weight_hbm_shape = (1, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        scale_hbm_shape = (1, INTERMEDIATE_SIZE // QUANT_BLOCK_K, 1, HIDDEN_SIZE)
        weight_scratch_shape = (T_PACKING, plan.bf, H_PER_T_PACKING)
        scale_scratch_shape = (T_PACKING, plan.bf // QUANT_BLOCK_K, 1, H_PER_T_PACKING)
    else:
        weight_hbm_shape = (1, HIDDEN_SIZE, INTERMEDIATE_SIZE)
        scale_hbm_shape = (1, HIDDEN_SIZE // QUANT_BLOCK_K, 1, INTERMEDIATE_SIZE)
        weight_scratch_shape = (T_PACKING, H_PER_T_PACKING, plan.bf)
        scale_scratch_shape = (T_PACKING, H_PER_T_PACKING // QUANT_BLOCK_K, 1, plan.bf)

    weight_source = jnp.ones(weight_hbm_shape, dtype=weight_dtype)
    scale_source = jnp.ones(scale_hbm_shape, dtype=scale_dtype)
    jax.block_until_ready(weight_source)
    jax.block_until_ready(scale_source)

    def kernel(weight_ref, scale_ref, out_ref, weight_scratch, scale_scratch, local_sems):
        del out_ref
        for p in range(T_PACKING):
            if is_w2:
                w_src = weight_ref.at[
                    0,
                    pl.ds(0, plan.bf),
                    pl.ds(p * H_PER_T_PACKING, H_PER_T_PACKING),
                ]
                s_src = scale_ref.at[
                    0,
                    pl.ds(0, plan.bf // QUANT_BLOCK_K),
                    pl.ds(0, 1),
                    pl.ds(p * H_PER_T_PACKING, H_PER_T_PACKING),
                ]
            else:
                w_src = weight_ref.at[
                    0,
                    pl.ds(p * H_PER_T_PACKING, H_PER_T_PACKING),
                    pl.ds(0, plan.bf),
                ]
                s_src = scale_ref.at[
                    0,
                    pl.ds(p * H_PER_T_PACKING // QUANT_BLOCK_K, H_PER_T_PACKING // QUANT_BLOCK_K),
                    pl.ds(0, 1),
                    pl.ds(0, plan.bf),
                ]
            pltpu.make_async_copy(
                src_ref=w_src,
                dst_ref=weight_scratch.at[p],
                sem=local_sems.at[0, sem_index],
            ).start()
            pltpu.make_async_copy(
                src_ref=s_src,
                dst_ref=scale_scratch.at[p],
                sem=local_sems.at[0, sem_index],
            ).start()

        # Mirror v2 wait_fetch_w*: two self-copy waits on the shared sem.
        pltpu.make_async_copy(
            src_ref=weight_scratch,
            dst_ref=weight_scratch,
            sem=local_sems.at[0, sem_index],
        ).wait()
        pltpu.make_async_copy(
            src_ref=scale_scratch,
            dst_ref=scale_scratch,
            sem=local_sems.at[0, sem_index],
        ).wait()

    @jax.jit
    def run_dma(weight_hbm, scale_hbm):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM(weight_scratch_shape, weight_dtype),
                    pltpu.VMEM(scale_scratch_shape, scale_dtype),
                    pltpu.SemaphoreType.DMA(LOCAL_SEMAPHORE_SHAPE),
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                has_side_effects=True,
                vmem_limit_bytes=VMEM_LIMIT_BYTES,
            ),
            name=f"layer1_fp8_weight_tile_dma_{plan.path}_bf{plan.bf}",
        )(weight_hbm, scale_hbm)

    jax.block_until_ready(run_dma(weight_source, scale_source))

    task = f"layer1_fp8_dma_{plan.path}_bf{plan.bf}"
    return multiple_iteration_timeit_from_trace(
        compute_func=run_dma,
        data_generator=lambda: (weight_source, scale_source),
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
    )
