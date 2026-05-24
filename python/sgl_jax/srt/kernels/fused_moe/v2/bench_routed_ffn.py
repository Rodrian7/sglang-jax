"""Standalone routed-FFN microbench for fused MoE v2.

This isolates the routed expert body from topk, metadata, scatter, gather, and
final token accumulation. Inputs are already expert-major:

  expert_tokens[E_local, capacity, t_packing, h_per_t]

The Pallas kernel keeps the same important routed-FFN structure as fused MoE v2:
weight DMA, bf/bts/btc tiling, optional FP8 direct-scaled-dot, optional full
dequant scratch, activation, W2 down, and expert output store.

Env vars:
  BENCH_TOKENS  — global source tokens used to derive avg routed load (default: 16384)
  BENCH_BT      — included only for tag parity with bench_v2 (default: 256)
  BENCH_BF      — comma-separated bf candidates (default: 1024)
  BENCH_BTC     — comma-separated btc candidates (default: 72)
  BENCH_BTS     — comma-separated bts candidates (default: 216)
  BENCH_BTS_TILES — comma-separated tiles per expert (default: 1)
  BENCH_FP8     — 1 to enable fp8 weights (default: 1)
  BENCH_QBK     — quant block K for fp8 (default: 128)
  BENCH_PREDEQUANT_FP8_WEIGHTS — quantize weights, then dequantize to bf16
                 before routed kernel launch; routed sees no scale (default: 0)
  BENCH_DIRECT_SCALED_DOT — fallback for FFN1/FFN2 direct flags
  BENCH_DIRECT_SCALED_DOT_FFN1/FFN2 — comma-separated 0/1 direct flags
  BENCH_APPLY_FP8_SCALE_FFN1/FFN2 — diagnostic 0/1 scale multiply toggles
  BENCH_DIRECT_SCALE_FFN1_BFC — direct-scaled FFN1 output chunk; 0 disables
  BENCH_DIRECT_SCALE_FFN2_BD2C — direct-scaled FFN2 hidden chunk; 0 disables
  BENCH_SG_CHUNK — scale-group unroll factor for FFN1 (default: 0 = full n_sg unroll).
                   When >0, fori_loop uses unroll=sg_chunk to reduce VRF pressure from
                   block-scaled matmul intermediates. Must divide n_sg (=h_per_t/qbk).
  BENCH_SG_VMEM_SPILL — periodic VMEM spill interval for FFN1 direct-scaled-dot
                   (default: 0 = disabled). When >0, keeps full unroll (no loop boundary)
                   but writes partial acc to VMEM scratch every N scale groups, then reads
                   back. This breaks VRF liveness of intermediate dot results without
                   breaking MXU pipeline. Must divide n_sg.
  BENCH_COMPUTE_REPEAT — repeat FFN compute per loaded bf tile (default: 1)
  BENCH_STREAM_FCHUNK — stream gate/up into W2 by F chunk; 0 disables (default: 0)
  BENCH_STAGE  — full|ffn1|ffn1_w1|ffn1_w1_reduce|ffn1_w1_nostore|
                 ffn2_scratch|ffn2_synth (default: full)
  BENCH_DELAY_STORE_WAIT — delay output DMA wait until b_y_stage reuse (default: 0)
  BENCH_GATE_UP_SCRATCH_DTYPE — f32|bf16 diagnostic scratch dtype (default: f32)
  BENCH_CAST_FFN1_INPUT_FP8 — diagnostic cast x to fp8 before W1/W3 dot (default: 0)
  BENCH_CAST_FFN2_INPUT_FP8 — diagnostic cast activation to fp8 before W2 dot (default: 0)
  BENCH_CAST_FFN2_INPUT_BF16 — diagnostic cast activation to bf16 before W2 dot (default: 0)
  BENCH_DOT_STYLE — v2|strix_fp8|strix_bf16_weight|native_bf16 (default: v2)
  BENCH_TOKEN_DTYPE — bf16|fp8 diagnostic input HBM dtype (default: bf16)
  DISABLE_X_LOAD / DISABLE_WEIGHT_LOAD / DISABLE_FFN_COMPUTE / DISABLE_EXPERT_STORE
  BENCH_WARMUP  — warmup iterations (default: 2)
  BENCH_ITERS   — timed iterations (default: 5)
  BENCH_WALL    — use wall timing instead of trace timing (default: 0)
  BENCH_D/F/E/TOPK — model dims (default: MiMo V2 Pro)
"""
from __future__ import annotations

import functools
import gzip
import itertools
import json
import os
import pathlib
import re
import time
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernel import activation_fn, get_dtype_packing


t0 = time.time()
TRACE_ROOT = "/tmp/tpu_logs/routed_ffn_trace"
KERNEL_NAME_RE = re.compile(r"routed-ffn-v2-.*")
P = jax.sharding.PartitionSpec

# TPU v7x, per JAX device / chiplet. These match the values recorded in
# core_doc/fused_moe_v2_rewrite_20260516/CONTEXT.md.
V7X_BF16_PEAK_TFLOPS = 1153.5
V7X_FP8_PEAK_TFLOPS = 2307.0
V7X_HBM_BW_TBPS = 3.69


def log(msg: str) -> None:
    print(f"[{time.time() - t0:.1f}s][p{jax.process_index()}] {msg}", flush=True)


def parse_csv_int(env_key: str, default: list[int]) -> list[int]:
    raw = os.environ.get(env_key)
    if raw is None:
        return default
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_csv_bool(env_key: str, default: list[bool]) -> list[bool]:
    raw = os.environ.get(env_key)
    if raw is None:
        return default
    out: list[bool] = []
    for item in raw.split(","):
        v = item.strip().lower()
        if v in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif v in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"Unsupported bool value {item!r} for {env_key}")
    return out


def parse_csv_str(env_key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(env_key)
    if raw is None:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def estimate_roofline(
    *,
    mean_ms: float,
    num_devices: int,
    d: int,
    f: int,
    e: int,
    bts: int,
    bts_tiles: int,
    use_fp8: bool,
    quant_block_k: int,
    apply_fp8_scale_ffn1: bool,
    apply_fp8_scale_ffn2: bool,
    disable_x_load: bool,
    disable_weight_load: bool,
    disable_ffn_compute: bool,
    disable_expert_store: bool,
    compute_repeat: int,
    stream_fchunk: int,
    direct_scale_ffn1_bfc: int,
    direct_scale_ffn2_bd2c: int,
    bench_stage: str,
    delay_store_wait: bool,
    gate_up_scratch_dtype: str,
    cast_ffn1_input_fp8: bool,
    cast_ffn2_input_fp8: bool,
    cast_ffn2_input_bf16: bool,
    dot_style: str,
    token_dtype: str,
) -> str:
    local_experts = e // num_devices
    routed_slots = local_experts * bts * bts_tiles
    run_ffn1 = bench_stage in (
        "full", "ffn1", "ffn1_w1", "ffn1_w1_reduce", "ffn1_w1_nostore"
    )
    run_ffn2 = bench_stage in ("full", "ffn2_scratch", "ffn2_synth")
    ffn1_mat_count = (
        1.0
        if bench_stage in ("ffn1_w1", "ffn1_w1_reduce", "ffn1_w1_nostore") else
        2.0
    )

    # Matmul-only FLOPs, counting FMA as 2 FLOPs:
    # W1 + W3 = 4 * M * D * F, W2 = 2 * M * D * F.
    flop_factor = 2.0 * float(run_ffn1) * ffn1_mat_count + 2.0 * float(run_ffn2)
    mxu_flops = 0.0 if disable_ffn_compute else (
        float(routed_slots) * flop_factor * float(d) * float(f) * float(compute_repeat)
    )

    weight_elem_bytes = 1 if use_fp8 else 2
    token_elem_bytes = 1.0 if token_dtype == "fp8" else 2.0
    token_bytes = 0.0 if disable_x_load else (
        float(routed_slots) * float(d) * token_elem_bytes
    )
    output_bytes = (
        0.0
        if disable_expert_store or not run_ffn2 else
        float(routed_slots) * float(d) * 2.0
    )
    weight_count = ffn1_mat_count * float(run_ffn1) + 1.0 * float(run_ffn2)
    weight_bytes = 0.0 if disable_weight_load else (
        float(local_experts)
        * float(bts_tiles)
        * weight_count
        * float(d)
        * float(f)
        * float(weight_elem_bytes)
    )
    scale_bytes = 0.0
    if use_fp8 and not disable_weight_load:
        # W1/W3 scales are (D / qbk, F), W2 scales are (F / qbk, D).
        scale_weight_count = float(
            int(run_ffn1) * int(apply_fp8_scale_ffn1) * ffn1_mat_count
            + int(run_ffn2) * int(apply_fp8_scale_ffn2)
        )
        scale_bytes = (
            float(local_experts)
            * float(bts_tiles)
            * scale_weight_count
            * float(d)
            * float(f)
            / float(quant_block_k)
            * 4.0
        )
    hbm_bytes = token_bytes + output_bytes + weight_bytes + scale_bytes

    peak_tflops = V7X_FP8_PEAK_TFLOPS if dot_style == "strix_fp8" else V7X_BF16_PEAK_TFLOPS
    achieved_tflops = mxu_flops / (mean_ms * 1e-3) / 1e12
    achieved_tbps = hbm_bytes / (mean_ms * 1e-3) / 1e12
    compute_floor_ms = mxu_flops / (peak_tflops * 1e12) * 1e3
    hbm_floor_ms = hbm_bytes / (V7X_HBM_BW_TBPS * 1e12) * 1e3
    roofline_ms = max(compute_floor_ms, hbm_floor_ms)
    gap = mean_ms / roofline_ms if roofline_ms > 0.0 else float("inf")
    oi = mxu_flops / hbm_bytes if hbm_bytes > 0.0 else float("inf")

    return (
        f"roofline stage={bench_stage} slots/dev={routed_slots} repeat={compute_repeat} "
        f"stream_fchunk={stream_fchunk} delay_store_wait={int(delay_store_wait)} "
        f"direct_f1_bfc={direct_scale_ffn1_bfc} "
        f"direct_f2_bd2c={direct_scale_ffn2_bd2c} "
        f"gateup={gate_up_scratch_dtype} "
        f"cast_f1_fp8={int(cast_ffn1_input_fp8)} "
        f"cast_f2_fp8={int(cast_ffn2_input_fp8)} "
        f"cast_f2_bf16={int(cast_ffn2_input_bf16)} "
        f"dot_style={dot_style} token_dtype={token_dtype} "
        f"mxu_flops={mxu_flops / 1e12:.3f}TF "
        f"hbm={hbm_bytes / 1e9:.3f}GB oi={oi:.1f}F/B "
        f"achieved={achieved_tflops:.1f}TF/s({achieved_tflops / peak_tflops * 100:.1f}%peak) "
        f"hbm_bw={achieved_tbps:.2f}TB/s({achieved_tbps / V7X_HBM_BW_TBPS * 100:.1f}%peak) "
        f"floor=max(compute={compute_floor_ms:.3f}ms,hbm={hbm_floor_ms:.3f}ms)"
        f"={roofline_ms:.3f}ms gap={gap:.2f}x"
    )


def _load_trace(trace_root: str) -> dict[str, Any]:
    trace_dir = pathlib.Path(trace_root) / "plugins" / "profile"
    if not trace_dir.exists():
        raise FileNotFoundError(f"No trace output under {trace_dir}")
    latest_dir = max(trace_dir.iterdir(), key=os.path.getmtime)
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace json.gz under {latest_dir}")
    combined: dict[str, Any] = {"traceEvents": []}
    for tf in sorted(trace_files):
        with gzip.open(tf, "rb") as fh:
            shard = json.load(fh)
        events = shard.get("traceEvents", [])
        if isinstance(events, list):
            combined["traceEvents"].extend(events)
    return combined


def _extract_durations_ms(trace: dict[str, Any]) -> list[float]:
    matched = [
        event
        for event in trace.get("traceEvents", [])
        if "name" in event and KERNEL_NAME_RE.match(event["name"])
    ]
    if not matched:
        return []
    by_pid: dict[int, list[dict[str, Any]]] = {}
    for event in matched:
        pid = event.get("pid")
        if isinstance(pid, int):
            by_pid.setdefault(pid, []).append(event)
    durations: dict[int, list[float]] = {}
    for pid, events in by_pid.items():
        events.sort(key=lambda event: float(event.get("ts", 0)))
        values: list[float] = []
        for event in events:
            args = event.get("args", {})
            if args.get("device_duration_ps"):
                values.append(float(args["device_duration_ps"]) / 1e9)
            elif "dur" in event:
                values.append(float(event["dur"]) / 1e3)
        if values:
            durations[pid] = values
    if not durations:
        return []
    return max(sorted(durations.items()), key=lambda kv: len(kv[1]))[1]


def trace_timeit(run_fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        out = run_fn()
        jax.block_until_ready(out)

    tag = f"{os.getpid()}_{int(time.time())}"
    trace_dir = os.path.join(TRACE_ROOT, f"run_{tag}")
    os.makedirs(trace_dir, exist_ok=True)
    with jax.profiler.trace(trace_dir):
        for _ in range(iters):
            out = run_fn()
            jax.block_until_ready(out)

    if jax.process_index() != 0:
        return []
    try:
        return _extract_durations_ms(_load_trace(trace_dir))
    except FileNotFoundError:
        return []


def wall_timeit(run_fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        out = run_fn()
        jax.block_until_ready(out)
    times: list[float] = []
    for _ in range(iters):
        start = time.monotonic()
        out = run_fn()
        jax.block_until_ready(out)
        times.append((time.monotonic() - start) * 1e3)
    return times


def _routed_ffn_kernel(
    expert_tokens_hbm,
    w1_hbm,
    w2_hbm,
    w3_hbm,
    w1_scale_hbm,
    w2_scale_hbm,
    w3_scale_hbm,
    output_hbm,
    b_x_vmem,
    b_w1_x2_vmem,
    b_w3_x2_vmem,
    b_w2_x2_vmem,
    b_w1_scale_x2_vmem,
    b_w3_scale_x2_vmem,
    b_w2_scale_x2_vmem,
    b_w1_dq_vmem,
    b_w3_dq_vmem,
    b_w2_dq_vmem,
    b_gate_acc_vmem,
    b_up_acc_vmem,
    b_y_acc_vmem,
    b_y_stage_vmem,
    x_sem,
    y_sem,
    local_sems,
    *,
    act_fn: str,
    direct_scaled_dot_ffn1: bool,
    direct_scaled_dot_ffn2: bool,
    apply_fp8_scale_ffn1: bool,
    apply_fp8_scale_ffn2: bool,
    disable_x_load: bool,
    disable_weight_load: bool,
    disable_ffn_compute: bool,
    disable_expert_store: bool,
    compute_repeat: int,
    stream_fchunk: int,
    direct_scale_ffn1_bfc: int,
    direct_scale_ffn2_bd2c: int,
    bench_stage: str,
    delay_store_wait: bool,
    gate_up_scratch_dtype: str,
    cast_ffn1_input_fp8: bool,
    cast_ffn2_input_fp8: bool,
    cast_ffn2_input_bf16: bool,
    dot_style: str,
    bt: int,
    bf: int,
    btc: int,
    bts: int,
    num_bts_tiles: int,
    quant_block_k: int | None,
    sg_chunk: int,
    sg_vmem_spill: int,
):
    del bt
    local_num_experts, capacity, t_packing, h_per_t = expert_tokens_hbm.shape
    intermediate_size = w2_hbm.shape[1]
    hidden_size = t_packing * h_per_t
    num_bf = pl.cdiv(intermediate_size, bf)
    num_btc_per_bts = bts // btc
    n_sg = h_per_t // quant_block_k if quant_block_k is not None else 1
    n_sg2 = bf // quant_block_k if quant_block_k is not None else 1

    def _sg_fori(n, fn, init, *, flush_fn=None):
        if sg_vmem_spill > 0 and sg_vmem_spill < n and flush_fn is not None:
            acc = init
            for sg_id in range(n):
                acc = fn(sg_id, acc)
                if (sg_id + 1) % sg_vmem_spill == 0 and sg_id < n - 1:
                    acc = flush_fn(acc)
            return acc
        if sg_chunk > 0 and sg_chunk < n:
            if sg_chunk == 1:
                return lax.fori_loop(0, n, fn, init, unroll=1)
            n_passes = n // sg_chunk
            def _outer(pass_id, carry):
                base = pass_id * sg_chunk
                def _inner(i, c):
                    return fn(base + i, c)
                return lax.fori_loop(0, sg_chunk, _inner, carry, unroll=sg_chunk)
            return lax.fori_loop(0, n_passes, _outer, init, unroll=1)
        return lax.fori_loop(0, n, fn, init, unroll=n)

    stream_chunks = bf // stream_fchunk if stream_fchunk > 0 else 0
    stream_sg_per_chunk = (
        stream_fchunk // quant_block_k
        if stream_fchunk > 0 and quant_block_k is not None else 0
    )
    ffn1_bfc = direct_scale_ffn1_bfc if direct_scale_ffn1_bfc > 0 else bf
    ffn1_bfc_chunks = bf // ffn1_bfc
    ffn2_hc = (
        direct_scale_ffn2_bd2c // t_packing
        if direct_scale_ffn2_bd2c > 0 else h_per_t
    )
    ffn2_hc_chunks = h_per_t // ffn2_hc
    run_ffn1 = bench_stage in (
        "full", "ffn1", "ffn1_w1", "ffn1_w1_reduce", "ffn1_w1_nostore"
    )
    run_ffn2 = bench_stage in ("full", "ffn2_scratch", "ffn2_synth")
    need_w3 = run_ffn1 and bench_stage not in (
        "ffn1_w1", "ffn1_w1_reduce", "ffn1_w1_nostore"
    )
    store_w1_only_gate = bench_stage == "ffn1_w1"
    reduce_w1_only_gate = bench_stage == "ffn1_w1_reduce"
    use_synthetic_w2_input = bench_stage == "ffn2_synth"
    use_full_fp8_dot = dot_style == "strix_fp8"
    use_native_bf16_dot = dot_style == "native_bf16"
    use_full_bf16_weight_dot = dot_style == "strix_bf16_weight"
    use_bf16_weight_dot = use_full_bf16_weight_dot or use_native_bf16_dot
    use_full_dot = use_full_fp8_dot or use_bf16_weight_dot
    fetch_w1_scale = (
        run_ffn1
        and not use_full_dot
        and w1_scale_hbm is not None
        and (apply_fp8_scale_ffn1 or not direct_scaled_dot_ffn1)
    )
    fetch_w3_scale = (
        need_w3
        and not use_full_dot
        and w3_scale_hbm is not None
        and (apply_fp8_scale_ffn1 or not direct_scaled_dot_ffn1)
    )
    fetch_w2_scale = (
        run_ffn2
        and not use_full_dot
        and w2_scale_hbm is not None
        and (apply_fp8_scale_ffn2 or not direct_scaled_dot_ffn2)
    )

    def maybe_cast_ffn1_input(x):
        return x.astype(jnp.float8_e4m3fn) if cast_ffn1_input_fp8 else x

    def maybe_cast_ffn2_input(x):
        if cast_ffn2_input_fp8:
            return x.astype(jnp.float8_e4m3fn)
        if cast_ffn2_input_bf16:
            return x.astype(jnp.bfloat16)
        return x

    def start_fetch_w1(local_e_id, slot, bf_id, priority=1):
        if disable_weight_load or not run_ffn1:
            return
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[
                    local_e_id,
                    pl.ds(p * h_per_t, h_per_t),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w1_x2_vmem.at[slot, p],
                sem=local_sems.at[slot, 0],
            ).start(priority=priority)
            if fetch_w1_scale:
                pltpu.make_async_copy(
                    src_ref=w1_scale_hbm.at[
                        local_e_id,
                        pl.ds(p * h_per_t // quant_block_k, h_per_t // quant_block_k),
                        pl.ds(0, 1),
                        pl.ds(bf_id * bf, bf),
                    ],
                    dst_ref=b_w1_scale_x2_vmem.at[slot, p],
                    sem=local_sems.at[slot, 0],
                ).start(priority=priority)

    def start_fetch_w3(local_e_id, slot, bf_id, priority=1):
        if disable_weight_load or not need_w3:
            return
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w3_hbm.at[
                    local_e_id,
                    pl.ds(p * h_per_t, h_per_t),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w3_x2_vmem.at[slot, p],
                sem=local_sems.at[slot, 1],
            ).start(priority=priority)
            if fetch_w3_scale:
                pltpu.make_async_copy(
                    src_ref=w3_scale_hbm.at[
                        local_e_id,
                        pl.ds(p * h_per_t // quant_block_k, h_per_t // quant_block_k),
                        pl.ds(0, 1),
                        pl.ds(bf_id * bf, bf),
                    ],
                    dst_ref=b_w3_scale_x2_vmem.at[slot, p],
                    sem=local_sems.at[slot, 1],
                ).start(priority=priority)

    def start_fetch_w2(local_e_id, slot, bf_id, priority=1):
        if disable_weight_load or not run_ffn2:
            return
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[
                    local_e_id,
                    pl.ds(bf_id * bf, bf),
                    pl.ds(p * h_per_t, h_per_t),
                ],
                dst_ref=b_w2_x2_vmem.at[slot, p],
                sem=local_sems.at[slot, 2],
            ).start(priority=priority)
            if fetch_w2_scale:
                pltpu.make_async_copy(
                    src_ref=w2_scale_hbm.at[
                        local_e_id,
                        pl.ds(bf_id * bf // quant_block_k, bf // quant_block_k),
                        pl.ds(0, 1),
                        pl.ds(p * h_per_t, h_per_t),
                    ],
                    dst_ref=b_w2_scale_x2_vmem.at[slot, p],
                    sem=local_sems.at[slot, 2],
                ).start(priority=priority)

    def wait_fetch(slot, sem_id, ref):
        if disable_weight_load:
            return
        pltpu.make_async_copy(src_ref=ref, dst_ref=ref, sem=local_sems.at[slot, sem_id]).wait()

    def wait_y_store():
        if disable_expert_store or not run_ffn2:
            return
        pltpu.make_async_copy(
            src_ref=b_y_stage_vmem, dst_ref=b_y_stage_vmem, sem=y_sem
        ).wait()

    def wait_fetch_w1(slot):
        wait_fetch(slot, 0, b_w1_x2_vmem.at[slot])
        if fetch_w1_scale:
            wait_fetch(slot, 0, b_w1_scale_x2_vmem.at[slot])

    def wait_fetch_w3(slot):
        wait_fetch(slot, 1, b_w3_x2_vmem.at[slot])
        if fetch_w3_scale:
            wait_fetch(slot, 1, b_w3_scale_x2_vmem.at[slot])

    def wait_fetch_w2(slot):
        wait_fetch(slot, 2, b_w2_x2_vmem.at[slot])
        if fetch_w2_scale:
            wait_fetch(slot, 2, b_w2_scale_x2_vmem.at[slot])

    def start_fetch_w13_w2(local_e_id, slot, bf_id):
        start_fetch_w1(local_e_id, slot, bf_id)
        start_fetch_w3(local_e_id, slot, bf_id)
        start_fetch_w2(local_e_id, slot, bf_id)

    def dequant_w1(slot):
        if use_full_dot or w1_scale_hbm is None or direct_scaled_dot_ffn1:
            return
        for p in range(t_packing):
            def _dq(sg_id, _):
                sg_off = sg_id * quant_block_k
                w_fp8 = b_w1_x2_vmem[slot, p, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)]
                scale = b_w1_scale_x2_vmem[slot, p, pl.ds(sg_id, 1), 0, pl.ds(0, bf)]
                w_bf16 = (
                    w_fp8.astype(jnp.float32)
                    * jnp.broadcast_to(scale.reshape(1, bf), (quant_block_k, bf))
                ).astype(jnp.bfloat16)
                b_w1_dq_vmem.at[p, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)][...] = w_bf16
                return None
            lax.fori_loop(0, n_sg, _dq, None, unroll=n_sg)

    def dequant_w3(slot):
        if use_full_dot or not need_w3 or w3_scale_hbm is None or direct_scaled_dot_ffn1:
            return
        for p in range(t_packing):
            def _dq(sg_id, _):
                sg_off = sg_id * quant_block_k
                w_fp8 = b_w3_x2_vmem[slot, p, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)]
                scale = b_w3_scale_x2_vmem[slot, p, pl.ds(sg_id, 1), 0, pl.ds(0, bf)]
                w_bf16 = (
                    w_fp8.astype(jnp.float32)
                    * jnp.broadcast_to(scale.reshape(1, bf), (quant_block_k, bf))
                ).astype(jnp.bfloat16)
                b_w3_dq_vmem.at[p, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)][...] = w_bf16
                return None
            lax.fori_loop(0, n_sg, _dq, None, unroll=n_sg)

    def dequant_w2(slot):
        if use_full_dot or w2_scale_hbm is None or direct_scaled_dot_ffn2:
            return
        for p in range(t_packing):
            def _dq(sg_id, _):
                sg_off = sg_id * quant_block_k
                w_fp8 = b_w2_x2_vmem[slot, p, pl.ds(sg_off, quant_block_k), pl.ds(0, h_per_t)]
                scale = b_w2_scale_x2_vmem[slot, p, pl.ds(sg_id, 1), 0, pl.ds(0, h_per_t)]
                w_bf16 = (
                    w_fp8.astype(jnp.float32)
                    * jnp.broadcast_to(scale.reshape(1, h_per_t), (quant_block_k, h_per_t))
                ).astype(jnp.bfloat16)
                b_w2_dq_vmem.at[p, pl.ds(sg_off, quant_block_k), pl.ds(0, h_per_t)][...] = w_bf16
                return None
            lax.fori_loop(0, n_sg2, _dq, None, unroll=n_sg2)

    def seed_gate_up_from_x():
        if bench_stage != "ffn2_scratch":
            return

        def seed_btc(btc_id, _):
            seed = b_x_vmem[
                pl.ds(btc_id * btc, btc),
                0,
                pl.ds(0, bf),
            ].astype(jnp.float32)
            b_gate_acc_vmem.at[
                pl.ds(btc_id * btc, btc), pl.ds(0, bf)
            ][...] = seed
            b_up_acc_vmem.at[
                pl.ds(btc_id * btc, btc), pl.ds(0, bf)
            ][...] = seed
            return None

        lax.fori_loop(0, num_btc_per_bts, seed_btc, None)

    def run_bts_tile(local_e_id, bts_id, store_pending):
        tile_start = bts_id * bts
        if not disable_x_load:
            pltpu.make_async_copy(
                src_ref=expert_tokens_hbm.at[
                    local_e_id, pl.ds(tile_start, bts), pl.ds(0, t_packing), pl.ds(0, h_per_t)
                ],
                dst_ref=b_x_vmem,
                sem=x_sem,
            ).start(priority=1)
            pltpu.make_async_copy(src_ref=b_x_vmem, dst_ref=b_x_vmem, sem=x_sem).wait()

        start_fetch_w13_w2(local_e_id, 0, 0)
        if num_bf >= 2:
            start_fetch_w13_w2(local_e_id, 1, 1)

        for bf_id in range(num_bf):
            slot = bf_id % 2
            if run_ffn1:
                wait_fetch_w1(slot)
                if need_w3:
                    wait_fetch_w3(slot)

            if disable_ffn_compute:
                if run_ffn2:
                    wait_fetch_w2(slot)
            else:
                for repeat_id in range(compute_repeat):
                    use_stream = (
                        run_ffn1
                        and run_ffn2
                        and stream_fchunk > 0
                        and direct_scaled_dot_ffn1
                        and direct_scaled_dot_ffn2
                        and w1_scale_hbm is not None
                        and w2_scale_hbm is not None
                        and w3_scale_hbm is not None
                    )
                    if use_stream:
                        for stream_chunk_id in range(stream_chunks):
                            f_off = stream_chunk_id * stream_fchunk
                            for btc_id in range(num_btc_per_bts):
                                gate = jnp.zeros((btc, stream_fchunk), dtype=jnp.float32)
                                up = jnp.zeros((btc, stream_fchunk), dtype=jnp.float32)
                                for p_id in range(t_packing):
                                    def _sg(sg_id, carry):
                                        gate_acc, up_acc = carry
                                        sg_off = sg_id * quant_block_k
                                        x_slice = b_x_vmem[
                                            pl.ds(btc_id * btc, btc), p_id,
                                            pl.ds(sg_off, quant_block_k)
                                        ]
                                        x_slice = maybe_cast_ffn1_input(x_slice)
                                        w1_tile = b_w1_x2_vmem[
                                            slot, p_id, pl.ds(sg_off, quant_block_k),
                                            pl.ds(f_off, stream_fchunk)
                                        ]
                                        w3_tile = b_w3_x2_vmem[
                                            slot, p_id, pl.ds(sg_off, quant_block_k),
                                            pl.ds(f_off, stream_fchunk)
                                        ]
                                        d1 = jnp.dot(x_slice, w1_tile,
                                                     preferred_element_type=jnp.float32)
                                        d3 = jnp.dot(x_slice, w3_tile,
                                                     preferred_element_type=jnp.float32)
                                        if apply_fp8_scale_ffn1:
                                            s1 = b_w1_scale_x2_vmem[
                                                slot, p_id, pl.ds(sg_id, 1), 0,
                                                pl.ds(f_off, stream_fchunk)
                                            ].reshape(1, stream_fchunk)
                                            s3 = b_w3_scale_x2_vmem[
                                                slot, p_id, pl.ds(sg_id, 1), 0,
                                                pl.ds(f_off, stream_fchunk)
                                            ].reshape(1, stream_fchunk)
                                            gate_acc += d1 * jnp.broadcast_to(s1, d1.shape)
                                            up_acc += d3 * jnp.broadcast_to(s3, d3.shape)
                                        else:
                                            gate_acc += d1
                                            up_acc += d3
                                        return gate_acc, up_acc
                                    def _flush_gu_stream(carry):
                                        g, u = carry
                                        gs = pl.ds(btc_id * btc, btc), pl.ds(f_off, stream_fchunk)
                                        b_gate_acc_vmem.at[gs][...] = g
                                        b_up_acc_vmem.at[gs][...] = u
                                        return b_gate_acc_vmem[gs], b_up_acc_vmem[gs]
                                    gate, up = _sg_fori(n_sg, _sg, (gate, up), flush_fn=_flush_gu_stream)

                                if repeat_id == 0 and stream_chunk_id == 0 and btc_id == 0:
                                    wait_fetch_w2(slot)

                                for p_id in range(t_packing):
                                    partial = jnp.zeros((btc, h_per_t), dtype=jnp.float32)
                                    for local_sg_id in range(stream_sg_per_chunk):
                                        local_f_off = local_sg_id * quant_block_k
                                        global_f_off = f_off + local_f_off
                                        gate_slice = gate[
                                            :, local_f_off:local_f_off + quant_block_k
                                        ]
                                        up_slice = up[
                                            :, local_f_off:local_f_off + quant_block_k
                                        ]
                                        act_slice = activation_fn(gate_slice, up_slice, act_fn)
                                        act_slice = maybe_cast_ffn2_input(act_slice)
                                        w2_tile = b_w2_x2_vmem[
                                            slot, p_id, pl.ds(global_f_off, quant_block_k),
                                            pl.ds(0, h_per_t)
                                        ]
                                        d = jnp.dot(act_slice, w2_tile,
                                                    preferred_element_type=jnp.float32)
                                        if apply_fp8_scale_ffn2:
                                            scale = b_w2_scale_x2_vmem[
                                                slot, p_id, pl.ds(global_f_off // quant_block_k, 1),
                                                0, pl.ds(0, h_per_t)
                                            ].reshape(1, h_per_t)
                                            partial += d * jnp.broadcast_to(scale, d.shape)
                                        else:
                                            partial += d
                                    acc_ref = b_y_acc_vmem.at[
                                        pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                                    ]
                                if bf_id == 0 and repeat_id == 0 and stream_chunk_id == 0:
                                    acc_ref[...] = partial
                                else:
                                    acc_ref[...] = acc_ref[...] + partial
                    elif run_ffn1 and use_full_dot:
                        def gate_up_full_dot_btc(btc_id, _):
                            gate = jnp.zeros((btc, bf), dtype=jnp.float32)
                            up = jnp.zeros((btc, bf), dtype=jnp.float32)
                            for p_id in range(t_packing):
                                x_slice = b_x_vmem[
                                    pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                                ]
                                if use_full_fp8_dot:
                                    x_slice = x_slice.astype(jnp.float8_e4m3fn)
                                    w1_tile = b_w1_x2_vmem[slot, p_id]
                                    gate += jnp.dot(
                                        x_slice, w1_tile,
                                        preferred_element_type=jnp.float32,
                                    )
                                    if need_w3:
                                        w3_tile = b_w3_x2_vmem[slot, p_id]
                                        up += jnp.dot(
                                            x_slice, w3_tile,
                                            preferred_element_type=jnp.float32,
                                        )
                                else:
                                    x_slice = x_slice.astype(jnp.bfloat16)
                                    w1_tile = b_w1_x2_vmem[slot, p_id]
                                    if use_full_bf16_weight_dot:
                                        w1_tile = w1_tile.astype(jnp.bfloat16)
                                    gate += jnp.dot(
                                        x_slice, w1_tile,
                                        preferred_element_type=jnp.float32,
                                    )
                                    if need_w3:
                                        w3_tile = b_w3_x2_vmem[slot, p_id]
                                        if use_full_bf16_weight_dot:
                                            w3_tile = w3_tile.astype(jnp.bfloat16)
                                        up += jnp.dot(
                                            x_slice, w3_tile,
                                            preferred_element_type=jnp.float32,
                                        )
                            if store_w1_only_gate:
                                b_gate_acc_vmem.at[
                                    pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                ][...] = gate
                            elif reduce_w1_only_gate:
                                b_gate_acc_vmem.at[
                                    pl.ds(btc_id * btc, 1), pl.ds(0, 1)
                                ][...] = jnp.sum(gate).reshape((1, 1))
                            else:
                                b_gate_acc_vmem.at[
                                    pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                ][...] = gate
                                b_up_acc_vmem.at[
                                    pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                ][...] = up
                            return None
                        lax.fori_loop(0, num_btc_per_bts, gate_up_full_dot_btc, None)
                    elif (
                        run_ffn1
                        and not need_w3
                        and direct_scaled_dot_ffn1
                        and w1_scale_hbm is not None
                        and direct_scale_ffn1_bfc > 0
                    ):
                        def gate_direct_chunked_btc(btc_id, _):
                            for chunk_id in range(ffn1_bfc_chunks):
                                f_off = chunk_id * ffn1_bfc
                                gate = jnp.zeros((btc, ffn1_bfc), dtype=jnp.float32)
                                for p_id in range(t_packing):
                                    def _sg(sg_id, gate_acc):
                                        sg_off = sg_id * quant_block_k
                                        x_slice = b_x_vmem[
                                            pl.ds(btc_id * btc, btc), p_id,
                                            pl.ds(sg_off, quant_block_k)
                                        ]
                                        x_slice = maybe_cast_ffn1_input(x_slice)
                                        w1_tile = b_w1_x2_vmem[
                                            slot, p_id, pl.ds(sg_off, quant_block_k),
                                            pl.ds(f_off, ffn1_bfc)
                                        ]
                                        d1 = jnp.dot(x_slice, w1_tile,
                                                     preferred_element_type=jnp.float32)
                                        if apply_fp8_scale_ffn1:
                                            s1 = b_w1_scale_x2_vmem[
                                                slot, p_id, pl.ds(sg_id, 1), 0,
                                                pl.ds(f_off, ffn1_bfc)
                                            ].reshape(1, ffn1_bfc)
                                            gate_acc += d1 * jnp.broadcast_to(s1, d1.shape)
                                        else:
                                            gate_acc += d1
                                        return gate_acc
                                    def _flush_g_chunked(g):
                                        gs = pl.ds(btc_id * btc, btc), pl.ds(f_off, ffn1_bfc)
                                        b_gate_acc_vmem.at[gs][...] = g
                                        return b_gate_acc_vmem[gs]
                                    gate = _sg_fori(n_sg, _sg, gate, flush_fn=_flush_g_chunked)
                                if store_w1_only_gate:
                                    b_gate_acc_vmem.at[
                                        pl.ds(btc_id * btc, btc), pl.ds(f_off, ffn1_bfc)
                                    ][...] = gate
                                if reduce_w1_only_gate:
                                    b_gate_acc_vmem.at[
                                        pl.ds(btc_id * btc, 1), pl.ds(0, 1)
                                    ][...] = jnp.sum(gate).reshape((1, 1))
                            return None
                        lax.fori_loop(0, num_btc_per_bts, gate_direct_chunked_btc, None)
                    elif (
                        run_ffn1
                        and not need_w3
                        and direct_scaled_dot_ffn1
                        and w1_scale_hbm is not None
                    ):
                        def gate_direct_btc(btc_id, _):
                            gate = jnp.zeros((btc, bf), dtype=jnp.float32)
                            for p_id in range(t_packing):
                                def _sg(sg_id, gate_acc):
                                    sg_off = sg_id * quant_block_k
                                    x_slice = b_x_vmem[
                                        pl.ds(btc_id * btc, btc), p_id,
                                        pl.ds(sg_off, quant_block_k)
                                    ]
                                    x_slice = maybe_cast_ffn1_input(x_slice)
                                    w1_tile = b_w1_x2_vmem[
                                        slot, p_id, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)
                                    ]
                                    d1 = jnp.dot(x_slice, w1_tile,
                                                 preferred_element_type=jnp.float32)
                                    if apply_fp8_scale_ffn1:
                                        s1 = b_w1_scale_x2_vmem[
                                            slot, p_id, pl.ds(sg_id, 1), 0, pl.ds(0, bf)
                                        ].reshape(1, bf)
                                        gate_acc += d1 * jnp.broadcast_to(s1, d1.shape)
                                    else:
                                        gate_acc += d1
                                    return gate_acc
                                def _flush_g(g):
                                    gs = pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                    b_gate_acc_vmem.at[gs][...] = g
                                    return b_gate_acc_vmem[gs]
                                gate = _sg_fori(n_sg, _sg, gate, flush_fn=_flush_g)
                            if store_w1_only_gate:
                                b_gate_acc_vmem.at[
                                    pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                ][...] = gate
                            if reduce_w1_only_gate:
                                b_gate_acc_vmem.at[
                                    pl.ds(btc_id * btc, 1), pl.ds(0, 1)
                                ][...] = jnp.sum(gate).reshape((1, 1))
                            return None
                        lax.fori_loop(0, num_btc_per_bts, gate_direct_btc, None)
                    elif (
                        run_ffn1
                        and direct_scaled_dot_ffn1
                        and w1_scale_hbm is not None
                        and direct_scale_ffn1_bfc > 0
                    ):
                        def gate_up_direct_chunked_btc(btc_id, _):
                            for chunk_id in range(ffn1_bfc_chunks):
                                f_off = chunk_id * ffn1_bfc
                                gate = jnp.zeros((btc, ffn1_bfc), dtype=jnp.float32)
                                up = jnp.zeros((btc, ffn1_bfc), dtype=jnp.float32)
                                for p_id in range(t_packing):
                                    def _sg(sg_id, carry):
                                        gate_acc, up_acc = carry
                                        sg_off = sg_id * quant_block_k
                                        x_slice = b_x_vmem[
                                            pl.ds(btc_id * btc, btc), p_id,
                                            pl.ds(sg_off, quant_block_k)
                                        ]
                                        x_slice = maybe_cast_ffn1_input(x_slice)
                                        w1_tile = b_w1_x2_vmem[
                                            slot, p_id, pl.ds(sg_off, quant_block_k),
                                            pl.ds(f_off, ffn1_bfc)
                                        ]
                                        w3_tile = b_w3_x2_vmem[
                                            slot, p_id, pl.ds(sg_off, quant_block_k),
                                            pl.ds(f_off, ffn1_bfc)
                                        ]
                                        d1 = jnp.dot(x_slice, w1_tile,
                                                     preferred_element_type=jnp.float32)
                                        d3 = jnp.dot(x_slice, w3_tile,
                                                     preferred_element_type=jnp.float32)
                                        if apply_fp8_scale_ffn1:
                                            s1 = b_w1_scale_x2_vmem[
                                                slot, p_id, pl.ds(sg_id, 1), 0,
                                                pl.ds(f_off, ffn1_bfc)
                                            ].reshape(1, ffn1_bfc)
                                            s3 = b_w3_scale_x2_vmem[
                                                slot, p_id, pl.ds(sg_id, 1), 0,
                                                pl.ds(f_off, ffn1_bfc)
                                            ].reshape(1, ffn1_bfc)
                                            gate_acc += d1 * jnp.broadcast_to(s1, d1.shape)
                                            up_acc += d3 * jnp.broadcast_to(s3, d3.shape)
                                        else:
                                            gate_acc += d1
                                            up_acc += d3
                                        return gate_acc, up_acc
                                    def _flush_gu_chunked(carry):
                                        g, u = carry
                                        gs = pl.ds(btc_id * btc, btc), pl.ds(f_off, ffn1_bfc)
                                        b_gate_acc_vmem.at[gs][...] = g
                                        b_up_acc_vmem.at[gs][...] = u
                                        return b_gate_acc_vmem[gs], b_up_acc_vmem[gs]
                                    gate, up = _sg_fori(n_sg, _sg, (gate, up), flush_fn=_flush_gu_chunked)
                                b_gate_acc_vmem.at[
                                    pl.ds(btc_id * btc, btc), pl.ds(f_off, ffn1_bfc)
                                ][...] = gate
                                b_up_acc_vmem.at[
                                    pl.ds(btc_id * btc, btc), pl.ds(f_off, ffn1_bfc)
                                ][...] = up
                            return None
                        lax.fori_loop(0, num_btc_per_bts, gate_up_direct_chunked_btc, None)
                    elif run_ffn1 and direct_scaled_dot_ffn1 and w1_scale_hbm is not None:
                        def gate_up_direct_btc(btc_id, _):
                            gate = jnp.zeros((btc, bf), dtype=jnp.float32)
                            up = jnp.zeros((btc, bf), dtype=jnp.float32)
                            for p_id in range(t_packing):
                                def _sg(sg_id, carry):
                                    gate_acc, up_acc = carry
                                    sg_off = sg_id * quant_block_k
                                    x_slice = b_x_vmem[
                                        pl.ds(btc_id * btc, btc), p_id,
                                        pl.ds(sg_off, quant_block_k)
                                    ]
                                    x_slice = maybe_cast_ffn1_input(x_slice)
                                    w1_tile = b_w1_x2_vmem[
                                        slot, p_id, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)
                                    ]
                                    w3_tile = b_w3_x2_vmem[
                                        slot, p_id, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)
                                    ]
                                    d1 = jnp.dot(x_slice, w1_tile,
                                                 preferred_element_type=jnp.float32)
                                    d3 = jnp.dot(x_slice, w3_tile,
                                                 preferred_element_type=jnp.float32)
                                    if apply_fp8_scale_ffn1:
                                        s1 = b_w1_scale_x2_vmem[
                                            slot, p_id, pl.ds(sg_id, 1), 0, pl.ds(0, bf)
                                        ].reshape(1, bf)
                                        s3 = b_w3_scale_x2_vmem[
                                            slot, p_id, pl.ds(sg_id, 1), 0, pl.ds(0, bf)
                                        ].reshape(1, bf)
                                        gate_acc += d1 * jnp.broadcast_to(s1, d1.shape)
                                        up_acc += d3 * jnp.broadcast_to(s3, d3.shape)
                                    else:
                                        gate_acc += d1
                                        up_acc += d3
                                    return gate_acc, up_acc
                                def _flush_gu(carry):
                                    g, u = carry
                                    gs = pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                    b_gate_acc_vmem.at[gs][...] = g
                                    b_up_acc_vmem.at[gs][...] = u
                                    return b_gate_acc_vmem[gs], b_up_acc_vmem[gs]
                                gate, up = _sg_fori(n_sg, _sg, (gate, up), flush_fn=_flush_gu)
                            b_gate_acc_vmem.at[
                                pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                            ][...] = gate
                            b_up_acc_vmem.at[
                                pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                            ][...] = up
                            return None
                        lax.fori_loop(0, num_btc_per_bts, gate_up_direct_btc, None)
                    elif run_ffn1:
                        dequant_w1(slot)
                        dequant_w3(slot)

                        def gate_up_btc(btc_id, _):
                            gate = jnp.zeros((btc, bf), dtype=jnp.float32)
                            up = jnp.zeros((btc, bf), dtype=jnp.float32)
                            for p_id in range(t_packing):
                                x_slice = b_x_vmem[
                                    pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                                ]
                                x_slice = maybe_cast_ffn1_input(x_slice)
                                w1_tile = (
                                    b_w1_dq_vmem[p_id]
                                    if w1_scale_hbm is not None else b_w1_x2_vmem[slot, p_id]
                                )
                                w3_tile = (
                                    b_w3_dq_vmem[p_id]
                                    if w3_scale_hbm is not None else b_w3_x2_vmem[slot, p_id]
                                )
                                gate += jnp.dot(x_slice, w1_tile,
                                                preferred_element_type=jnp.float32)
                                up += jnp.dot(x_slice, w3_tile,
                                              preferred_element_type=jnp.float32)
                            b_gate_acc_vmem.at[
                                pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                            ][...] = gate
                            b_up_acc_vmem.at[
                                pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                            ][...] = up
                            return None
                        lax.fori_loop(0, num_btc_per_bts, gate_up_btc, None)

                    if repeat_id == 0 and bench_stage == "ffn2_scratch":
                        seed_gate_up_from_x()

                    if repeat_id == 0 and run_ffn2:
                        wait_fetch_w2(slot)
                        dequant_w2(slot)

                    def act_down_btc(btc_id, _):
                        use_direct_w2 = direct_scaled_dot_ffn2 and w2_scale_hbm is not None
                        use_full_dot_w2 = use_full_dot and run_ffn2
                        if use_full_dot_w2:
                            if use_synthetic_w2_input:
                                act = b_x_vmem[
                                    pl.ds(btc_id * btc, btc), 0, pl.ds(0, bf)
                                ].astype(jnp.float32)
                            else:
                                gate = b_gate_acc_vmem[
                                    pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                ].astype(jnp.float32)
                                up_val = b_up_acc_vmem[
                                    pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                ].astype(jnp.float32)
                                act = activation_fn(gate, up_val, act_fn)
                            act = maybe_cast_ffn2_input(act)
                            if use_full_fp8_dot:
                                act = act.astype(jnp.float8_e4m3fn)
                            else:
                                act = act.astype(jnp.bfloat16)
                        elif not use_direct_w2:
                            if use_synthetic_w2_input:
                                act = b_x_vmem[
                                    pl.ds(btc_id * btc, btc), 0, pl.ds(0, bf)
                                ].astype(jnp.float32)
                            else:
                                gate = b_gate_acc_vmem[
                                    pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                ].astype(jnp.float32)
                                up_val = b_up_acc_vmem[
                                    pl.ds(btc_id * btc, btc), pl.ds(0, bf)
                                ].astype(jnp.float32)
                                act = activation_fn(gate, up_val, act_fn)
                            act = maybe_cast_ffn2_input(act)
                        for p_id in range(t_packing):
                            if use_full_dot_w2:
                                w2_tile = b_w2_x2_vmem[slot, p_id]
                                if use_full_bf16_weight_dot:
                                    w2_tile = w2_tile.astype(jnp.bfloat16)
                                partial = jnp.dot(
                                    act, w2_tile,
                                    preferred_element_type=jnp.float32,
                                )
                            elif use_direct_w2:
                                if direct_scale_ffn2_bd2c > 0:
                                    for hc_id in range(ffn2_hc_chunks):
                                        h_off = hc_id * ffn2_hc

                                        def _sg(sg_id, partial_acc):
                                            sg_off = sg_id * quant_block_k
                                            if use_synthetic_w2_input:
                                                act_slice = b_x_vmem[
                                                    pl.ds(btc_id * btc, btc),
                                                    0,
                                                    pl.ds(sg_off, quant_block_k),
                                                ].astype(jnp.float32)
                                            else:
                                                gate_slice = b_gate_acc_vmem[
                                                    pl.ds(btc_id * btc, btc),
                                                    pl.ds(sg_off, quant_block_k)
                                                ].astype(jnp.float32)
                                                up_slice = b_up_acc_vmem[
                                                    pl.ds(btc_id * btc, btc),
                                                    pl.ds(sg_off, quant_block_k)
                                                ].astype(jnp.float32)
                                                act_slice = activation_fn(
                                                    gate_slice, up_slice, act_fn
                                                )
                                            act_slice = maybe_cast_ffn2_input(act_slice)
                                            w2_tile = b_w2_x2_vmem[
                                                slot, p_id, pl.ds(sg_off, quant_block_k),
                                                pl.ds(h_off, ffn2_hc)
                                            ]
                                            d = jnp.dot(act_slice, w2_tile,
                                                        preferred_element_type=jnp.float32)
                                            if apply_fp8_scale_ffn2:
                                                scale = b_w2_scale_x2_vmem[
                                                    slot, p_id, pl.ds(sg_id, 1), 0,
                                                    pl.ds(h_off, ffn2_hc)
                                                ].reshape(1, ffn2_hc)
                                                return partial_acc + d * jnp.broadcast_to(
                                                    scale, d.shape
                                                )
                                            return partial_acc + d

                                        partial = lax.fori_loop(
                                            0,
                                            n_sg2,
                                            _sg,
                                            jnp.zeros((btc, ffn2_hc), dtype=jnp.float32),
                                            unroll=n_sg2,
                                        )
                                        acc_ref = b_y_acc_vmem.at[
                                            pl.ds(btc_id * btc, btc),
                                            p_id,
                                            pl.ds(h_off, ffn2_hc),
                                        ]
                                        if bf_id == 0 and repeat_id == 0:
                                            acc_ref[...] = partial
                                        else:
                                            acc_ref[...] = acc_ref[...] + partial
                                    continue
                                else:
                                    def _sg(sg_id, partial_acc):
                                        sg_off = sg_id * quant_block_k
                                        if use_synthetic_w2_input:
                                            act_slice = b_x_vmem[
                                                pl.ds(btc_id * btc, btc),
                                                0,
                                                pl.ds(sg_off, quant_block_k),
                                            ].astype(jnp.float32)
                                        else:
                                            gate_slice = b_gate_acc_vmem[
                                                pl.ds(btc_id * btc, btc),
                                                pl.ds(sg_off, quant_block_k)
                                            ].astype(jnp.float32)
                                            up_slice = b_up_acc_vmem[
                                                pl.ds(btc_id * btc, btc),
                                                pl.ds(sg_off, quant_block_k)
                                            ].astype(jnp.float32)
                                            act_slice = activation_fn(gate_slice, up_slice, act_fn)
                                        act_slice = maybe_cast_ffn2_input(act_slice)
                                        w2_tile = b_w2_x2_vmem[
                                            slot, p_id, pl.ds(sg_off, quant_block_k),
                                            pl.ds(0, h_per_t)
                                        ]
                                        d = jnp.dot(act_slice, w2_tile,
                                                    preferred_element_type=jnp.float32)
                                        if apply_fp8_scale_ffn2:
                                            scale = b_w2_scale_x2_vmem[
                                                slot, p_id, pl.ds(sg_id, 1), 0,
                                                pl.ds(0, h_per_t)
                                            ].reshape(1, h_per_t)
                                            return partial_acc + d * jnp.broadcast_to(scale, d.shape)
                                        return partial_acc + d
                                    partial = lax.fori_loop(
                                        0,
                                        n_sg2,
                                        _sg,
                                        jnp.zeros((btc, h_per_t), dtype=jnp.float32),
                                        unroll=n_sg2,
                                    )
                            else:
                                w2_tile = (
                                    b_w2_dq_vmem[p_id]
                                    if w2_scale_hbm is not None else b_w2_x2_vmem[slot, p_id]
                                )
                                partial = jnp.dot(act, w2_tile,
                                                  preferred_element_type=jnp.float32)
                            acc_ref = b_y_acc_vmem.at[
                                pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                            ]
                            if bf_id == 0 and repeat_id == 0:
                                acc_ref[...] = partial
                            else:
                                acc_ref[...] = acc_ref[...] + partial
                        return None
                    if run_ffn2:
                        lax.fori_loop(0, num_btc_per_bts, act_down_btc, None)

            next_bf_id = bf_id + 2
            if next_bf_id < num_bf:
                start_fetch_w13_w2(local_e_id, slot, next_bf_id)

        if run_ffn2 and not disable_expert_store:
            if delay_store_wait:
                @pl.when(store_pending != 0)
                def _wait_previous_store():
                    wait_y_store()

            def writeback_btc(btc_id, _):
                for p_id in range(t_packing):
                    acc_slice = b_y_acc_vmem[
                        pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                    ]
                    b_y_stage_vmem.at[
                        pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                    ][...] = acc_slice.astype(output_hbm.dtype)
                return None
            lax.fori_loop(0, num_btc_per_bts, writeback_btc, None)
            pltpu.make_async_copy(
                src_ref=b_y_stage_vmem,
                dst_ref=output_hbm.at[
                    local_e_id,
                    pl.ds(tile_start, bts),
                    pl.ds(0, t_packing),
                    pl.ds(0, h_per_t),
                ],
                sem=y_sem,
            ).start()
            if delay_store_wait:
                return jnp.int32(1)
            wait_y_store()
            return jnp.int32(0)
        return store_pending

    def expert_body(local_e_id, store_pending):
        for bts_id in range(num_bts_tiles):
            store_pending = run_bts_tile(local_e_id, bts_id, store_pending)
        return store_pending

    final_store_pending = lax.fori_loop(
        0, local_num_experts, expert_body, jnp.int32(0), unroll=False
    )
    if delay_store_wait:
        @pl.when(final_store_pending != 0)
        def _wait_final_store():
            wait_y_store()


def build_routed_ffn(
    mesh: jax.sharding.Mesh,
    *,
    act_fn: str,
    bt: int,
    bf: int,
    btc: int,
    bts: int,
    num_bts_tiles: int,
    quant_block_k: int | None,
    direct_scaled_dot_ffn1: bool,
    direct_scaled_dot_ffn2: bool,
    apply_fp8_scale_ffn1: bool,
    apply_fp8_scale_ffn2: bool,
    disable_x_load: bool,
    disable_weight_load: bool,
    disable_ffn_compute: bool,
    disable_expert_store: bool,
    compute_repeat: int,
    stream_fchunk: int,
    direct_scale_ffn1_bfc: int,
    direct_scale_ffn2_bd2c: int,
    bench_stage: str,
    delay_store_wait: bool,
    gate_up_scratch_dtype: str,
    cast_ffn1_input_fp8: bool,
    cast_ffn2_input_fp8: bool,
    cast_ffn2_input_bf16: bool,
    dot_style: str,
    sg_chunk: int,
    sg_vmem_spill: int,
):
    dp_axis_name = "data"
    tp_axis_name = "tensor"
    scope_name = (
        f"routed-ffn-v2-bt_{bt}_{bts}_{btc}-bf_{bf}"
        f"-tiles_{num_bts_tiles}"
        f"-direct_f1_{int(direct_scaled_dot_ffn1)}"
        f"_f2_{int(direct_scaled_dot_ffn2)}"
        f"-scale_f1_{int(apply_fp8_scale_ffn1)}"
        f"_f2_{int(apply_fp8_scale_ffn2)}"
        f"-repeat_{compute_repeat}"
        f"-stream_{stream_fchunk}"
        f"-f1bfc_{direct_scale_ffn1_bfc}"
        f"-f2bd2c_{direct_scale_ffn2_bd2c}"
        f"-stage_{bench_stage}"
        f"-delay_store_{int(delay_store_wait)}"
        f"-gateup_{gate_up_scratch_dtype}"
        f"-castf1_{int(cast_ffn1_input_fp8)}"
        f"_castf2_{int(cast_ffn2_input_fp8)}"
        f"_castf2bf16_{int(cast_ffn2_input_bf16)}"
        f"-dot_{dot_style}"
        f"-sgc_{sg_chunk}"
        f"-sgvs_{sg_vmem_spill}"
    )
    if disable_x_load:
        scope_name += "-no_x"
    if disable_weight_load:
        scope_name += "-no_weight"
    if disable_ffn_compute:
        scope_name += "-no_compute"
    if disable_expert_store:
        scope_name += "-no_store"

    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    def make_call(expert_tokens, w1, w2, w3, w1_scale, w2_scale, w3_scale):
        local_num_experts, capacity, t_packing, h_per_t = expert_tokens.shape
        _, intermediate_size, hidden_size = w2.shape
        wb_slots = 2
        use_full_dot = dot_style in ("strix_fp8", "strix_bf16_weight", "native_bf16")
        use_w1_dq = w1_scale is not None and not direct_scaled_dot_ffn1 and not use_full_dot
        use_w3_dq = w3_scale is not None and not direct_scaled_dot_ffn1 and not use_full_dot
        use_w2_dq = w2_scale is not None and not direct_scaled_dot_ffn2 and not use_full_dot
        gate_up_dtype = jnp.bfloat16 if gate_up_scratch_dtype == "bf16" else jnp.float32
        run_ffn1 = bench_stage in (
            "full", "ffn1", "ffn1_w1", "ffn1_w1_reduce", "ffn1_w1_nostore"
        )
        run_ffn2 = bench_stage in ("full", "ffn2_scratch", "ffn2_synth")
        need_w3 = run_ffn1 and bench_stage not in (
            "ffn1_w1", "ffn1_w1_reduce", "ffn1_w1_nostore"
        )
        scratch_shapes = (
            pltpu.VMEM((bts, t_packing, h_per_t), expert_tokens.dtype),
            (None if not run_ffn1 else
                pltpu.VMEM((wb_slots, t_packing, h_per_t, bf), w1.dtype)),
            (None if not need_w3 else
                pltpu.VMEM((wb_slots, t_packing, h_per_t, bf), w3.dtype)),
            (None if not run_ffn2 else
                pltpu.VMEM((wb_slots, t_packing, bf, h_per_t), w2.dtype)),
            (None if not run_ffn1 or w1_scale is None else
                pltpu.VMEM((wb_slots, t_packing, h_per_t // quant_block_k, 1, bf), jnp.float32)),
            (None if not need_w3 or w3_scale is None else
                pltpu.VMEM((wb_slots, t_packing, h_per_t // quant_block_k, 1, bf), jnp.float32)),
            (None if not run_ffn2 or w2_scale is None else
                pltpu.VMEM((wb_slots, t_packing, bf // quant_block_k, 1, h_per_t), jnp.float32)),
            (None if not run_ffn1 or not use_w1_dq else
                pltpu.VMEM((t_packing, h_per_t, bf), jnp.bfloat16)),
            (None if not need_w3 or not use_w3_dq else
                pltpu.VMEM((t_packing, h_per_t, bf), jnp.bfloat16)),
            (None if not run_ffn2 or not use_w2_dq else
                pltpu.VMEM((t_packing, bf, h_per_t), jnp.bfloat16)),
            (None if not run_ffn1 and bench_stage != "ffn2_scratch" else
                pltpu.VMEM((bts, bf), gate_up_dtype)),
            (None if not run_ffn1 and bench_stage != "ffn2_scratch" else
                pltpu.VMEM((bts, bf), gate_up_dtype)),
            (None if not run_ffn2 else
                pltpu.VMEM((bts, t_packing, h_per_t), jnp.float32)),
            (None if not run_ffn2 else
                pltpu.VMEM((bts, t_packing, h_per_t), expert_tokens.dtype)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, 3)),
        )

        call = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _routed_ffn_kernel,
                    act_fn=act_fn,
                    direct_scaled_dot_ffn1=direct_scaled_dot_ffn1,
                    direct_scaled_dot_ffn2=direct_scaled_dot_ffn2,
                    apply_fp8_scale_ffn1=apply_fp8_scale_ffn1,
                    apply_fp8_scale_ffn2=apply_fp8_scale_ffn2,
                    disable_x_load=disable_x_load,
                    disable_weight_load=disable_weight_load,
                    disable_ffn_compute=disable_ffn_compute,
                    disable_expert_store=disable_expert_store,
                    compute_repeat=compute_repeat,
                    stream_fchunk=stream_fchunk,
                    direct_scale_ffn1_bfc=direct_scale_ffn1_bfc,
                    direct_scale_ffn2_bd2c=direct_scale_ffn2_bd2c,
                    bench_stage=bench_stage,
                    delay_store_wait=delay_store_wait,
                    gate_up_scratch_dtype=gate_up_scratch_dtype,
                    cast_ffn1_input_fp8=cast_ffn1_input_fp8,
                    cast_ffn2_input_fp8=cast_ffn2_input_fp8,
                    cast_ffn2_input_bf16=cast_ffn2_input_bf16,
                    dot_style=dot_style,
                    bt=bt,
                    bf=bf,
                    btc=btc,
                    bts=bts,
                    num_bts_tiles=num_bts_tiles,
                    quant_block_k=quant_block_k,
                    sg_chunk=sg_chunk,
                    sg_vmem_spill=sg_vmem_spill,
                ),
                out_shape=jax.ShapeDtypeStruct(expert_tokens.shape, expert_tokens.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    in_specs=[
                        hbm_spec,
                        hbm_spec,
                        hbm_spec,
                        hbm_spec,
                        None if w1_scale is None else hbm_spec,
                        None if w2_scale is None else hbm_spec,
                        None if w3_scale is None else hbm_spec,
                    ],
                    out_specs=hbm_spec,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    collective_id=0,
                    allow_collective_id_without_custom_barrier=True,
                    has_side_effects=True,
                    vmem_limit_bytes=64 * 1024 * 1024,
                ),
                name=scope_name,
            )
        )
        return call(
            pltpu.with_memory_space_constraint(expert_tokens, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3, pltpu.HBM),
            None if w1_scale is None else pltpu.with_memory_space_constraint(w1_scale, pltpu.HBM),
            None if w2_scale is None else pltpu.with_memory_space_constraint(w2_scale, pltpu.HBM),
            None if w3_scale is None else pltpu.with_memory_space_constraint(w3_scale, pltpu.HBM),
        )

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            P((dp_axis_name, tp_axis_name)),
            P((dp_axis_name, tp_axis_name)),
            P((dp_axis_name, tp_axis_name)),
            P((dp_axis_name, tp_axis_name)),
            None if quant_block_k is None else P((dp_axis_name, tp_axis_name)),
            None if quant_block_k is None else P((dp_axis_name, tp_axis_name)),
            None if quant_block_k is None else P((dp_axis_name, tp_axis_name)),
        ),
        out_specs=P((dp_axis_name, tp_axis_name)),
        check_vma=False,
    )
    def kernel(expert_tokens, w1, w2, w3, w1_scale, w2_scale, w3_scale):
        return make_call(expert_tokens, w1, w2, w3, w1_scale, w2_scale, w3_scale)

    return kernel


def main() -> None:
    jax.distributed.initialize()
    log(f"initialized: {jax.device_count()} devices, {jax.process_count()} procs")

    num_devices = jax.device_count()
    devices = np.array(jax.devices()).reshape(1, num_devices)
    mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
    ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

    d = int(os.environ.get("BENCH_D", "6144"))
    f = int(os.environ.get("BENCH_F", "2048"))
    e = int(os.environ.get("BENCH_E", "384"))
    top_k = int(os.environ.get("BENCH_TOPK", "8"))
    num_tokens = int(os.environ.get("BENCH_TOKENS", "16384"))
    bt = int(os.environ.get("BENCH_BT", "256"))
    bf_candidates = parse_csv_int("BENCH_BF", [1024])
    btc_candidates = parse_csv_int("BENCH_BTC", [72])
    bts_candidates = parse_csv_int("BENCH_BTS", [216])
    bts_tile_candidates = parse_csv_int("BENCH_BTS_TILES", [1])
    warmup = int(os.environ.get("BENCH_WARMUP", "2"))
    iters = int(os.environ.get("BENCH_ITERS", "5"))
    use_wall = os.environ.get("BENCH_WALL", "0") == "1"
    use_fp8 = os.environ.get("BENCH_FP8", "1") == "1"
    predequant_fp8_weights = os.environ.get("BENCH_PREDEQUANT_FP8_WEIGHTS", "0") == "1"
    quant_block_k = int(os.environ.get("BENCH_QBK", "128"))
    direct_scaled_dot = os.environ.get("BENCH_DIRECT_SCALED_DOT", "1") == "1"
    direct_f1_modes = parse_csv_bool("BENCH_DIRECT_SCALED_DOT_FFN1", [direct_scaled_dot])
    direct_f2_modes = parse_csv_bool("BENCH_DIRECT_SCALED_DOT_FFN2", [direct_scaled_dot])
    apply_scale_f1_modes = parse_csv_bool("BENCH_APPLY_FP8_SCALE_FFN1", [True])
    apply_scale_f2_modes = parse_csv_bool("BENCH_APPLY_FP8_SCALE_FFN2", [True])
    disable_x_load = os.environ.get("DISABLE_X_LOAD", "0") == "1"
    disable_weight_load = os.environ.get("DISABLE_WEIGHT_LOAD", "0") == "1"
    disable_ffn_compute = os.environ.get("DISABLE_FFN_COMPUTE", "0") == "1"
    disable_expert_store = os.environ.get("DISABLE_EXPERT_STORE", "0") == "1"
    compute_repeat = int(os.environ.get("BENCH_COMPUTE_REPEAT", "1"))
    stream_fchunk = int(os.environ.get("BENCH_STREAM_FCHUNK", "0"))
    direct_scale_ffn1_bfc = int(os.environ.get("BENCH_DIRECT_SCALE_FFN1_BFC", "0"))
    direct_scale_ffn2_bd2c = int(os.environ.get("BENCH_DIRECT_SCALE_FFN2_BD2C", "0"))
    sg_chunk_candidates = parse_csv_int("BENCH_SG_CHUNK", [0])
    sg_vmem_spill_candidates = parse_csv_int("BENCH_SG_VMEM_SPILL", [0])
    bench_stage = os.environ.get("BENCH_STAGE", "full")
    delay_store_wait = os.environ.get("BENCH_DELAY_STORE_WAIT", "0") == "1"
    gate_up_scratch_dtype = os.environ.get("BENCH_GATE_UP_SCRATCH_DTYPE", "f32")
    cast_ffn1_input_fp8 = os.environ.get("BENCH_CAST_FFN1_INPUT_FP8", "0") == "1"
    cast_ffn2_input_fp8 = os.environ.get("BENCH_CAST_FFN2_INPUT_FP8", "0") == "1"
    cast_ffn2_input_bf16 = os.environ.get("BENCH_CAST_FFN2_INPUT_BF16", "0") == "1"
    dot_styles = parse_csv_str("BENCH_DOT_STYLE", ["v2"])
    token_dtype_name = os.environ.get("BENCH_TOKEN_DTYPE", "bf16")
    valid_stages = (
        "full", "ffn1", "ffn1_w1", "ffn1_w1_reduce", "ffn1_w1_nostore",
        "ffn2_scratch", "ffn2_synth"
    )
    if bench_stage not in valid_stages:
        raise ValueError(f"{bench_stage=} must be one of {valid_stages}")
    if gate_up_scratch_dtype not in ("f32", "bf16"):
        raise ValueError(f"{gate_up_scratch_dtype=} must be f32 or bf16")
    valid_dot_styles = ("v2", "strix_fp8", "strix_bf16_weight", "native_bf16")
    invalid_dot_styles = [style for style in dot_styles if style not in valid_dot_styles]
    if invalid_dot_styles:
        raise ValueError(f"{invalid_dot_styles=} must be within {valid_dot_styles}")
    if predequant_fp8_weights and not use_fp8:
        raise ValueError("BENCH_PREDEQUANT_FP8_WEIGHTS=1 requires BENCH_FP8=1.")
    if predequant_fp8_weights and dot_styles != ["native_bf16"]:
        raise ValueError(
            "BENCH_PREDEQUANT_FP8_WEIGHTS=1 currently requires "
            "BENCH_DOT_STYLE=native_bf16."
        )
    if use_fp8 and "native_bf16" in dot_styles and not predequant_fp8_weights:
        raise ValueError("BENCH_DOT_STYLE=native_bf16 requires BENCH_FP8=0.")
    if cast_ffn2_input_fp8 and cast_ffn2_input_bf16:
        raise ValueError("BENCH_CAST_FFN2_INPUT_FP8 and BF16 are mutually exclusive.")
    if token_dtype_name not in ("bf16", "fp8"):
        raise ValueError(f"{token_dtype_name=} must be bf16 or fp8")

    token_dtype = jnp.float8_e4m3fn if token_dtype_name == "fp8" else jnp.bfloat16
    t_packing = get_dtype_packing(token_dtype)
    if d % get_dtype_packing(jnp.bfloat16) != 0:
        raise ValueError(f"{d=} must be divisible by bf16 packing")
    if e % num_devices != 0:
        raise ValueError(f"{e=} must be divisible by {num_devices=}")
    if compute_repeat < 1:
        raise ValueError(f"{compute_repeat=} must be >= 1")
    if stream_fchunk < 0:
        raise ValueError(f"{stream_fchunk=} must be >= 0")
    if stream_fchunk and (stream_fchunk % quant_block_k != 0):
        raise ValueError(f"{stream_fchunk=} must be divisible by {quant_block_k=}")
    if direct_scale_ffn1_bfc < 0:
        raise ValueError(f"{direct_scale_ffn1_bfc=} must be >= 0")
    if direct_scale_ffn1_bfc and (
        f % direct_scale_ffn1_bfc != 0
        or direct_scale_ffn1_bfc % 128 != 0
    ):
        raise ValueError(
            f"{direct_scale_ffn1_bfc=} must divide {f=} and be 128-aligned."
        )
    if direct_scale_ffn2_bd2c < 0:
        raise ValueError(f"{direct_scale_ffn2_bd2c=} must be >= 0")
    if direct_scale_ffn2_bd2c and (
        d % direct_scale_ffn2_bd2c != 0
        or direct_scale_ffn2_bd2c % (get_dtype_packing(token_dtype) * 128) != 0
    ):
        raise ValueError(
            f"{direct_scale_ffn2_bd2c=} must divide {d=} and be "
            f"{get_dtype_packing(token_dtype) * 128}-aligned."
        )

    h_per_t = d // t_packing
    avg_routed = num_tokens * top_k / e
    log(
        f"model: E={e} d={d} f={f} k={top_k} ep={num_devices} "
        f"avg_routed={avg_routed:.1f} fp8={use_fp8} "
        f"predequant_fp8_weights={predequant_fp8_weights} "
        f"dot_styles={dot_styles} token_dtype={token_dtype_name}"
    )

    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    def make_sharded(rng_key, shape, dtype, scale=1.0):
        local_shape = (shape[0] // num_devices, *shape[1:])
        per_device = []
        for i, dev in enumerate(jax.local_devices()):
            sk = jax.random.fold_in(rng_key, jax.process_index() * len(jax.local_devices()) + i)
            shard = jax.random.normal(sk, local_shape, dtype=dtype) * scale
            per_device.append(jax.device_put(shard, dev))
        return jax.make_array_from_single_device_arrays(shape, ep_sharding, per_device)

    log("creating weight arrays...")
    w1 = make_sharded(k1, (e, d, f), jnp.bfloat16, 0.01)
    w2 = make_sharded(k2, (e, f, d), jnp.bfloat16, 0.01)
    w3 = make_sharded(k3, (e, d, f), jnp.bfloat16, 0.01)

    w1_scale = w2_scale = w3_scale = None
    qbk_arg = None
    if use_fp8:
        log(f"quantizing weights to fp8 (qbk={quant_block_k})...")

        @jax.jit
        @jax.shard_map(
            mesh=mesh,
            in_specs=(P(("data", "tensor")),),
            out_specs=(P(("data", "tensor")), P(("data", "tensor"))),
            check_vma=False,
        )
        def quantize_shard_map(w):
            local_w = w
            e_loc, k_dim, n_dim = local_w.shape
            w_f32 = local_w.astype(jnp.float32).reshape(
                e_loc, k_dim // quant_block_k, quant_block_k, n_dim
            )
            amax = jnp.max(jnp.abs(w_f32), axis=2, keepdims=True)
            scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
            w_q = (w_f32 / scale).astype(jnp.float8_e4m3fn)
            return w_q.reshape(e_loc, k_dim, n_dim), scale.astype(jnp.float32)

        w1, w1_scale = quantize_shard_map(w1)
        w2, w2_scale = quantize_shard_map(w2)
        w3, w3_scale = quantize_shard_map(w3)
        qbk_arg = quant_block_k
        log("fp8 quantization done")

        if predequant_fp8_weights:
            log("predequantizing fp8 weights to bf16 outside routed kernel...")

            @jax.jit
            @jax.shard_map(
                mesh=mesh,
                in_specs=(P(("data", "tensor")), P(("data", "tensor"))),
                out_specs=P(("data", "tensor")),
                check_vma=False,
            )
            def dequantize_shard_map(w_q, scale):
                local_w = w_q
                e_loc, k_dim, n_dim = local_w.shape
                w_f32 = local_w.astype(jnp.float32).reshape(
                    e_loc, k_dim // quant_block_k, quant_block_k, n_dim
                )
                w_dq = w_f32 * scale
                return w_dq.reshape(e_loc, k_dim, n_dim).astype(jnp.bfloat16)

            w1 = dequantize_shard_map(w1, w1_scale)
            w2 = dequantize_shard_map(w2, w2_scale)
            w3 = dequantize_shard_map(w3, w3_scale)
            w1_scale = w2_scale = w3_scale = None
            qbk_arg = None
            log("predequantization done")

    results: list[tuple[str, float, list[float]]] = []
    for bf, btc, bts, bts_tiles, direct_f1, direct_f2, apply_scale_f1, apply_scale_f2, dot_style, sg_chunk, sg_vmem_spill in itertools.product(
        bf_candidates,
        btc_candidates,
        bts_candidates,
        bts_tile_candidates,
        direct_f1_modes,
        direct_f2_modes,
        apply_scale_f1_modes,
        apply_scale_f2_modes,
        dot_styles,
        sg_chunk_candidates,
        sg_vmem_spill_candidates,
    ):
        if bts % btc != 0:
            log(f"skip invalid bts={bts} btc={btc}")
            continue
        if f % bf != 0:
            log(f"skip invalid f={f} bf={bf}")
            continue
        if stream_fchunk and bf % stream_fchunk != 0:
            log(f"skip invalid stream_fchunk={stream_fchunk} bf={bf}")
            continue
        if direct_scale_ffn1_bfc and bf % direct_scale_ffn1_bfc != 0:
            log(f"skip invalid ffn1_bfc={direct_scale_ffn1_bfc} bf={bf}")
            continue
        n_sg_check = h_per_t // quant_block_k if use_fp8 else 1
        if sg_chunk > 0 and (n_sg_check == 0 or n_sg_check % sg_chunk != 0):
            log(f"skip invalid sg_chunk={sg_chunk} n_sg={n_sg_check}")
            continue
        if sg_vmem_spill > 0 and (n_sg_check == 0 or n_sg_check % sg_vmem_spill != 0):
            log(f"skip invalid sg_vmem_spill={sg_vmem_spill} n_sg={n_sg_check}")
            continue
        if sg_chunk > 0 and sg_vmem_spill > 0:
            log(f"skip sg_chunk={sg_chunk} + sg_vmem_spill={sg_vmem_spill} (mutually exclusive)")
            continue
        capacity = bts * bts_tiles
        expert_tokens = make_sharded(
            k4, (e, capacity, t_packing, h_per_t), token_dtype, 1.0
        )
        run = build_routed_ffn(
            mesh,
            act_fn="swigluoai",
            bt=bt,
            bf=bf,
            btc=btc,
            bts=bts,
            num_bts_tiles=bts_tiles,
            quant_block_k=qbk_arg,
            direct_scaled_dot_ffn1=direct_f1,
            direct_scaled_dot_ffn2=direct_f2,
            apply_fp8_scale_ffn1=apply_scale_f1,
            apply_fp8_scale_ffn2=apply_scale_f2,
            disable_x_load=disable_x_load,
            disable_weight_load=disable_weight_load,
            disable_ffn_compute=disable_ffn_compute,
            disable_expert_store=disable_expert_store,
            compute_repeat=compute_repeat,
            stream_fchunk=stream_fchunk,
            direct_scale_ffn1_bfc=direct_scale_ffn1_bfc,
            direct_scale_ffn2_bd2c=direct_scale_ffn2_bd2c,
            bench_stage=bench_stage,
            delay_store_wait=delay_store_wait,
            gate_up_scratch_dtype=gate_up_scratch_dtype,
            cast_ffn1_input_fp8=cast_ffn1_input_fp8,
            cast_ffn2_input_fp8=cast_ffn2_input_fp8,
            cast_ffn2_input_bf16=cast_ffn2_input_bf16,
            dot_style=dot_style,
            sg_chunk=sg_chunk,
            sg_vmem_spill=sg_vmem_spill,
        )
        tag = (
            f"bt={bt},bf={bf},btc={btc},bts={bts},tiles={bts_tiles},"
            f"direct_f1={int(direct_f1)},direct_f2={int(direct_f2)},"
            f"scale_f1={int(apply_scale_f1)},scale_f2={int(apply_scale_f2)},"
            f"predeq={int(predequant_fp8_weights)},"
            f"repeat={compute_repeat},stream={stream_fchunk},"
            f"f1_bfc={direct_scale_ffn1_bfc},"
            f"f2_bd2c={direct_scale_ffn2_bd2c},"
            f"x={int(not disable_x_load)},"
            f"stage={bench_stage},delay_store={int(delay_store_wait)},"
            f"gateup={gate_up_scratch_dtype},"
            f"cast_f1={int(cast_ffn1_input_fp8)},"
            f"cast_f2={int(cast_ffn2_input_fp8)},"
            f"cast_f2bf16={int(cast_ffn2_input_bf16)},"
            f"dot={dot_style},token={token_dtype_name},"
            f"sgc={sg_chunk},sgvs={sg_vmem_spill},"
            f"compute={int(not disable_ffn_compute)},"
            f"store={int(not disable_expert_store)},weight={int(not disable_weight_load)}"
        )
        log(f"compile/run {tag}")
        out = run(expert_tokens, w1, w2, w3, w1_scale, w2_scale, w3_scale)
        jax.block_until_ready(out)
        timeit = wall_timeit if use_wall else trace_timeit
        times = timeit(
            lambda: run(expert_tokens, w1, w2, w3, w1_scale, w2_scale, w3_scale),
            warmup,
            iters,
        )
        if jax.process_index() == 0:
            arr = np.asarray(times, dtype=np.float64)
            if arr.size == 0:
                log(f"RESULT {tag}: no trace durations")
                continue
            mean_ms = float(arr.mean())
            log(
                f"RESULT {tag}: mean={mean_ms:.3f}ms "
                f"min={arr.min():.3f} max={arr.max():.3f} "
                f"samples={[round(float(v), 3) for v in arr.tolist()]}"
            )
            log(
                "RESULT "
                + tag
                + ": "
                + estimate_roofline(
                    mean_ms=mean_ms,
                    num_devices=num_devices,
                    d=d,
                    f=f,
                    e=e,
                    bts=bts,
                    bts_tiles=bts_tiles,
                    use_fp8=use_fp8 and not predequant_fp8_weights,
                    quant_block_k=quant_block_k,
                    apply_fp8_scale_ffn1=apply_scale_f1,
                    apply_fp8_scale_ffn2=apply_scale_f2,
                    disable_x_load=disable_x_load,
                    disable_weight_load=disable_weight_load,
                    disable_ffn_compute=disable_ffn_compute,
                    disable_expert_store=disable_expert_store,
                    compute_repeat=compute_repeat,
                    stream_fchunk=stream_fchunk,
                    direct_scale_ffn1_bfc=direct_scale_ffn1_bfc,
                    direct_scale_ffn2_bd2c=direct_scale_ffn2_bd2c,
                    bench_stage=bench_stage,
                    delay_store_wait=delay_store_wait,
                    gate_up_scratch_dtype=gate_up_scratch_dtype,
                    cast_ffn1_input_fp8=cast_ffn1_input_fp8,
                    cast_ffn2_input_fp8=cast_ffn2_input_fp8,
                    cast_ffn2_input_bf16=cast_ffn2_input_bf16,
                    dot_style=dot_style,
                    token_dtype=token_dtype_name,
                )
            )
            results.append((tag, mean_ms, arr.tolist()))

    if jax.process_index() == 0 and results:
        best = min(results, key=lambda item: item[1])
        log(f"BEST {best[1]:.3f}ms [{best[0]}]")
    log("done")


if __name__ == "__main__":
    main()
