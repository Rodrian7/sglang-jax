"""Standalone TC/Pallas communication microbench.

This isolates the communication primitive used by fused MoE v2 scatter/gather:
`pltpu.make_async_remote_copy` inside a Pallas kernel.  It is intentionally
separate from the MoE kernel so experiments do not perturb the production path.

Env vars:
  BENCH_MODE       — local, remote, remote_overlap, remote_batch, remote_moe, or all (default: all)
  BENCH_ROWS       — comma-separated rows per payload (default: 1,8,64)
  BENCH_HIDDEN     — hidden size in bf16 elements per row (default: 3072)
  BENCH_REPEATS    — remote-copy repeats inside one kernel call (default: 1)
  BENCH_EXPERTS    — local expert semaphore slots for remote_moe (default: 12)
  BENCH_COMPUTE    — dummy matmul repeats for remote_overlap (default: 8)
  BENCH_WARMUP     — warmup iterations (default: 3)
  BENCH_ITERS      — timed iterations (default: 20)
  BENCH_WALL       — use wall timing instead of trace timing (default: 0)
"""
from __future__ import annotations

import functools
import gzip
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


t0 = time.time()
TRACE_ROOT = "/tmp/tpu_logs/tc_comm_trace"
KERNEL_NAME_RE = re.compile(r"tc-comm-.*")


def log(msg: str) -> None:
    print(f"[{time.time() - t0:.1f}s][p{jax.process_index()}] {msg}", flush=True)


def parse_csv_int(env_key: str, default: list[int]) -> list[int]:
    raw = os.environ.get(env_key)
    if raw is None:
        return default
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_csv_str(env_key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(env_key)
    if raw is None:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_trace(trace_root: str) -> dict[str, Any]:
    trace_dir = pathlib.Path(trace_root) / "plugins" / "profile"
    if not trace_dir.exists():
        raise FileNotFoundError(f"No trace output under {trace_dir}")
    latest_dir = max(trace_dir.iterdir(), key=os.path.getmtime)
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace json.gz under {latest_dir}")
    combined: dict[str, Any] = {"traceEvents": []}
    for trace_file in sorted(trace_files):
        with gzip.open(trace_file, "rb") as fh:
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


def _tc_comm_kernel(
    x_hbm,
    y_hbm,
    output_hbm,
    local_sem,
    send_sem,
    recv_sem,
    a_vmem,
    b_vmem,
    *,
    mode: str,
    repeats: int,
    compute_repeats: int,
    dp_axis_name: str,
    tp_axis_name: str,
    rows: int,
    hidden_size: int,
    local_num_experts: int,
):
    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = dp_size * tp_size
    next_id = (my_id + jnp.int32(1)) % num_devices
    prev_id = (my_id + num_devices - jnp.int32(1)) % num_devices

    def get_mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

    def local_copy_body(i, _):
        sem_id = i & jnp.int32(1)
        copy = pltpu.make_async_copy(
            src_ref=x_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            dst_ref=y_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            sem=local_sem.at[sem_id],
        )
        copy.start()
        copy.wait()
        return None

    def remote_copy_body(i, _):
        sem_id = i & jnp.int32(1)
        send = send_sem.at[sem_id]
        recv = recv_sem.at[sem_id]
        pltpu.make_async_remote_copy(
            src_ref=x_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            dst_ref=y_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            send_sem=send,
            recv_sem=recv,
            device_id=get_mesh_device_id(next_id),
            device_id_type=pltpu.DeviceIdType.MESH,
        ).start()
        recv_ref = y_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)]
        pltpu.make_async_copy(src_ref=recv_ref, dst_ref=recv_ref, sem=recv).wait()
        send_ref = x_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)]
        pltpu.make_async_copy(src_ref=send_ref, dst_ref=send_ref, sem=send).wait()
        return None

    def remote_overlap_body(i, acc):
        sem_id = i & jnp.int32(1)
        send = send_sem.at[sem_id]
        recv = recv_sem.at[sem_id]
        pltpu.make_async_remote_copy(
            src_ref=x_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            dst_ref=y_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            send_sem=send,
            recv_sem=recv,
            device_id=get_mesh_device_id(next_id),
            device_id_type=pltpu.DeviceIdType.MESH,
        ).start()
        for _ in range(compute_repeats):
            acc += a_vmem[...].astype(jnp.float32) * b_vmem[...].astype(jnp.float32)
        recv_ref = y_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)]
        pltpu.make_async_copy(src_ref=recv_ref, dst_ref=recv_ref, sem=recv).wait()
        send_ref = x_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)]
        pltpu.make_async_copy(src_ref=send_ref, dst_ref=send_ref, sem=send).wait()
        return acc

    def remote_batch_start_body(i, _):
        send = send_sem.at[0]
        recv = recv_sem.at[0]
        pltpu.make_async_remote_copy(
            src_ref=x_hbm.at[pl.ds(i, 1), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            dst_ref=y_hbm.at[pl.ds(i, 1), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            send_sem=send,
            recv_sem=recv,
            device_id=get_mesh_device_id(next_id),
            device_id_type=pltpu.DeviceIdType.MESH,
        ).start()
        return None

    def remote_batch_wait():
        send = send_sem.at[0]
        recv = recv_sem.at[0]
        recv_ref = y_hbm.at[
            pl.ds(0, repeats), pl.ds(0, 2), pl.ds(0, hidden_size // 2)
        ]
        pltpu.make_async_copy(src_ref=recv_ref, dst_ref=recv_ref, sem=recv).wait()
        send_ref = x_hbm.at[
            pl.ds(0, repeats), pl.ds(0, 2), pl.ds(0, hidden_size // 2)
        ]
        pltpu.make_async_copy(src_ref=send_ref, dst_ref=send_ref, sem=send).wait()

    def remote_moe_start_body(i, _):
        slot = i % jnp.int32(local_num_experts)
        # Deterministic all-to-all remote routing: each source spreads token rows
        # over all non-local destination devices, while sharing per-expert sems.
        dest = (my_id + jnp.int32(1) + (i % (num_devices - jnp.int32(1)))) % num_devices
        pltpu.make_async_remote_copy(
            src_ref=x_hbm.at[pl.ds(i, 1), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            dst_ref=y_hbm.at[pl.ds(i, 1), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            send_sem=send_sem.at[slot],
            recv_sem=recv_sem.at[slot],
            device_id=get_mesh_device_id(dest),
            device_id_type=pltpu.DeviceIdType.MESH,
        ).start()
        return None

    def remote_moe_count_recv(slot):
        def _count_one(n, acc):
            send_id = n // jnp.int32(repeats)
            row = n % jnp.int32(repeats)
            dest = (
                send_id + jnp.int32(1) + (row % (num_devices - jnp.int32(1)))
            ) % num_devices
            row_slot = row % jnp.int32(local_num_experts)
            should_count = jnp.logical_and(dest == my_id, row_slot == slot)
            return acc + should_count.astype(jnp.int32)

        return lax.fori_loop(
            0, num_devices * jnp.int32(repeats), _count_one, jnp.int32(0),
            unroll=False,
        )

    def remote_moe_count_send(slot):
        def _count_one(row, acc):
            row_slot = row % jnp.int32(local_num_experts)
            return acc + (row_slot == slot).astype(jnp.int32)

        return lax.fori_loop(0, jnp.int32(repeats), _count_one, jnp.int32(0))

    def remote_moe_wait_slot(slot, _):
        recv_count = remote_moe_count_recv(slot)

        @pl.when(recv_count != 0)
        def _wait_recv(recv_count=recv_count, slot=slot):
            recv_ref = y_hbm.at[
                pl.ds(0, recv_count), pl.ds(0, 2), pl.ds(0, hidden_size // 2)
            ]
            pltpu.make_async_copy(
                src_ref=recv_ref, dst_ref=recv_ref, sem=recv_sem.at[slot],
            ).wait()

        send_count = remote_moe_count_send(slot)

        @pl.when(send_count != 0)
        def _wait_send(send_count=send_count, slot=slot):
            send_ref = x_hbm.at[
                pl.ds(0, send_count), pl.ds(0, 2), pl.ds(0, hidden_size // 2)
            ]
            pltpu.make_async_copy(
                src_ref=send_ref, dst_ref=send_ref, sem=send_sem.at[slot],
            ).wait()

        return None

    del prev_id

    def copy_y_to_output():
        copy = pltpu.make_async_copy(
            src_ref=y_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            dst_ref=output_hbm.at[pl.ds(0, rows), pl.ds(0, 2), pl.ds(0, hidden_size // 2)],
            sem=local_sem.at[0],
        )
        copy.start()
        copy.wait()

    if mode == "local":
        lax.fori_loop(0, repeats, local_copy_body, None, unroll=False)
        copy_y_to_output()
        return
    if mode == "remote":
        lax.fori_loop(0, repeats, remote_copy_body, None, unroll=False)
        copy_y_to_output()
        return
    if mode == "remote_overlap":
        a_vmem[...] = jnp.ones_like(a_vmem)
        b_vmem[...] = jnp.ones_like(b_vmem)
        acc = jnp.zeros((128, 128), jnp.float32)
        acc = lax.fori_loop(0, repeats, remote_overlap_body, acc, unroll=False)
        copy_y_to_output()
        return
    if mode == "remote_batch":
        lax.fori_loop(0, repeats, remote_batch_start_body, None, unroll=False)
        remote_batch_wait()
        copy_y_to_output()
        return
    if mode == "remote_moe":
        lax.fori_loop(0, repeats, remote_moe_start_body, None, unroll=False)
        lax.fori_loop(
            0, jnp.int32(local_num_experts), remote_moe_wait_slot, None,
            unroll=False,
        )
        copy_y_to_output()
        return
    raise RuntimeError(f"Unsupported mode: {mode}")


def build_tc_comm(
    mesh: jax.sharding.Mesh,
    *,
    mode: str,
    rows: int,
    hidden_size: int,
    repeats: int,
    compute_repeats: int,
    local_num_experts: int,
):
    dp_axis_name = "data"
    tp_axis_name = "tensor"
    payload_bytes = rows * hidden_size * jnp.dtype(jnp.bfloat16).itemsize
    scope_name = (
        f"tc-comm-{mode}-bytes{payload_bytes}"
        f"-rep{repeats}-compute{compute_repeats}"
    )

    kernel_call = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _tc_comm_kernel,
                mode=mode,
                repeats=repeats,
                compute_repeats=compute_repeats,
                dp_axis_name=dp_axis_name,
                tp_axis_name=tp_axis_name,
                rows=rows,
                hidden_size=hidden_size,
                local_num_experts=local_num_experts,
            ),
            out_shape=jax.ShapeDtypeStruct((rows, 2, hidden_size // 2), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA((
                        local_num_experts if mode == "remote_moe" else 2,
                    )),
                    pltpu.SemaphoreType.DMA((
                        local_num_experts if mode == "remote_moe" else 2,
                    )),
                    pltpu.SemaphoreType.DMA((2,)),
                    pltpu.VMEM((128, 128), jnp.bfloat16),
                    pltpu.VMEM((128, 128), jnp.bfloat16),
                ),
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

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            jax.sharding.PartitionSpec((dp_axis_name, tp_axis_name)),
            jax.sharding.PartitionSpec((dp_axis_name, tp_axis_name)),
        ),
        out_specs=jax.sharding.PartitionSpec((dp_axis_name, tp_axis_name)),
        check_vma=False,
    )
    def run(x, y):
        return kernel_call(
            pltpu.with_memory_space_constraint(x, pltpu.HBM),
            pltpu.with_memory_space_constraint(y, pltpu.HBM),
        )

    return run


def main() -> None:
    jax.distributed.initialize()
    log(f"initialized: {jax.device_count()} devices, {jax.process_count()} procs")

    modes = parse_csv_str("BENCH_MODE", ["all"])
    if modes == ["all"]:
        modes = ["local", "remote", "remote_overlap"]
    bad_modes = [
        mode
        for mode in modes
        if mode not in {
            "local", "remote", "remote_overlap", "remote_batch", "remote_moe",
        }
    ]
    if bad_modes:
        raise ValueError(f"Unsupported BENCH_MODE values: {bad_modes}")

    row_list = parse_csv_int("BENCH_ROWS", [1, 8, 64])
    hidden_size = int(os.environ.get("BENCH_HIDDEN", "3072"))
    if hidden_size % 2 != 0:
        raise ValueError(f"{hidden_size=} must be divisible by 2")
    repeats = int(os.environ.get("BENCH_REPEATS", "1"))
    local_num_experts = int(os.environ.get("BENCH_EXPERTS", "12"))
    compute_repeats = int(os.environ.get("BENCH_COMPUTE", "8"))
    warmup = int(os.environ.get("BENCH_WARMUP", "3"))
    iters = int(os.environ.get("BENCH_ITERS", "20"))
    use_wall = os.environ.get("BENCH_WALL", "0") == "1"

    num_devices = jax.device_count()
    devices = np.array(jax.devices()).reshape(1, num_devices)
    mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(("data", "tensor")),
    )

    log(
        f"modes={modes} rows={row_list} hidden={hidden_size} repeats={repeats} "
        f"compute_repeats={compute_repeats} experts={local_num_experts} "
        f"timing={'wall' if use_wall else 'trace'}"
    )

    for rows in row_list:
        dtype_bytes = jnp.dtype(jnp.bfloat16).itemsize
        payload_bytes = rows * hidden_size * dtype_bytes
        key = jax.random.key(0)
        local_shape = (rows, 2, hidden_size // 2)
        per_device_x = []
        per_device_y = []
        for idx, dev in enumerate(jax.local_devices()):
            sk = jax.random.fold_in(key, jax.process_index() * len(jax.local_devices()) + idx)
            x_local = jax.random.normal(sk, local_shape, dtype=jnp.bfloat16)
            y_local = jnp.zeros(local_shape, dtype=jnp.bfloat16)
            per_device_x.append(jax.device_put(x_local, dev))
            per_device_y.append(jax.device_put(y_local, dev))
        x = jax.make_array_from_single_device_arrays(
            (rows * num_devices, 2, hidden_size // 2), sharding, per_device_x,
        )
        y = jax.make_array_from_single_device_arrays(
            (rows * num_devices, 2, hidden_size // 2), sharding, per_device_y,
        )

        for mode in modes:
            if mode == "remote_batch" and repeats > rows:
                raise ValueError(
                    f"remote_batch requires {repeats=} <= {rows=} so each copy "
                    "has a distinct row."
                )
            run = build_tc_comm(
                mesh,
                mode=mode,
                rows=rows,
                hidden_size=hidden_size,
                repeats=repeats,
                compute_repeats=compute_repeats,
                local_num_experts=local_num_experts,
            )
            log(f"compile/run mode={mode} rows={rows} bytes={payload_bytes}")
            out = run(x, y)
            jax.block_until_ready(out)
            times = (
                wall_timeit(lambda: run(x, y), warmup, iters)
                if use_wall else
                trace_timeit(lambda: run(x, y), warmup, iters)
            )
            if jax.process_index() == 0:
                arr = np.asarray(times, dtype=np.float64)
                if arr.size == 0:
                    log(
                        f"RESULT mode={mode} rows={rows} "
                        f"bytes={payload_bytes} no trace durations"
                    )
                else:
                    log(
                        f"RESULT mode={mode} rows={rows} bytes={payload_bytes} "
                        f"mean={arr.mean():.4f}ms min={arr.min():.4f}ms "
                        f"max={arr.max():.4f}ms samples="
                        f"{[round(float(v), 4) for v in arr.tolist()]}"
                    )

    log("done")


if __name__ == "__main__":
    main()
