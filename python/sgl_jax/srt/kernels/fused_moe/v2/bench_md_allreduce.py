"""Minimal repro for async_metadata_broadcast hang on 2D mesh.

Usage (4 pods, EP=32):
  python bench_md_allreduce.py --mode butterfly   # should work
  python bench_md_allreduce.py --mode broadcast    # hangs in serving
  python bench_md_allreduce.py --mode broadcast_barrier # all-at-once with bracketing barriers
  python bench_md_allreduce.py --mode scan         # prefix scan only
  python bench_md_allreduce.py --mode scan_owner128 # prefix scan + aligned owner exchange
  python bench_md_allreduce.py --mode broadcast_1d # works (bench_v2 config)

Set MESH=1d to use (1, 32) mesh (bench_v2 style), MESH=2d for (8, 4) (serving style).
"""

from __future__ import annotations

import argparse
import math
import os
import time

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def make_kernel(
    *,
    mode: str,
    num_devices: int,
    padded_num_experts: int,
    dp_axis_name: str,
    tp_axis_name: str,
    add_scatter: bool = False,
    scatter_tokens: int = 64,
):
    hidden_size = 6144
    t_packing = 4
    h_per_t = hidden_size // t_packing  # 1536

    def _kernel(
        d2e_input_hbm,   # (num_devices, 1, padded_num_experts) int32
        d2e_count_smem,   # (1, 1, padded_num_experts) int32 — output
        d2e_count_vmem,   # (num_devices, 1, padded_num_experts) int32 — scratch
        scan_work_vmem,   # (1, padded_num_experts) int32 — prefix-scan scratch
        scan_recv_vmem,   # (1, padded_num_experts) int32 — prefix-scan scratch
        scan_prefix_vmem, # (1, padded_num_experts) int32 — inclusive prefix
        scan_sizes_vmem,  # (1, padded_num_experts) int32 — global sizes
        barrier_vmem,     # (1,) int32
        md_send_sem,      # DMA scalar
        md_recv_sem,      # DMA scalar
        barrier_sem,      # BARRIER scalar
    ):
        dp_rank = lax.axis_index(dp_axis_name)
        tp_rank = lax.axis_index(tp_axis_name)
        tp_size = lax.axis_size(tp_axis_name)
        my_id = dp_rank * tp_size + tp_rank

        def get_mesh_device_id(ep_rank):
            return (ep_rank // tp_size, ep_rank % tp_size)

        def sync_barrier():
            for i in range(num_devices):
                pltpu.semaphore_signal(
                    barrier_sem,
                    device_id=get_mesh_device_id(i),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pltpu.semaphore_wait(barrier_sem, num_devices)

        # Load local data from HBM (input is sharded, local shape = (1, 1, padded))
        load = pltpu.async_copy(
            src_ref=d2e_input_hbm.at[pl.ds(0, 1)],
            dst_ref=d2e_count_vmem.at[pl.ds(my_id, 1)],
            sem=md_recv_sem,
        )
        load.wait()

        if mode == "broadcast":
            # All-to-all broadcast — hangs on 2D mesh in serving
            for step in range(num_devices):
                peer = (my_id + jnp.int32(step)) % jnp.int32(num_devices)
                pltpu.make_async_remote_copy(
                    src_ref=d2e_count_vmem.at[
                        my_id,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    dst_ref=d2e_count_vmem.at[
                        my_id,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    send_sem=md_send_sem,
                    recv_sem=md_recv_sem,
                    device_id=get_mesh_device_id(peer),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            for _ in range(num_devices):
                recv_ref = d2e_count_vmem.at[
                    0, pl.ds(0, 1), pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=recv_ref, dst_ref=recv_ref,
                    sem=md_recv_sem,
                ).wait()

            for _ in range(num_devices):
                send_ref = d2e_count_vmem.at[
                    my_id, pl.ds(0, 1), pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=send_ref, dst_ref=send_ref,
                    sem=md_send_sem,
                ).wait()

        elif mode == "broadcast_barrier":
            # Full allgather broadcast with one bracketing barrier before and
            # after the outstanding remote copies. This tests whether Pallas
            # needs barriers around the collective region rather than one
            # barrier per recursive-doubling round.
            sync_barrier()

            for step in range(1, num_devices):
                peer = (my_id + jnp.int32(step)) % jnp.int32(num_devices)
                pltpu.make_async_remote_copy(
                    src_ref=d2e_count_vmem.at[
                        my_id,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    dst_ref=d2e_count_vmem.at[
                        my_id,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    send_sem=md_send_sem,
                    recv_sem=md_recv_sem,
                    device_id=get_mesh_device_id(peer),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            for _ in range(num_devices - 1):
                recv_ref = d2e_count_vmem.at[
                    0, pl.ds(0, 1), pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=recv_ref, dst_ref=recv_ref,
                    sem=md_recv_sem,
                ).wait()

            for _ in range(num_devices - 1):
                send_ref = d2e_count_vmem.at[
                    my_id, pl.ds(0, 1), pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=send_ref, dst_ref=send_ref,
                    sem=md_send_sem,
                ).wait()

            sync_barrier()

        elif mode == "broadcast_ds":
            # Same broadcast but using pl.ds for all indexing
            for step in range(num_devices):
                peer = (my_id + jnp.int32(step)) % jnp.int32(num_devices)
                pltpu.make_async_remote_copy(
                    src_ref=d2e_count_vmem.at[
                        pl.ds(my_id, 1),
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    dst_ref=d2e_count_vmem.at[
                        pl.ds(my_id, 1),
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    send_sem=md_send_sem,
                    recv_sem=md_recv_sem,
                    device_id=get_mesh_device_id(peer),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            for _ in range(num_devices):
                recv_ref = d2e_count_vmem.at[
                    pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=recv_ref, dst_ref=recv_ref,
                    sem=md_recv_sem,
                ).wait()

            for _ in range(num_devices):
                send_ref = d2e_count_vmem.at[
                    pl.ds(my_id, 1), pl.ds(0, 1), pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=send_ref, dst_ref=send_ref,
                    sem=md_send_sem,
                ).wait()

        elif mode == "butterfly":
            # Butterfly allreduce with barriers — works
            sync_barrier()

            if num_devices > 0 and (num_devices & (num_devices - 1)) == 0:
                rounds = int(math.log2(num_devices))
                for round_id in range(rounds):
                    sync_barrier()

                    chunk = 1 << round_id
                    chunk_i32 = jnp.int32(chunk)
                    peer_id = my_id ^ chunk_i32

                    send_start = (my_id >> round_id) << round_id
                    recv_start = (peer_id >> round_id) << round_id

                    pltpu.make_async_remote_copy(
                        src_ref=d2e_count_vmem.at[
                            pl.ds(send_start, chunk),
                            pl.ds(0, 1),
                            pl.ds(0, padded_num_experts),
                        ],
                        dst_ref=d2e_count_vmem.at[
                            pl.ds(send_start, chunk),
                            pl.ds(0, 1),
                            pl.ds(0, padded_num_experts),
                        ],
                        send_sem=md_send_sem,
                        recv_sem=md_recv_sem,
                        device_id=get_mesh_device_id(peer_id),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                    recv_ref = d2e_count_vmem.at[
                        pl.ds(recv_start, chunk),
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ]
                    pltpu.make_async_copy(
                        src_ref=recv_ref, dst_ref=recv_ref,
                        sem=md_recv_sem,
                    ).wait()

                    send_ref = d2e_count_vmem.at[
                        pl.ds(send_start, chunk),
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ]
                    pltpu.make_async_copy(
                        src_ref=send_ref, dst_ref=send_ref,
                        sem=md_send_sem,
                    ).wait()

            sync_barrier()

        elif mode == "broadcast_no_self":
            # Broadcast without self-send
            for step in range(1, num_devices):
                peer = (my_id + jnp.int32(step)) % jnp.int32(num_devices)
                pltpu.make_async_remote_copy(
                    src_ref=d2e_count_vmem.at[
                        my_id,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    dst_ref=d2e_count_vmem.at[
                        my_id,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    send_sem=md_send_sem,
                    recv_sem=md_recv_sem,
                    device_id=get_mesh_device_id(peer),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            for _ in range(num_devices - 1):
                recv_ref = d2e_count_vmem.at[
                    0, pl.ds(0, 1), pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=recv_ref, dst_ref=recv_ref,
                    sem=md_recv_sem,
                ).wait()

            for _ in range(num_devices - 1):
                send_ref = d2e_count_vmem.at[
                    my_id, pl.ds(0, 1), pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=send_ref, dst_ref=send_ref,
                    sem=md_send_sem,
                ).wait()

        elif mode == "shift":
            # Shift broadcast: 1 send + 1 recv wait + 1 send wait per step
            for step in range(1, num_devices):
                peer = (my_id + jnp.int32(step)) % jnp.int32(num_devices)
                pltpu.make_async_remote_copy(
                    src_ref=d2e_count_vmem.at[
                        my_id,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    dst_ref=d2e_count_vmem.at[
                        my_id,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ],
                    send_sem=md_send_sem,
                    recv_sem=md_recv_sem,
                    device_id=get_mesh_device_id(peer),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

                src_peer = (my_id + jnp.int32(num_devices - step)) % jnp.int32(num_devices)
                recv_ref = d2e_count_vmem.at[
                    src_peer,
                    pl.ds(0, 1),
                    pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=recv_ref, dst_ref=recv_ref,
                    sem=md_recv_sem,
                ).wait()

                send_ref = d2e_count_vmem.at[
                    my_id,
                    pl.ds(0, 1),
                    pl.ds(0, padded_num_experts),
                ]
                pltpu.make_async_copy(
                    src_ref=send_ref, dst_ref=send_ref,
                    sem=md_send_sem,
                ).wait()

        elif mode in ("scan", "scan_owner", "scan_owner128"):
            # Candidate for fused_moe_v2 metadata:
            #   1. recursive-doubling prefix scan over the full expert-count vector,
            #      producing this rank's starts plus global sizes;
            #   2. all-to-owner exchange of only each owner's local expert slice, so
            #      owner ranks can drive the gather return path.
            #
            # This preserves the metadata semantics used by the full kernel while
            # reducing payload from full allgather's 31 * padded_num_experts ints
            # to roughly (log2(P) + 1) * padded_num_experts ints per device.
            local_num_experts = padded_num_experts // num_devices
            local_owner_start = my_id * jnp.int32(local_num_experts)

            scan_work_vmem[...] = d2e_count_vmem[my_id]
            scan_prefix_vmem[...] = d2e_count_vmem[my_id]

            sync_barrier()
            if num_devices > 0 and (num_devices & (num_devices - 1)) == 0:
                rounds = int(math.log2(num_devices))
                for round_id in range(rounds):
                    sync_barrier()

                    chunk_i32 = jnp.int32(1 << round_id)
                    peer_id = my_id ^ chunk_i32

                    pltpu.make_async_remote_copy(
                        src_ref=scan_work_vmem,
                        dst_ref=scan_recv_vmem,
                        send_sem=md_send_sem,
                        recv_sem=md_recv_sem,
                        device_id=get_mesh_device_id(peer_id),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                    pltpu.make_async_copy(
                        src_ref=scan_recv_vmem,
                        dst_ref=scan_recv_vmem,
                        sem=md_recv_sem,
                    ).wait()
                    pltpu.make_async_copy(
                        src_ref=scan_work_vmem,
                        dst_ref=scan_work_vmem,
                        sem=md_send_sem,
                    ).wait()

                    @pl.when((my_id & chunk_i32) != 0)
                    def _():
                        scan_prefix_vmem[...] = scan_prefix_vmem[...] + scan_recv_vmem[...]

                    scan_work_vmem[...] = scan_work_vmem[...] + scan_recv_vmem[...]

            sync_barrier()
            scan_sizes_vmem[...] = scan_work_vmem[...]
            scan_prefix_vmem[...] = scan_prefix_vmem[...] - d2e_count_vmem[my_id]

            if mode != "scan":
                # Exchange local expert slices to their owner ranks. Natural
                # local_num_experts is 12 for EP=32,E=384, but Mosaic requires
                # VMEM slices along this axis to be 128-aligned. scan_owner is
                # the natural layout and is expected to fail that constraint;
                # scan_owner128 measures the viable padded-owner layout.
                owner_width = 128 if mode == "scan_owner128" else local_num_experts

                sync_barrier()
                for owner_id in range(num_devices):
                    owner_i32 = jnp.int32(owner_id)
                    owner_start = owner_id * local_num_experts
                    src_start = 0 if mode == "scan_owner128" else owner_start
                    dst_start = 0 if mode == "scan_owner128" else owner_start

                    @pl.when(owner_i32 != my_id)
                    def _():
                        pltpu.make_async_remote_copy(
                            src_ref=(
                                scan_work_vmem.at[
                                    pl.ds(0, 1),
                                    pl.ds(0, owner_width),
                                ]
                                if mode == "scan_owner128" else
                                d2e_count_vmem.at[
                                    my_id,
                                    pl.ds(0, 1),
                                    pl.ds(src_start, owner_width),
                                ]
                            ),
                            dst_ref=d2e_count_vmem.at[
                                my_id,
                                pl.ds(0, 1),
                                pl.ds(dst_start, owner_width),
                            ],
                            send_sem=md_send_sem,
                            recv_sem=md_recv_sem,
                            device_id=get_mesh_device_id(owner_i32),
                            device_id_type=pltpu.DeviceIdType.MESH,
                        ).start()

                for _ in range(num_devices - 1):
                    recv_ref = d2e_count_vmem.at[
                        0,
                        pl.ds(0, 1),
                        pl.ds(0 if mode == "scan_owner128" else local_owner_start, owner_width),
                    ]
                    pltpu.make_async_copy(
                        src_ref=recv_ref,
                        dst_ref=recv_ref,
                        sem=md_recv_sem,
                    ).wait()

                for _ in range(num_devices - 1):
                    send_ref = (
                        scan_work_vmem.at[
                            pl.ds(0, 1),
                            pl.ds(0, owner_width),
                        ]
                        if mode == "scan_owner128" else
                        d2e_count_vmem.at[
                            my_id,
                            pl.ds(0, 1),
                            pl.ds(0, owner_width),
                        ]
                    )
                    pltpu.make_async_copy(
                        src_ref=send_ref,
                        dst_ref=send_ref,
                        sem=md_send_sem,
                    ).wait()

                sync_barrier()

            # Store this rank's exclusive prefix vector as a correctness
            # signature. The owner-slice exchange above is still executed, but
            # this avoids scalar VMEM stores in the standalone repro.
            d2e_count_vmem[my_id] = scan_prefix_vmem[...]

        # Copy my_id's row to output (output is sharded, 1 row per device)
        store = pltpu.async_copy(
            src_ref=d2e_count_vmem.at[pl.ds(my_id, 1)],
            dst_ref=d2e_count_smem.at[pl.ds(0, 1)],
            sem=md_send_sem,
        )
        store.wait()

    return _kernel


def run(
    mesh: jax.sharding.Mesh,
    mode: str,
    dp_axis_name: str,
    tp_axis_name: str,
    iters: int = 100,
    warmup: int = 5,
):
    num_devices = mesh.size
    num_experts = 384
    padded_num_experts = ((num_experts + 127) // 128) * 128  # 384

    kernel_fn = make_kernel(
        mode=mode,
        num_devices=num_devices,
        padded_num_experts=padded_num_experts,
        dp_axis_name=dp_axis_name,
        tp_axis_name=tp_axis_name,
    )

    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    pallas_call = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct(
            (1, 1, padded_num_experts), jnp.int32
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm_spec],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[
                pltpu.VMEM((num_devices, 1, padded_num_experts), jnp.int32),
                pltpu.VMEM((1, padded_num_experts), jnp.int32),
                pltpu.VMEM((1, padded_num_experts), jnp.int32),
                pltpu.VMEM((1, padded_num_experts), jnp.int32),
                pltpu.VMEM((1, padded_num_experts), jnp.int32),
                pltpu.VMEM((1,), jnp.int32),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
        ),
        name=f"md_allreduce_{mode}",
    )

    from jax.sharding import NamedSharding, PartitionSpec as P

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P((dp_axis_name, tp_axis_name)),),
        out_specs=P((dp_axis_name, tp_axis_name)),
        check_vma=False,
    )
    def allreduce(d2e_input):
        return pallas_call(d2e_input)

    # Create test data: each device fills its row with device_id + 1
    total_data = jnp.zeros((num_devices, 1, padded_num_experts), dtype=jnp.int32)
    for d in range(num_devices):
        total_data = total_data.at[d, 0, :num_experts].set(d + 1)
    sharding = NamedSharding(mesh, P((dp_axis_name, tp_axis_name)))
    d2e_input = jax.device_put(total_data, sharding)

    print(f"Mode: {mode}, Mesh: {mesh.shape}, Devices: {num_devices}")
    print(f"Compiling...")
    t0 = time.time()
    result = allreduce(d2e_input)
    result.block_until_ready()
    print(f"Compile + first run: {time.time() - t0:.2f}s")

    # Verify correctness on local shards only
    try:
        local_shards = result.addressable_shards
        print(f"  Got {len(local_shards)} local shards, first shard shape: {local_shards[0].data.shape}")
        local_data = local_shards[0].data
        # Each shard should be (1, 1, padded) — the device's own row after allreduce
        if local_data.shape == (1, 1, padded_num_experts):
            if mode in ("scan", "scan_owner", "scan_owner128"):
                sig = [int(local_data[0, 0, i]) for i in range(3)]
                print(f"  First shard signature[0:3]: {sig}")
            else:
                val = int(local_data[0, 0, 0])
                print(f"  First shard value[0]: {val} (expected: device_id+1)")
        else:
            print(f"  Unexpected shard shape: {local_data.shape}")
    except Exception as e:
        print(f"  Verification skipped: {e}")

    print(f"Warmup {warmup} iters...")
    for _ in range(warmup):
        result = allreduce(d2e_input)
        result.block_until_ready()

    print(f"Running {iters} iters...")
    times = []
    for i in range(iters):
        t0 = time.time()
        result = allreduce(d2e_input)
        result.block_until_ready()
        elapsed = time.time() - t0
        times.append(elapsed * 1000)
        if (i + 1) % 20 == 0:
            print(f"  iter {i+1}/{iters}: {elapsed*1000:.3f} ms")

    times.sort()
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    mean = sum(times) / len(times)
    print(f"Results: mean={mean:.3f}ms p50={p50:.3f}ms p90={p90:.3f}ms")
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="broadcast",
                        choices=["broadcast", "broadcast_barrier",
                                 "broadcast_ds", "butterfly",
                                 "broadcast_no_self", "shift", "scan",
                                 "scan_owner", "scan_owner128"],
                        help="Allreduce mode to test")
    parser.add_argument("--mesh", type=str, default=None,
                        help="Mesh shape: '2d' for (8,4), '1d' for (1,32), "
                             "or auto-detect from device count")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--all", action="store_true",
                        help="Run all modes")
    args = parser.parse_args()

    if os.environ.get("BENCH_SKIP_DIST_INIT", "0") != "1":
        jax.distributed.initialize()

    import numpy as np
    num_devices = jax.device_count()
    print(f"JAX devices: {num_devices}")

    if args.mesh == "1d":
        devices = np.array(jax.devices()).reshape(1, num_devices)
        dp_axis_name, tp_axis_name = "data", "tensor"
    elif args.mesh == "2d":
        dp_size = num_devices // 4
        tp_size = 4
        devices = np.array(jax.devices()).reshape(dp_size, tp_size)
        dp_axis_name, tp_axis_name = "data", "tensor"
    else:
        # Auto-detect: use 2D if num_devices > 4, else 1D
        if num_devices > 4:
            tp_size = min(4, num_devices)
            dp_size = num_devices // tp_size
            devices = np.array(jax.devices()).reshape(dp_size, tp_size)
        else:
            devices = np.array(jax.devices()).reshape(1, num_devices)
        dp_axis_name, tp_axis_name = "data", "tensor"

    mesh = jax.sharding.Mesh(devices, axis_names=(dp_axis_name, tp_axis_name))
    print(f"Mesh shape: {mesh.shape}")

    if args.all:
        modes = ["butterfly", "scan", "scan_owner128", "scan_owner",
                 "broadcast_barrier", "broadcast", "broadcast_ds",
                 "broadcast_no_self", "shift"]
    else:
        modes = [args.mode]

    for mode in modes:
        print(f"\n{'='*60}")
        try:
            run(mesh, mode, dp_axis_name, tp_axis_name,
                iters=args.iters, warmup=args.warmup)
        except Exception as e:
            print(f"Mode {mode} FAILED: {e}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
