"""Standalone test for sc_scatter_kernel — runs on 4-pod EP=32 v7x.

Verifies the SC scatter kernel:
1. Compiles successfully (Mosaic IR generation works)
2. Produces output (DMAs complete without hang)
3. (loose) data movement is plausible

Run via:
  python python/sgl_jax/srt/kernels/fused_moe/v2/sc_prefetch/test_sc_scatter.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.experimental import multihost_utils

from sgl_jax.srt.kernels.fused_moe.v2.sc_prefetch.sc_scatter_kernel import (
    sc_bt0_scatter,
)


def main():
    jax.distributed.initialize()

    num_devices = jax.device_count()
    process_id = jax.process_index()
    if process_id == 0:
        print(f"Total devices: {num_devices}, processes: {jax.process_count()}")

    # Mesh matching fused_moe_v2 EP=32 setup
    if num_devices == 32:
        devices = np.array(jax.devices()).reshape(8, 4)
        mesh = Mesh(devices, axis_names=("data", "tensor"))
    else:
        # smaller mesh for debugging
        devices = np.array(jax.devices()).reshape(num_devices, 1)
        mesh = Mesh(devices, axis_names=("data", "tensor"))

    # Workload params (MiMo V2 Pro)
    local_num_tokens = 512
    top_k = 8
    num_experts = 384
    local_num_experts = num_experts // num_devices  # 12
    hidden_size = 6144
    t_packing = 2  # bf16
    h_per_t = hidden_size // t_packing  # 3072
    bt = 256
    bt_start = 0
    a2a_max_tokens = 8208  # matches fused_moe_v2's buffer sizing

    if process_id == 0:
        print(f"Local tokens: {local_num_tokens}, bt: {bt}, local_experts: {local_num_experts}")

    # Synthetic inputs
    rng = jax.random.PRNGKey(0)
    rng_tokens, rng_topk = jax.random.split(rng)

    tokens_full_shape = (num_devices * local_num_tokens, t_packing, h_per_t)
    tokens_local = jax.random.uniform(
        rng_tokens,
        (local_num_tokens, t_packing, h_per_t),
        dtype=jnp.bfloat16,
    )
    # Distribute tokens across data axis
    tokens_sh = jax.device_put(
        tokens_local,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "tensor"))),
    )

    # Top-k IDs (random valid experts)
    topk_ids_local = jax.random.randint(
        rng_topk,
        (local_num_tokens, top_k),
        minval=0,
        maxval=num_experts,
        dtype=jnp.int32,
    )
    topk_ids_sh = jax.device_put(
        topk_ids_local,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "tensor"))),
    )

    # Expert starts (precomputed — for test, use zeros + simple cumsum)
    padded_num_experts = ((num_experts + 127) // 128) * 128  # 384 already aligned to 128
    expert_starts = jnp.zeros((padded_num_experts,), dtype=jnp.int32)
    expert_starts_sh = jax.device_put(
        expert_starts,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )

    if process_id == 0:
        print("Launching SC scatter kernel...")

    try:
        out = sc_bt0_scatter(
            tokens_sh,
            topk_ids_sh,
            expert_starts_sh,
            bt=bt,
            bt_start=bt_start,
            top_k=top_k,
            local_num_experts=local_num_experts,
            num_devices=num_devices,
            a2a_max_tokens=a2a_max_tokens,
            dp_axis_name="data",
            tp_axis_name="tensor",
            mesh=mesh,
        )
        out.block_until_ready()
        if process_id == 0:
            print(f"PASS: SC scatter kernel compiled and ran")
            print(f"  output shape: {out.shape}")
            print(f"  output dtype: {out.dtype}")
            # Check that something was written (output is non-zero somewhere)
            global_out = multihost_utils.process_allgather(out, tiled=True)
            host_out = np.array(global_out)
            nonzero_count = np.count_nonzero(host_out)
            print(f"  nonzero elements: {nonzero_count} / {host_out.size}")
    except Exception as e:
        if process_id == 0:
            print(f"FAIL: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
