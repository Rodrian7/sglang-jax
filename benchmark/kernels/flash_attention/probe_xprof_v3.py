"""T1.5 xprof probe — capture per-config traces of RPA v3 decode for analysis.

Goals:
  1. Print pltpu.get_tpu_info().vmem_capacity_bytes  (resolve T0.3 v6e VMEM)
  2. A/B vmem_limit_bytes path:
       (a) heuristic block sizes WITH vmem_limit_bytes=get_vmem_limit()  (probe path)
       (b) heuristic block sizes WITH vmem_limit_bytes=None              (bench path)
  3. A/B winner block sizes:
       (c) (sz=2048, csz=512) with vmem_limit_bytes=get_vmem_limit()
       (d) (sz=2048, csz=512) with vmem_limit_bytes=None
  4. Capture xprof trace per config under $ARTIFACT_LOCAL_DIR/<tag>/tensorboard/...

After the exp succeeds, query each trace with the xprof MCP tools to read MXU
active%, HBM bytes, sem wait, and pl.when overhead.

Single mid-kv distribution [1024, 2048) (matching CI bench's data-gen).
Single shape: q=4 kv=1 hd=128 ps=128 mnt=128 max_ctx=40960.
"""

import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

# Reuse the probe's data-gen so both files stay in sync.
from probe_kv_len_v3 import make_decode_inputs

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    RpaCase,
    get_default_block_sizes,
    get_vmem_limit,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import get_device_name

SHAPE = dict(
    max_context_len=40960,
    max_kv_cache_tokens=600000,
    batch_size=128,
    q_head_num=4,
    kv_head_num=1,
    head_dim=128,
    page_size=128,
)
PREFIX_LO = 1024
PREFIX_HI = 2048


def run_one(tag: str, sz: int, csz: int, pass_vmem_limit: bool, *, profile_root: str):
    """Run one config under jax.profiler.trace, return mean time (ms)."""
    inputs = make_decode_inputs(PREFIX_LO, PREFIX_HI, **SHAPE)
    q, k, v, kvc, kvl, pi, cql, ckl, dist = inputs
    block_sizes = (1, sz, 1, csz)
    vmem_arg = get_vmem_limit() if pass_vmem_limit else None

    @functools.partial(jax.jit, static_argnames=["sm_scale", "d_block_sizes", "vmem_limit_bytes"])
    def attn(q, k, v, kvc, kvl, pi, cql, ckl, dist, sm_scale, d_block_sizes, vmem_limit_bytes):
        return ragged_paged_attention(
            q,
            k,
            v,
            kvc,
            kvl,
            pi,
            cql,
            ckl,
            dist,
            custom_mask=None,
            causal=1,
            sm_scale=sm_scale,
            d_block_sizes=d_block_sizes,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    bound = functools.partial(
        attn,
        q,
        k,
        v,
        kvc,
        kvl,
        pi,
        cql,
        ckl,
        dist,
        SHAPE["head_dim"] ** -0.5,
        block_sizes,
        vmem_arg,
    )

    # Warmup outside the profile.
    jax.block_until_ready(bound())

    # Profile run — multiple iterations for trace + timing in one pass.
    trace_dir = os.path.join(profile_root, tag, "tensorboard", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(trace_dir, exist_ok=True)
    with jax.profiler.trace(trace_dir):
        for i in range(5):
            out = bound()
            jax.block_until_ready(out)

    # Standalone timing pass (separate from the profiled iters, since profiler adds overhead).
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: bound(),
        data_generator=lambda: (),
        task=f"xprof-{tag}",
        tries=1,
    )
    mean_ms = float(np.mean(times)) if times else float("nan")
    return mean_ms, trace_dir


def main():
    profile_root = os.environ.get("ARTIFACT_LOCAL_DIR", "/tmp/operator-artifact") + "/profiles"
    print(f"# Device: {get_device_name()}")
    print(f"# Profile root: {profile_root}")

    # T0.3 resolution: dump exact vmem_capacity_bytes.
    info = pltpu.get_tpu_info()
    print(f"# pltpu.get_tpu_info().vmem_capacity_bytes = {info.vmem_capacity_bytes:,} bytes")
    print(f"#   = {info.vmem_capacity_bytes / 1024 / 1024:.1f} MB")
    print(f"# get_vmem_limit() = {get_vmem_limit():,} bytes")
    print()

    # Heuristic block sizes for context.
    pages_per_seq = cdiv(SHAPE["max_context_len"], SHAPE["page_size"])
    heur = get_default_block_sizes(
        jnp.bfloat16,
        jnp.bfloat16,
        SHAPE["q_head_num"],
        SHAPE["kv_head_num"],
        SHAPE["head_dim"],
        SHAPE["page_size"],
        SHAPE["batch_size"],
        SHAPE["batch_size"],
        pages_per_seq,
        case=RpaCase.DECODE,
        vmem_limit_bytes=get_vmem_limit(),
    )
    heur_sz, heur_csz = heur["bkv_sz"], heur["bkv_csz"]
    print(f"# Heuristic decode block sizes: bkv_sz={heur_sz}, bkv_csz={heur_csz}")
    print()

    # 4 configs.
    configs = [
        ("heur_with_vmem", heur_sz, heur_csz, True),  # probe path
        ("heur_no_vmem", heur_sz, heur_csz, False),  # bench path (vmem_limit_bytes=None)
        ("winner_with_vmem", 2048, 512, True),
        ("winner_no_vmem", 2048, 512, False),
    ]

    print(f"# Shape: {SHAPE}")
    print(
        f"# Prefix range: [{PREFIX_LO}, {PREFIX_HI}) — actual_kv ≈ {(PREFIX_LO + PREFIX_HI) // 2}"
    )
    print()

    results = []
    for tag, sz, csz, pass_vmem in configs:
        print(f"### config: {tag} (sz={sz}, csz={csz}, pass_vmem_limit={pass_vmem})")
        try:
            t_ms, trace_dir = run_one(tag, sz, csz, pass_vmem, profile_root=profile_root)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append((tag, None, None))
            continue
        print(f"  mean: {t_ms:.4f} ms")
        print(f"  trace: {trace_dir}")
        results.append((tag, t_ms, trace_dir))
        print()

    print("=== summary ===")
    for tag, t, _ in results:
        print(f"  {tag:>20}: {f'{t:.4f}ms' if t else 'FAILED'}")


if __name__ == "__main__":
    main()
