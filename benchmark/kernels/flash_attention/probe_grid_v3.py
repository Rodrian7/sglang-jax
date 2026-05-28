"""Minimum probe to answer Q1-Q4 before deciding RPA v3 schema.

Q1: Does optimal block config diverge across kv distributions?
Q2: Does csz<sz help any shape (real signal vs noise)?
Q3: What is measurement variance / noise floor?
Q4: Are conclusions consistent across CI shapes?

Grid: 2 shapes × 1 mnt × 3 kv distributions × 6 (sz, csz) configs × 3 tries
    = 108 measurements
Wall: ~30 min on a v6e single VM

Measurement uses the same task=<pallas_call scope name> trick the CI bench uses,
so what we report is kernel-only device time (apples-to-apples with CI).
Each config also captures a jax.profiler.trace under
$ARTIFACT_LOCAL_DIR/profiles/<tag>/ so we can drill into HLO if needed.
"""

import functools
import os
import statistics

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu
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

# Two CI shapes covering the failing case + a different (q, kv, ps) combo
# to test consistency.
SHAPES = [
    dict(
        tag="A_q4kv1_ps128_mnt128",
        q_head_num=4,
        kv_head_num=1,
        head_dim=128,
        page_size=128,
        batch_size=128,
    ),
    dict(
        tag="B_q8kv4_ps256_mnt128",
        q_head_num=8,
        kv_head_num=4,
        head_dim=128,
        page_size=256,
        batch_size=128,
    ),
]

KV_RANGES = [
    (256, 512),  # short
    (1024, 2048),  # mid (CI default)
    (16384, 32768),  # long
]

# (sz, csz, label). 'heuristic' marker means use heuristic block sizes for the shape.
CONFIGS = [
    ("heuristic", None, None),
    ("sz1024_csz1024", 1024, 1024),
    ("sz2048_csz2048", 2048, 2048),
    ("sz32768_csz32768", 32768, 32768),
    ("sz2048_csz512", 2048, 512),  # validate short-kv csz<sz signal
    ("sz32768_csz4096", 32768, 4096),  # csz=sz/8 ratio for long-kv
]

MAX_CONTEXT_LEN = 40960
MAX_KV_CACHE_TOKENS = 600000


def make_inputs(shape: dict, prefix_lo: int, prefix_hi: int):
    return make_decode_inputs(
        prefix_lo,
        prefix_hi,
        max_context_len=MAX_CONTEXT_LEN,
        max_kv_cache_tokens=MAX_KV_CACHE_TOKENS,
        batch_size=shape["batch_size"],
        q_head_num=shape["q_head_num"],
        kv_head_num=shape["kv_head_num"],
        head_dim=shape["head_dim"],
        page_size=shape["page_size"],
    )


def heuristic_block_sizes(shape: dict):
    pages_per_seq = cdiv(MAX_CONTEXT_LEN, shape["page_size"])
    h = get_default_block_sizes(
        jnp.bfloat16,
        jnp.bfloat16,
        shape["q_head_num"],
        shape["kv_head_num"],
        shape["head_dim"],
        shape["page_size"],
        shape["batch_size"],
        shape["batch_size"],
        pages_per_seq,
        case=RpaCase.DECODE,
        vmem_limit_bytes=get_vmem_limit(),
    )
    return h["bkv_sz"], h["bkv_csz"]


def run_cell(shape: dict, kv_range, sz: int, csz: int, *, profile_root: str, tries: int):
    inputs = make_inputs(shape, *kv_range)
    q, k, v, kvc, kvl, pi, cql, ckl, dist = inputs
    block_sizes = (1, sz, 1, csz)

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
        shape["head_dim"] ** -0.5,
        block_sizes,
        get_vmem_limit(),
    )

    # Warmup outside the profile.
    jax.block_until_ready(bound())

    # Match CI bench: task name == the pallas_call scope name so the regex
    # in multiple_iteration_timeit_from_trace returns kernel-only device time.
    task = f"RPAd-p_{shape['page_size']}-bq_1_1-bkv_{sz}_{csz}"

    # multiple_iteration_timeit_from_trace already manages its own
    # jax.profiler.trace context. Don't wrap it again here.
    cell_trace_root = os.path.join(
        profile_root,
        shape["tag"],
        f"kv{kv_range[0]}-{kv_range[1]}",
        f"sz{sz}_csz{csz}",
    )
    os.makedirs(cell_trace_root, exist_ok=True)

    samples = []
    for t in range(tries):
        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: bound(),
            data_generator=lambda: (),
            task=task,
            tries=1,
            trace_root=cell_trace_root,
        )
        if times:
            samples.append(float(np.mean(times)))
        else:
            samples.append(float("nan"))
    return samples, cell_trace_root


def main():
    profile_root = os.environ.get("ARTIFACT_LOCAL_DIR", "/tmp/operator-artifact") + "/profiles"
    print(f"# Device: {get_device_name()}")
    print(f"# Profile root: {profile_root}")
    info = pltpu.get_tpu_info()
    print(f"# VMEM capacity: {info.vmem_capacity_bytes / 1024 / 1024:.1f} MB")
    print(f"# get_vmem_limit() = {get_vmem_limit() / 1024 / 1024:.1f} MB")
    print()

    # rows[(shape_tag, kv_range, config_label)] = (sz, csz, samples_ms, mean_ms, std_ms)
    rows = []
    for shape in SHAPES:
        h_sz, h_csz = heuristic_block_sizes(shape)
        print(f"### shape {shape['tag']}: heuristic decode block_sizes = (sz={h_sz}, csz={h_csz})")
        for kv_range in KV_RANGES:
            print(f"  --- kv range [{kv_range[0]}, {kv_range[1]}) ---")
            for label, raw_sz, raw_csz in CONFIGS:
                if label == "heuristic":
                    sz, csz = h_sz, h_csz
                else:
                    sz, csz = raw_sz, raw_csz
                if sz % csz != 0:
                    print(f"    {label:>20s} (sz={sz}, csz={csz}): SKIP (sz % csz != 0)")
                    continue
                try:
                    samples, _ = run_cell(
                        shape, kv_range, sz, csz, profile_root=profile_root, tries=3
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"    {label:>20s} (sz={sz}, csz={csz}): SKIP {type(e).__name__}: {e}")
                    continue
                mean = statistics.mean(samples)
                std = statistics.stdev(samples) if len(samples) > 1 else 0.0
                rel_std = (std / mean * 100.0) if mean else 0.0
                print(
                    f"    {label:>20s} (sz={sz:>5d}, csz={csz:>5d}): "
                    f"{mean:.4f}ms ± {std:.4f}ms ({rel_std:.1f}%)  "
                    f"raw={[f'{s:.4f}' for s in samples]}"
                )
                rows.append((shape["tag"], kv_range, label, sz, csz, samples, mean, std, rel_std))
            print()

    # ---- per-(shape, kv_range) winner table ----
    print("=" * 80)
    print("WINNERS PER (shape, kv_range)")
    print("=" * 80)
    seen = set()
    for shape_tag, kv_range, label, sz, csz, samples, mean, std, rel_std in rows:
        key = (shape_tag, kv_range)
        if key in seen:
            continue
        # find min mean for this key
        candidates = [r for r in rows if (r[0], r[1]) == key]
        winner = min(candidates, key=lambda r: r[6])
        wlabel, wsz, wcsz, wmean, wstd = winner[2], winner[3], winner[4], winner[6], winner[7]
        # heuristic baseline for this key
        heur = next((r for r in candidates if r[2] == "heuristic"), None)
        if heur is not None:
            speedup = heur[6] / wmean
            heur_str = f"heur={heur[6]:.4f}ms"
        else:
            speedup = float("nan")
            heur_str = "heur=?"
        print(
            f"  {shape_tag}  kv[{kv_range[0]},{kv_range[1]}):  "
            f"winner={wlabel} (sz={wsz}, csz={wcsz})  "
            f"{wmean:.4f}ms ± {wstd:.4f}ms  vs  {heur_str}  speedup={speedup:.2f}x"
        )
        seen.add(key)

    # ---- universal candidate ----
    print()
    print("=" * 80)
    print("UNIVERSAL CANDIDATE CHECK")
    print("=" * 80)
    print("For each (sz, csz) config, compute worst-case slowdown across all (shape, kv) cells:")
    by_config = {}
    for shape_tag, kv_range, label, sz, csz, samples, mean, std, rel_std in rows:
        if label == "heuristic":
            continue
        cell_key = (shape_tag, kv_range)
        cell_best = min((r[6] for r in rows if (r[0], r[1]) == cell_key))
        ratio = mean / cell_best
        by_config.setdefault((label, sz, csz), []).append((cell_key, ratio))
    for (label, sz, csz), cell_ratios in sorted(by_config.items()):
        if len(cell_ratios) < len(SHAPES) * len(KV_RANGES):
            continue  # didn't run on all cells
        worst = max(r for _, r in cell_ratios)
        flag = "  ← UNIVERSAL CANDIDATE" if worst < 1.10 else ""
        cell_str = " ".join(f"{ck[0][0]}{ck[1][0]}:{r:.2f}x" for ck, r in cell_ratios)
        print(
            f"  {label:>20s} (sz={sz:>5d}, csz={csz:>5d})  worst={worst:.2f}x  ({cell_str}){flag}"
        )


if __name__ == "__main__":
    main()
