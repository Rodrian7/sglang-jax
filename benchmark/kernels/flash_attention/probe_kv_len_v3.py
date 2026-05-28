"""One-off probe: does the optimal RPA v3 decode block-config depend on
actual kv length, or is mnt+shape sufficient as the table key?

For ONE shape (the v6e CI failure: q=4 kv=1 hd=128 ps=128 mnt=128) sweep
bkv candidates across N prefix-length ranges. If the winning bkv clusters
across ranges, the existing key (no kv_len bucket) is fine; if it diverges,
TUNED_BLOCK_SIZES_V3 needs a kv_len_bucket dimension.

Self-contained — does not touch get_block_spec_config_v3.py / utils.py.
Delete this file once the bucket-or-not decision lands.

Usage:
    python probe_kv_len_v3.py
    # narrower / fuller sweep:
    python probe_kv_len_v3.py --prefix-ranges 256:512,1024:2048,16384:32768
"""

import argparse
import functools
from math import inf

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    RpaCase,
    get_default_block_sizes,
    get_vmem_limit,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import get_dtype_packing
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import get_device_name


def make_decode_inputs(
    prefix_lo: int,
    prefix_hi: int,
    *,
    max_context_len: int,
    max_kv_cache_tokens: int,
    batch_size: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    page_size: int,
    dtype=jnp.bfloat16,
    seed: int = 42,
):
    """Mirror utils.create_decode_uniform_data with parameterized prefix range."""
    if prefix_hi <= prefix_lo or prefix_hi > max_context_len:
        raise ValueError(f"need 0 < {prefix_lo=} < {prefix_hi=} <= {max_context_len=}")

    key = jax.random.PRNGKey(seed)
    prefix_lens = jax.random.randint(key, (batch_size,), prefix_lo, prefix_hi)
    seq_lens = prefix_lens + 1  # +1 = the decode token itself

    cu_q_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(jnp.ones(batch_size, dtype=jnp.int32)),
        ]
    )
    cu_kv_lens = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lens)])

    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(keys[0], (batch_size, q_head_num, head_dim), dtype=dtype)
    k = jax.random.normal(keys[1], (batch_size, kv_head_num, head_dim), dtype=dtype)
    v = jax.random.normal(keys[2], (batch_size, kv_head_num, head_dim), dtype=dtype)

    packing = get_dtype_packing(dtype)
    total_pages = cdiv(max_kv_cache_tokens, page_size)
    kv_cache = jax.random.normal(
        keys[1],
        (total_pages, page_size, kv_head_num * 2 // packing, packing, head_dim),
        dtype=dtype,
    )

    # Page indices: reuse the same layout as utils.create_page_indices_data
    cache_loc = jnp.arange(0, int(seq_lens.sum().item()), dtype=jnp.int32)
    cache_start = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lens)])
    pieces = []
    for i in range(batch_size):
        s, e = cache_start[i], cache_start[i] + seq_lens[i]
        pieces.append(
            jnp.pad(cache_loc[s:e], (0, max_context_len - seq_lens[i]), constant_values=0)
        )
    page_indices = jnp.concatenate(pieces)[0::page_size] // page_size

    distribution = jnp.array([batch_size, batch_size, batch_size], dtype=jnp.int32)
    return (
        q,
        k,
        v,
        kv_cache,
        seq_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
    )


def benchmark_one(
    bkv_sz: int,
    *,
    inputs,
    head_dim: int,
):
    q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution = inputs
    block_sizes = (1, bkv_sz, 1, bkv_sz)  # decode: bq=1, csz=sz

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
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        head_dim**-0.5,
        block_sizes,
        get_vmem_limit(),
    )
    jax.block_until_ready(bound())  # warmup
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: bound(),
        data_generator=lambda: (),
        task=f"probe-bkv{bkv_sz}",
        tries=1,
    )
    return float(np.mean(times)) if times else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prefix-ranges",
        default="256:512,1024:2048,16384:32768",
        help="comma-list of LO:HI ranges; each is one sweep",
    )
    p.add_argument("--bkv-candidates", default="256,512,1024,2048,4096,8192,16384,32768")
    p.add_argument("--max-context-len", type=int, default=40960)
    p.add_argument("--max-kv-cache-tokens", type=int, default=600000)
    p.add_argument("--batch-size", type=int, default=128)  # mnt for decode
    p.add_argument("--q-heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=128)
    args = p.parse_args()

    ranges = []
    for tok in args.prefix_ranges.split(","):
        lo, hi = tok.split(":")
        ranges.append((int(lo), int(hi)))
    bkv_list = [int(x) for x in args.bkv_candidates.split(",")]

    # Heuristic baseline (max_context_len-aware) for context.
    pages_per_seq = cdiv(args.max_context_len, args.page_size)
    heur = get_default_block_sizes(
        jnp.bfloat16,
        jnp.bfloat16,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.page_size,
        args.batch_size,
        args.batch_size,
        pages_per_seq,
        case=RpaCase.DECODE,
        vmem_limit_bytes=get_vmem_limit(),
    )
    print(f"# Device: {get_device_name()}")
    print(
        f"# Shape: q={args.q_heads} kv={args.kv_heads} hd={args.head_dim} "
        f"ps={args.page_size} mnt={args.batch_size} max_ctx={args.max_context_len}"
    )
    print(f"# Heuristic decode bkv_sz: {heur['bkv_sz']}")
    print()

    # Per range, sweep bkv. Print table.
    all_winners: list[tuple[tuple[int, int], int | None, float]] = []
    for lo, hi in ranges:
        inputs = make_decode_inputs(
            lo,
            hi,
            max_context_len=args.max_context_len,
            max_kv_cache_tokens=args.max_kv_cache_tokens,
            batch_size=args.batch_size,
            q_head_num=args.q_heads,
            kv_head_num=args.kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
        )
        actual_kv = (lo + hi) // 2
        print(f"=== prefix [{lo},{hi}) — actual_kv ≈ {actual_kv} ===")
        best_bkv, best_t = None, inf
        for bkv in bkv_list:
            try:
                t = benchmark_one(bkv, inputs=inputs, head_dim=args.head_dim)
            except Exception as e:  # noqa: BLE001
                print(f"  bkv={bkv:>5}  SKIP ({type(e).__name__}: {e})")
                continue
            mark = ""
            if t < best_t:
                best_t, best_bkv = t, bkv
                mark = "  ← winner so far"
            print(f"  bkv={bkv:>5}  {t*1000:.4f}ms{mark}")
        print(f"  WINNER: bkv={best_bkv} @ {best_t*1000:.4f}ms")
        print()
        all_winners.append(((lo, hi), best_bkv, best_t))

    # Verdict.
    print("=== verdict ===")
    winners = {bkv for _, bkv, _ in all_winners}
    if len(winners) == 1:
        print(f"All ranges agree on bkv={winners.pop()} → kv_len bucketing NOT needed.")
    else:
        print(f"Winners diverge across ranges: {all_winners}")
        print("→ TUNED_BLOCK_SIZES_V3 likely needs a kv_len_bucket key dimension.")


if __name__ == "__main__":
    main()
