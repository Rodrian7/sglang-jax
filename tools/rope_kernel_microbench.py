"""Micro-benchmark the RPA v3 kernel with apply_qrope True vs False on a single
host, to isolate the in-kernel q-rope integration cost (vs external rope) without
a full 4-pod e2e cycle. Prefill-shaped (large bq) and decode-shaped (q=1, long kv).
Run on bench-4: PYTHONPATH=/tmp/sglang-rope/python python /tmp/rope_kernel_microbench.py
"""

import logging
import time

logging.disable(logging.WARNING)
import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention,
)

print("jax", jax.__version__, "| dev", jax.devices()[0].device_kind)

NQ, NKV, D, RD, PAGE = 16, 1, 192, 192, 256  # q heads, kv heads, head_dim, rotary_dim, page
THETA, NEOX = 1e6, True
KVP = 2  # bf16 kv packing


def make_inputs(q_len, ctx_len):
    """One sequence: q_len new query tokens, total kv context = ctx_len."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (q_len, NQ, D), jnp.bfloat16)
    k = jax.random.normal(jax.random.split(key)[0], (q_len, NKV, D), jnp.bfloat16)
    v = jax.random.normal(jax.random.split(key)[1], (q_len, NKV, D), jnp.bfloat16)
    npages = (ctx_len + PAGE - 1) // PAGE
    cache = jnp.zeros((npages, PAGE, (NKV * 2) // KVP, KVP, D), jnp.bfloat16)
    kv_lens = jnp.array([ctx_len], jnp.int32)
    page_indices = jnp.arange(npages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, q_len], jnp.int32)
    cu_kv_lens = jnp.array([0, ctx_len], jnp.int32)
    # decode_end, prefill_end, mixed_end -> 1 prefill seq
    distribution = jnp.array([0, 1, 1], jnp.int32)
    return q, k, v, cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution


def make_decode_batch(nseq, ctx_len):
    """nseq decode sequences, each 1 new query token, kv context = ctx_len each."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (nseq, NQ, D), jnp.bfloat16)
    k = jax.random.normal(jax.random.split(key)[0], (nseq, NKV, D), jnp.bfloat16)
    v = jax.random.normal(jax.random.split(key)[1], (nseq, NKV, D), jnp.bfloat16)
    pages_per_seq = (ctx_len + PAGE - 1) // PAGE
    npages = pages_per_seq * nseq
    cache = jnp.zeros((npages, PAGE, (NKV * 2) // KVP, KVP, D), jnp.bfloat16)
    kv_lens = jnp.array([ctx_len] * nseq, jnp.int32)
    page_indices = jnp.arange(npages, dtype=jnp.int32)
    cu_q_lens = jnp.arange(nseq + 1, dtype=jnp.int32)  # 1 q token per seq
    cu_kv_lens = jnp.arange(0, (nseq + 1) * ctx_len, ctx_len, dtype=jnp.int32)
    distribution = jnp.array([nseq, nseq, nseq], jnp.int32)  # all decode
    return q, k, v, cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution


def bench_inputs(inp, q_len, apply_qrope, iters=50):
    q, k, v, cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution = inp

    @jax.jit
    def run(q, k, v, cache):
        out, _ = ragged_paged_attention(
            q,
            k,
            v,
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            None,
            causal=1,
            sm_scale=float(1.0 / np.sqrt(D)),
            chunk_prefill_size=max(q_len, 1),
            apply_qrope=apply_qrope,
            rope_theta=THETA,
            rotary_dim=RD,
            is_neox=NEOX,
        )
        return out

    o = run(q, k, v, cache)
    o.block_until_ready()  # warmup / compile
    t0 = time.perf_counter()
    for _ in range(iters):
        o = run(q, k, v, cache)
    o.block_until_ready()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/call


def bench(q_len, ctx_len, apply_qrope, iters=50):
    return bench_inputs(make_inputs(q_len, ctx_len), q_len, apply_qrope, iters)


for name, q_len, ctx_len in [
    ("prefill q=2048 kv=2048", 2048, 2048),
    ("prefill q=2048 kv=8192", 2048, 8192),
    ("decode   q=1   kv=16384", 1, 16384),
]:
    t_off = bench(q_len, ctx_len, False)
    t_on = bench(q_len, ctx_len, True)
    delta = (t_on - t_off) / t_off * 100
    print(
        f"{name:28s}  off={t_off:8.4f}ms  on={t_on:8.4f}ms  "
        f"delta={t_on - t_off:+.4f}ms ({delta:+.1f}%)"
    )

# Realistic decode: a batch of nseq sequences each emitting 1 token (continuous batching).
for nseq, ctx_len in [(64, 4096), (64, 16384), (256, 4096)]:
    inp = make_decode_batch(nseq, ctx_len)
    t_off = bench_inputs(inp, 1, False)
    t_on = bench_inputs(inp, 1, True)
    delta = (t_on - t_off) / t_off * 100
    print(
        f"decode batch nseq={nseq:3d} kv={ctx_len:5d}  off={t_off:8.4f}ms  on={t_on:8.4f}ms  "
        f"delta={t_on - t_off:+.4f}ms ({delta:+.1f}%)"
    )
