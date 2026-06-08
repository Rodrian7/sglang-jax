"""Pallas-kernel cost via pure-JAX reference implementations.

Per goal 1: instead of hand-written FLOP formulas, derive each Pallas kernel's
FLOPs/transcendentals from a pure-JAX *reference* of its math via
``jax.experimental.pallas.estimate_cost`` (the pattern used by
``multimodal/kernels/flash_attention.py::_fwd_cost_estimate``). Bytes are taken
from the kernel's real I/O (not the naive reference, which would over-count
intermediates like the [S,S] attention scores). Kernels stay atomic -- we do
NOT model intra-kernel overlap.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .quant import BF16, QuantSpec


def _bytes(shape, dtype) -> int:
    return math.prod(shape) * jnp.dtype(dtype).itemsize


def _ref_flops(ref_fn, *abstract_args) -> tuple[int, int]:
    """(flops, transcendentals) of a pure-JAX reference via estimate_cost."""
    c = pl.estimate_cost(ref_fn, *abstract_args)
    return int(c.flops), int(c.transcendentals)


# --------------------------------------------------------------------------
# Reference math (pure JAX) -- only used to COUNT flops, never executed for real
# --------------------------------------------------------------------------
def attention_ref(q, k, v):
    # q:[T,Hq,D]  k:[S,Hkv,D]  v:[S,Hkv,Dv]  (Hkv broadcast to Hq for GQA)
    d = q.shape[-1]
    scores = jnp.einsum("thd,shd->hts", q, k) * (1.0 / math.sqrt(d))
    p = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hts,shd->thd", p, v)


def moe_ffn_ref(x, w_gate, w_up, w_down):
    # x:[T,d]  w_gate/w_up:[d,f]  w_down:[f,d]  -> gate/up GEMM, SiLU, down GEMM
    a = jnp.dot(x, w_gate, preferred_element_type=jnp.float32)
    b = jnp.dot(x, w_up, preferred_element_type=jnp.float32)
    h = (jax.nn.silu(a) * b).astype(x.dtype)
    return jnp.dot(h, w_down, preferred_element_type=jnp.float32)


def gemm_ref(x, w):
    return jnp.dot(x, w, preferred_element_type=jnp.float32)


# --------------------------------------------------------------------------
# Per-kernel cost: flops from reference, bytes from real kernel I/O + quant
# --------------------------------------------------------------------------
def attention_cost(
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    v_head_dim,
    q_tokens,
    kv_tokens,
    total_interactions,
    bq=32,
    q_dtype=jnp.bfloat16,
    kv_dtype=jnp.bfloat16,
) -> dict:
    """FLOPs from attention_ref (1 query block vs full KV, then scaled to the
    real interaction count); bytes from flash-style I/O (Q,O,KV; no S^2)."""
    # estimate_cost on small ref gives per-(T,S) flop density; scale to real interactions
    T0, S0 = 8, 256
    q = jax.ShapeDtypeStruct((T0, num_q_heads, head_dim), q_dtype)
    k = jax.ShapeDtypeStruct((S0, num_q_heads, head_dim), kv_dtype)
    v = jax.ShapeDtypeStruct((S0, num_q_heads, v_head_dim), kv_dtype)
    f0, t0 = _ref_flops(attention_ref, q, k, v)
    flops_per_interaction = f0 / (T0 * S0)
    flops = int(flops_per_interaction * total_interactions)
    transcendentals = int(t0 / (T0 * S0) * total_interactions)
    q_bytes = q_tokens * num_q_heads * head_dim * jnp.dtype(q_dtype).itemsize
    o_bytes = q_tokens * num_q_heads * v_head_dim * jnp.dtype(q_dtype).itemsize
    kv_read = (
        (total_interactions // max(bq, 1))
        * num_kv_heads
        * 2
        * head_dim
        * jnp.dtype(kv_dtype).itemsize
    )
    kv_write = q_tokens * num_kv_heads * 2 * head_dim * jnp.dtype(kv_dtype).itemsize
    return {
        "flops": flops,
        "transcendentals": transcendentals,
        "hbm_bytes": int(q_bytes + o_bytes + kv_read + kv_write),
    }


def moe_experts_cost(*, tokens_per_device, local_experts, d, f, qspec: QuantSpec = BF16) -> dict:
    """FLOPs from moe_ffn_ref; weight(+scale) bytes via quant over local experts."""
    x = jax.ShapeDtypeStruct((max(1, tokens_per_device), d), jnp.bfloat16)
    wg = jax.ShapeDtypeStruct((d, f), jnp.bfloat16)
    wu = jax.ShapeDtypeStruct((d, f), jnp.bfloat16)
    wd = jax.ShapeDtypeStruct((f, d), jnp.bfloat16)
    f0, t0 = _ref_flops(moe_ffn_ref, x, wg, wu, wd)
    per_expert_w = (
        2 * (qspec.w_bytes(d, f) + qspec.weight_scale_bytes(d, f))
        + qspec.w_bytes(f, d)
        + qspec.weight_scale_bytes(f, d)
    )
    weight_bytes = int(local_experts * per_expert_w)
    act = 2 * tokens_per_device * d * 2  # in + out, bf16
    return {
        "flops": int(f0),
        "transcendentals": int(t0),
        "hbm_bytes": int(weight_bytes + act),
        "peak_kind": qspec.peak_kind(),
    }


def gemm_cost(*, m, k, n, qspec: QuantSpec = BF16) -> dict:
    """FLOPs from gemm_ref; bytes via quant (weight+scale+act+out)."""
    f0, t0 = _ref_flops(
        gemm_ref,
        jax.ShapeDtypeStruct((m, k), jnp.bfloat16),
        jax.ShapeDtypeStruct((k, n), jnp.bfloat16),
    )
    w = qspec.w_bytes(k, n) + qspec.weight_scale_bytes(k, n)
    a = qspec.a_bytes(m, k) + qspec.act_scale_bytes(m, k)
    return {
        "flops": int(f0),
        "transcendentals": int(t0),
        "hbm_bytes": int(w + a + m * n * 2),
        "peak_kind": qspec.peak_kind(),
    }
