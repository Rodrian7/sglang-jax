"""Per-op cost primitives returning ``jax.experimental.roofline.RooflineResult``.

XLA ops (projections, gate, norm, rope, lm_head) are costed by *leveraging*
``jax.experimental.roofline`` on a tiny reference function (fusion-aware FLOPs +
HBM bytes, single-device / unsharded). Pallas kernels (attention, MoE experts,
fp8 QKV) cannot be walked by roofline (its interpreter crashes on Pallas Refs),
so they are costed by closed-form formulas ported from the kernels' own
``cost_estimate`` (RPA ``ragged_paged_attention.py``, GMM ``gmm_v2.py``).

These return *full / unsharded* op costs; the descriptor scales per-device
(divides by the parallel degree) and adds collective ICI bytes.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import roofline as _roofline
from jax.experimental.roofline import RooflineResult

from .quant import BF16, QuantSpec
from .report import OpRoofline, PeakKind


def _itemsize(dtype) -> float:
    return jnp.dtype(dtype).itemsize


# --------------------------------------------------------------------------
# Bridge: RooflineResult (jax) -> OpRoofline (report row)
# --------------------------------------------------------------------------
def to_op(
    rr: RooflineResult,
    label: str,
    category: str,
    *,
    peak_kind: PeakKind = "bf16",
    source: str = "",
    count: int = 1,
) -> OpRoofline:
    ici = sum(rr.ici_bytes.values()) if rr.ici_bytes else 0
    return OpRoofline(
        label=label,
        category=category,
        source=source,
        count=count,
        flops=int(rr.flops),
        hbm_bytes=int(rr.hbm_bytes),
        ici_bytes=int(ici),
        peak_kind=peak_kind,
    )


def _rr(flops: int, hbm_bytes: int, ici_bytes: int = 0) -> RooflineResult:
    out = dataclasses.replace(
        RooflineResult.zeros(),
        flops=int(flops),
        unfused_flops=int(flops),
        hbm_bytes=int(hbm_bytes),
        unfused_hbm_bytes=int(hbm_bytes),
    )
    if ici_bytes:
        out = dataclasses.replace(out, ici_bytes={"ep": int(ici_bytes)})
    return out


# --------------------------------------------------------------------------
# XLA ops via jax.experimental.roofline (single-device / unsharded)
# --------------------------------------------------------------------------
def xla_roofline(fn, *abstract_args) -> RooflineResult:
    """Run roofline (no mesh => no shard_map) on ``fn`` to get its cost.

    ``abstract_args`` are ``jax.ShapeDtypeStruct`` so nothing is allocated.
    """
    _, res = _roofline.roofline(fn)(*abstract_args)
    return res


def linear(
    m: int, k: int, n: int, *, w_dtype=jnp.bfloat16, act_dtype=jnp.bfloat16
) -> RooflineResult:
    """Dense matmul [m,k]@[k,n] -> [m,n] (fusion-aware via roofline)."""

    def f(x, w):
        return jnp.dot(x, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    return xla_roofline(
        f,
        jax.ShapeDtypeStruct((m, k), act_dtype),
        jax.ShapeDtypeStruct((k, n), w_dtype),
    )


def rms_norm(m: int, h: int, *, dtype=jnp.bfloat16) -> RooflineResult:
    """RMSNorm: closed-form HBM (memory-bound; roofline gives 0 for pure
    elementwise fusions). Reads x[m,h] + weight[h], writes out[m,h]."""
    isz = _itemsize(dtype)
    hbm = 2 * m * h * isz + h * isz
    flops = 4 * m * h  # square + reduce + rsqrt + 2 muls (negligible vs matmuls)
    return _rr(flops, int(hbm))


def rope(m: int, q_size: int, k_size: int, *, dtype=jnp.bfloat16) -> RooflineResult:
    """RoPE on q[m,q_size] and k[m,k_size]: closed-form HBM (read+write q,k)."""
    isz = _itemsize(dtype)
    hbm = 2 * (q_size + k_size) * m * isz  # read + write q and k
    flops = 6 * (q_size + k_size) * m  # rotate-half: a few muls/adds per elem
    return _rr(flops, int(hbm))


def elementwise(m: int, h: int, *, n_inputs: int = 2, dtype=jnp.bfloat16) -> RooflineResult:
    """Residual add / activation: read ``n_inputs`` tensors, write one."""
    isz = _itemsize(dtype)
    hbm = (n_inputs + 1) * m * h * isz
    return _rr(m * h, int(hbm))


def router_gate(m: int, h: int, num_experts: int, topk: int) -> RooflineResult:
    def f(x, wg):
        logits = jnp.dot(x.astype(jnp.float32), wg)
        probs = jax.nn.softmax(logits, axis=-1)
        w, _ = jax.lax.top_k(probs, topk)
        return w / jnp.sum(w, axis=-1, keepdims=True)

    return xla_roofline(
        f,
        jax.ShapeDtypeStruct((m, h), jnp.bfloat16),
        jax.ShapeDtypeStruct((h, num_experts), jnp.float32),
    )


# --------------------------------------------------------------------------
# Pallas kernels: closed-form (roofline cannot walk Pallas)
# --------------------------------------------------------------------------
def attention(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    q_tokens: int,
    total_interactions: int,
    bq: int = 32,
    q_dtype=jnp.bfloat16,
    kv_dtype=jnp.bfloat16,
) -> RooflineResult:
    """Ragged-paged attention cost (formula from ragged_paged_attention.py:1605)."""
    flops = 4 * num_q_heads * head_dim * total_interactions
    q_bytes = q_tokens * num_q_heads * head_dim * _itemsize(q_dtype)
    o_bytes = q_tokens * num_q_heads * v_head_dim * _itemsize(q_dtype)
    kv_read = (total_interactions // max(bq, 1)) * num_kv_heads * 2 * head_dim * _itemsize(kv_dtype)
    kv_write = q_tokens * num_kv_heads * 2 * head_dim * _itemsize(kv_dtype)
    return _rr(flops, int(q_bytes + o_bytes + kv_read + kv_write))


def moe_experts(
    *,
    tokens_per_device: int,
    local_experts: float,
    d: int,
    f: int,
    qspec: QuantSpec = BF16,
    act_dtype=jnp.bfloat16,
    ici_bytes: int = 0,
) -> RooflineResult:
    """Fused EP-MoE expert FFN: 3 GEMMs/token (gate,up: d->f; down: f->d).

    flops = 2 * tokens_per_device * 3 * d * f.
    weight (+scale) bytes are summed over ``local_experts`` experts using qspec.
    """
    flops = 2 * tokens_per_device * 3 * d * f
    per_expert_w = (
        2 * (qspec.w_bytes(d, f) + qspec.weight_scale_bytes(d, f))  # gate + up
        + qspec.w_bytes(f, d)
        + qspec.weight_scale_bytes(f, d)  # down
    )
    weight_bytes = int(local_experts * per_expert_w)
    act_in = tokens_per_device * d * _itemsize(act_dtype)
    act_out = tokens_per_device * d * _itemsize(act_dtype)
    return _rr(flops, int(weight_bytes + act_in + act_out), ici_bytes=ici_bytes)


def gemm(m: int, k: int, n: int, qspec: QuantSpec = BF16) -> RooflineResult:
    """Quant-aware dense matmul [m,k]@[k,n]->[m,n] (bf16 output).

    Captures: quantized weight bytes + scale-tensor bytes (per-channel or
    block-wise), quantized activation bytes + act-scale bytes. The MXU rate
    penalty of block-wise quant is carried by ``qspec.peak_kind()`` at the
    caller, not here.
    """
    flops = 2 * m * k * n
    w = qspec.w_bytes(k, n) + qspec.weight_scale_bytes(k, n)
    a = qspec.a_bytes(m, k) + qspec.act_scale_bytes(m, k)
    out = m * n * 2  # bf16 output
    return _rr(flops, int(w + a + out))


def estimate_cost_flops(ref_fn, *abstract_args) -> int:
    """Cross-check helper: auto-derive FLOPs of a Pallas kernel's math from a
    pure-JAX reference via ``jax.experimental.pallas.estimate_cost`` (Tier 2)."""
    from jax.experimental import pallas as pl

    return int(pl.estimate_cost(ref_fn, *abstract_args).flops)
