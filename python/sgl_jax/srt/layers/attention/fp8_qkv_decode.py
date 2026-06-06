"""FP8 W8A16 fused-QKV decode kernel for MiMo-V2 attention.

Decode (M≈8 tokens/device) is HBM-bound: reading the QKV weights dominates. Storing
the weights as fp8 (half the bytes of bf16) and upcasting in-VMEM lets a continuous
double-buffered stream beat XLA bf16 at decode while staying numerically accurate
(W8A16: weight fp8, activation bf16; re-quant of the already-fp8-derived bf16 weight
is near lossless, validated rel-err ~2.5e-3 vs the deployed bf16 path).

Prefill stays on bf16 XLA (LinearBase) — the per-128-K-block scale makes an fp8 prefill
GEMM cap at ~0.85x XLA on this MXU, so fp8 is decode-only.

The three col-parallel q/k/v projections are fused into one weight so decode issues a
single streamed kernel (one fill/drain instead of three). Under TP the merged weight is
*stripe-interleaved*: device d owns ``[q_d | k_d | v_d | pad]`` (see MergedColumnParallelLinear).
``build_fused_qkv_fp8`` builds that layout; the per-device split happens inside ``shard_map``.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)
BLK = 128  # weight K-quantization block (fixed by the model's [128,128] fp8 quant)
BN = 256  # output N-tile for the stream (128-aligned; decode sweet spot)


def quantize_block_k(w: jax.Array):
    """Per-128-K-block fp8 quantization. w[K,N] bf16 -> (wq[K,N] fp8, ws[K//128,N] f32)."""
    k, n = w.shape
    nk = k // BLK
    wf = w.astype(jnp.float32).reshape(nk, BLK, n)
    amax = jnp.max(jnp.abs(wf), axis=1, keepdims=True)
    s = jnp.where(amax == 0, 1.0, amax / FP8_MAX)
    wq = (wf / s).clip(-FP8_MAX, FP8_MAX).astype(FP8).reshape(k, n)
    return wq, s.reshape(nk, n).astype(jnp.float32)


def build_fused_qkv_fp8(wq_bf16, wk_bf16, wv_bf16, tp: int):
    """Global (single-device / test) build of the TP-interleaved fused QKV fp8 weight.

    Device d must own ``[q_d | k_d | v_d | pad]`` so the post-kernel split (inside shard_map)
    is local. Each device block is padded to a multiple of ``BN`` (every N-tile 128-aligned).
    For the real (sharded) model use :func:`build_qkv_fp8_local` under shard_map instead.

    Returns: (wq_fused fp8 [K, tp*nf_pad], ws_fused f32 [K//128, tp*nf_pad],
              (q_local, k_local, v_local), nf_local, nf_pad).
    """
    k = wq_bf16.shape[0]
    q_local = wq_bf16.shape[1] // tp
    k_local = wk_bf16.shape[1] // tp
    v_local = wv_bf16.shape[1] // tp
    nf_local = q_local + k_local + v_local
    nf_pad = ((nf_local + BN - 1) // BN) * BN
    q3 = wq_bf16.reshape(k, tp, q_local)
    k3 = wk_bf16.reshape(k, tp, k_local)
    v3 = wv_bf16.reshape(k, tp, v_local)
    block = jnp.concatenate([q3, k3, v3], axis=2)  # [K, tp, nf_local], per-device block
    if nf_pad > nf_local:
        block = jnp.pad(block, ((0, 0), (0, 0), (0, nf_pad - nf_local)))
    merged = block.reshape(k, tp * nf_pad)  # device d => columns [d*nf_pad:(d+1)*nf_pad]
    wq_fused, ws_fused = quantize_block_k(merged)
    return wq_fused, ws_fused, (q_local, k_local, v_local), nf_local, nf_pad


def build_qkv_fp8_local(q, k, v, nf_pad: int):
    """Per-device fused-QKV fp8 build — call INSIDE shard_map (in/out specs P(None,"tensor")).

    q/k/v are this device's col-parallel shards ``[K, *_local]``. Concatenating them gives the
    device's ``[q_d | k_d | v_d]`` block; shard_map assembles the TP-interleaved global weight.
    Pads to ``nf_pad`` (multiple of BN) so the decode kernel's N-tiles stay 128-aligned.

    Returns (wq[K, nf_pad] fp8, ws[K//128, nf_pad] f32).
    """
    block = jnp.concatenate([q, k, v], axis=1)  # [K, nf_local]
    nf_local = block.shape[1]
    if nf_pad > nf_local:
        block = jnp.pad(block, ((0, 0), (0, nf_pad - nf_local)))
    return quantize_block_k(block)


def fp8_qkv_w8a16_local(x, wq, ws, bn: int = BN):
    """Per-device W8A16 decode matmul (call inside shard_map).

    x[M,K] bf16, wq[K,Nf] fp8, ws[K//128,Nf] f32 -> out[M,Nf] bf16.
    Continuous stream: grid=(1,), weights in HBM(ANY), 2-slot double buffer prefetched by
    N-tile, each tile dequant'd fp8->bf16 (apply per-128-K-block scale) then a full-K bf16 dot.
    """
    m, k = x.shape
    n = wq.shape[1]
    nk = k // BLK
    nt = n // bn

    def kern(x_ref, wq_h, ws_h, o_ref, wv, sv, w_sem, s_sem):
        def fetch(t, slot):
            pltpu.make_async_copy(
                wq_h.at[:, pl.ds(t * bn, bn)], wv.at[slot], w_sem.at[slot]
            ).start()
            pltpu.make_async_copy(
                ws_h.at[:, pl.ds(t * bn, bn)], sv.at[slot], s_sem.at[slot]
            ).start()

        def wait(slot):
            pltpu.make_async_copy(wv.at[slot], wv.at[slot], w_sem.at[slot]).wait()
            pltpu.make_async_copy(sv.at[slot], sv.at[slot], s_sem.at[slot]).wait()

        fetch(0, 0)

        def loop(t, _):
            cur = lax.rem(t, 2)
            nxt = lax.rem(t + 1, 2)

            @pl.when(t + 1 < nt)
            def _():
                fetch(t + 1, nxt)

            wait(cur)
            wqf = wv[cur].astype(jnp.float32).reshape(nk, BLK, bn)
            wd = (wqf * sv[cur].reshape(nk, 1, bn)).reshape(k, bn).astype(jnp.bfloat16)
            d = jax.lax.dot_general(
                x_ref[...], wd, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            o_ref[:, pl.ds(t * bn, bn)] = d.astype(o_ref.dtype)
            return _

        lax.fori_loop(0, nt, loop, None)

    return pl.pallas_call(
        kern,
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m, k), lambda i: (0, 0)),
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.ANY),
        ],
        out_specs=pl.BlockSpec((m, n), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        scratch_shapes=[
            pltpu.VMEM((2, k, bn), FP8),
            pltpu.VMEM((2, nk, bn), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )(x, wq, ws)
