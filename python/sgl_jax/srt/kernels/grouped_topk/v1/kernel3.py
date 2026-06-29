"""Grouped top-k via argmax-selection — weights gathered once after the pick loop.

Same routing as kernel2 (id-for-id with gate.py, lowest-index tie-break) but the per-pick weight
reduction is dropped: the loop selects ids only (max + masked-min), then weights are gathered in one
`take_along_axis` from the pre-bias logits. Cuts the hot loop from 3 lane-reductions to 2.
"""

from __future__ import annotations

import functools
import logging
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

try:
    from sgl_jax.srt.kernels.grouped_topk.tuned_block_sizes import get_tuned_bt
except Exception:  # noqa: BLE001

    def get_tuned_bt(*_a, **_k):  # noqa: ANN002, ANN003
        return None


logger = logging.getLogger(__name__)

NEG_INF = -jnp.inf
SAFE_AUTO_BT = 2048
FULL_UNROLL_ELEM_BUDGET = 10_000_000


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _largest_safe_divisor(bs: int, cap: int = SAFE_AUTO_BT, align: int = 8) -> int | None:
    hi = (min(cap, bs) // align) * align
    for d in range(hi, 0, -align):
        if bs % d == 0:
            return d
    return None


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _grouped_topk_kernel(
    logits_ref,
    bias_ref,
    w_ref,
    ids_ref,
    *,
    n_group: int,
    topk_group: int,
    topk: int,
    num_experts: int,
    padded_topk: int,
    full_unroll: bool,
):
    S = num_experts // n_group
    logits = logits_ref[...].astype(jnp.float32)
    scores = logits + bias_ref[...][None, :]
    bt = scores.shape[0]

    # group score = sum of top-2 per group (2-pass max)
    with jax.named_scope("group_top2"):
        g = []
        for gi in range(n_group):
            sl = scores[:, gi * S : (gi + 1) * S]
            io = jax.lax.broadcasted_iota(jnp.int32, sl.shape, 1)
            v1 = jnp.max(sl, axis=1, keepdims=True)
            sl2 = jnp.where(io == jnp.argmax(sl, axis=1, keepdims=True), NEG_INF, sl)
            g.append(v1 + jnp.max(sl2, axis=1, keepdims=True))
        group_scores = jnp.concatenate(g, axis=1)

    # select topk_group groups (lowest-index tie-break)
    with jax.named_scope("group_select"):
        g_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, n_group), 1)
        group_mask = jnp.zeros((bt, n_group), jnp.bool_)
        tmp = group_scores
        for _ in range(topk_group):
            gmax = jnp.max(tmp, axis=1, keepdims=True)
            gi = jnp.min(jnp.where(tmp == gmax, g_iota, n_group), axis=1, keepdims=True)
            hit = g_iota == gi
            group_mask |= hit
            tmp = jnp.where(hit, NEG_INF, tmp)

    with jax.named_scope("expert_mask"):
        masked = jnp.concatenate(
            [
                jnp.where(group_mask[:, gi : gi + 1], scores[:, gi * S : (gi + 1) * S], NEG_INF)
                for gi in range(n_group)
            ],
            axis=1,
        )

    # pick topk ids into a width-E order buffer (2 reductions/pick), defer weights
    with jax.named_scope("final_select"):
        e_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, num_experts), 1)

        def _pick(k, carry):
            cur, order = carry
            cmax = jnp.max(cur, axis=1, keepdims=True)
            idx = jnp.min(jnp.where(cur == cmax, e_iota, num_experts), axis=1, keepdims=True)
            order = jnp.where(e_iota == k, idx, order)  # write pick k into column k
            cur = jnp.where(e_iota == idx, NEG_INF, cur)
            return cur, order

        order0 = jnp.zeros((bt, num_experts), jnp.int32)
        _, order = jax.lax.fori_loop(
            0, topk, _pick, (masked, order0), unroll=topk if full_unroll else 1
        )

    # gather all weights in one shot (Mosaic gather needs output shape == input shape -> width E)
    with jax.named_scope("gather_weights"):
        w_full = jnp.take_along_axis(logits, order, axis=1)  # [BT, E]

    # columns >= topk are filler (order=0); the wrapper slices [:, :topk]. padded_topk <= E always
    # here (E is a multiple of 128), so the width-E buffers cover the output tile.
    ids_ref[...] = order[:, :padded_topk]
    w_ref[...] = w_full[:, :padded_topk]


def grouped_topk_pallas(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    unroll: bool | None = None,
    interpret: bool | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Biased grouped top-k via argmax-selection. Returns (topk_weights[BS,k], topk_ids[BS,k])."""
    bs, e = router_logits.shape
    router_logits = router_logits.astype(jnp.float32)
    bias = correction_bias.astype(jnp.float32)

    if block_tokens == "auto":
        tuned = get_tuned_bt(bs, e, num_expert_group, topk_group, topk)
        if tuned is not None and bs % tuned == 0:
            bt = tuned
        elif bs % 512 == 0:
            bt = min(512, bs)
        else:
            bt = _largest_safe_divisor(bs) or bs
        if bt > SAFE_AUTO_BT:
            logger.warning(
                "grouped_topk: auto block_tokens fell back to whole-batch BT=%d (BS=%d has no "
                "VMEM-safe divisor); pass an explicit block_tokens.",
                bt,
                bs,
            )
    else:
        bt = min(block_tokens, bs)
        if bs % bt != 0:
            raise ValueError(f"BS={bs} must be divisible by block_tokens={bt}")
    if interpret is None:
        interpret = get_interpret()

    padded_topk = _align_to(topk, 128)
    full_unroll = topk * bt * e <= FULL_UNROLL_ELEM_BUDGET if unroll is None else bool(unroll)
    kernel = functools.partial(
        _grouped_topk_kernel,
        n_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_experts=e,
        padded_topk=padded_topk,
        full_unroll=full_unroll,
    )
    compiler_params = None
    if vmem_limit_bytes is not None:
        import jax.experimental.pallas.tpu as pltpu

        params_cls = getattr(pltpu, "CompilerParams", None) or pltpu.TPUCompilerParams
        compiler_params = params_cls(vmem_limit_bytes=vmem_limit_bytes)
    weights, ids = pl.pallas_call(
        kernel,
        grid=(bs // bt,),
        in_specs=[
            pl.BlockSpec((bt, e), lambda i: (i, 0)),
            pl.BlockSpec((e,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((bt, padded_topk), lambda i: (i, 0)),
            pl.BlockSpec((bt, padded_topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((bs, padded_topk), jnp.float32),
            jax.ShapeDtypeStruct((bs, padded_topk), jnp.int32),
        ],
        interpret=interpret,
        name="grouped-topk",
        compiler_params=compiler_params,
    )(router_logits, bias)
    return weights[:, :topk], ids[:, :topk]
