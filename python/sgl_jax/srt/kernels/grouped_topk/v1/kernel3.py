"""Grouped top-k via count-rank (vectorized, no iterative argmax).

Same routing/result as kernel2 (id-for-id with gate.py, lowest-index tie-break) but the final select
is loop-free: each expert's rank is its count of strictly-better experts (plus equal-but-lower-index
for the tie-break), which is exactly its position in `jax.lax.top_k`'s descending order. Experts with
rank < topk are the winners and scatter straight into their output column.

Trades the scalar-bound iterative argmax for wide vector compare+reduce (uses the otherwise-idle
VPU), at the cost of a 3D [BT,E,S] working tensor — verify it lowers on the target TPU, and keep
block_tokens modest so [BT,E,S] fits VMEM.
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
):
    S = num_experts // n_group
    logits = logits_ref[...].astype(jnp.float32)
    scores = logits + bias_ref[...][None, :]
    bt = scores.shape[0]

    with jax.named_scope("group_top2"):
        g = []
        for gi in range(n_group):
            sl = scores[:, gi * S : (gi + 1) * S]
            io = jax.lax.broadcasted_iota(jnp.int32, sl.shape, 1)
            v1 = jnp.max(sl, axis=1, keepdims=True)
            sl2 = jnp.where(io == jnp.argmax(sl, axis=1, keepdims=True), NEG_INF, sl)
            g.append(v1 + jnp.max(sl2, axis=1, keepdims=True))
        group_scores = jnp.concatenate(g, axis=1)

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

    # rank[i] = #{j: masked[j] > masked[i]} + #{j<i: masked[j] == masked[i]}  == top_k position of i.
    # Accumulated per group so the 3D working tensor is [BT, E, S], not [BT, E, E].
    with jax.named_scope("count_rank"):
        i_glob = jax.lax.broadcasted_iota(jnp.int32, (bt, num_experts), 1)
        si = masked[:, :, None]  # [BT, E, 1]
        rank = jnp.zeros((bt, num_experts), jnp.int32)
        for gi in range(n_group):
            sj = masked[:, None, gi * S : (gi + 1) * S]  # [BT, 1, S]
            j_glob = gi * S + jax.lax.broadcasted_iota(jnp.int32, (bt, num_experts, S), 2)
            beats = (sj > si) | ((sj == si) & (j_glob < i_glob[:, :, None]))
            rank += jnp.sum(beats.astype(jnp.int32), axis=2)

    # scatter each winner (rank < topk) into output column = its rank
    with jax.named_scope("scatter"):
        col = jax.lax.broadcasted_iota(jnp.int32, (bt, num_experts, topk), 2)
        one = rank[:, :, None] == col  # [BT, E, topk]
        ids = jnp.sum(jnp.where(one, i_glob[:, :, None], 0), axis=1).astype(jnp.int32)
        w = jnp.sum(jnp.where(one, logits[:, :, None], 0.0), axis=1)
        pad = padded_topk - topk
        if pad > 0:
            ids = jnp.concatenate([ids, jnp.full((bt, pad), -1, jnp.int32)], axis=1)
            w = jnp.concatenate([w, jnp.zeros((bt, pad), jnp.float32)], axis=1)

    ids_ref[...] = ids
    w_ref[...] = w


def grouped_topk_pallas(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    unroll: bool | None = None,  # noqa: ARG001  (accepted for a uniform call signature; unused)
    interpret: bool | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Biased grouped top-k via count-rank. Returns (topk_weights[BS,k], topk_ids[BS,k])."""
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
    else:
        bt = min(block_tokens, bs)
        if bs % bt != 0:
            raise ValueError(f"BS={bs} must be divisible by block_tokens={bt}")
    if interpret is None:
        interpret = get_interpret()

    padded_topk = _align_to(topk, 128)
    kernel = functools.partial(
        _grouped_topk_kernel,
        n_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_experts=e,
        padded_topk=padded_topk,
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
