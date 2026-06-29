"""Standalone Pallas TPU kernel for biased grouped top-k MoE routing.

This is the routing of `gate.py:TopK._biased_grouped_topk` (DeepSeek-V3 noaux_tc) done
WITHOUT any `sort` — entirely via `max`/`argmax` selection, fully VMEM-resident inside one
Pallas kernel. It mirrors the in-kernel `get_top_k` of the v1 fused-MoE kernel
(`kernels/fused_moe/v1/kernel.py`) but is a self-contained, separately benchmarkable op.

Why no sort: on TPU `jax.lax.top_k` lowers to a `stablehlo.sort` (a bitonic comparison
network) that is bound by the VPU's cross-lane permute throughput (~8% of VPU peak). Selecting
top-k by iterated `argmax` (+ masking the winner) is a sequence of plain reduces — it runs on
the much faster reduce path and touches fewer elements for small k. See
`work/group-topk-kernel/analysis-zh.md`.

Algorithm (matches `_biased_grouped_topk` exactly, id-for-id, INCLUDING ties):
  scores = router_logits + correction_bias                         # post-bias "scores_for_choice"
  ① group score = sum of top-2 per group, via 2-pass max           # no sort
  ② select `topk_group` groups, via max + masked-min               # no sort, lowest-index tie-break
  ③ mask dropped groups to -inf, select `topk` experts likewise    # no sort, lowest-index tie-break
  weights = router_logits[selected_ids]   (PRE-bias logits)        # like gate.py
Renormalize / routed_scaling are left to the caller (as in `TopK.__call__`).

Tie-break: selection uses `max` + masked `min` (smallest index achieving the max) rather than
`jnp.argmax`, because TPU Mosaic's lane-reduction argmax does NOT break ties toward the lowest
index; this reproduces `jax.lax.top_k`'s lowest-index order exactly, even on equal scores.
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
except Exception:  # noqa: BLE001  (e.g. base64-embedded standalone copy without the package)

    def get_tuned_bt(*_a, **_k):  # noqa: ANN002, ANN003
        return None


logger = logging.getLogger(__name__)

NEG_INF = -jnp.inf

# Largest BT the multi-block (grid>1) path is known to fit in v7x VMEM (double-buffered [BT,E]
# inputs; see tuned_block_sizes / analysis-zh.md). The "auto" path never tiles above this without
# warning. The kernel's per-block working set is ~independent of topk (the final-select runs as a
# fori_loop carrying a single [BT,E], not an unrolled O(topk) chain), so this bound holds for any k.
SAFE_AUTO_BT = 2048

# Final-select unroll policy. A full unroll overlaps all `topk` picks (fast) but keeps O(topk) live
# [BT,E] temporaries; a rolled loop keeps one (safe, ~15-44% slower). Pallas only allows fori_loop
# unroll=1 or unroll=num_steps (no partial), so we pick per-call: full-unroll when the working set
# `topk * BT * E` fits VMEM, else roll. The budget is calibrated to TPU v6e/v7x (~32 MiB scoped
# VMEM; OOM observed at ~16.8M elems). Being conservative only costs speed — the rolled fallback
# always fits — so lower it for smaller-VMEM TPU generations if needed.
FULL_UNROLL_ELEM_BUDGET = 10_000_000  # max topk * BT * E to take the full-unroll fast path


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _largest_safe_divisor(bs: int, cap: int = SAFE_AUTO_BT, align: int = 8) -> int | None:
    """Largest d that divides bs with d <= cap and d % align == 0.

    Pallas TPU tiling needs the token block to be a multiple of 8 (sublane), so an auto block size
    must be both a divisor of bs (for an even grid) and 8-aligned. Returns None when bs has no such
    divisor (e.g. bs is prime / not 8-aligned), so the caller can fall back to a single block.
    """
    hi = (min(cap, bs) // align) * align
    for d in range(hi, 0, -align):
        if bs % d == 0:
            return d
    return None


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _grouped_topk_kernel(
    logits_ref,  # [BT, E] f32  (router_logits, post-score-func, PRE-bias)
    bias_ref,  # [E]     f32  (correction_bias)
    w_ref,  # [BT, padded_topk] f32  out: weights
    ids_ref,  # [BT, padded_topk] i32  out: expert ids
    *,
    n_group: int,
    topk_group: int,
    topk: int,
    num_experts: int,
    padded_topk: int,
    full_unroll: bool,
):
    S = num_experts // n_group
    logits = logits_ref[...].astype(jnp.float32)  # pre-bias
    with jax.named_scope("bias_add"):
        scores = logits + bias_ref[...][None, :]  # post-bias [BT, E]
    bt = scores.shape[0]

    # ① group score = sum of top-2 within each group, via 2-pass max (no sort)
    with jax.named_scope("group_top2"):
        g_scores = []
        for g in range(n_group):
            sl = scores[:, g * S : (g + 1) * S]  # [BT, S]
            v1 = jnp.max(sl, axis=1, keepdims=True)
            i1 = jnp.argmax(sl, axis=1, keepdims=True)
            io = jax.lax.broadcasted_iota(jnp.int32, sl.shape, 1)
            sl_masked = jnp.where(io == i1, NEG_INF, sl)
            v2 = jnp.max(sl_masked, axis=1, keepdims=True)
            g_scores.append(v1 + v2)
        group_scores = jnp.concatenate(g_scores, axis=1)  # [BT, G]

    # ② select `topk_group` groups, lowest-index tie-break (matches jax.lax.top_k). TPU Mosaic's
    #    argmax does NOT break ties toward the lowest index, so use max + masked-min instead.
    with jax.named_scope("group_select"):
        group_mask = jnp.zeros((bt, n_group), dtype=jnp.bool_)
        g_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, n_group), 1)
        tmp = group_scores
        for _ in range(topk_group):
            gmax = jnp.max(tmp, axis=1, keepdims=True)
            gi = jnp.min(jnp.where(tmp == gmax, g_iota, n_group), axis=1, keepdims=True)
            m = g_iota == gi
            group_mask = jnp.logical_or(group_mask, m)
            tmp = jnp.where(m, NEG_INF, tmp)

    # build NARROW candidates: top-t experts per group (t = max a single group can give
    #     to the global top-k). Dropped groups are forced to NEG_INF so they rank last.
    #     Replaces the [BT,E] `masked` working array with a [BT, C] one (C = n_group*t).
    with jax.named_scope("candidates"):
        t = min(topk, S)                       # max picks any one group can supply
        loc_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, S), 1)
        cand_score, cand_id, cand_wt = [], [], []
        for g in range(n_group):
            gm = group_mask[:, g : g + 1]                                   # [BT,1]
            sl = jnp.where(gm, scores[:, g * S : (g + 1) * S], NEG_INF)     # [BT,S] post-bias (rank)
            lg = logits[:, g * S : (g + 1) * S]                             # [BT,S] pre-bias (weight)
            cur = sl
            for _ in range(t):                 # extend the same max-mask trick as group_top2
                m = jnp.max(cur, axis=1, keepdims=True)
                li = jnp.min(jnp.where(cur == m, loc_iota, S), axis=1, keepdims=True)  # local idx
                sel = loc_iota == li
                cand_score.append(m)                                        # [BT,1] rank key
                cand_id.append((li + g * S).astype(jnp.int32))              # [BT,1] GLOBAL id
                cand_wt.append(jnp.sum(jnp.where(sel, lg, 0.0), axis=1, keepdims=True))  # [BT,1]
                cur = jnp.where(sel, NEG_INF, cur)
        
        # C = n_group * t
        cand_score = jnp.concatenate(cand_score, axis=1)                    # [BT,C]
        cand_id = jnp.concatenate(cand_id, axis=1)                          # [BT,C] (all unique)
        cand_wt = jnp.concatenate(cand_wt, axis=1)                          # [BT,C]

    #  global top-k on the NARROW [BT,C] array. Tie-break by lowest GLOBAL id (matches
    #    jax.lax.top_k) using cand_id directly — ids are unique so `cand_id == pick_id` is one-hot.
    with jax.named_scope("final_select"):
        col_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, padded_topk), 1)
        ids_init = jnp.full((bt, padded_topk), -1, dtype=jnp.int32)
        w_init = jnp.zeros((bt, padded_topk), dtype=jnp.float32)

        def _pick(k, carry):
            cur, ids_buf, w_buf = carry                                     # cur: [BT,C]
            cmax = jnp.max(cur, axis=1, keepdims=True)
            pick_id = jnp.min(
                jnp.where(cur == cmax, cand_id, num_experts), axis=1, keepdims=True
            )                                                              # [BT,1] lowest global id
            sel = cand_id == pick_id                                        # [BT,C] one-hot
            pick_wt = jnp.sum(jnp.where(sel, cand_wt, 0.0), axis=1, keepdims=True)
            write = col_iota == k
            ids_buf = jnp.where(write, pick_id, ids_buf)
            w_buf = jnp.where(write, pick_wt, w_buf)
            cur = jnp.where(sel, NEG_INF, cur)
            return cur, ids_buf, w_buf

        _, ids_out, w_out = jax.lax.fori_loop(
            0, topk, _pick, (cand_score, ids_init, w_init),
            unroll=topk if full_unroll else 1,
        )

    ids_ref[...] = ids_out  # [BT, padded_topk]
    w_ref[...] = w_out  # [BT, padded_topk]


def grouped_topk_pallas(
    router_logits: jax.Array,  # [BS, E] (any float; cast to f32 inside)
    correction_bias: jax.Array,  # [E]
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    unroll: bool | None = None,
    interpret: bool | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Biased grouped top-k via argmax-selection. Returns (topk_weights[BS,k], topk_ids[BS,k]).

    Drop-in for `gate.py:TopK._biased_grouped_topk` (renormalize / routed_scaling_factor are
    applied by the caller, exactly as in `TopK.__call__`).

    `block_tokens="auto"` (default) looks up the tuned BT for this device + (BS, E, G, Gtop, k)
    and falls back to a safe default on a miss; an explicit int forces that block size.

    `unroll` selects the final-select loop form: None (default) full-unrolls when the working set
    `topk*BT*E` fits VMEM (`FULL_UNROLL_ELEM_BUDGET`) else rolls; True/False force the choice. Full
    unroll is faster but keeps O(topk) live [BT,E] (needs a smaller BT); rolling allows a larger BT.

    `vmem_limit_bytes` (None = compiler default) caps the Mosaic scoped-VMEM budget; binary-search
    it to measure a config's real peak VMEM (the smallest value that still compiles).
    """
    bs, e = router_logits.shape
    router_logits = router_logits.astype(jnp.float32)
    bias = correction_bias.astype(jnp.float32)

    if block_tokens == "auto":
        tuned = get_tuned_bt(bs, e, num_expert_group, topk_group, topk)
        if tuned is not None and bs % tuned == 0:
            bt = tuned
        elif bs % 512 == 0:
            bt = min(512, bs)  # 512 divides bs -> safe default tile
        else:
            # Odd serving bucket (bs divisible by neither the tuned BT nor 512): pick the largest
            # VMEM-safe divisor of bs rather than silently tiling the whole batch as one block.
            bt = _largest_safe_divisor(bs) or bs
        if bt > SAFE_AUTO_BT:
            logger.warning(
                "grouped_topk: auto block_tokens fell back to whole-batch BT=%d (BS=%d has no "
                "VMEM-safe divisor); a single [%d,%d] tile may exceed VMEM. Pad local tokens to a "
                "power-of-2 or a multiple of 512, or pass an explicit block_tokens.",
                bt,
                bs,
                bs,
                e,
            )
    else:
        bt = min(block_tokens, bs)
        if bs % bt != 0:
            raise ValueError(f"BS={bs} must be divisible by block_tokens={bt}")
    if interpret is None:
        interpret = get_interpret()

    padded_topk = _align_to(topk, 128)  # TPU VMEM output tile needs a 128-multiple minor dim
    
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

        # jax<0.8 exposed this as TPUCompilerParams; 0.8+ renamed it to CompilerParams.
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
