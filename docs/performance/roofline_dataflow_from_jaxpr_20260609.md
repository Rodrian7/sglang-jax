# Roofline dataflow: auto-derive the layer graph from a traced jaxpr (iteration)

**Date:** 2026-06-09  ·  **Branch:** `feat/theoretical-roofline`

## Why

The dataflow side of the theoretical roofline (`graph.build_layer_graph` +
`critical_path`) was a **hand-written transcription** of one MiMo decoder layer.
Three weaknesses (see review): (1) hand-coded per-model — a new model needs new
graph code, and the `source="file.py:line"` strings rot; (2) granularity is
conceptual ops, disconnected from what XLA actually runs (can't show fusion);
(3) the CPM critical path is near-trivial for a serial transformer.

This iteration attacks (1) and (2): **derive the layer dataflow graph
automatically from the descriptor's traced reference forward (a jaxpr)** instead
of transcribing it by hand.

## What was built

`python/sgl_jax/srt/utils/roofline/graph_from_jaxpr.py`:

- `build_graph_from_jaxpr(jaxpr, pallas_coster)` — walks the jaxpr: **nodes =
  vars (tensors), edges = equations (ops)**. Per-eqn theoretical cost
  (`_dot_flops` for matmuls, element-count for elementwise/reduce, 0 for
  movement) and **real source attribution** via `eqn.source_info`. Pallas /
  `custom_call` are opaque to a static walk, so their cost comes from a
  pluggable `pallas_coster` (reuses the same `references.attention_cost` /
  `moe_experts_cost` the descriptor uses).
- `fuse(graph)` — a light XLA-like fusion model: only **materialised** tensors
  (graph i/o, anchor in/outputs, fan-out > 1) cost an HBM round-trip;
  elementwise/movement chains between matmul/Pallas anchors are fused away.
  FLOPs unchanged. Pallas weight bytes (not in the jaxpr) are preserved from the
  coster.
- `analyze_reference(arch, …)` + `render_auto_graph` — trace → graph → fuse →
  critical path, with a cross-check against `build_layer_graph`.
- CLI: `model_roofline.py --view f` prints the auto-derived graph.

## Result: cross-validation vs the hand-written graph

Unsharded (tp=1, ep=1) one full+MoE MiMo layer, decode (batch 8, ctx 4096):

| graph | ops | FLOPs | HBM | t_critical |
|---|---|---|---|---|
| AUTO (jaxpr, unfused) | 51 | 5.208 GFLOP | 9857.0 MB | 2.667 ms |
| AUTO (jaxpr, **fused**) | 51 | 5.208 GFLOP | 9853.6 MB | 2.666 ms |
| hand-written `build_layer_graph` | 11 | 5.208 GFLOP | 9850.4 MB | 2.670 ms |

**FLOPs match exactly; HBM within 0.04% (decode) / 2.5% (prefill); t_critical
within 0.13%.** The auto-derived 51-op graph (which auto-expands rms_norm into
`integer_pow/reduce_sum/rsqrt/mul`, softmax into `reduce_max/sub/exp/reduce_sum/
div`, etc.) reproduces the hand-coded 11-op graph — so the derivation is
faithful and can replace the hand-written one, while adding: faithful-by-
construction, generalises to any registered reference forward, real per-op
source attribution, finer granularity. The 2.5% prefill HBM gap is the fused
model counting large prefill activation round-trips the hand-written graph
rolled into cheaper hand-tuned costs.

## "Can jaxpr show what XLA fused?" — no; optimized HLO can

Confirmed empirically. The reference layer is **51 jaxpr equations**; after XLA
optimization (`jax.jit(fn).lower(*args).compile().as_text()`) it collapses to
**10 fusions + 3 dots + 3 custom-calls**, and the fusion names tell you what got
fused (`add_rsqrt_fusion` = RMSNorm, `subtract_exponential_fusion` /
`broadcast_divide_fusion` = softmax). So:

- **jaxpr** = pre-fusion (what we derive the graph from now).
- **optimized HLO** = post-fusion ground truth. Our `fuse()` is a cheap static
  approximation of it.

## Limitations & recommended next steps

1. **Source attribution points at the reference forward**, not the real model
   (`descriptors.py:_mimo_v2_flash_reference`). It auto-tracks the reference's
   logical fns (`rms_norm`/`linear`/`layer`) — better than rotting strings, but
   to attribute to *real* model lines we'd trace the real model's forward.
2. **Unsharded.** The traced reference is full-size; tp/ep sharding is applied
   by the descriptor (views B–E), not yet by the auto-graph.
3. **Fusion is a heuristic.** Next iteration: derive the graph from the
   **optimized HLO** (parse `compile().as_text()`), giving the *real* fused op
   set + async-collective (overlap) structure — still pure theory (compile-only,
   no profiling). NB: compile on the **TPU** backend; CPU fusion/custom-calls are
   not representative.
4. Multi-output eqns (`top_k`) expose only their first output in the CPM edges
   (cheap, off the critical path).

## Run

```
python tools/model_roofline.py --model-path <ckpt> --view f
python tools/roofline_jaxpr_validate.py      # the cross-validation above
```
