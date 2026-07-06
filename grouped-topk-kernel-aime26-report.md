# Grouped-Top-K Kernel A/B — AIME26 Accuracy (Ling-2.6-flash)

**Date:** 2026-07-06
**Branch:** `bench/grouped-topk`
**Question:** Does enabling the Pallas grouped-top-k routing kernel
(`--enable-grouped-topk-kernel`) change AIME26 accuracy for Ling-2.6-flash vs
the default pure-JAX biased grouped-top-k path?

**Answer:** No meaningful difference. The two paths are accuracy-equivalent;
observed pass@k deltas are ±1 question, consistent with `temperature=0.6`
sampling noise on a 30-question benchmark.

---

## 1. Setup

Two `falcon` experiments on **`tpuv7x-64-node`** (v7x, `2x2x1`, 8 chips each),
run **in parallel**, identical except for the single kernel flag.

| Variant | EXP_ID | `enable_grouped_topk_kernel` | Routing path |
|---|---|---|---|
| disabled | `exp-h8r2wk3zr1` | `False` | pure JAX (`_biased_grouped_topk_jax`) |
| enabled  | `exp-cvs4sd0t7c` | `True`  | Pallas `grouped_topk_pallas` |

The kernel only activates for `device=="tpu"` and biased grouped top-k routing.
Ling-2.6-flash (`bailing_moe`, `n_group>0` + `correction_bias`) exercises exactly
that path, so the flag genuinely toggles the routing computation.

### Server (identical on both boxes)

```
python -m sgl_jax.launch_server \
  --model-path /models/Ling-2.6-flash \
  --served-model-name inclusionAI/Ling-2.6-flash \
  --trust-remote-code \
  --tp-size 8 --dp-size 2 --ep-size 8 \
  --moe-backend fused_v2 \
  --device tpu --dtype bfloat16 \
  --page-size 128 --context-length 131072 \
  --chunked-prefill-size 2048 --mem-fraction-static 0.85 \
  --max-running-requests 256 --attention-backend fa \
  --dp-schedule-policy round_robin \
  --disable-radix-cache --skip-server-warmup \
  [--enable-grouped-topk-kernel]   # enabled box only
```

> `--served-model-name inclusionAI/Ling-2.6-flash` is required so it matches
> evalscope's `--model`; otherwise requests 404.

### Evaluation (identical on both boxes)

```
evalscope eval \
  --model inclusionAI/Ling-2.6-flash \
  --api-url http://127.0.0.1:30000/v1 --api-key EMPTY \
  --datasets aime26 \
  --generation-config temperature=0.6,top_p=0.95,max_tokens=32768 \
  --repeats 4 --eval-batch-size 8 \
  --dataset-args '{"aime26": {"aggregation": "mean_and_pass_at_k"}}'
```

AIME26 = 30 questions × 4 repeats = **120 samples**.

### Cold-start timing (per box, from process start to "ready")

| Stage | disabled | enabled |
|---|---|---|
| Weight load (GCS → TPU, JAX lazy loader) | ~6m19s | ~5m18s |
| KV cache allocated | 105.6 GB | 105.6 GB |
| EXTEND precompile | 90s | 112s |
| DECODE precompile | 104s | (same order) |
| **Total to ready** | **~9m53s** | **~10–11m** |
| evalscope wall-clock | 12m16s | 12m41s |

---

## 2. Results — AIME26

| Metric | disabled (JAX) | enabled (Pallas) | Δ |
|---|---|---|---|
| **pass@1** (mean_acc) | 71.67% | **75.83%** | **+4.16%** |
| pass@2 | 84.44% | 82.78% | −1.66% |
| pass@3 | 88.33% | 85.00% | −3.33% |
| **pass@4** (macro_score) | **90.00%** | 86.67% | −3.33% |
| avg output tok/s | 102.83 | 105.72 | +2.89 |
| total input tokens | 20620 | 20620 | 0 |
| total output tokens | 547505 | 577080 | +29575 |

---

## 3. Interpretation

- With only **30 questions**, each pass@k step of **0.0333 = exactly one question**.
  So pass@3/@4 −3.33% = 1 question each; pass@2 −1.66% = one flip in one of the
  4 repeats.
- Deltas go **both directions** (pass@1 +4 pts for the kernel, pass@2/3/4 −1
  question). A one-question, bidirectional wobble at `temperature=0.6` over 30
  questions is **sampling variance, not a systematic kernel regression**.
- Input tokens are byte-identical (20620); output token totals differ ~5% purely
  because different sampled trajectories generate different lengths.

**Conclusion: the Pallas grouped-top-k kernel is accuracy-equivalent to the pure
JAX path. Safe to enable.**

---

## 4. Caveats & follow-ups

- This is an **end-to-end "accuracy does not regress"** check, not a numerical
  equivalence proof. To prove bit-level correctness, rerun with
  `temperature=0` (greedy) — token sequences should then match near-exactly.
- Per-question `predictions/` and `reviews/` are saved under `eval-results/` for
  drilling into which specific questions flipped.

---

## 5. Artifacts & reproduction

Files in repo root on `bench/grouped-topk`:

| File | Purpose |
|---|---|
| `benchmark-kernel-disabled.yaml` | falcon exp, kernel OFF, auto server+eval |
| `benchmark-kernel-enabled.yaml`  | falcon exp, kernel ON, auto server+eval |
| `poll_gtopk_eval.sh` | polls both exps for `DONE`, pulls `eval-out/` down |
| `compare_gtopk_eval.py` | parses both runs, prints the metric-diff table |

```bash
# submit
falcon exp create -f benchmark-kernel-disabled.yaml
falcon exp create -f benchmark-kernel-enabled.yaml
# after both DONE, fetch + compare
python compare_gtopk_eval.py eval-results/disabled eval-results/enabled
```

Raw results: `eval-results/disabled/20260706_030042/`,
`eval-results/enabled/20260706_030255/`.

> Remember to `falcon exp abort exp-h8r2wk3zr1 exp-cvs4sd0t7c` to free the v7x
> boxes (they `sleep infinity` after eval).
