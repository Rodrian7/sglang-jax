# Roofline parallelism input + validation + correct theoretical communication

**Date:** 2026-06-09  ·  **Branch:** `feat/theoretical-roofline`

## Why

Parallelism degrees (tp/ep/dp) are **deployment choices, not in `config.json`** —
the tool cannot derive them, so they must be inputs, and a wrong/guessed layout
silently produces wrong sharding + communication. The old tool took `--tp`/`--ep`
defaults (8/32) with **no validation** and modelled communication against the
wrong groups. Investigating the real framework (workflow `roofline-comm-model`,
findings verified against the code) exposed a bigger error than just comm.

## The key finding: the mesh is 2D, `--tp` is the mesh total

`scheduler.py:296` builds `create_device_mesh(ici_parallelism=[dp_size, tp_size//dp_size])`.
So the runtime mesh is **`[data=dp, tensor=tp//dp]`, total = `tp_size` = device
count**. Consequences (all verified in code):

- `tp_size` is the **mesh total = #devices**, *not* the tensor-parallel degree.
- the real **tensor-parallel degree** for attention + row/col-parallel linears is
  **`t = tp//dp`** (`model_runner.py:78 attention_tp_size`).
- the fused/fused_v2 MoE **expert-parallel group is the full mesh = `tp_size` =
  devices**; `--ep-size` is **ignored** by the kernel (`scheduler.py:301-312`,
  `fused_moe.py:138` shards experts over `P(('data','tensor'),…)`).
- when `t > num_kv_heads`, KV heads are **replicated** to `t` (1/device), not
  sharded below 1 (`model_config.get_total_num_kv_heads_with_replication`).

For the validated MiMo-V2-Pro baseline `--tp 32 --dp 8` on 32 devices: mesh =
`[data=8, tensor=4]`, so **attention/linears are tensor-parallel over 4, not 32**.
The old tool sharded them by 32 → **under-counted per-device attention/linear
cost by `dp`=8×.**

## What changed

New `parallelism.py`:
- `resolve(config, par)` → `ParallelLayout` + **validation** (raises `ValueError`):
  `devices == tp`; `tp % dp == 0`; `num_attention_heads % (tp//dp) == 0`;
  `n_routed_experts % devices == 0` (fused MoE); warns when `--ep-size != devices`.
- `kv_heads_per_device` (replication-aware); collective-volume helpers
  (`all_reduce_bytes`, `reduce_scatter_bytes`, `all_gather_bytes`).
- `row_parallel_reduce_bytes(tokens, H, lp)` — the correct o_proj / MoE-output
  reduce, **SP-aware**:
  - **DP / SP-below-threshold**: all-reduce over the tensor axis `t` = `2(t-1)/t·msg`.
  - **SP** (`enable_sp` and `should_scatter(tokens) = tokens ≥ devices·128 and
    tokens % devices == 0`): reduce-scatter over the full mesh + residual
    all-gather = `2(D-1)/D·msg`.

`descriptors.py` + `graph.py` now resolve the layout and use `t` for
attention/linear sharding (not `tp`), `ep = devices` for the MoE a2a, KV
replication, and the SP-aware reduce. CLI: `--tp` (=mesh total), `--dp`,
`--enable-sp`, `--moe-backend`; `--devices` defaults to `--tp`; the layout is
validated up-front (`ap.error` on violation) and the EP-override warning printed.

## Verified behaviour (baseline tp=32, dp=8, devices=32, +SP)

- layout: `t=4`, `ep=4·8=32`, header `mesh[data=8 x tensor=4] attn_tp=4 ep=32 +SP`.
- **decode** (batch 64): `should_scatter(64)=False` → o_proj **all-reduce over t=4**
  (1.18 MB/layer); MoE a2a over devices=32 + reduce.
- **prefill** (chunk 16384): `should_scatter=True` → o_proj **reduce-scatter +
  all-gather over 32** (390 MB/layer · 60 SWA = 23.4 GB, matches the model row).
- validation rejects the old impossible default (`devices=32 != tp=8`) and
  `n_routed_experts` not divisible by devices.
- View F cross-check unchanged (FLOPs ×0.9999, HBM ×1.0006); all views B–F OK.

## Caveats (kept honest)

- The collective volumes are **balanced theoretical lower bounds**; real all-to-all
  is imbalance-bound (MEMORY: prefill a2a at the torus floor).
- SP gating uses `tpu_scatter_min_local_size` (default 128); the tool hard-codes
  128 — read the runtime value if it is overridden.
- Only the fused/fused_v2 MoE EP semantics are modelled (the relevant path).
- lm_head SP hidden all-gather is not modelled (≈0; omitted).
