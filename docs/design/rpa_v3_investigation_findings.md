# RPA v3 Investigation Findings (2026-05-28)

Triggered by a v6e CI regression: the `pallas-kernel-benchmark` perf
guard for `('decode', 128, 128, 4, 1, 128, 600000)` measured 1.093ms
against the 0.640ms threshold (~70% slower). What started as a
"tune block sizes for v6e" task surfaced a set of deeper structural
issues across the kernel, the tuner, the bench, and the production
attention path. This doc records what we learned so the next person
who touches RPA v3 doesn't have to rediscover it.

All measurements below come from a single v6e probe pod
(`exp-4iv0vyoeke`, `tpuv6e-256-node`, single VM `device_count=4
device_topo=2x2 replica=1`) and an existing MiMo production trace at
`/tmp/profile-analysis/baseline-bs64-osl1/` (rank0 v7x-32, MoE).

## TL;DR

| Claim before this work | What we actually found |
|---|---|
| "vmem capacity bump (`//2` → full) is the CI breakage" | Yes — heuristic `bkv` jumped 16384 → 32768 because vmem_budget doubled. But that's a symptom; the heuristic was always thin. |
| "csz=sz is the right rule, validated empirically" | Validated only at q=16 kv=2 hd=256 ps=256 mid-kv on v7x. Outside that scope the rule starts to leak. |
| "v7x tuned table is the production-validated baseline we should copy to v6e" | v7x table covers ONE shape × ONE kv distribution. CI bench uses a totally different shape (q=4 kv=1 hd=128 ps=128). The "no production complaints" argument is not validation — it's blind-spot. |
| "kv_len bucketing in the table key is needed for production correctness" | Real but smaller than expected. Kernel already does runtime early-exit at `bkv_csz` granularity via `pl.when`. csz<sz with sz=fixed buys 3-6% on short kv but loses 20-30% on long kv (loop overhead). |
| "tuning block sizes is the path to fixing CI and production performance" | Fixing CI: yes. Fixing production: ~0.16% global ROI. Real production levers are elsewhere (layout coercion, MoE kernel). |

## Investigation chronology

1. **Probe 1 (kv-len winner divergence)** — at fixed shape, swept bkv ∈
   {256..32768} at three prefix-length distributions {[256,512),
   [1024,2048), [16384,32768)}. Winners diverged dramatically (bkv=512
   for short, bkv=1024 for medium, bkv=32768 for long). Concluded
   kv_len bucketing was needed; recommended schema change.
2. **Tier-0 audit** — read `multiple_iteration_timeit_from_trace`,
   traced the magic-number provenance, recovered the full empirical
   record behind the csz=sz removal (`exp-kc4leh9wnt`).
3. **Tier-1.5 xprof probe (`exp-4iv0vyoeke`)** — captured per-config
   xprof traces on v6e with vmem_limit_bytes A/B and winner block
   sizes A/B. Got real device-side timing and HLO breakdown.
4. **Production trace re-analysis** — drilled into MiMo trace's
   `jit(jitted_run_model)/FlashAttention/shard_map/jit(ragged_paged_attention)`
   scope to find the actual production attention bottleneck.

## Key findings

### F1. Probe vs CI numeric mismatch was a `task` regex problem, not a measurement bug

`multiple_iteration_timeit_from_trace(task=...)` matches event names
by regex. CI bench passes
`task=f"RPA{rpa_case.symbol}-p_{ps}-bq_..."` which lands inside the
HLO event name `jit(...)/RPAd-p_128-bq_1_1-bkv_32768_32768/pallas_call`
and so returns kernel-only `device_duration_ps`. The probe used
`task=f"xprof-{tag}"`, which doesn't appear in any HLO event name,
fell back to MARKER scope, and so returned the whole-step time
including `copy.13` and IDLE.

This explains the 2× discrepancy between probe (2.30ms) and CI bench
(1.09ms) for the same shape × heuristic. **Both numbers are
correct**; they just measure different things. CI's number reflects
production behavior; the probe's whole-step number contains
bench-only overheads.

Action: in any future probe / bench script, name the bench step
deliberately so the regex matches the HLO event you want to measure.

### F2. The vmem capacity bump was the proximate trigger of CI red

Before commit `1861ca945` (`fix(rpa-v3): use full VMEM capacity as
default tuner budget`):

- `get_vmem_limit() = vmem_capacity_bytes // 2 = 64 MB` on v6e
- Inside `get_default_block_sizes`,
  `vmem_budget = vmem_limit_bytes * 0.30 = 19.2 MB`
- For the failing shape, the heuristic shrink loop reduced
  `bkv` from 32768 to **16384** (vmem_est at 16384 ≈ 17.9 MB ≤ 19.2 MB)

After the commit:

- `get_vmem_limit() = 128 MB` (Trillium full capacity)
- `vmem_budget = 38.4 MB`
- Initial `bkv = 32768` (vmem_est ≈ 35.7 MB ≤ 38.4 MB) → no shrink → **32768**

The kernel-level cost of this shift, from probe data:

| `bkv` | mid-kv kernel time |
|---:|---:|
| 16384 | ~1.7 ms (interpolated) |
| 32768 | 1.09 ms × ~1.6 = **2.0 ms** if csz also = sz |

Wait — that doesn't match. CI sees 1.09ms with the new heuristic
output `(1, 32768, 1, 32768)`. Re-reading: heuristic shrink loop is
keyed on `bkv_sz`, not the *observed* runtime. The 32768 bkv pays
its real cost in compute waste (95% of the matmul tile is masked).
Pre-bump heuristic landed at 16384 and only paid ~50% waste.

Action: the `0.30` `vmem_budget` factor and the `// 2` capacity
factor are both empirical safety margins copied from the upstream
`tpu-inference` repo without rationale. The heuristic is brittle to
either being changed without re-tuning. **A regression like this is
expected if either constant moves and the table isn't refreshed.**

### F3. The 16 MB / 4 / 0.30 magic numbers are imported, not derived

`min_bkv_sz_to_peak = 16 * 1024 * 1024 * kv_packing // 4 // head_dim // num_kv_heads_x2`
came from commit `4dd1a9ded` ("integrate from upstream
tpu-inference"). No comment in our code explains the 16 MB target
or the `// 4`. Same for the `0.30` `vmem_budget` factor.
Only `MAX_BQ_SZ = 32` has a documented rationale (bf16 rounding
divergence in flash-attention accumulation).

Action: file an issue to either (a) re-derive these constants from
first principles (HBM bandwidth × pipeline-depth target / per-token
KV bytes) or (b) at minimum document them with the upstream source
and note that they are empirical safety knobs.

### F4. v6e VMEM = 128 MB; repo conventions are inconsistent

`pltpu.get_tpu_info().vmem_capacity_bytes = 134,217,728` on v6e
(verified live in `exp-4iv0vyoeke`). This is the full Trillium
spec.

Repo uses different numbers in different kernels:

| File | v6e value |
|---|---|
| `quantized_matmul/.../tuned_block_sizes.py` `DEVICE_VMEM_LIMIT` | 96 MB |
| `simple_gla/simple_gla.py` (default) | 128 MB |
| `multimodal/.../flash_attention.py` (default) | 128 MB |
| `fused_moe/v1/kernel.py` (hardcoded) | 96 MB |
| `ragged_paged_attention_v3.py` `get_vmem_limit()` | full capacity (128) |

96 MB is conservatively safe; 128 MB is the actual limit. There's
no single source of truth in the repo. Action: pick one
authoritative `DEVICE_VMEM_LIMIT` and use it consistently, or at
minimum document why each kernel chose its number.

### F5. The csz-removal commit's empirical evidence was solid in scope, just over-extrapolated

`exp-kc4leh9wnt` (the experiment that justified commit `e5abcaf39`
"simplify tuner to v2-style — no csz dim") **did fully sweep
`(bq, bkv, bq_csz, bkv_csz)` combinations**, not only `csz=sz`. All
five reported decode winners (mnt 32-512) had `csz=sz`. The
conclusion was correct for that scope.

But the scope was: q=16, kv=2, hd=256, ps=256, prefix=[1024,2048),
v7x. Outside that scope the rule starts to leak — Probe-1 found
`(sz=1024, csz=512)` 3.6% faster than `(sz=1024, csz=1024)` for
short kv at q=4 kv=1 hd=128 on v6e. Not a contradiction; just an
under-tested subspace.

Action: when introducing or removing a tuning dimension based on
empirical data, the commit message + table comment should record
the dimensions of the tested grid. If a future shape lies outside
that grid, the dimension may need to come back.

### F6. csz=sz vs csz<sz tradeoff: short kv wins go to csz<sz, long kv wins go to csz=sz

Probe data on v6e q=4 kv=1 hd=128 ps=128 mnt=128 (whole-step times,
divide by ~2 for kernel-only equivalent):

| range (actual_kv) | best `csz=sz` | best `csz=512` |
|---|---:|---:|
| [256, 512) (~384) | sz=1024,csz=1024: 1.21ms | sz=1024,csz=512: **1.16ms** |
| [1024, 2048) (~1536) | sz=1024,csz=1024: 1.35ms | sz=2048,csz=512: 1.34ms (tie) |
| [16384, 32768) (~24576) | sz=32768,csz=32768: **2.90ms** | sz=32768,csz=512: 3.66ms |

Why csz<sz hurts long kv: the inner `pl.loop(0, sz/csz)` runs more
iterations as csz shrinks. Each iteration carries a runtime
`pl.when` predicate, sem-wait, register-state setup. With csz=512
on actual_kv=24576 → 48 iterations of overhead, ~26% wall-clock
penalty over the 1-iteration csz=32768 path.

Why csz<sz helps short kv: the kernel's compute granularity (matmul
tile size) is `bkv_csz`. For short actual_kv with bkv_sz=1024 and
csz=1024, the single tile computes 1024 tokens of matmul but only
~384 are valid → 60% FLOP waste. With csz=512, two iterations of
512-token matmul, the second early-exits via `pl.when`, ~25% waste.

Implication: the kernel **already has runtime early-exit at the
`bkv_csz` granularity**. The "compute on real tokens only" intent
of the v3 design works, but only at csz granularity, not at token
granularity (matmul tile size is static). For a workload-blind
universal csz this means a compromise.

A "hardware constant csz" idea (e.g., csz=512 or 256, just enough
to saturate MXU) was tested in the second probe and **rejected by
data**: csz=512 universal had a 1.32× slowdown vs per-range
optimal on long kv. There is no universal csz that's within 10% of
optimal across short/medium/long actual_kv.

### F7. `copy.13` (614 MB / iter, 0.42 ms) is a bench-only artifact, not a production issue

In the probe trace, the largest per-iteration HLO op was
`copy.13`:

```
%copy.13 = bf16[4688,128,1,2,128]{4,3,2,1,0:T(2,128)(2,1)}
        copy(bf16[4688,128,1,2,128]{4,3,2,1,0:T(2,128)(2,1)} %kvc.1)
```

bytes_accessed = 614 MB, 88% HBM bandwidth utilized for ~424 µs.
Same shape, same layout — a defensive HBM→HBM full kv-cache copy.

Cause: the probe wrapped `ragged_paged_attention` in an outer
`@jax.jit` without `donate_argnums`. The kernel's inner
`donate_argnames` for `kv_cache_fused` cannot take effect when the
outer jit treats the buffer as a regular input, so XLA inserts a
defensive copy before the inner pallas_call.

Production does NOT have this. `model_runner.py:207`:

```python
@partial(jax.jit, donate_argnames=["memory_pools"], ...)
def jitted_run_model(...): ...
```

The outer jit donates `memory_pools` (which contains the kv-cache
pool), and donation propagates through the layer call → shard_map →
inner kernel.

Confirmed in the production trace
(`/tmp/profile-analysis/baseline-bs64-osl1/`): no >50MB copy event
exists. KV cache is NOT being defensively copied per call in
production.

Action: probe / bench measurement scripts that try to reflect
production end-to-end behavior need to add `donate_argnums` to
their outer jit wrapper. CI bench reports kernel-only time and
filters `copy.13` out via the `task` regex (F1), so its number is
already production-representative.

### F8. The real production attention bottleneck is XLA layout coercion (`reshape`), not pallas_call

Production trace HLO breakdown for ragged_paged_attention:

| op | total time | % of attention | per-instance |
|---|---:|---:|---:|
| `reshape` (layout coercion family) | 582 ms | 74% | 77 µs / 50 MB |
| `pad` | 171 ms | 22% | 77 µs / similar |
| `RPAd + RPAm pallas_call` (incl. SWA) | 34 ms | 4% | 45-289 µs |

The biggest reshape, with HLO long_name:

```
%reshape.8239 = bf16[2048,2,3072]{2,1,0:T(2,128)(2,1)}
       reshape(bf16[2048,6144]{1,0:T(8,128)(2,1)} %fusion.76)
```

This is **not a logical-shape reshape** — it's a **physical layout
change** from tile `T(8, 128)` to tile `T(2, 128)`. Tile shape is
how XLA stores tensors in HBM on TPU. Different upstream / downstream
ops want different tiles. The change requires reading the tensor
with one tile layout and writing it with another → HBM-bound copy.

Where it comes from: the upstream Q projection (`linear_base/dot_general`)
emits its result in `T(8, 128)` (matmul-friendly). The downstream
pallas_call (RPA) BlockSpec with `pltpu.HBM` wants `T(2, 128)` for
VMEM-friendly access. XLA inserts the layout coercion. There are
~25 of these reshapes per layer × ~50 forward passes in the trace.

Total ROI calibration on this MoE workload (16-second trace):

| Optimization | global wallclock saved |
|---|---:|
| `fused-moe-v2 kernel` +5% | ~3.2% |
| Eliminate RPA layout reshape | ~3.5% |
| RPA `pallas_call` 4.7× faster (block-tuning ceiling) | ~0.16% |

**Block tuning is well below noise level for production end-to-end
on this workload.** That doesn't mean don't tune (CI is a kernel
regression gate, valid use case). It does mean: don't expect block
tuning to move production latency, and don't pretend it's the
bottleneck.

Action items deferred to issues (see below).

### F9. CI bench measurement scope is narrow

The benchmark filters by HLO instruction name to report kernel-only
time. This is the right thing for kernel regression detection. But
it gives zero signal on:

- Layout coercion overhead inside the kernel's jit boundary (F8)
- KV cache donation breakage (F7) (in our code paths it doesn't break,
  but if someone introduces a donate-killing wrapper it won't show)
- `prepare_inputs` / `prepare_kv_cache_fused` overhead

Result: structural regressions in surrounding ops can ship without
hitting CI.

Action: add a complementary "RPA jit-boundary end-to-end" CI metric
that measures from the outer jit entry to exit, not just the
pallas_call. Provide both numbers per case — kernel-only for kernel
regression, e2e for graph-level regression.

## Recommended follow-up issues

These are the action items, prioritized by ROI:

### I1. P0 — fix v6e CI

Add tuned entries for `TUNED_BLOCK_SIZES_V3["TPU v6e"]` covering
the 30 CI bench cases (`q ∈ {4,8} × kv ∈ {1,2,4} × ps ∈ {128,256} ×
mnt ∈ {128, 256, 1024, 4096}`). Tune in mid-kv distribution
(matching `bench_flashattention.py`'s default `prefix_lens ∈
[1024, 2048)`). `csz=sz`, no kv_len bucket (acceptable for the
fixed CI workload). Document in the table comment:

```
# Tuned for actual_kv ∈ [1024, 2048) on v6e.
# Long-context workloads (>16K kv) will see ~30% suboptimal
# performance — see issue I3.
```

Cost: half a day. CI passes.

### I2. P1 — investigate / mitigate XLA layout coercion (reshape) in production attention

This is the single biggest production-side lever in this MoE
workload (~3.5% global). Two angles:

- **Q projection layout hint**: can `linear_base/dot_general` emit
  its result already in tile `T(2, 128)` so the coercion is unneeded?
- **Pallas BlockSpec**: can `ragged_paged_attention`'s BlockSpec
  for `q` accept `T(8, 128)` directly? The kernel reads through
  VMEM scratch, so HBM tile shape may be flexible.

Either way requires a deeper dive into XLA layout assignment.
Before doing this, run an A/B: compare a forced-layout build
against the current build to confirm the savings are real and
isolated.

### I3. P1 — kv_len bucketing for RPA v3 table

The current table key is empirically valid only for actual_kv in
the bench's tuned range. Long-context production workloads will
see suboptimal perf. Two options:

- **Caller-passes-hint**: add `kv_len_hint: int | None = None`
  parameter to `ragged_paged_attention`. Server passes a value
  derived from its expected workload (e.g., from configured
  max_context_len or live-measured average kv length). Lookup uses
  `next_pow2(kv_len_hint)` as an extra key dimension.
- **Multi-bucket precompile + runtime dispatch**: server precompiles
  multiple block-size variants per shape and dispatches at runtime
  based on `kv_lens.max()` or similar. More complex; gives best
  results for highly variable workloads.

Either requires re-tuning v6e and v7x at multiple kv buckets.

### I4. P1 — eliminate `prepare_kv_cache_fused` redundant pad

`memory_pool.py:54` pads the kv-cache to `head_dim_aligned` at
storage time. `ragged_paged_attention_v3.py:1255` pads it again at
each call. If the second pad is a no-op (already aligned), it's
still in the HLO graph and may be costing layout-fix time. Verify
and remove if redundant.

### I5. P2 — re-derive (or document) the heuristic magic numbers

`min_bkv_sz_to_peak`'s 16 MB and 4 divisors, the 0.30
`vmem_budget` factor — derive from first principles or document
their upstream provenance and the expected operating range. Include
a unit test that fails if any one of these is changed without
re-tuning the table.

### I6. P2 — provenance & staleness for tuned table entries

Each entry in `TUNED_BLOCK_SIZES_V3` should record `(exp_id,
kernel_sha, vmem_config, prefix_range, date)` either inline or in
a sidecar. A maintenance script flags entries whose `kernel_sha`
no longer matches the current kernel as "potentially stale".
Without this, kernel changes silently invalidate tuned values.

### I7. P2 — broaden CI bench measurement scope

Add a second CI metric that measures end-to-end from the outer
jit entry to exit (whole-step including layout coercion, pad,
copy). Continue reporting kernel-only as the regression gate,
but the broader number gives early warning when surrounding-graph
costs creep up.

### I8. P3 — RPA tuning input space coverage matrix

Maintain a `docs/design/rpa_tuning_coverage.md` (or similar) that
lists, per (model × hardware × workload-type), the active
attention shape, expected kv distribution, mnt buckets. Tuning
work targets cells in this matrix. New models on a new shape =
must add a row before going live with that model on that hardware.

## What this investigation does NOT change

- The v6e CI fix path (I1) is straightforward and unblocked.
- `csz=sz` remains the sane default for the table; the csz<sz
  exploration only matters at the boundaries where MXU saturation,
  loop overhead, and tail compute waste trade off (F6).
- The kernel itself (HLO, mosaic, donation, runtime early-exit) is
  fundamentally fine. The issues we found are at the boundary
  layers (heuristic thinness, table coverage gaps, layout coercion,
  bench scope), not in the kernel logic.

## References

- v6e CI failure: `bench_flashattention.py:268-373` test threshold
  table; failing case `('decode', 128, 128, 4, 1, 128, 600000)`,
  baseline 0.582 ms × 1.10 = 0.640 ms threshold, observed 1.093 ms.
- vmem cap commit: `1861ca945 fix(rpa-v3): use full VMEM capacity
  as default tuner budget`.
- csz-removal commit: `e5abcaf39 perf(rpa_v3): simplify tuner to
  v2-style — one vmem knob, no csz dim`.
- Tier-0/Tier-1 probe exp: `exp-4iv0vyoeke` on
  `tpuv6e-256-node`. Trace data at
  `gs://inference-model-storage-poc-tpu-hns/experiments/exp-4iv0vyoeke/artifacts/art-t6t26jgiqc/profiles/`.
- Production trace analyzed:
  `/tmp/profile-analysis/baseline-bs64-osl1/plugins/profile/2026_05_26_12_23_41/`.
- csz=sz removal exp: `exp-kc4leh9wnt` (q=16 kv=2 hd=256 ps=256
  mid-kv only).

Author: investigation conducted 2026-05-28 in worktree
`feat/fused-qkv-mimo-pro` → branch `probe/rpa-v3-kv-len-v6e`.
