# PD Performance Overnight Handoff, 2026-07-05

## Current Status

Completed overnight items:

- Expanded the PD performance report with 1P1D, non-PD comparison, AIME24, and 2P1D multi-prefill results.
- Added the non-PD baseline document:
  `docs/developer_guide/pd_disaggregation/nonpd_mimo_v2_flash_16k4k_baseline_20260705.md`.
- Fixed multi-prefill routing correctness:
  - Router now aligns `bootstrap_room` to the selected prefill index.
  - Decode now preserves the Raiden endpoint descriptor host instead of rewriting it to the bootstrap registry host.
- Reduced hot-path transfer log noise by demoting per-chunk `RAIDEN-D start_read*` logs from warning to debug.
- Ran 2P1D C64/C128 16K/4K benchmark after the fix.

## Code Changes

Changed files:

- `python/sgl_jax/srt/disaggregation/mini_lb_helpers.py`
  - Added `align_bootstrap_room_to_prefill`.
  - Batched requests now use `room + i * prefill_count` so all items map to the same selected prefill.
- `python/sgl_jax/srt/disaggregation/mini_lb.py`
  - `select_pair()` returns `prefill_index`.
  - Router injects `prefill_index/prefill_count` into bootstrap field generation.
- `python/sgl_jax/srt/disaggregation/decode.py`
  - `_raiden_endpoint_for_dp()` keeps the host from Raiden's advertised endpoint.
  - This fixes the observed bad connection attempt `10.125.130.4:34189` when the actual extra prefill endpoint was `10.125.132.39:34189`.
- `python/sgl_jax/srt/disaggregation/jax_transfer/conn.py`
  - Per-chunk start-read logs are `debug` level.
- `python/sgl_jax/test/disaggregation/test_pd_mini_lb_helpers.py`
  - Adds multi-prefill bootstrap-room alignment tests.
- `python/sgl_jax/test/test_pd_swa_basic.py`
  - Adds endpoint-host preservation test and hot-path logging test.

## Benchmark Summary

16K input / 4K output:

| Mode | C | total tok/s | input tok/s | output tok/s | peak output tok/s | mean TTFT ms | mean ITL ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| non-PD | 128 | 8617 | 6894 | 1723 | 4504 | 83256 | 53.54 |
| PD 1P1D | 128 | 13642 | 10913 | 2728 | 3483 | 57602 | 28.73 |
| PD 2P1D | 128 | 15307 | 12246 | 3061 | 3968 | 20114 | 35.08 |

Main conclusion:

- PD 1P1D already proves the value of separation: C128 total throughput is about `58%` higher than non-PD.
- 2P1D mainly helps high concurrency. C128 total throughput is about `12%` higher than 1P1D and mean TTFT is much lower. C64 only improves about `3%`.
- One decode host remains the long-output limiter. More prefill helps queueing and prefill pressure, but does not fundamentally change decode ITL.
- Per-request transfer cost is stable in 2P1D: decode `kv_wait` is about `2.31s`, prefill `transfer` about `2.56s`.

## Remote State

Main Falcon exp:

- `exp-5uqgg64144`
- rank0: bootstrap + original prefill, IP `10.125.130.4`
- rank1: decode + router + benchmark driver, IP `10.125.129.4`

Extra prefill Falcon exp:

- `exp-ahgyl3g479`
- rank0: extra prefill, IP `10.125.132.39`

At the end of the run, the 2P1D services were left running:

- rank0 original prefill is still registered.
- extra prefill is still heartbeating to rank0 bootstrap.
- rank1 decode/router are running from:
  `/tmp/e2e_logs/pd_2p1d_16k_4k_fixed_1783269610`

Useful health checks:

```bash
falcon exp exec exp-5uqgg64144 --rank 1 -- \
  "curl -sf http://localhost:30000/health"

falcon exp exec exp-5uqgg64144 --rank 0 -- \
  "curl -sf http://localhost:8998/list_prefills"
```

## Raw Artifacts

Main report:

- `docs/developer_guide/pd_disaggregation/pd_mimo_v2_flash_final_perf_report_20260705.md`

Non-PD report:

- `docs/developer_guide/pd_disaggregation/nonpd_mimo_v2_flash_16k4k_baseline_20260705.md`

PD 1P1D raw logs:

- `tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/`

Non-PD raw logs:

- `tmp/e2e_logs/nonpd_16k_4k_1783265639/`

PD 2P1D raw logs:

- `tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/`
- parsed summary: `tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/parsed_summary.json`

## Verification

Local targeted tests run during this work:

```bash
.venv/bin/python -m pytest \
  python/sgl_jax/test/disaggregation/test_pd_mini_lb_helpers.py \
  python/sgl_jax/test/disaggregation/test_pd_router_admission.py -q

.venv/bin/python -m pytest \
  python/sgl_jax/test/test_pd_swa_basic.py \
  python/sgl_jax/test/disaggregation/test_pd_overlap_schedule.py \
  python/sgl_jax/test/disaggregation/test_pd_time_stats.py \
  python/sgl_jax/test/disaggregation/test_pd_internal_state.py -q
```

The endpoint-host regression test was first run before the fix and failed as expected, then passed after the decode change.

## Next Work

Recommended next steps:

1. Keep focusing on transfer and host/device scheduling overlap. Router admission is not the important remaining cost.
2. Investigate moving transfer discovery/progress off the decode event-loop tick so decode forward and incoming KV progress overlap more cleanly.
3. Evaluate 2P1D only for high concurrency or production-like burst patterns. It is useful at C128, but not very useful at C64.
4. Treat precompile cache as a startup optimization, not a runtime throughput issue. Current precompile itself is about tens of seconds after model load; model load/layout conversion dominates restart time.
