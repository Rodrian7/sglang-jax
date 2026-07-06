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

2026-07-06 follow-up:

- Added pod-count-fair non-PD C64 comparison: two ordinary non-PD servers,
  one per Falcon rank, behind a thin streaming round-robin proxy.
- Reran AIME24 on that two-host non-PD endpoint.

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
| non-PD serve-level DP | 64 | 11700 | 9360 | 2340 | 3884 | 13675 | 23.89 |
| PD 1P1D | 128 | 13642 | 10913 | 2728 | 3483 | 57602 | 28.73 |
| PD 2P1D | 128 | 15307 | 12246 | 3061 | 3968 | 20114 | 35.08 |

Main conclusion:

- The original single-host non-PD baseline proves that same-device prefill/decode contention is real, but it is not pod-count fair.
- The 2026-07-06 two-host non-PD C64 run is stronger than PD at C64: `11.70K total tok/s` vs PD 1P1D `10.55K` and PD 2P1D `10.83K`.
- PD 1P1D still improves C128 total throughput by about `58%` over the single-host non-PD C128 baseline. A fair two-host non-PD C128 run has not yet been done.
- 2P1D mainly helps high concurrency. C128 total throughput is about `12%` higher than 1P1D and mean TTFT is much lower. C64 only improves about `3%`.
- One decode host remains the long-output limiter. More prefill helps queueing and prefill pressure, but does not fundamentally change decode ITL.
- Per-request transfer cost is stable in 2P1D: decode `kv_wait` is about `2.31s`, prefill `transfer` about `2.56s`.
- AIME24 follow-up on non-PD serve-level DP got `0.8667` (26/30). Previous PD endpoint run got `0.7667` (23/30). With `temperature=1`, treat this as sampling variance rather than evidence of a precision regression.

## Remote State

Main Falcon exp:

- `exp-5uqgg64144`
- rank0: currently non-PD server on port `30010`, IP `10.125.130.4`
- rank1: currently non-PD server on port `30010` + proxy on port `30000`, IP `10.125.129.4`

Extra prefill Falcon exp:

- `exp-ahgyl3g479`
- rank0: extra prefill, IP `10.125.132.39`

After the 2026-07-06 follow-up, the main Falcon exp is no longer running the
PD services. It is running the two-host non-PD follow-up topology:

- rank0: non-PD server on port `30010`.
- rank1: non-PD server on port `30010`.
- rank1: round-robin proxy on port `30000`.
- Run dir: `/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840`.

The extra prefill Falcon exp `exp-ahgyl3g479` was not used for this follow-up.
Its old prefill process may still be running, but the main rank0 bootstrap was
stopped when switching the main exp to non-PD.

Useful health checks:

```bash
falcon exp exec exp-5uqgg64144 --rank 1 -- \
  "curl -sf http://localhost:30000/health"

falcon exp exec exp-5uqgg64144 --rank 0 -- \
  "curl -sf http://localhost:30010/health"
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

Non-PD two-host C64/AIME24 raw logs:

- `tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/`
- parsed summary: `tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_parsed_summary.json`

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
3. Add a fair two-host non-PD C128 run before making strong C128 pod-count claims.
4. Evaluate 2P1D only for high concurrency or production-like burst patterns. It is useful at C128, but not very useful at C64.
4. Treat precompile cache as a startup optimization, not a runtime throughput issue. Current precompile itself is about tens of seconds after model load; model load/layout conversion dominates restart time.
