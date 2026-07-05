# MiMo-V2-Flash PD Disaggregation Performance Report, 2026-07-05

## TL;DR

- 本轮小优化是在 decode overlap loop 里，当前 batch `run_batch` launch 后、上一批 `process_batch_result` 前，额外 poll 一次 transfer queue。单测通过，但 C32/C64/C128 回归显示吞吐收益很小，基本在噪声范围内。
- 16K input / 4K output 最好点仍是 C128：client 全程平均 `13.64K total tok/s`，其中 `10.91K input tok/s`、`2.73K output tok/s`；client peak output `3.48K tok/s`。
- 用 server log 看 device 能力：prefill forward-only 估计约 `22.8K input tok/s`（router prefill inflight=4 反推），但包含 transfer/tail 后的实际 active prefill 只有 `12.8K input tok/s`；decode highwater steady 约 `3.18K output tok/s`，max `3.41K output tok/s`。
- 额外做了 `2 prefill : 1 decode` 探索。C64 收益很小（`10.83K total tok/s`，比 1P1D C64 高约 `2.7%`）；C128 提升更明显（`15.31K total tok/s`，比 1P1D C128 高约 `12.2%`，mean TTFT 从 `57.6s` 降到 `20.1s`），但长输出阶段仍由单 decode device 主导。
- AIME24 完整 30 题结果为 `0.7667`（23/30），和 PDF 中的 AIME24 结果一致，没有看到精度异常。

## Tested Code

Remote benchmark code:

- Falcon exp: `exp-5uqgg64144`
- Remote repo: `/tmp/sglang-jax`
- Remote git head: `c6105f1cb09119ce40462d9f65776198a312737b`
- Remote working tree is dirty; exact dirty status is saved in:
  `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/env.json`
- This round added the extra decode transfer poll at [decode.py](/Users/jiongxuan/workspace/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py:301).
- Multi-prefill follow-up fixes:
  - Router aligns injected `bootstrap_room` with the selected prefill index so bootstrap registry selection matches the actual forwarded prefill URL.
  - Decode preserves the Raiden endpoint descriptor's advertised host instead of rewriting it to the bootstrap registry host. This fixed a real `2P1D` failure where decode tried to connect to `10.125.130.4:34189` even though that Raiden endpoint belonged to `10.125.132.39:34189`.
  - Per-chunk `RAIDEN-D start_read*` logs were demoted from warning to debug; this removes hot-path log noise without changing transfer behavior.

Local verification code:

- Local git head: `a7ba33c4a7a07c05a091076994bcc00eaf8668ac`
- Added regression test at [test_pd_overlap_schedule.py](/Users/jiongxuan/workspace/sglang-jax/python/sgl_jax/test/disaggregation/test_pd_overlap_schedule.py:137).
- Verification: `.venv/bin/python -m pytest python/sgl_jax/test/disaggregation/test_pd_overlap_schedule.py python/sgl_jax/test/disaggregation/test_pd_time_stats.py python/sgl_jax/test/disaggregation/test_pd_internal_state.py -q` -> `18 passed in 3.35s`.

## Environment

- Rank 0: PD bootstrap + prefill server
- Rank 1: decode server + PD router + benchmark/eval driver
- Model: `/models/MiMo-V2-Flash`
- Python: `3.12.12`
- Runtime env:

```bash
export TPU_PROCESS_ADDRESSES=localhost:8471
export TPU_WORKER_HOSTNAMES=localhost
export TPU_PROCESS_PORT=8471
export TPU_WORKER_ID=0
export TPU_HOST_BOUNDS=1,1,1
export HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1
export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jit_cache
export LIBTPU_INIT_ARGS=--xla_tpu_dvfs_p_state=7
export PYTHONPATH=/tmp/tpu-raiden-cached/tpu-raiden:/tmp/sglang-jax:${PYTHONPATH:-}
export SGLANG_JAX_USE_RAIDEN=1
```

Precompile cache was enabled through `JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jit_cache`. In this restart, both prefill/decode precompile finished in about 17s after model load.

## Serve Commands

Bootstrap, rank 0:

```bash
/usr/local/bin/python -m sgl_jax.srt.disaggregation.run_bootstrap \
  --host 0.0.0.0 --port 8998
```

Prefill, rank 0:

```bash
/usr/local/bin/python -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash --trust-remote-code \
  --tp-size 8 --ep-size 8 --moe-backend fused_v2 \
  --nnodes 1 --node-rank 0 --page-size 256 --context-length 262144 \
  --disable-radix-cache --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --max-running-requests 256 \
  --dp-size 2 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-enable-d2h --disaggregation-use-raiden \
  --enable-metrics --enable-request-time-stats-logging \
  --host 0.0.0.0 --port 10000 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-url http://localhost:8998 \
  --disaggregation-max-inflight-transfers 8
```

Decode, rank 1:

```bash
/usr/local/bin/python -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash --trust-remote-code \
  --tp-size 8 --ep-size 8 --moe-backend fused_v2 \
  --nnodes 1 --node-rank 0 --page-size 256 --context-length 262144 \
  --disable-radix-cache --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --max-running-requests 256 \
  --dp-size 2 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-enable-d2h --disaggregation-use-raiden \
  --enable-metrics --enable-request-time-stats-logging \
  --host 0.0.0.0 --port 10001 \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-url http://falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local:8998 \
  --disaggregation-max-inflight-transfers 32
```

Router, rank 1:

```bash
/usr/local/bin/python -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local:10000 8998 \
  --decode http://localhost:10001 \
  --prefill-bootstrap-host falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local \
  --max-concurrent-requests 256 \
  --pd-prefill-max-inflight-requests 4 \
  --pd-decode-prealloc-soft-limit 0 \
  --pd-decode-oldest-prealloc-wait-ms-soft-limit 0 \
  --pd-router-admission-poll-ms 50 \
  --host 0.0.0.0 --port 30000
```

## Benchmark Commands

16K/4K throughput:

```bash
for C in 32 64 128; do
  NUM=$((C * 3))
  /usr/local/bin/python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --base-url http://localhost:30000 \
    --model /models/MiMo-V2-Flash \
    --tokenizer /models/MiMo-V2-Flash \
    --dataset-name random \
    --random-input-len 16384 \
    --random-output-len 4096 \
    --random-range-ratio 1.0 \
    --num-prompts "${NUM}" \
    --request-rate inf \
    --max-concurrency "${C}" \
    --warmup-requests 0 \
    --seed 12345 \
    --output-details \
    --extra-request-body '{"sampling_params":{"temperature":0.1,"top_p":0.95,"max_new_tokens":4096,"ignore_eos":true}}' \
    --output-file "/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/bench_c${C}.jsonl"
done
```

AIME24:

```bash
/usr/local/bin/python -m pip install -q evalscope==0.17.1
/usr/local/bin/python -m evalscope.run \
  --model /models/MiMo-V2-Flash \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets aime24 \
  --eval-batch-size 16 \
  --timeout 6000000 \
  --work-dir /aime24_workdir \
  --generation-config '{"temperature":1,"top_p":0.95,"max_tokens":30000,"chat_template_kwargs":{"enable_thinking":true}}'
```

## Throughput Results

Client full-run numbers, 16K/4K:

| Run | C | req/s | input tok/s | output tok/s | client peak output tok/s | total tok/s | mean TTFT ms | p99 TTFT ms | mean ITL ms | p99 ITL ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before extra poll | 32 | 0.39 | 6350 | 1588 | 1984 | 7938 | 10175 | 41913 | 16.68 | 42.24 |
| after extra poll | 32 | 0.38 | 6270 | 1568 | 2016 | 7838 | 10266 | 41479 | 16.96 | 41.19 |
| before extra poll | 64 | 0.52 | 8589 | 2147 | 2655 | 10736 | 16939 | 83534 | 23.55 | 81.01 |
| after extra poll | 64 | 0.51 | 8437 | 2109 | 2653 | 10546 | 16476 | 81917 | 24.13 | 82.07 |
| before extra poll | 128 | 0.66 | 10867 | 2717 | 3491 | 13584 | 61510 | 162593 | 27.91 | 88.66 |
| after extra poll | 128 | 0.67 | 10913 | 2728 | 3483 | 13642 | 57602 | 161325 | 28.73 | 89.26 |

Server-side prefill input capacity:

| Run | C | observed active input tok/s | observed active req/s @16K | forward mean ms | forward-only capacity @4 inflight | prefill total mean ms |
|---|---:|---:|---:|---:|---:|---:|
| before extra poll | 32 | 8322 | 0.508 | 2844 | 23044 tok/s | 4056 |
| after extra poll | 32 | 8150 | 0.497 | 2860 | 22922 tok/s | 4168 |
| before extra poll | 64 | 10736 | 0.655 | 2689 | 24372 tok/s | 3491 |
| after extra poll | 64 | 10556 | 0.644 | 2693 | 24343 tok/s | 3501 |
| before extra poll | 128 | 12684 | 0.774 | 2899 | 22608 tok/s | 4838 |
| after extra poll | 128 | 12788 | 0.780 | 2879 | 22764 tok/s | 4799 |

Interpretation: P-side pure forward capacity is roughly in the `22K-24K input tok/s` band when router keeps 4 prefill requests in flight. The realized active prefill ingress rate is much lower, around `12.8K input tok/s` at C128, because transfer/prealloc/tail still sit on the critical path.

Server-side decode output capacity:

| Run | C | max running | all mean output tok/s | all max output tok/s | highwater steady mean | highwater steady max |
|---|---:|---:|---:|---:|---:|---:|
| before extra poll | 32 | 32 | 1596 | 1983 | 1847 | 1983 |
| after extra poll | 32 | 32 | 1594 | 2016 | 1852 | 2016 |
| before extra poll | 64 | 64 | 2069 | 2648 | 2505 | 2648 |
| after extra poll | 64 | 64 | 2039 | 2607 | 2462 | 2607 |
| before extra poll | 128 | 98 | 2591 | 3374 | 3161 | 3374 |
| after extra poll | 128 | 100 | 2616 | 3414 | 3180 | 3414 |

The decode peak is therefore around `3.2K output tok/s` steady/highwater and `3.4K output tok/s` instantaneous serve-log max. The extra poll does not materially change this.

## Steady Windows

The steady windows below are computed from server logs in the benchmark window:

- Prefill active window: first to last `Prefill batch` line for the concurrency case.
- Decode highwater steady: rows where `running-req >= 0.9 * max_observed_running_req` for that case. For C128, max observed running on decode was 100 rather than 128, so this highwater definition is more useful than the PDF's `0.9 * bs` rule.

| C | prefill active window UTC | prefill active duration s | prefill active input tok/s | decode highwater threshold | decode steady window UTC | decode steady duration s | highwater mean output tok/s | highwater max output tok/s |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 32 | 13:22:56-13:26:09 | 193 | 8150 | >= 29/32 | 13:23:35-13:26:43 | 188 | 1852 | 2016 |
| 64 | 13:27:42-13:32:40 | 298 | 10556 | >= 58/64 | 13:28:59-13:32:58 | 239 | 2462 | 2607 |
| 128 | 13:34:40-13:42:52 | 492 | 12788 | >= 90/100 | 13:37:41-13:43:02 | 321 | 3180 | 3414 |

## Transfer / Time Stats

Mean per-request PD time stats:

| Run | C | P forward ms | P transfer ms | P transfer_tail ms | P total ms | D prealloc_wait ms | D kv_wait ms | D transfer_tail ms | D total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before extra poll | 32 | 2844 | 2534 | 307 | 4056 | 1710 | 2279 | 50 | 3990 |
| after extra poll | 32 | 2860 | 2544 | 308 | 4168 | 1819 | 2284 | 46 | 4103 |
| before extra poll | 64 | 2689 | 2395 | 264 | 3491 | 1270 | 2182 | 48 | 3453 |
| after extra poll | 64 | 2693 | 2401 | 268 | 3501 | 1258 | 2180 | 46 | 3439 |
| before extra poll | 128 | 2899 | 2591 | 329 | 4838 | 2521 | 2320 | 55 | 4842 |
| after extra poll | 128 | 2879 | 2567 | 322 | 4799 | 2497 | 2293 | 45 | 4790 |

Main conclusion:

- Prefill chunk transfer is partially overlapped with forward, but the remaining transfer/tail still reduces realized prefill capacity from about `22.8K input tok/s` forward-only to about `12.8K input tok/s` active end-to-end at C128.
- Decode receive path is still effectively serial at request level: `prealloc_wait + kv_wait ~= total`. For C128 after extra poll, `2497 + 2293 ~= 4790 ms`.
- The extra after-launch poll trims little or no measurable time. It is low-risk, but it is not the bottleneck.

Detailed prefill per-request stage stats after the extra poll:

| C | stage | n | mean ms | p50 ms | p95 ms | p99 ms | max ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | queue | 96 | 997.5 | 1242.8 | 1700.7 | 1915.7 | 2509.7 |
| 32 | forward | 96 | 2859.7 | 2901.4 | 3039.2 | 3051.6 | 3057.5 |
| 32 | chunk_register_span | 96 | 2234.0 | 2257.1 | 2374.9 | 2398.5 | 2403.5 |
| 32 | sender_done_wait | 96 | 308.2 | 323.8 | 340.1 | 346.1 | 359.6 |
| 32 | transfer_tail | 96 | 308.4 | 324.0 | 340.4 | 346.5 | 359.8 |
| 32 | transfer | 96 | 2543.8 | 2578.4 | 2705.2 | 2715.2 | 2724.6 |
| 32 | total | 96 | 4167.5 | 4442.9 | 5019.7 | 5080.9 | 5675.6 |
| 64 | queue | 192 | 537.5 | 0.9 | 1692.0 | 1765.3 | 2509.0 |
| 64 | forward | 192 | 2692.9 | 2774.7 | 3030.5 | 3058.0 | 3070.7 |
| 64 | chunk_register_span | 192 | 2130.8 | 2190.5 | 2360.2 | 2395.1 | 2445.9 |
| 64 | sender_done_wait | 192 | 267.9 | 276.2 | 339.5 | 344.2 | 348.3 |
| 64 | transfer_tail | 192 | 268.4 | 276.7 | 340.0 | 344.7 | 348.8 |
| 64 | transfer | 192 | 2400.5 | 2461.6 | 2695.5 | 2721.5 | 2748.6 |
| 64 | total | 192 | 3500.7 | 3061.5 | 5029.7 | 5056.1 | 5720.4 |
| 128 | queue | 384 | 1595.6 | 1625.4 | 1699.4 | 1743.3 | 2422.0 |
| 128 | forward | 384 | 2879.0 | 2920.1 | 3032.1 | 3047.8 | 3140.1 |
| 128 | chunk_register_span | 384 | 2243.6 | 2273.2 | 2366.8 | 2419.5 | 2510.2 |
| 128 | sender_done_wait | 384 | 320.5 | 326.1 | 342.5 | 346.5 | 361.7 |
| 128 | transfer_tail | 384 | 322.3 | 327.5 | 344.2 | 348.3 | 363.3 |
| 128 | transfer | 384 | 2567.1 | 2597.9 | 2703.1 | 2749.8 | 2835.3 |
| 128 | total | 384 | 4798.8 | 4860.0 | 5031.4 | 5072.2 | 5669.1 |

Detailed decode per-request stage stats after the extra poll:

| C | stage | n | mean ms | p50 ms | p95 ms | p99 ms | max ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | metadata_wait | 96 | 1818.8 | 2002.3 | 2627.6 | 2772.2 | 3384.4 |
| 32 | prealloc_wait | 96 | 1819.2 | 2002.6 | 2628.0 | 2772.4 | 3384.7 |
| 32 | first_chunk_wait | 96 | 4.6 | 5.0 | 6.6 | 8.0 | 9.3 |
| 32 | chunk_start_span | 96 | 2230.4 | 2251.9 | 2368.6 | 2399.6 | 2405.2 |
| 32 | transfer_tail | 96 | 45.8 | 44.8 | 58.5 | 60.6 | 62.3 |
| 32 | kv_wait | 96 | 2283.6 | 2310.4 | 2422.7 | 2454.6 | 2454.7 |
| 32 | enqueue_decode | 96 | 2.7 | 2.7 | 3.4 | 3.8 | 4.0 |
| 32 | total | 96 | 4103.5 | 4320.9 | 4953.6 | 5079.1 | 5687.2 |
| 64 | metadata_wait | 192 | 1257.9 | 733.2 | 2649.0 | 2719.0 | 3377.2 |
| 64 | prealloc_wait | 192 | 1258.3 | 734.0 | 2649.3 | 2719.3 | 3377.5 |
| 64 | first_chunk_wait | 192 | 3.9 | 3.5 | 6.1 | 7.1 | 7.8 |
| 64 | chunk_start_span | 192 | 2127.1 | 2188.3 | 2355.5 | 2387.4 | 2442.8 |
| 64 | transfer_tail | 192 | 46.3 | 47.2 | 58.3 | 66.2 | 72.1 |
| 64 | kv_wait | 192 | 2180.1 | 2239.2 | 2412.6 | 2443.5 | 2500.7 |
| 64 | enqueue_decode | 192 | 2.8 | 3.0 | 3.8 | 4.3 | 4.4 |
| 64 | total | 192 | 3439.2 | 3007.4 | 5039.4 | 5081.8 | 5682.9 |
| 128 | metadata_wait | 384 | 2496.7 | 2537.9 | 2643.2 | 2705.3 | 3238.2 |
| 128 | prealloc_wait | 384 | 2497.0 | 2538.2 | 2643.5 | 2705.6 | 3238.7 |
| 128 | first_chunk_wait | 384 | 6.2 | 6.1 | 7.5 | 9.1 | 10.7 |
| 128 | chunk_start_span | 384 | 2237.3 | 2271.2 | 2361.4 | 2414.7 | 2501.3 |
| 128 | transfer_tail | 384 | 45.2 | 44.4 | 55.1 | 59.6 | 60.9 |
| 128 | kv_wait | 384 | 2292.5 | 2324.7 | 2418.8 | 2466.6 | 2565.5 |
| 128 | enqueue_decode | 384 | 3.8 | 3.8 | 4.5 | 4.9 | 5.4 |
| 128 | total | 384 | 4790.2 | 4851.1 | 5029.1 | 5088.0 | 5551.3 |

## 2P1D Multi-Prefill Probe

This run used two prefill TPU hosts feeding one decode TPU host:

- Original prefill: `exp-5uqgg64144` rank 0, `10.125.130.4`.
- Extra prefill: `exp-ahgyl3g479` rank 0, `10.125.132.39`.
- Decode/router/driver: `exp-5uqgg64144` rank 1, `10.125.129.4`.
- Router used both prefill URLs and one decode URL, with the same `--pd-prefill-max-inflight-requests 4`.
- Raw run id: `pd_2p1d_16k_4k_bench_1783270151`.

Extra prefill serve command is the same as the rank0 prefill command above except:

```bash
--disaggregation-bootstrap-url http://falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local:8998
```

2P1D router command:

```bash
/usr/local/bin/python -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local:10000 8998 \
  --prefill http://10.125.132.39:10000 8998 \
  --decode http://localhost:10001 \
  --prefill-bootstrap-host falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local \
  --max-concurrent-requests 256 \
  --pd-prefill-max-inflight-requests 4 \
  --pd-decode-prealloc-soft-limit 0 \
  --pd-decode-oldest-prealloc-wait-ms-soft-limit 0 \
  --pd-router-admission-poll-ms 50 \
  --host 0.0.0.0 --port 30000
```

Client results, 16K/4K:

| Mode | C | success | req/s | input tok/s | output tok/s | peak output tok/s | total tok/s | mean TTFT ms | p99 TTFT ms | mean ITL ms | p99 ITL ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PD 1P1D | 64 | 192/192 | 0.51 | 8437 | 2109 | 2653 | 10546 | 16476 | 81917 | 24.13 | 82.07 |
| PD 2P1D | 64 | 192/192 | 0.53 | 8663 | 2166 | 2844 | 10829 | 11488 | 47663 | 25.84 | 79.60 |
| PD 1P1D | 128 | 384/384 | 0.67 | 10913 | 2728 | 3483 | 13642 | 57602 | 161325 | 28.73 | 89.26 |
| PD 2P1D | 128 | 384/384 | 0.75 | 12246 | 3061 | 3968 | 15307 | 20114 | 91477 | 35.08 | 128.74 |

2P1D per-request stage stats:

| C | role | n | forward mean ms | transfer / kv_wait mean ms | total mean ms | p50 total ms | p95 total ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | prefill | 192 | 1649 | 2574 transfer | 4323 | 4875 | 5243 |
| 64 | decode | 192 | n/a | 2321 kv_wait | 4279 | 4853 | 5233 |
| 128 | prefill | 384 | 1512 | 2564 transfer | 4407 | 4844 | 5272 |
| 128 | decode | 384 | n/a | 2313 kv_wait | 4369 | 4838 | 5273 |

Important observations:

- `2P1D` improves C128 total throughput by about `12.2%` over 1P1D (`15.31K / 13.64K`) and reduces mean TTFT by about `65%` (`57.6s -> 20.1s`). It helps because prefill queueing pressure is split across two P hosts.
- C64 barely improves (`10.83K / 10.55K`, about `2.7%`), so one prefill is already close enough for this concurrency.
- Decode is still the limiting role for long 4K output. C128 output throughput improves to `3.06K tok/s` with a `3.97K tok/s` client peak, but ITL worsens (`28.73ms -> 35.08ms`) because one decode host is carrying a larger effective running set.
- Transfer itself did not regress with remote extra prefill: decode `kv_wait` stays around `2.31s`; prefill transfer stays around `2.56s`.

## AIME24

| Dataset | Num | Score | Correct |
|---|---:|---:|---:|
| AIME24 / `HuggingFaceH4/aime_2024` | 30 | 0.7667 | 23 |

This matches the PDF number (`0.7667`) under the non-greedy reasoning config.

## Raw Artifacts

Local artifacts:

- Parsed summary: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/parsed_summary.json`
- 2P1D parsed summary: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/parsed_summary.json`
- Rank 1 raw tar: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/rank1.tar.gz`
- Rank 0 raw tar: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/rank0.tar.gz`
- Benchmark logs:
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/raw/bench_c32.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/raw/bench_c64.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/raw/bench_c128.log`
- Server logs:
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/prefill_extra_poll_server.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/decode_extra_poll_server.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/router_extra_poll.log`
- AIME24:
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/raw/aime24_eval.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/aime24_workdir/20260705_135200/reports/MiMo-V2-Flash/aime24.json`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/aime24_workdir/20260705_135200/predictions/MiMo-V2-Flash/aime24_default.jsonl`
- 2P1D:
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/pd_2p1d_16k_4k_bench_1783270151/raw/bench_c64.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/pd_2p1d_16k_4k_bench_1783270151/raw/bench_c128.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/pd_2p1d_16k_4k_fixed_1783269610/raw/decode_server.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/rank0_prefill_logs/prefill_extra_poll_server.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/extra_prefill_2p1d_1783268093/raw/extra_prefill_server.log`

## Next Optimization Direction

1. Transfer path should be the next target, not router admission. The useful gap is between forward-only `~22.8K input tok/s` and realized active prefill `~12.8K input tok/s`.
2. Decode receiver needs true overlap between metadata/prealloc and kv transfer wait. Today `prealloc_wait + kv_wait` almost equals total, so it is still serial.
3. Prefill sender tail is still visible: C128 `transfer_tail` is about `322 ms`, and transfer span is about `2.57 s`. Reducing bootstrap/register polling and Raiden done-sending tail should directly improve prefill realized bandwidth.
4. 2P1D is worth keeping for high-concurrency production-like loads, but the current best cost/perf point is workload-dependent: C64 has negligible gain, C128 has clear TTFT and throughput gain.
5. The next code experiment should be larger-grain host/device scheduling overlap: make transfer discovery/progress independent of the decode event-loop tick, or pipeline the next request's transfer setup while the current decode forward is in flight.
