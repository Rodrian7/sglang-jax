# MiMo-V2-Flash PD Disaggregation Performance Report, 2026-07-05

## TL;DR

- 本轮小优化是在 decode overlap loop 里，当前 batch `run_batch` launch 后、上一批 `process_batch_result` 前，额外 poll 一次 transfer queue。单测通过，但 C32/C64/C128 回归显示吞吐收益很小，基本在噪声范围内。
- 16K input / 4K output 最好点仍是 C128：client 全程平均 `13.64K total tok/s`，其中 `10.91K input tok/s`、`2.73K output tok/s`；client peak output `3.48K tok/s`。
- 用 server log 看 device 能力：prefill forward-only 估计约 `22.8K input tok/s`（router prefill inflight=4 反推），但包含 transfer/tail 后的实际 active prefill 只有 `12.8K input tok/s`；decode highwater steady 约 `3.18K output tok/s`，max `3.41K output tok/s`。
- AIME24 完整 30 题结果为 `0.7667`（23/30），和 PDF 中的 AIME24 结果一致，没有看到精度异常。

## Tested Code

Remote benchmark code:

- Falcon exp: `exp-5uqgg64144`
- Remote repo: `/tmp/sglang-jax`
- Remote git head: `c6105f1cb09119ce40462d9f65776198a312737b`
- Remote working tree is dirty; exact dirty status is saved in:
  `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/env.json`
- This round added the extra decode transfer poll at [decode.py](/Users/jiongxuan/workspace/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py:301).

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

## AIME24

| Dataset | Num | Score | Correct |
|---|---:|---:|---:|
| AIME24 / `HuggingFaceH4/aime_2024` | 30 | 0.7667 | 23 |

This matches the PDF number (`0.7667`) under the non-greedy reasoning config.

## Raw Artifacts

Local artifacts:

- Parsed summary: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/parsed_summary.json`
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

## Next Optimization Direction

1. Transfer path should be the next target, not router admission. The useful gap is between forward-only `~22.8K input tok/s` and realized active prefill `~12.8K input tok/s`.
2. Decode receiver needs true overlap between metadata/prealloc and kv transfer wait. Today `prealloc_wait + kv_wait` almost equals total, so it is still serial.
3. Prefill sender tail is still visible: C128 `transfer_tail` is about `322 ms`, and transfer span is about `2.57 s`. Reducing bootstrap/register polling and Raiden done-sending tail should directly improve prefill realized bandwidth.
4. The next code experiment should be larger-grain host/device scheduling overlap: make transfer discovery/progress independent of the decode event-loop tick, or pipeline the next request's transfer setup while the current decode forward is in flight.
