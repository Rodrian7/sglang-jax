# MiMo-V2-Flash Non-PD 16K/4K Baseline, 2026-07-05

## Summary

The original 2026-07-05 run used one v7x-8 host as a normal non-PD server.
That is useful as a single-host baseline, but it is not a pod-count-fair
comparison against PD 1P1D.

The 2026-07-06 follow-up adds a fairer C64 comparison: two normal non-PD
servers, one on each Falcon rank, behind a thin streaming round-robin proxy.
This is equivalent to doing data parallelism at the serve layer instead of P/D
disaggregation.

Main result:

- Best non-PD client throughput is C128: `8.62K total tok/s`, with `6.89K input tok/s` and `1.72K output tok/s`.
- PD 1P1D C128 is `13.64K total tok/s`, with `10.91K input tok/s` and `2.73K output tok/s`.
- Non-PD C128 decode serve-log highwater is strong: `3.87K output tok/s` mean, `4.10K max`. The end-to-end loss is not decode kernel weakness; it is same-device prefill/decode contention and long prefill queueing.
- Non-PD C128 had `383/384` successful requests. The failed request is included in the raw benchmark log and makes this point slightly noisy, but it does not change the conclusion.
- The two-host non-PD C64 follow-up reached `11.70K total tok/s`, `9.36K input tok/s`, and `2.34K output tok/s`. This is higher than PD 1P1D C64 (`10.55K total tok/s`) and PD 2P1D C64 (`10.83K total tok/s`).
- Therefore, the fair C64 conclusion is: serve-level DP is better than PD at this concurrency. PD's advantage should be argued from higher pressure, role isolation, and separate prefill/decode capacity rather than from the original single-host C64 comparison.

## Tested Code / Environment

- Falcon exp: `exp-5uqgg64144`, rank 1.
- Remote repo: `/tmp/sglang-jax`.
- Remote run dir: `/tmp/e2e_logs/nonpd_16k_4k_1783265639`.
- Local raw archive: `tmp/e2e_logs/nonpd_16k_4k_1783265639/nonpd_16k_4k_1783265639.tar.gz`.
- Parsed summary: `tmp/e2e_logs/nonpd_16k_4k_1783265639/parsed_summary.json`.
- Model: `/models/MiMo-V2-Flash`.
- JAX compilation cache: `/tmp/tpu_logs/jit_cache`.

## Serve Command

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
  --enable-metrics --enable-request-time-stats-logging \
  --host 0.0.0.0 --port 30000
```

## Benchmark Command

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
    --output-file "/tmp/e2e_logs/nonpd_16k_4k_1783265639/bench_c${C}.jsonl"
done
```

## Client Results

| C | success | req/s | input tok/s | output tok/s | peak output tok/s | total tok/s | mean TTFT ms | p99 TTFT ms | mean ITL ms | p99 ITL ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 96/96 | 0.29 | 4737 | 1184 | 1952 | 5922 | 22086 | 42060 | 21.63 | 18.15 |
| 64 | 192/192 | 0.35 | 5685 | 1421 | 2688 | 7106 | 42904 | 83541 | 34.55 | 41.88 |
| 128 | 383/384 | 0.42 | 6894 | 1723 | 4504 | 8617 | 83256 | 162879 | 53.54 | 317.23 |

## Serve-Log Steady Results

| C | prefill active window UTC | prefill span s | prefill active input tok/s | prefill max queue | decode highwater window UTC | decode highwater mean tok/s | decode highwater max tok/s |
|---:|---|---:|---:|---:|---|---:|---:|
| 32 | 15:39:16-15:43:39 | 263 | 5980 | 30 | 15:39:59-15:44:48 | 1880 | 1942 |
| 64 | 15:45:26-15:52:58 | 452 | 6960 | 62 | 15:46:51-15:54:39 | 2576 | 2687 |
| 128 | 15:55:26-16:08:17 | 771 | 8160 | 124 | 15:58:14-16:10:31 | 3872 | 4103 |

## PD vs Non-PD

| Mode | Hosts | C | client total tok/s | client input tok/s | client output tok/s | serve prefill active input tok/s | serve decode highwater output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| non-PD single server | 1 | 32 | 5922 | 4737 | 1184 | 5980 | 1880 |
| PD 1P1D | 2 | 32 | 7838 | 6270 | 1568 | 8150 | 1852 |
| non-PD single server | 1 | 64 | 7106 | 5685 | 1421 | 6960 | 2576 |
| non-PD serve-level DP | 2 | 64 | 11700 | 9360 | 2340 | 11893 | 3761 |
| PD 1P1D | 2 | 64 | 10546 | 8437 | 2109 | 10556 | 2462 |
| PD 2P1D | 3 | 64 | 10829 | 8663 | 2166 | n/a | n/a |
| non-PD single server | 1 | 128 | 8617 | 6894 | 1723 | 8160 | 3872 |
| PD 1P1D | 2 | 128 | 13642 | 10913 | 2728 | 12788 | 3180 |
| PD 2P1D | 3 | 128 | 15307 | 12246 | 3061 | n/a | n/a |

Interpretation:

- The original single-host non-PD baseline is not a fair pod-count comparison against PD.
- At C64 with two hosts, serve-level DP is about `10.9%` higher than PD 1P1D by client total tok/s (`11.70K / 10.55K`) and about `8.0%` higher than PD 2P1D C64 (`11.70K / 10.83K`), while also showing lower mean TTFT (`13.7s` vs PD 1P1D `16.5s`).
- This makes sense: at C64, two full non-PD replicas split the burst and keep KV local, avoiding PD transfer overhead.
- PD still has a clear role-isolation story at higher pressure. The PD 1P1D C128 result beats single-host non-PD C128 by about `58%`, and 2P1D further improves high-concurrency TTFT/throughput. A fair two-host non-PD C128 was not run in this follow-up.

## Two-Host Serve-Level DP C64 Follow-Up

Run id:

```text
nonpd_2host_c64_aime24_1783295840
```

Topology:

- Falcon exp: `exp-5uqgg64144`.
- Rank 0: normal non-PD server, `http://rank0:30010`.
- Rank 1: normal non-PD server, `http://localhost:30010`.
- Rank 1 proxy: `http://localhost:30000`, thin streaming round-robin proxy.

The proxy performed no admission or scheduling optimization beyond round-robin
request distribution. After the C64 benchmark it reported `196` total forwarded
requests, `98` per backend. That includes the 192 benchmark requests plus
metadata/health requests.

Server command on both ranks:

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
  --enable-metrics --enable-request-time-stats-logging \
  --host 0.0.0.0 --port 30010
```

Benchmark command:

```bash
/usr/local/bin/python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --base-url http://127.0.0.1:30000 \
  --model /models/MiMo-V2-Flash \
  --tokenizer /models/MiMo-V2-Flash \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 4096 \
  --random-range-ratio 1.0 \
  --num-prompts 192 \
  --request-rate inf \
  --max-concurrency 64 \
  --warmup-requests 0 \
  --seed 12345 \
  --output-details \
  --extra-request-body '{"sampling_params":{"temperature":0.1,"top_p":0.95,"max_new_tokens":4096,"ignore_eos":true}}' \
  --output-file /tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/bench_c64.jsonl
```

Client result:

| C | success | duration s | req/s | input tok/s | output tok/s | peak output tok/s | total tok/s | mean TTFT ms | p99 TTFT ms | mean ITL ms | p99 ITL ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 192/192 | 336.08 | 0.57 | 9360 | 2340 | 3884 | 11700 | 13675 | 40857 | 23.89 | 171.86 |

Serve-log summary for the C64 window `00:06:13-00:12:08 UTC`:

| Rank | prefill window UTC | prefill span s | prefill input tok/s | prefill max queue | decode high-load mean tok/s | decode max tok/s | max running |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 00:06:29-00:10:53 | 264 | 5958 | 30 | 1879 | 1970 | 40 |
| 1 | 00:06:29-00:10:54 | 265 | 5935 | 30 | 1882 | 1963 | 48 |
| combined | n/a | n/a | 11893 | n/a | 3761 | 3933 | n/a |

## AIME24 Follow-Up

The same two-host non-PD serve-level DP endpoint was used for an AIME24 rerun:

```bash
/usr/local/bin/python -m evalscope.run \
  --model /models/MiMo-V2-Flash \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets aime24 \
  --eval-batch-size 16 \
  --timeout 6000000 \
  --work-dir /tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/aime24_workdir \
  --generation-config '{"temperature":1,"top_p":0.95,"max_tokens":30000,"chat_template_kwargs":{"enable_thinking":true}}'
```

Result:

| Endpoint | Dataset | Num | Score | Correct |
|---|---|---:|---:|---:|
| non-PD serve-level DP | AIME24 / `HuggingFaceH4/aime_2024` | 30 | 0.8667 | 26 |

The earlier PD run got `0.7667` (23/30) with the same non-greedy generation
shape. Because the config uses `temperature=1`, this difference should be
treated as sampling variance unless a deterministic accuracy protocol is added.
Both results are within a plausible band and do not suggest a precision bug.

## Raw Logs

- Client logs: `raw/bench_c32.log`, `raw/bench_c64.log`, `raw/bench_c128.log`.
- Server log: `raw/nonpd_server.log`.
- JSONL details: `bench_c32.jsonl`, `bench_c64.jsonl`, `bench_c128.jsonl`.
- Window markers: `c32.window`, `c64.window`, `c128.window`.
- Two-host C64/AIME24 local artifacts:
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_aime24_1783295840_rank0.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_aime24_1783295840_rank1.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/rank1_extract/nonpd_2host_c64_aime24_1783295840/raw/bench_c64.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/rank1_extract/nonpd_2host_c64_aime24_1783295840/aime24_workdir/20260706_001210/reports/MiMo-V2-Flash/aime24.json`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_parsed_summary.json`
