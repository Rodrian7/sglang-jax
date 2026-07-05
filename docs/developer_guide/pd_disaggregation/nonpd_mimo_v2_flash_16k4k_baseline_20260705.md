# MiMo-V2-Flash Non-PD 16K/4K Baseline, 2026-07-05

## Summary

This run uses one v7x-8 host as a normal non-PD server. It is the direct
comparison point for the PD 1P1D 16K input / 4K output results in
`pd_mimo_v2_flash_final_perf_report_20260705.md`.

Main result:

- Best non-PD client throughput is C128: `8.62K total tok/s`, with `6.89K input tok/s` and `1.72K output tok/s`.
- PD 1P1D C128 is `13.64K total tok/s`, with `10.91K input tok/s` and `2.73K output tok/s`.
- Non-PD C128 decode serve-log highwater is strong: `3.87K output tok/s` mean, `4.10K max`. The end-to-end loss is not decode kernel weakness; it is same-device prefill/decode contention and long prefill queueing.
- Non-PD C128 had `383/384` successful requests. The failed request is included in the raw benchmark log and makes this point slightly noisy, but it does not change the conclusion.

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

| Mode | C | client total tok/s | client input tok/s | client output tok/s | serve prefill active input tok/s | serve decode highwater output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| non-PD | 32 | 5922 | 4737 | 1184 | 5980 | 1880 |
| PD 1P1D | 32 | 7838 | 6270 | 1568 | 8150 | 1852 |
| non-PD | 64 | 7106 | 5685 | 1421 | 6960 | 2576 |
| PD 1P1D | 64 | 10546 | 8437 | 2109 | 10556 | 2462 |
| non-PD | 128 | 8617 | 6894 | 1723 | 8160 | 3872 |
| PD 1P1D | 128 | 13642 | 10913 | 2728 | 12788 | 3180 |

Interpretation:

- PD improves C128 total throughput by about `58%` over non-PD (`13.64K / 8.62K`).
- Non-PD decode can hit higher serve-log decode throughput at C128 because the single server lets decode accumulate a large running batch, but the same device is also spending a long span on prefill. Client output throughput is still much lower than PD.
- PD should therefore be evaluated by device-role peak capacity: prefill input processing on P and output token generation on D, not only client TTFT.

## Raw Logs

- Client logs: `raw/bench_c32.log`, `raw/bench_c64.log`, `raw/bench_c128.log`.
- Server log: `raw/nonpd_server.log`.
- JSONL details: `bench_c32.jsonl`, `bench_c64.jsonl`, `bench_c128.jsonl`.
- Window markers: `c32.window`, `c64.window`, `c128.window`.
