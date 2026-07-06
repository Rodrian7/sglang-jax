# PD Steady-State Advantage Notes, 2026-07-06

## Purpose

This note isolates a benchmark view where PD disaggregation looks better than
serve-level DP: high-pressure `16K input / 4K output / C128`.

The main point is to avoid reading client TTFT as pure device capability.
Client TTFT includes the benchmark's burst submission, router/proxy wait,
server queues, and final drain effects. For runtime capacity, use:

- Prefill active input tok/s from `Prefill batch` lines.
- Decode highwater output tok/s from `Decode batch` lines.
- PD serve-internal handoff time from `PD-TIME-STATS`.

## Data Sources

Local raw artifacts:

- PD 1P1D: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/`
- PD 2P1D: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/`
- non-PD two-host serve-level DP C128: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/`

Code version for the latest committed local report:

```text
10cce94737d15b778d9f016af3b55ca6b8cc2024
```

Remote benchmark code for the original PD 1P1D run was:

```text
c6105f1cb09119ce40462d9f65776198a312737b
```

## Steady-State Definition

For these notes:

- **Prefill active window**: first to last `Prefill batch` line inside the
  benchmark window.
- **Decode highwater window**: `Decode batch` rows where
  `running-req >= 0.9 * max_observed_running_req`.

This differs slightly from the PDF-style rule `running-req >= 0.9 * C`. The
adjusted definition is more useful for runs where C128 never reaches 128 active
decode requests.

## Best PD C128 Numbers

`16K/4K C128`, random dataset, request rate `inf`, `num_prompts=384`.

| Mode | Hosts | client total tok/s | client input tok/s | client output tok/s | client peak output tok/s | client mean TTFT | client mean ITL |
|---|---:|---:|---:|---:|---:|---:|---:|
| non-PD serve-level DP | 2 | 12.78K | 10.22K | 2.56K | 5.02K | 46.49s | 34.56ms |
| PD 1P1D | 2 | 13.64K | 10.91K | 2.73K | 3.48K | 57.60s | 28.73ms |
| PD 2P1D | 3 | 15.31K | 12.25K | 3.06K | 3.97K | 20.11s | 35.08ms |

Pod-count-fair comparison:

- PD 1P1D beats two-host non-PD C128 by about `6.8%` on client total tok/s.
- PD 1P1D has worse client mean TTFT because client TTFT includes burst queueing
  and P/D handoff, but it has better mean ITL.

Higher-throughput comparison:

- PD 2P1D beats two-host non-PD C128 by about `19.8%` on client total tok/s,
  but uses one extra prefill host.
- PD 2P1D also cuts client mean TTFT from PD 1P1D `57.6s` to `20.1s`, because
  two prefill workers reduce the burst prefill backlog.

## Serve-Log Steady Numbers

| Mode | Prefill active window UTC | Prefill active input tok/s | Decode highwater window UTC | Decode highwater output tok/s | Decode highwater max output tok/s |
|---|---|---:|---|---:|---:|
| non-PD serve-level DP C128 | both ranks `00:42:56-00:51:55` | 11.64K combined | rank windows are not aligned | 4.56K rank-local highwater sum | 5.36K sum of rank max |
| PD 1P1D C128 | `13:34:40-13:42:52` | 12.79K | `13:37:41-13:43:02` | 3.18K | 3.41K |
| PD 2P1D C128 | rank0 `16:56:18-17:02:58`, rank2 `16:56:18-17:02:59` | 15.71K combined | `16:57:38-17:04:04` | 3.63K | 3.95K |

Interpretation:

- PD 1P1D is the cleanest pod-count-fair C128 win. It has a higher sustained
  prefill active rate than two-host non-PD: `12.79K` vs `11.64K` input tok/s.
- non-PD two-host shows strong rank-local decode highwater, but the two ranks'
  highwater windows are not aligned and the run still averages only `2.56K`
  output tok/s at the client. The bottleneck is not raw decode kernel ability;
  it is same-device prefill/decode interference plus queue/tail behavior.
- PD 2P1D raises prefill active capacity to `15.71K` input tok/s and reaches
  full `128` observed decode running requests, giving the best measured total
  throughput in this round.

## Serve-Internal TTFT View

PD request-time stats give a request-level handoff view after the request has
entered the PD serving path. This is not the same as client TTFT.

`16K/4K C128`, mean values:

| Mode | P queue | P forward | P transfer | P transfer tail | P total | D prealloc wait | D KV wait | D transfer tail | D total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PD 1P1D | 1596ms | 2879ms | 2567ms | 322ms | 4799ms | 2497ms | 2293ms | 45ms | 4790ms |
| PD 2P1D | 1387ms | mixed log schemas | 2564ms | n/a | 4407ms | 2055ms | 2313ms | 51ms | 4369ms |

The practical serve-internal first-token handoff cost is therefore around:

- PD 1P1D C128: `~4.80s` mean.
- PD 2P1D C128: `~4.37s` mean.

This is much smaller than client TTFT (`57.6s` / `20.1s`) because client TTFT
also includes benchmark burst wait and queueing before a request reaches the
active service path. For device-capacity reporting, the server-side phase stats
are the better diagnostic.

## Why This Is A PD-Favorable Scenario

This C128 run has three properties that public PD reports repeatedly identify as
favorable:

1. Long prefill: `16K` input makes prefill expensive enough that colocating
   prefill and decode creates visible interference.
2. Long decode: `4K` output keeps decode busy long enough for a highwater steady
   region to exist.
3. Burst pressure: `request_rate=inf` and `C128` expose queue/tail behavior.

At C64 the two-host non-PD serve-level DP result is better, so the claim should
not be generalized. The current measured statement is:

```text
For 16K/4K under C128 burst pressure, PD 1P1D gives a pod-count-fair total
throughput win, and PD 2P1D gives the best absolute throughput in the tested
set.
```

## Public PD Reports To Mirror

Useful public references:

- [DistServe OSDI 2024](https://arxiv.org/html/2401.09670v3): defines
  goodput as the largest request rate that satisfies TTFT/TPOT SLOs, and reports
  large gains under stricter SLOs. The key lesson for us is to report goodput
  and SLO attainment, not only average tok/s.
- [Splitwise](https://arxiv.org/html/2311.18677v2): production traces motivate
  prefill/decode separation because prefill is compute-heavy while decode is
  memory-bandwidth-heavy. Its published gains are strongest when phase-specific
  resources can be matched separately.
- [TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html):
  reports gains for long-input cases such as `4400/1200`, `8192/256`, and
  `8192/1024`, and recommends first measuring context req/s/GPU and generation
  tok/s/user, then doing rate matching.
- [vLLM MORI-IO KV connector](https://vllm.ai/blog/2026-04-07-moriio-kv-connector):
  shows a PD-style setup improving SLO goodput for a `2000/1000` workload,
  with the tradeoff shifting failures from ITL spikes toward TTFT.
- [dstack SGLang PD ratio benchmark](https://dstack.ai/blog/benchmarking-pd-ratios/):
  compares `3P:1D`, `2P:2D`, and `1P:3D` at C32/C64/C128. Its strongest
  guidance for us is to test decode-heavy ratios too; `3P:1D` is not always the
  best use of hosts.
- [NVIDIA Dynamo disaggregated serving docs](https://docs.nvidia.com/dynamo/v-0-7-1/design-docs/disaggregated-serving):
  explicitly notes that remote prefill helps long-context requests, while short
  prompts or high prefix-cache hits may be better served locally.
- [NVIDIA NIM / GenAI-Perf metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html):
  useful for metric definitions. TTFT commonly includes queueing and network
  time, so server-side phase metrics should be reported alongside client-visible
  metrics.

## Suggested Next Test Matrix

Priority order:

1. Keep `16K/4K C128` as the PD-favorable anchor. Report both client throughput
   and server-side steady input/output tok/s.
2. Add open-loop request-rate sweeps for C128-equivalent pressure and compute
   goodput under SLOs, for example TTFT `<30s` or `<60s`, ITL `<40ms` or
   `<60ms`.
3. Test ratios if hosts are available:
   - `1P:1D` as the pod-count-fair baseline.
   - `2P:1D` as the current best absolute throughput.
   - `1P:2D` and `1P:3D` to check whether long output/reasoning becomes decode
     constrained enough that more decode hosts beat more prefill hosts.
4. Add length variants:
   - `16K/512`: long input, short output; expected to favor more prefill.
   - `2K/4K`: decode-heavy; expected to favor more decode.
   - `2K/2K`: balanced.
5. Add a mixed-load case, for example 80% `1K/256` plus 20% `16K/4K`, to test
   whether PD protects short-request ITL/TPOT from long-prefill interference.
6. Add prefix-cache hit variants: 0%, 50%, 80%. Public systems caution that
   high cache hit can make local prefill better than remote prefill.

## Reporting Template

For each run, report:

- Client-visible: total tok/s, input tok/s, output tok/s, peak output tok/s,
  mean/p99 TTFT, mean/p99 ITL, success count.
- Server prefill: active window, active input tok/s, max queue, forward/transfer
  stage means if PD.
- Server decode: highwater window, highwater output tok/s, max running request
  count, max queue.
- PD handoff: prealloc wait, KV wait, transfer tail, total.
- Goodput: max request rate satisfying explicit TTFT and ITL SLOs.
