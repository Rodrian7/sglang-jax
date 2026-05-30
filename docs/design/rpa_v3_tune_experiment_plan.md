# RPA v3 tune 实验计划

> **术语简写**：本文用 `num_tokens` 指代 `max_num_batched_tokens`（lookup key 字段全名）。仅文档简写，代码 / key schema 仍用全名。

> **开发流程**：本次 tune 走 branch-only 流程，**不开 PR**。Phase 0 五项修复在专用 branch `tune/rpa-v3-bench-fixes` 上开发，push 到 remote 后 falcon manifest 直接 clone 该 branch 跑实验。所有 sweep 跑完、数据写进 `tuned_block_sizes_v3.py` 之后，由用户亲自走 PR 合主干。

## 背景

`ragged_paged_attention` v3 内部包含 3 个 pallas_call（D / P / M），每次 forward 全部 launch，由 `distribution: i32[3]` 数组决定每个 pallas_call 处理的 seq 范围。生产中 scheduler 实际只产生两种 distribution：

| 生产 forward | distribution | 实际工作 pallas_call |
|---|---|---|
| forward_mode=DECODE | `[N, N, N]` | D（处理 N 个 seq, q_len=1） |
| forward_mode=EXTEND（含 chunked-prefill / mix_with_running） | `[0, N, N]` | P（处理 N 个 seq, q_len 视情况） |

**M pallas_call 在生产中永远空跑**（scheduler 从不设 distribution[2] > distribution[1]）。本次 sweep 只 tune D 和 P 两种 stage，M 写表时复制 P 的 winner（避免 lookup miss 退化到 heuristic）。

---

## 通用配置

```
max_context_len      = 40960
max_kv_cache_tokens  = 600000
chunk_prefill_size   = 4096       # 跟 sgl-jax server_args 默认对齐
tries                = 2          # tuner default 从 3 降到 2
write_threshold_pct  = 0
decode actual_kv_len ∈ [1024, 2048)
```

## Lookup table key

```
key = (stage, sliding_window, q_dtype, kv_dtype, q_heads, kv_heads,
       head_dim, page_size, max_num_batched_tokens)
value = (bq_sz, bkv_sz, bq_csz, bkv_csz)

stage:    {d, p, m}    # m 写表但值复制自 p, 不实测
dtype:    bfloat16     # 其他 dtype 字段保留, 本次不填
```

---

## Phase 0：前置修复（branch 开发，不开 PR）

工作流：

```bash
# 从 main 拉新 branch
git checkout main && git pull
git checkout -b tune/rpa-v3-bench-fixes

# 完成下面 5 项修复 + 本地验证

# push 到 remote
git push -u origin tune/rpa-v3-bench-fixes

# 锁定 commit SHA, Phase 1 manifest 用此 SHA clone
git rev-parse HEAD   # → 写进 manifest 的 COMMIT_SHA
```

**总时间预算 ~2h active 工作**：编码 ~1h，本地 v6e 验证 ~1h，push <5 min。

本地验证清单：
- 1 个 d cell（`d hd=128 ps=128 swNone q=4 kv=1 num_tokens=128`）：tuner 跑通
- 1 个 p cell（`p hd=128 ps=256 swNone q=8 kv=1 num_tokens=4096`）：chunk_prefill_size 参数生效
- 跑完后 `/tmp/sglang_jax_moe_trace/` 应为空：trace cleanup 工作
- 第二次跑同 cell：JIT compile 时间应明显缩短（cache 命中）

### 0.1 bench prefill chunk 拆分参数化

文件：`benchmark/kernels/flash_attention/utils.py`，函数 `create_prefill_uniform_data`

```python
def create_prefill_uniform_data(
    ...,
    chunk_prefill_size: int = 4096,    # 新增参数, 默认 = sgl-jax 当前 default
):
    if max_num_batched_tokens > chunk_prefill_size:
        batch_size = cdiv(max_num_batched_tokens, chunk_prefill_size)
        seq_lens_list = [chunk_prefill_size] * (batch_size - 1) + [
            max_num_batched_tokens - chunk_prefill_size * (batch_size - 1)
        ]
    else:
        batch_size = 1
        seq_lens_list = [max_num_batched_tokens]
    ...
```

### 0.2 tuner SMEM 估算修正

文件：`benchmark/kernels/flash_attention/get_block_spec_config_v3.py`，函数 `_smem_estimate_bytes`

```python
def _smem_estimate_bytes(
    stage: str,
    max_num_batched_tokens: int,
    pages_per_seq: int,
    chunk_prefill_size: int = 4096,
) -> int:
    if stage == "d":
        num_seqs = max_num_batched_tokens
    else:
        num_seqs = max(1, (max_num_batched_tokens + chunk_prefill_size - 1) // chunk_prefill_size)
    return num_seqs * pages_per_seq * 4 + 10 * num_seqs * 4
```

### 0.3 trace 目录 cleanup

文件：`python/sgl_jax/srt/kernels/utils/perf.py`，函数 `multiple_iteration_timeit_from_trace`

每次 call 在 `/tmp/sglang_jax_moe_trace/` 创一个 trace dir，sweep 跑 1000+ 次会撑爆 ephemeral 存储。函数末尾 `try/finally` 加 cleanup：

```python
import shutil

def multiple_iteration_timeit_from_trace(...):
    ...
    try:
        with jax.profiler.trace(trace_dir):
            ...
        return _extract_marker_durations_ms(...)
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)
```

### 0.4 tuner 候选 block size 剪枝

文件：`benchmark/kernels/flash_attention/get_block_spec_config_v3.py`

基于已测 81 个 v6e + v7x winner 的分布观察：
- `bkv ≥ 4096` 从未赢过（最大 winner 是 2048）
- `bq ∈ {1, 2, 4, 8, 16, 32}` 在不同 num_tokens 下都可能赢，不能剪

```python
def _bkv_candidates(page_size, kv_packing, max_kv):
    raw = [256, 512, 1024, 2048, 4096]   # 从 8 个削到 5 个 (保 4096 作上界探针)
    ...

# _bq_candidates 不变
```

### 0.5 JIT compilation cache 持久化

每候选 block size 触发一次完整 ragged_paged_attention 重 JIT，3 个 pallas_call 全编译。多 manifest 跑相同 (bq, bkv) 时大量重复编译。

manifest 启动命令前置：

```bash
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jax_cache
```

`/tmp/tpu_logs/` 是 GKE TPU host 持久化目录（不计 ephemeral 配额，参考 [[reference_tmp_tpu_logs_special]]），cache 跨 manifest 复用。

---

## 真实模型 (q_heads, kv_heads, head_dim, sliding_window) 元组（47 个）

只测下面 5 个 `(head_dim, sliding_window)` 组合，每组合下列出真实模型出现的 `(q_heads, kv_heads)` per-shard 对。

### A. (head_dim=128, sliding_window=None) — 24 对

涵盖 Llama / Qwen / Mistral 非 SWA / GLM / Phi-3 / MoE 类。GQA ratio 1, 2, 4, 8, 16 全覆盖：

```
ratio=1:  (1,1) (2,2) (4,4) (8,8) (16,16) (32,32)
ratio=2:  (2,1) (4,2) (8,4) (16,8) (32,16) (64,32)
ratio=4:  (4,1) (8,2) (16,4) (32,8) (64,16)
ratio=8:  (8,1) (16,2) (32,4) (64,8)
ratio=16: (16,1) (32,2) (64,4)
```

### B. (head_dim=128, sliding_window=4096) — 9 对

Mistral-7B SWA + Gemma-2-27B SWA。ratio ∈ {2, 4}：

```
(2,1) (4,1) (4,2) (8,2) (8,4) (16,4) (16,8) (32,8) (32,16)
```

### C. (head_dim=256, sliding_window=None) — 7 对

Gemma-2-9B 全 attn 层 + MiMo-V2-Pro 全 attn 层：

```
(2,1) (4,2) (8,4) (16,8) (16,1) (32,2) (64,4)
```

### D. (head_dim=256, sliding_window=128) — 3 对

MiMo-V2-Pro / Flash SWA 层（MiMo attention chunk）：

```
(64,4) (32,2) (16,1)
```

### E. (head_dim=256, sliding_window=4096) — 4 对

Gemma-2-9B SWA 层：

```
(2,1) (4,2) (8,4) (16,8)
```

---

## stage × num_tokens

取生产 padding 落点的代表值，跨度大到 winner 大概率不同。

| stage | num_tokens 取值 | 数量 |
|---|---|---|
| d | 1, 8, 32, 128, 512, 2048 | **6 个** |
| p | 1024, 4096, 16384 | **3 个** |
| m | 不实测，写表时复制 p 的 winner | — |

中间未测的桶（比如 num_tokens = 4 / 16 / 64 / 256）lookup miss → 退化到最近代表值的 heuristic 行为。

## page_size

`{128, 256}`，**2 个**，全测。

## SMEM 通过率（修 bench 后）

```
pages_per_seq = ceil(40960 / page_size)
  page_size=128 → 320
  page_size=256 → 160

decode (num_seqs = num_tokens):
  ps=128: num_tokens × 320 × 4 ≤ 1MB → ≤ 819
    通过: {1, 8, 32, 128, 512}                          5/6
  ps=256: ≤ 1638
    通过: {1, 8, 32, 128, 512}                          5/6
    （num_tokens=2048 在 ps=256 也超 SMEM, 1.31 MB > 1 MB）

prefill (修 bench 后 num_seqs = ceil(num_tokens / chunk_prefill_size)):
  num_tokens 取值: {1024, 4096, 16384} → num_seqs = {1, 1, 4}
  page_indices ≤ 5 KB
  通过: 3/3 全部
```

---

## Phase 1（v6e + v7x sweep，只测 D + P）

### sweep 维度

| 维度 | 取值 | 数量 |
|---|---|---|
| (head_dim, sliding_window) 组合 | A, B, C, D, E | **5 组** |
| (q_heads, kv_heads) | 各组全集 | **共 47 元组** |
| page_size | 128, 256 | **2 个** |
| stage | d, p | **2 个**（m 不实测） |
| num_tokens d | 1, 8, 32, 128, 512, 2048 | **6 个** |
| num_tokens p | 1024, 4096, 16384 | **3 个** |
| target | v6e, v7x | **2 个**（并行提交） |

### cell 数

每 `(q_heads, kv_heads, head_dim, sliding_window)` 元组的 cell 数：

```
d 阶段:
  ps=128 通过 5 个 num_tokens
  ps=256 通过 5 个 num_tokens
  小计: 10 cell

p 阶段:
  3 个 num_tokens × 2 个 ps = 6 cell

每元组合计: 10 + 6 = 16 cell
```

每组合的 cell 总数：

| (head_dim, sliding_window) 组 | (q,kv) 对数 | 元组小计 cell |
|---|---|---|
| A: (128, None) | 24 | 24 × 16 = 384 |
| B: (128, 4096) | 9 | 9 × 16 = 144 |
| C: (256, None) | 7 | 7 × 16 = 112 |
| D: (256, 128) | 3 | 3 × 16 = 48 |
| E: (256, 4096) | 4 | 4 × 16 = 64 |
| **合计 per target** | **47** | **752 cell** |

减已有数据 + VMEM 5%：
- v6e：减已有 56 entry + VMEM ~37 → **~659 net new**
- v7x：减已有 25 entry + VMEM ~37 → **~690 net new**

**两 target 合计 ~1,349 net new cell**

---

## Manifest 拆分

按 `(target, stage, head_dim, page_size, sliding_window)` 拆。每 (head_dim, sliding_window) 组合下 stage × ps = 4 个 manifest，共 5 组合 = **20 manifest per target**。

### v6e 20 manifest 列表

每 manifest cell 数 = `(q,kv) 对数 × num_tokens 通过数`：

| stage | (head_dim, sliding_window) | page_size | (q,kv) 对数 | num_tokens 通过数 | cell |
|---|---|---|---|---|---|
| d | A (128, None) | 128 | 24 | 5 | 120 |
| d | A (128, None) | 256 | 24 | 5 | 120 |
| d | B (128, 4096) | 128 | 9 | 5 | 45 |
| d | B (128, 4096) | 256 | 9 | 5 | 45 |
| d | C (256, None) | 128 | 7 | 5 | 35 |
| d | C (256, None) | 256 | 7 | 5 | 35 |
| d | D (256, 128) | 128 | 3 | 5 | 15 |
| d | D (256, 128) | 256 | 3 | 5 | 15 |
| d | E (256, 4096) | 128 | 4 | 5 | 20 |
| d | E (256, 4096) | 256 | 4 | 5 | 20 |
| p | A (128, None) | 128 | 24 | 3 | 72 |
| p | A (128, None) | 256 | 24 | 3 | 72 |
| p | B (128, 4096) | 128 | 9 | 3 | 27 |
| p | B (128, 4096) | 256 | 9 | 3 | 27 |
| p | C (256, None) | 128 | 7 | 3 | 21 |
| p | C (256, None) | 256 | 7 | 3 | 21 |
| p | D (256, 128) | 128 | 3 | 3 | 9 |
| p | D (256, 128) | 256 | 3 | 3 | 9 |
| p | E (256, 4096) | 128 | 4 | 3 | 12 |
| p | E (256, 4096) | 256 | 4 | 3 | 12 |
| **合计 v6e** | | | | | **752 cell** |

v7x 同样 20 manifest，**两 target 合计 40 manifest，1,504 cell**。

### Manifest 时长

每 cell ~3 min（decode）或 ~10 min（prefill / mixed），Phase 0 候选剪枝 + JIT cache 之后：

- 最大 d manifest = 120 cell × 3 min = ~6h
- 最大 p manifest = 72 cell × 10 min = ~12h ← **超 8h timeout**

p stage 的 A 组 manifest 需要按 (q,kv) 切两半：

| 切片后 | (q,kv) | cell | 时长 |
|---|---|---|---|
| p-A-ps128-half1 | 12 | 36 | ~6h |
| p-A-ps128-half2 | 12 | 36 | ~6h |

p stage A 组 4 个 manifest 各切两半 → 20 → **24 manifest per target，48 manifest total**。

### Manifest 命名

```
sglang-jax-rpa-v3-tune-{target}-{stage}-hd{head_dim}-ps{page_size}-sw{sliding_window}[-shardN]-{date}
```

例：
- `sglang-jax-rpa-v3-tune-v6e-d-hd128-ps128-swNone-20260530`
- `sglang-jax-rpa-v3-tune-v6e-p-hd128-ps128-swNone-shard0-20260530`

### Falcon 配置

```yaml
exp_type: PROFILING
cluster_name: tpuv6e-256-node          # v6e 部分
cluster_name: tpuv7x-64-node           # v7x 部分
device_count: 4
device_type: v6e (or v7x)
device_topo: 2x2
replica: 1
image: us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1
envs:
    JAX_COMPILATION_CACHE_DIR: /tmp/tpu_logs/jax_cache
```

---

## tuner CLI

每 manifest 调用 `get_block_spec_config_v3.py`：

```bash
python3 -u get_block_spec_config_v3.py \
    --stages {stage} \
    --page-sizes {page_size} \
    --head-dims {head_dim} \
    --head-combos {q1}:{kv1},{q2}:{kv2},...,{qN}:{kvN} \
    --decode-mnt {csv}      \    # stage=d 时
    --prefill-mnt {csv}     \    # stage=p 时
    --sliding-window {sw_or_omit} \
    --tries 2 \
    --write-threshold-pct 0.0
```

注：tuner CLI 现行 flag `--decode-mnt` / `--prefill-mnt` 是历史命名（mnt 来自最初 `max_num_tokens` 的缩写），沿用不改。

---

## Phase 4（子表 sub-experiment，follow-up）

Phase 1 数据回来后再决定要不要做。

### 4A — `max_context_len=131072` 敏感性

固定 5 个 representative shape，sweep `max_context_len ∈ {40960, 131072}`，验证 winner 是否对 `max_context_len` 敏感：

| q_heads | kv_heads | head_dim | page_size | num_tokens |
|---|---|---|---|---|
| 4 | 1 | 128 | 128 | 128 |
| 4 | 1 | 128 | 256 | 256 |
| 8 | 1 | 128 | 256 | 256 |
| 16 | 2 | 128 | 256 | 256 |
| 32 | 2 | 256 | 256 | 256 |

cell 数：5 shape × 2 max_context_len × 3 tries = **30 cell**

### 4B — 长 actual_kv_len sweep

固定 6 shape，sweep `actual_kv_len ∈ {[1024,2048), [12288,16384)}` × `bkv_sz ∈ {1024, 2048, 4096}`：

| q_heads | kv_heads | head_dim | page_size | num_tokens |
|---|---|---|---|---|
| 4 | 1 | 128 | 128 | 128 |
| 4 | 1 | 128 | 256 | 256 |
| 8 | 1 | 128 | 256 | 256 |
| 16 | 2 | 128 | 256 | 256 |
| 4 | 1 | 256 | 256 | 256 |
| 32 | 2 | 256 | 256 | 256 |

cell 数：6 shape × 2 actual_kv_len × 3 bkv_sz × 3 tries = **108 cell**

### 4C — mix_with_running 工况验证

P pallas_call 在生产里实际处理 mixed q_lens（`[1]*N_decode + [chunk_prefill_size]*N_prefill`），但我们 bench 是 uniform q_len。验证 winner 是否在 mixed-chunk 工况下仍是 winner：

| q_heads | kv_heads | head_dim | page_size | num_tokens | layout |
|---|---|---|---|---|---|
| 8 | 1 | 128 | 256 | 4124 | uniform: `[4096, 28]` |
| 8 | 1 | 128 | 256 | 4124 | mixed: `[1]*28 + [4096]` |
| 32 | 2 | 256 | 256 | 4124 | uniform |
| 32 | 2 | 256 | 256 | 4124 | mixed |
| 4 | 1 | 256 | 256 | 8192 | uniform: `[4096, 4096]` |
| 4 | 1 | 256 | 256 | 8192 | mixed: `[1]*16 + [4096]*2` |

cell 数：6 shape × 2 layout × 3 tries = **36 cell**

判定：
- mixed layout winner ≈ uniform layout winner（差 <5%）→ 现有 P tune 够用
- 显著不同（≥10%）→ 加 mix_ratio 维度到 schema，重做 P sweep

---

## 写表

每 manifest 跑完产出 `[WIN]` lines，聚合脚本：

```bash
python tools/aggregate_rpa_tune.py \
    --logs /tmp/operator-artifacts/sglang-jax-rpa-v3-tune-*/rank-0/tuner.log \
    --device "TPU v6e" \
    --output python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py
```

合并规则：
- 每 (key) 取 [WIN] 行的 4-tuple
- 跟现有 entry 冲突 → 新数据覆盖
- **m stage entry 自动从 p stage 同 (q, kv, hd, sw, ps, num_tokens) 复制**（Phase 1 不实测 m）
- SMEM/VMEM 剪掉的 cell 不写表，lookup miss 时退到 heuristic

---

## 提交批次

| 批次 | 内容 | manifest 数 |
|---|---|---|
| 0 | Phase 0 修复（branch `tune/rpa-v3-bench-fixes` 开发 + push, 不开 PR） | - |
| 1 | Phase 1：v6e + v7x sweep（D + P only），manifest 锁定 Phase 0 commit SHA | 48 |
| 2 | 写 m stage entry（脚本复制 p 值） | 0（脚本内执行） |
| 3 | Phase 4：sub-experiment（看 Phase 1 结果决定） | 0~3 |
| 4 | **用户亲自走 PR**：合 Phase 0 修复 + 新 tuned table 进主干 | - |
| **合计** | | **~48-51 manifest** |

---

## 实验追踪

每 manifest 提交后追加到 `~/sp_exps_templates/rpa_tune_log.md`：
- exp_id
- manifest 名（含 target / stage / head_dim / page_size / sliding_window / shard）
- 提交时间 / 完成时间
- cell 总数 / WIN 数
- 输出 log 路径
