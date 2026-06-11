# Roofline HTML 报告 · 使用 / 读图指南

> 配套文档:[`roofline_trace_html_howto.md`](./roofline_trace_html_howto.md) 讲**怎么生成**这份 HTML。本文讲**拿到 HTML 后怎么读、怎么从它推到一个优化决策**。

报告完全自包含:下载那个 `.html`,任意浏览器离线打开即可,无需装任何东西。所有数字**拖左侧旋钮实时重算**,不用重新生成。

---

## 30 秒上手:三步工作流

1. **设场景** —— 左侧旋钮调到你关心的工况:phase(decode / prefill)、`tp`/`dp`、token 数、量化方式。
2. **看 bound** —— 左下 summary 框直接告诉你瓶颈是哪个引擎:
   - `bound: HBM` → 访存瓶颈(读权重 / KV / 激活)
   - `bound: compute` → 算力瓶颈(MXU 打满)
   - `bound: ICI` → 通信瓶颈(跨设备 collective)
   - 还有 `step ≈ X ms`(单步理论下界)和 compute / HBM / ICI 各自的 ms。
3. **进对应 tab 找杠杆** —— bound 决定你该看哪个 tab、能动哪个旋钮:

| bound | 去哪个 tab | 典型杠杆 |
|---|---|---|
| **HBM** | Kernel(找字节最重的 kernel)+ Fusion | fp8 权重 / 更大 EP(↓本地专家)/ 砍中间激活往返 / 提 OI |
| **compute** | Kernel | 提 MXU 速率(W8A8,非 block-fp8)/ 砍 flops |
| **ICI** | Overlap(先看 comm 是否被藏住) | 若暴露且 ΣI > 算/访存墙 → **overlap 没用,只能砍通信**:更小 chunk / EP 局部性 / 拓扑 / 少跨 host |

> 核心公式贯穿全报告:**`理论 step ≈ max(compute, HBM, ICI)`**。哪个引擎最大,就是 bound;砍别的引擎不会让 step 变快。

---

## 左侧旋钮

| 旋钮 | 作用 |
|---|---|
| **phase**(decode / prefill) | 切换两种工况。decode 看单 token 解码,prefill 看 chunk 预填。 |
| **SP** | sequence-parallel 开关(影响 TP reduce 的形态)。 |
| **weight quant** | bf16 / fp8 per-tensor / per-channel / block-wise(+ block size)。block-wise fp8 的 MXU 速率按 **bf16** 算(per-block scale 打断 K 累加),工具已正确处理。 |
| **activation** | W8A16(激活 bf16)/ W8A8(激活也 fp8,才能吃到 fp8 MXU 速率)。 |
| **tp / dp** | 下拉**只列合法组合**。注意:`tp` 是 **mesh 总设备数**,真实 tensor 并行度 `t = tp/dp`,fused-MoE 的 `ep = tp`。 |
| **decode batch / KV context / prefill chunk** | workload 滑块。**token 口径是「每 DP 组(per-dp)」**;MoE 全局 token = per-dp × dp。 |

调任意旋钮,右侧所有 tab + summary 立即重算。

---

## 五个 tab,逐个怎么读

### Overview —— 整体 roofline + 逐类成本
- **回答**:整体在哪个 roofline 区、每类算子各花多少。
- **怎么读**:log-log roofline 图,每个点是一类算子;悬停看 achieved TFLOP/s、GB/s、%peak。**画成 ✕ 的点是 ICI-bound**(掉在 roof 下方,算力/访存都没打满,卡在通信)。下方成本表给每类的 TFLOP / HBM GB / ICI GB / ideal ms / bound。
- **看到 → 做什么**:先在这里确认整体 bound 和「谁最贵」,再决定钻哪个 tab。

### Overlap —— 通信能不能藏在计算后面
- **回答**:ICI 通信是被算力/访存**藏住(hidden)**了,还是**暴露(exposed)**在关键路径上。
- **怎么读**:一张合并表,每行一个 collective:
  - `type (model)` = 理论预测能否流水(pipelineable / barrier)。
  - **`XLA actual (HLO)`** = 编译器实际怎么调度的(✓ 吻合 / ⚠ 模型说能藏但实际暴露)。**这列是实测证据,不是理论**(需生成时带 `--hlo`)。
  - `hidden / exposed` ms + 顶部 comm budget 条 + step 分解条。
- **关键判读**:看底部 verdict —
  - 若 **ΣI > 算/访存墙** → **comm-bound**,即便完美 overlap 也降不到通信时间以下 → overlap **不是**杠杆,必须**砍通信**(更小 chunk / EP 局部性 / 拓扑)。
  - 若 exposed ≈ 0 → 通信不是瓶颈,别在这浪费精力,回去砍 HBM/flops。
- **MiMo 实测要点**:MoE a2a 在融合 kernel 内(SparseCore),**不是 XLA collective**,XLA 管不到;且实测在 torus 带宽下沿**暴露**。能不能藏住是 kernel/device-trace 问题。

### Kernel —— 该攻哪个 kernel、怎么攻
- **回答**:按 ideal ms 排序的算子 + 每个 Pallas kernel 的深度拆解。
- **怎么读**:每张 kernel 卡(fused-MoE-v2、RPA full、RPA SWA)给:
  - 三引擎时间条:**HBM(weight+act)/ compute(MXU)/ ICI**,最长的就是 bound。
  - **OI vs ridge**:operational intensity 低于 ridge → 访存瓶颈;高于 → 算力瓶颈。
  - **VMEM working set / 64MB**(v7x 单 device 上限 64MB,**取自 kernel 自己的估算器**)—— 接近 64MB 说明大 chunk 下逼近 spill。
  - **tuned block config** 随 token 旋钮变(查 kernel 自己的 tuned 表)。
  - bound-aware 的 **lever** 文案:直接告诉你这个 kernel 该砍字节还是提算力。
- **看到 → 做什么**:照着卡上 lever 走。例:MoE `weight-HBM-bound` → ① fp8 权重 ② 更大 EP(↓本地专家数)③ 提 OI(更大 batch/chunk → 每专家更多 token)。

### Fusion —— 哪些中间激活的 HBM 往返能省
- **回答**:把 producer→consumer 的中间激活折进相邻 matmul/kernel,省掉它的 HBM 往返。
- **怎么读**:按「省下的 step」排序;`status` 列是**编译器 HLO 实测**(✓ 已被 XLA 融成 matmul epilogue / ✗ 没融 / 在 Pallas kernel 内)。
- **看到 → 做什么**:仅当模型 **HBM-bound** 时,省字节 ≈ 省 step,这里才值得动;compute/ICI-bound 时融合基本无感。

### Trace —— 这个模型真实的 forward
- **回答**:报告里的「类别」对应到**真实代码**哪一行。
- **怎么读**:Code path = 每个算子角色 → 它实际的 `models/*.py` 调用链(如 `qkv ← mimo_v2_flash.py:310`);层数是从 trace 涌现的,不是手写。Pallas kernels = 真实 kernel 名 + per-device avals + shard_map 调用点。
- **看到 → 做什么**:在别的 tab 锁定一个贵的类别后,来这里定位到具体代码去改。

---

## 一个完整例子(MiMo-V2-Pro,tp32 / dp8)

**Prefill, chunk 256**:
1. summary → `bound: ICI`,step 由 a2a 主导。
2. Overlap → MoE a2a 那行 `XLA actual` 是 **⚠**(模型说可流水,实测暴露在 torus 下沿);ΣI > 算/访存墙 → verdict **comm-bound**。
3. 结论:别指望 overlap,**砍通信** —— 更小 chunk / EP 局部性 / 减少跨 host 跳数。

**Decode, batch 适中**:
1. summary → `bound: HBM`,step ≈ 10ms 级,由 **MoE 权重读**主导(每步读一次,与 batch 无关)。
2. Kernel → fused-MoE-v2 卡 `weight-HBM-bound`,权重占 HBM ~100%。
3. 结论:**fp8 权重**(quant 旋钮)/ **更大 EP**;attention 的 KV 读相比之下很小(单序列)。

---

## 关键概念速查

- **roofline / bound**:`ideal = max(compute, HBM, ICI)`;最大的那个是瓶颈。这是**理论下界**,不是实测吞吐。
- **OI(operational intensity)= flops / bytes**;`ridge = MXU峰值 / HBM带宽`。OI < ridge → 访存瓶颈;> ridge → 算力瓶颈。
- **exposed vs hidden(ICI)**:暴露的通信加在关键路径上;藏住的被计算盖住。`step ≈ max(ΣC,ΣH) + exposed comm`。
- **VMEM working set**:kernel 实际占的片上内存(取自 kernel 自己的估算器),对 64MB(v7x 上限)看余量。
- **t / ep**:`t = tp/dp`(tensor 轴),`ep = tp`(fused-MoE 专家并行)。

---

## 常见误读 / 边界

- **是理论下界,不是实测**。`ideal_ms = max(...)` 不预测跨算子运行时 overlap 等涌现效应;Overlap/Kernel 的最终确认仍需 device trace。报告里 `XLA actual` 列是编译器实测证据(需 `--hlo`),但「kernel 能否打到天花板」要真机 profile。
- **token 是 per-dp**:旋钮里的 token 是每 DP 组;MoE 全局 = per-dp × dp。
- **`--seq-len` 只影响 decode 的 KV 读**;prefill 在 chunk 内因果,上下文取 chunk。
- **bound 比绝对 ms 可信**:各类同构,放缩不改变「哪个引擎最大」;所以「该砍什么」的结论很稳,绝对 ms 当量级看。
- 这套深度拆解(RPA / fused-MoE-v2 卡、tuned block、VMEM)目前是 **MiMo 系**专用;换架构(如 Ling3 的 KDA/MLA)需要先做适配。
