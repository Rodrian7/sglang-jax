# 生成 Roofline HTML 报告(server-free,任何注册模型)

一条命令,在 **CPU 上**(无需 TPU、无需加载权重、无需起 server)trace 任意已注册 sglang-jax 模型的**真实 forward**,产出一个**自包含的交互式 HTML** roofline 报告。

- 工具:`tools/trace_roofline.py`(分支 `feat/theoretical-roofline`)
- 核心模块:`python/sgl_jax/srt/utils/roofline/`(`standalone_trace.py` / `trace_analyze.py` / `report_html.py` / `forward_jaxpr_dump.py`)

---

## TL;DR

```bash
# 在任意有 sglang-jax 依赖的环境(CPU pod 的 /opt/venv 即可),仓库根目录下:
PYTHONPATH=python python tools/trace_roofline.py \
    --model-path /models/MiMo-V2-Pro-Private \
    --tp 32 --dp 8 --seq-len 4096 \
    --html /tmp/roofline.html
```

输出:
- 终端打印 prefill + decode 的逐类 cost 表(TFLOP / HBM / ICI / ideal ms / bound)
- `/tmp/roofline.html` —— **单文件、零外部依赖、可离线打开**的交互报告

整个过程 **~1 分钟**(绝大部分是 sglang-jax 的 import;trace 本身是秒级)。

---

## 运行环境

不需要 TPU。需要的只是能 import sglang-jax 的 Python 环境:`jax`、`flax`、`transformers`、`numpy` + 仓库本身。

- **最省事**:在任意一个 bench pod 上用现成的 `/opt/venv`(已装好全部依赖),仓库在 `/tmp/sglang-rope`:
  ```bash
  cd /tmp/sglang-rope
  PYTHONPATH=/tmp/sglang-rope/python /opt/venv/bin/python tools/trace_roofline.py \
      --model-path /models/MiMo-V2-Pro-Private --tp 32 --dp 8 --html /tmp/roofline.html
  ```
- **本地**:任何装了 `jax[cpu]` / `flax` / `transformers` 的 venv 都行(模型目录里要有 `config.json`)。

CLI 会在 import jax **之前**自动设好 `JAX_PLATFORMS=cpu` 和 `--xla_force_host_platform_device_count=<devices>`,你不用手动设环境变量。若在 TPU 机器上运行,它会自动用 TPU(不影响结果结构)。

---

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model-path` | (必填) | 模型目录,内含 `config.json`(可以只有 config.json,不需要权重) |
| `--tp` | 32 | mesh 总设备数(= tp_size)。注意:这是 **mesh 总数**,真实 tensor 并行度 `t = tp/dp` |
| `--dp` | 8 | data 并行度。`t = tp/dp` |
| `--seq-len` | 4096 | decode 的 KV 上下文长度(workload 输入) |
| `--tokens` | 512 | prefill 的全局 chunk token 数 |
| `--phases` | `prefill,decode` | 要 trace 的阶段,逗号分隔 |
| `--html` | 无 | 写出自包含 HTML 报告 |
| `--dump` | 无 | 额外导出原始 jaxpr 结构 JSON(prefill) |
| `--devices` | = tp | CPU 假设备数(TPU 机器上忽略) |

---

## 产物:HTML 里有什么

左侧**实时旋钮**(拖动即重算,不需要重新 trace):
- phase 切换(decode / prefill)、SP(sequence-parallel)开关
- 权重量化方式(bf16 / fp8 per-tensor / per-channel / block-wise)+ block size + 激活精度(W8A8/W8A16)
- `tp` / `dp` 下拉(只列**合法**组合)
- decode batch / KV context / prefill chunk 滑块

右侧 **5 个 tab**:
- **Roofline** —— log-log roofline 图;悬停任一点看 achieved TFLOP/s、GB/s、%peak;ICI-bound 的算子画成 ✕(掉在 roof 下方)
- **Dataflow** —— 单层算子链 + 每步 bound 条形(compute/HBM/ICI)
- **Fusion** —— 结构性融合机会 + 省下的 HBM 往返
- **Code path** ⭐ —— **来自真实 trace**:每个算子角色 → 它实际的 `models/*.py` 调用链(如 `qkv ← mimo_v2_flash.py:310`、`o_proj ← :355`)。不是手写的描述符
- **Kernels** ⭐ —— **真实 Pallas kernel 清单**:RPAd/RPAm(full + SWA)、fused-moe-v2,带 per-device avals + shard_map 调用点

**分享**:HTML 完全自包含(数据/CSS/JS 全部 inline,roofline 用 `<canvas>` 现画,无 CDN/图片/网络请求)。直接把这**一个 `.html` 文件**发出去,对方下载后用任意浏览器离线打开即可,不需要装任何东西。

---

## 原理(为什么不需要 TPU / 权重 / server)

1. **抽象构造模型**:`nnx.eval_shape(lambda: model_class(config, dtype, mesh))` 建出零分配的抽象权重(loader 本来就用这套);跳过 checkpoint 加载和逐层 fp8 dequant(那是 7 分钟里的大头,对 jaxpr 毫无贡献)。
2. **CPU 假 mesh**:`--xla_force_host_platform_device_count` 造出 N 个 CPU 设备,按 `[data=dp, tensor=tp//dp]` 建 mesh。
3. **只 trace 不执行**:`jax.make_jaxpr(forward)(...)` 只做符号化 trace —— **从不 lower Mosaic**,所以 Pallas kernel 在 CPU 上照样 trace 出来。`patch_for_cpu(7)` 把设备伪装成 v7x,让 kernel 的 block-size 选择能解析(纯 host 端 Python,安全)。
4. **结构来自 trace,代价来自解析模型**:权重**数值**对 jaxpr 没有影响(只用形状)。所以量化、真实上下文长度是在 cost 模型里**解析地**套用(从 `config.json` + 并行度),而不是从 trace 里读。trace 提供的是:算子清单、真实层数、真实 source 归属、真实 kernel 身份。
5. **代价已校验**:`trace_analyze.analyze_trace` 复用已验证的 roofline 原语(`ops`/`references`/`quant`/`parallelism`),其逐类结果与手写解析模型 `descriptors.build` **逐类吻合到比值 1.000**(prefill + decode 都验证过)。HTML 旋钮用的就是这个已验证的闭式模型(浏览器里没法重 trace)。

---

## 换一个模型怎么办(零 per-model 代码)

只要模型已在 `python/sgl_jax/srt/models/` 注册(有 `EntryClass` + `load_weights`),直接换 `--model-path` 指向它的目录即可:

```bash
PYTHONPATH=python python tools/trace_roofline.py --model-path /models/<另一个模型> --tp 16 --dp 2 --html /tmp/x.html
```

角色分类(qkv / o_proj / gate_up / down / router / lm_head)和层数是从 trace 的**调用栈 + 形状**自动推出的(不硬编码行号),Pallas kernel 按 kernel 名注册表定价(RPA→attention、fused-moe-v2→experts)。新模型如果用了**新的 Pallas kernel**,在 `trace_analyze.py::_kernel_kind` / 定价分支里加一条即可(per-kernel,不是 per-model)。

---

## 校验工具(可选)

`/tmp/verify_analyze.py`(本会话用过,未入库)把 trace 分析器和 `descriptors.build` 并排对比逐类 cost。要重跑校验时可参考它对每个 phase 打印 `ratio`,理想值全为 1.000。

---

## 注意事项 / 已知边界

- **纯理论**:这是 roofline 下界(`ideal_ms = max(compute, HBM, ICI)`),不预测跨算子 overlap 等运行时涌现效应。
- block-wise fp8 在 trace 时被跳过(它需要 TPU dequant kernel);量化在 cost 模型里解析套用,且 block-wise fp8 的 MXU 速率被正确地按 bf16 计(per-block scale 打断 K 累加)。
- 旋钮里 token 口径是**每 DP 组**(per-dp);MoE 全局 token = per-dp × dp。
- `--seq-len` 只影响 decode 的 KV 读取量;prefill 在 chunk 内因果,上下文取 chunk。
