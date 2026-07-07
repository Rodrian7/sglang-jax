# sglang-jax

JAX backend for [SGLang](https://github.com/sgl-project/sglang) — a structured generation language for large language models.

## Overview

sglang-jax provides a JAX-based backend implementation for SGLang, enabling efficient LLM inference on TPU and GPU hardware through XLA compilation.

## Key Features

- **JAX/XLA compilation** — Automatic optimization for TPU/GPU execution
- **Structured generation** — Constrained decoding with grammar support
- **Batch inference** — Efficient batched request processing
- **TPU support** — Native Google Cloud TPU integration

## Quick Start

```bash
pip install sglang-jax
```

```python
import sglang_jax as sj

# Initialize engine
engine = sj.Engine(model="meta-llama/Llama-3-8B")

# Run inference
output = engine.generate("Hello, world!", max_tokens=100)
```

## Project Links

- [GitHub Repository](https://github.com/sgl-project/sglang-jax)
- [SGLang Main Project](https://github.com/sgl-project/sglang)

