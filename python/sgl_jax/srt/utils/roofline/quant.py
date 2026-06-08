"""Quantization model for the roofline.

Parses a checkpoint's ``quantization_config`` (e.g. fp8 block-wise with dynamic
activation quant, as MiMo uses) into per-op-role ``QuantSpec``s, and provides
the byte/peak accounting each GEMM needs: quantized weights save HBM but add
scale-tensor reads (per-channel = N, block = ceil(K/bk)*ceil(N/bn)), and W8A8
(both operands fp8) runs at the fp8 MXU rate while W8A16 stays at bf16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_BITS = {"bf16": 16, "fp16": 16, "fp8": 8, "int8": 8, "fp32": 32}


@dataclass(frozen=True)
class QuantSpec:
    weight: str = "bf16"  # bf16 | fp8 | int8
    act: str = "bf16"  # bf16 | fp8 | int8
    block: tuple[int, int] | None = None  # (bk, bn) block-wise; None => per-channel if quantized

    def w_bytes(self, k: int, n: int) -> int:
        return k * n * _BITS[self.weight] // 8

    def a_bytes(self, m: int, k: int) -> int:
        return m * k * _BITS[self.act] // 8

    def weight_scale_bytes(self, k: int, n: int) -> int:
        if self.weight == "bf16":
            return 0
        if self.block:
            bk, bn = self.block
            return math.ceil(k / bk) * math.ceil(n / bn) * 4  # fp32 scales
        return n * 4  # per-output-channel

    def act_scale_bytes(self, m: int, k: int) -> int:
        if self.act == "bf16":
            return 0
        if self.block:
            bk, _ = self.block
            return m * math.ceil(k / bk) * 4  # per-token, per-K-block (dynamic)
        return m * 4  # per-token

    def peak_kind(self) -> str:
        """Compute MXU rate. Block-wise fp8 does NOT reach the fp8 peak: the
        per-block (e.g. 128) scales force the K accumulation to break every
        block and re-dequantize, capping the right-matrix tile fed to the MXU,
        so it effectively runs at the bf16 rate. Only clean (non-block) fp8xfp8
        reaches the fp8 MXU peak."""
        if self.weight == "fp8" and self.act == "fp8" and self.block is None:
            return "fp8"
        return "bf16"

    def dequant_flops(self, m: int, k: int, n: int) -> int:
        """Vector-unit dequant work for block-wise quant: rescale the [M,N]
        partials once per K-block. ~ m*n*ceil(K/bk). 0 if not block-quantized."""
        if self.weight == "bf16" or not self.block:
            return 0
        bk = self.block[0]
        return m * n * math.ceil(k / bk)

    def tag(self) -> str:
        if self.weight == "bf16":
            return "bf16"
        g = f"blk{self.block[0]}" if self.block else "pc"
        return f"w{self.weight}-a{self.act}-{g}"


BF16 = QuantSpec()


def quant_specs_from_config(config: dict) -> dict[str, QuantSpec]:
    """Map op-role -> QuantSpec from ``config['quantization_config']``.

    Roles: qkv, o_proj, mlp (dense gate/up/down), experts, lm_head.
    Falls back to all-bf16 if there is no fp8 quantization_config.
    """
    qc = config.get("quantization_config") or {}
    if str(qc.get("quant_method", "")).lower() != "fp8":
        return {r: BF16 for r in ("qkv", "o_proj", "mlp", "experts", "lm_head")}

    block = qc.get("weight_block_size")
    block = tuple(block) if block else None
    act = "fp8" if qc.get("activation_scheme") == "dynamic" else "bf16"
    q = QuantSpec(weight="fp8", act=act, block=block)
    ignored = qc.get("ignored_layers") or []
    o_proj_ignored = any("o_proj" in str(x) for x in ignored)

    return {
        "qkv": q,
        "o_proj": BF16 if o_proj_ignored else q,
        "mlp": q,
        "experts": q,
        "lm_head": BF16,  # head is typically left in bf16
    }
