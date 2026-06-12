"""Theoretical whole-model roofline for sglang-jax.

Leverages ``jax.experimental.roofline`` for XLA ops + closed-form formulas for
Pallas kernels (attention / MoE / fp8 QKV), composed per layer-pattern into a
per-device, per-phase roofline. See ``descriptors`` for per-architecture wiring
and ``tools/model_roofline.py`` for the CLI.
"""

from .report import HardwarePeaks, ModelRoofline, OpRoofline, render_cost_views

__all__ = ["HardwarePeaks", "ModelRoofline", "OpRoofline", "render_cost_views"]
