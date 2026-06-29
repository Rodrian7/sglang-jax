"""Official grouped top-k Pallas kernel (stable lowest-index tie-break)."""

from python.sgl_jax.srt.kernels.grouped_topk.v1.kernel2 import grouped_topk_pallas

__all__ = ["grouped_topk_pallas"]
