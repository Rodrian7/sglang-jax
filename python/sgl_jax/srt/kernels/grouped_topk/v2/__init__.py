"""Token-in-lane grouped top-k Pallas kernel (v2, experimental).

Same routing as v1 but with the working layout transposed to `[E, BT]` (experts in the
sublane/major dim, tokens in the 128-wide lane/minor dim). Selection reductions run over the
sublane axis so all 128 token-lanes are processed in parallel and no cross-lane permute is needed —
targeting the VPU/cross-lane bottleneck of the v1 `[BT, E]` layout.
"""

from sgl_jax.srt.kernels.grouped_topk.v2.kernel import grouped_topk_pallas_v2

__all__ = ["grouped_topk_pallas_v2"]
