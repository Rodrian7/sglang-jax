"""Fused Expert-Parallel MoE layer using Pallas kernel."""

import logging

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe,
    get_dtype_packing,
    get_ep_size,
)
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor

logger = logging.getLogger(__name__)


def _expand_moe_block_scale(scale_3d: jax.Array, n_out: int, block_n: int) -> jax.Array:
    """Expand compact 2D MoE block scales to the kernel's fast 1D-ready layout."""
    scale_per_channel = jnp.repeat(scale_3d, block_n, axis=2)[..., :n_out]
    return scale_per_channel[:, :, None, :]


class FusedEPMoE(nnx.Module):
    """
    Expert Parallel MoE layer using fused TPU kernel.

    This layer wraps the optimized fused_ep_moe kernel which combines Top-K selection,
    expert computation, and aggregation into a single efficient operation.

    Key differences from EPMoE:
    - Weight format: w1/w3 are (num_experts, hidden_size, intermediate_size) for gate/up proj
      and w2 is (num_experts, intermediate_size, hidden_size) for down proj
    - Input: Takes router_logits directly instead of pre-computed topk_weights/topk_ids
    - Implementation: Uses Pallas kernel with manual memory management for TPU optimization

    Args:
        hidden_size: Hidden size of the model
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts to select per token (top_k)
        ep_size: Expert parallel size (number of devices to shard experts across)
        mesh: JAX mesh for distributed execution
        intermediate_dim: Intermediate dimension for expert FFN
        weight_dtype: Data type for weights
        dtype: Data type for computation
        activation: Activation function ("silu", "gelu", "swigluoai")
        layer_id: Layer index (for debugging)
        renormalize_topk_logits: Whether to renormalize top-k weights
        bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c: Tile size parameters (auto-selected if None)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        use_grouped_topk: bool = False,
        num_groups: int = 1,
        top_k_groups: int = 1,
        renormalize_topk_logits: bool = False,
        routed_scaling_factor: float | None = None,
        num_shared_experts: int = 0,
        moe_shared_expert_intermediate_size: int | None = None,
        quantization_config=None,
        enable_act_quant: bool = True,
        # Profiling / ablation flags (primarily for microbenching).
        disable_a2a: bool = False,
        disable_dynamic_ffn1: bool = False,
        disable_dynamic_ffn2: bool = False,
        disable_weight_load: bool = False,
        disable_a2a_s_tile_read: bool = False,
        disable_a2a_s_acc_tile_write: bool = False,
        disable_shared_expert: bool = False,
        disable_all_reduce_metadata: bool = False,
        disable_sync_barrier: bool = False,
        use_jax_allreduce_metadata: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.activation = activation
        self.use_grouped_topk = use_grouped_topk
        self.num_groups = num_groups
        self.top_k_groups = top_k_groups
        self.renormalize_topk_logits = renormalize_topk_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.num_shared_experts = num_shared_experts
        self.moe_shared_expert_intermediate_size = (
            moe_shared_expert_intermediate_size or intermediate_dim
        )
        self.mesh = mesh
        self.disable_a2a = disable_a2a
        self.disable_dynamic_ffn1 = disable_dynamic_ffn1
        self.disable_dynamic_ffn2 = disable_dynamic_ffn2
        self.disable_weight_load = disable_weight_load
        self.disable_a2a_s_tile_read = disable_a2a_s_tile_read
        self.disable_a2a_s_acc_tile_write = disable_a2a_s_acc_tile_write
        self.disable_shared_expert = disable_shared_expert
        self.disable_all_reduce_metadata = disable_all_reduce_metadata
        self.disable_sync_barrier = disable_sync_barrier
        self.use_jax_allreduce_metadata = use_jax_allreduce_metadata

        metadata = get_global_expert_location_metadata()
        if metadata is not None and layer_id is not None:
            self.num_experts = metadata.num_physical_experts
        else:
            self.num_experts = num_experts

        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )

        self.quantized_dtype = (
            quantization_config.get_moe_weight_dtype() if quantization_config else None
        )
        self.activation_quantized_dtype = (
            quantization_config.get_moe_activation_dtype() if quantization_config else None
        )
        # Optional explicit disable for in-kernel activation quantization. The
        # positive signal comes from quantization_config.moe_activation_dtype.
        self.enable_act_quant_cfg = enable_act_quant

        # Initialize weights.
        self.w1 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (self.num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )
        self.w3 = nnx.Param(
            jax.random.normal(
                jax.random.key(1),
                (self.num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w2 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (self.num_experts, intermediate_dim, hidden_size),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w1_scale = None
        self.w3_scale = None
        self.w2_scale = None

        if self.num_shared_experts > 0:
            se_inter_dim = self.moe_shared_expert_intermediate_size * self.num_shared_experts

            self.w1_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w2_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (se_inter_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w3_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )
        else:
            self.w1_shared = None
            self.w3_shared = None
            self.w2_shared = None

        self.w1_shared_scale = None
        self.w3_shared_scale = None
        self.w2_shared_scale = None

        # Read block-wise quantization settings from config.
        weight_block_size = (
            getattr(quantization_config, "weight_block_size", None) if quantization_config else None
        )
        if weight_block_size is not None and len(weight_block_size) == 2:
            self.quant_block_k = int(weight_block_size[1])  # block_k
            self.quant_block_n = int(weight_block_size[0])  # block_n
        else:
            self.quant_block_k = None
            self.quant_block_n = None

    def quantize_weights(self, is_static: bool = False):
        """Quantize MoE weights in-place. Call once after model loading."""
        if self.quantized_dtype is None:
            return

        # Determine quant_block_k. The v1 kernel requires a block size when
        # scales are provided, so per-channel fp8 must tile to block-256; the v2
        # kernel accepts per-channel (None).
        wsz = (
            self.quant_block_k
            if self.quant_block_k is not None
            else (None if isinstance(self, FusedEPMoEV2) else 256)
        )
        if hasattr(self, "quant_block_k"):
            del self.quant_block_k
        self.quant_block_k = wsz

        with jax.set_mesh(self.mesh):
            if is_static:
                ep_scale_sharding = P(("data", "tensor"), None, None, None)

                if wsz is None:
                    # Per-channel: scale shape (E, 1, 1, N)
                    w1_scale_shape = (
                        self.num_experts,
                        1,
                        1,
                        self.intermediate_dim,
                    )
                    w3_scale_shape = w1_scale_shape
                    w2_scale_shape = (
                        self.num_experts,
                        1,
                        1,
                        self.hidden_size,
                    )
                else:
                    # Block-wise: scale shape (E, K//block_k, 1, N)
                    w1_scale_shape = (
                        self.num_experts,
                        self.hidden_size // wsz,
                        1,
                        self.intermediate_dim,
                    )
                    w3_scale_shape = w1_scale_shape
                    w2_scale_shape = (
                        self.num_experts,
                        self.intermediate_dim // wsz,
                        1,
                        self.hidden_size,
                    )

                if hasattr(self, "w1_scale"):
                    del self.w1_scale
                self.w1_scale = nnx.Param(
                    jnp.zeros(w1_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if hasattr(self, "w3_scale"):
                    del self.w3_scale
                self.w3_scale = nnx.Param(
                    jnp.zeros(w3_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if hasattr(self, "w2_scale"):
                    del self.w2_scale
                self.w2_scale = nnx.Param(
                    jnp.zeros(w2_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if self.num_shared_experts > 0:
                    # fused kernel expects per-channel shared scale (1, 1, se_inter)
                    # — see _validate_fused_ep_moe_args / kernel.py:455-470
                    shared_scale_sharding = P(None, None, None)
                    se_inter = self.moe_shared_expert_intermediate_size * self.num_shared_experts

                    if hasattr(self, "w1_shared_scale"):
                        del self.w1_shared_scale
                    self.w1_shared_scale = nnx.Param(
                        jnp.zeros((1, 1, se_inter), dtype=jnp.float32),
                        out_sharding=shared_scale_sharding,
                    )

                    if hasattr(self, "w3_shared_scale"):
                        del self.w3_shared_scale
                    self.w3_shared_scale = nnx.Param(
                        jnp.zeros((1, 1, se_inter), dtype=jnp.float32),
                        out_sharding=shared_scale_sharding,
                    )

                    if hasattr(self, "w2_shared_scale"):
                        del self.w2_shared_scale
                    self.w2_shared_scale = nnx.Param(
                        jnp.zeros((1, 1, self.hidden_size), dtype=jnp.float32),
                        out_sharding=shared_scale_sharding,
                    )

                return

            # Replace original weights with quantized versions
            if self.quant_block_n is not None:
                # 2D block-wise quantization: scale shape (E, K//block_k, N//block_n)
                w1_value, w1_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
                w3_value, w3_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
                w2_value, w2_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
            else:
                # 1D sub-channel quantization: scale shape (E, K//wsz, N)
                w1_value, w1_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )
                w3_value, w3_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )
                w2_value, w2_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )

            # NOTE: Fused MoE shards the expert dimension across EP=(data*tensor).
            ep_sharding = P(("data", "tensor"), None, None)
            ep_scale_sharding = P(("data", "tensor"), None, None, None)

            self.w1 = nnx.Param(w1_value, out_sharding=ep_sharding)
            self.w3 = nnx.Param(w3_value, out_sharding=ep_sharding)
            self.w2 = nnx.Param(w2_value, out_sharding=ep_sharding)

            # Update scales (reshape to 4D for GMM kernel)
            if self.quant_block_n is not None:
                # 2D block-wise: expand block scales once so forward can run
                # through the fast 1D kernel path without changing semantics.
                w1_scale_4d = _expand_moe_block_scale(
                    w1_scale, self.intermediate_dim, self.quant_block_n
                )
                w3_scale_4d = _expand_moe_block_scale(
                    w3_scale, self.intermediate_dim, self.quant_block_n
                )
                w2_scale_4d = _expand_moe_block_scale(
                    w2_scale, self.hidden_size, self.quant_block_n
                )
            else:
                # (E, K//wsz, N) → (E, K//wsz, 1, N)
                w1_scale_4d = w1_scale.reshape(
                    w1_scale.shape[0], w1_scale.shape[1], 1, w1_scale.shape[2]
                )
                w3_scale_4d = w3_scale.reshape(
                    w3_scale.shape[0], w3_scale.shape[1], 1, w3_scale.shape[2]
                )
                w2_scale_4d = w2_scale.reshape(
                    w2_scale.shape[0], w2_scale.shape[1], 1, w2_scale.shape[2]
                )

            if hasattr(self, "w1_scale"):
                del self.w1_scale
            self.w1_scale = nnx.Param(
                w1_scale_4d,
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w3_scale"):
                del self.w3_scale
            self.w3_scale = nnx.Param(
                w3_scale_4d,
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w2_scale"):
                del self.w2_scale
            self.w2_scale = nnx.Param(
                w2_scale_4d,
                out_sharding=ep_scale_sharding,
            )

            if self.w1_shared is not None:
                w1_shared_value, w1_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1_shared.value,
                    axis=0,
                )
                w3_shared_value, w3_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3_shared.value,
                    axis=0,
                )
                w2_shared_value, w2_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2_shared.value,
                    axis=0,
                )

                self.w1_shared = nnx.Param(w1_shared_value, out_sharding=P(None, None))
                self.w3_shared = nnx.Param(w3_shared_value, out_sharding=P(None, None))
                self.w2_shared = nnx.Param(w2_shared_value, out_sharding=P(None, None))

                if hasattr(self, "w1_shared_scale"):
                    del self.w1_shared_scale
                self.w1_shared_scale = nnx.Param(
                    w1_shared_scale.reshape(
                        1,
                        1,
                        w1_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w3_shared_scale"):
                    del self.w3_shared_scale
                self.w3_shared_scale = nnx.Param(
                    w3_shared_scale.reshape(
                        1,
                        1,
                        w3_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w2_shared_scale"):
                    del self.w2_shared_scale
                self.w2_shared_scale = nnx.Param(
                    w2_shared_scale.reshape(
                        1,
                        1,
                        w2_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

    def __call__(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        block_config: FusedMoEBlockConfig | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Forward pass through the fused MoE layer."""
        assert hidden_states.ndim == 2

        w1_shared_val = self.w1_shared.value if self.w1_shared is not None else None
        w3_shared_val = self.w3_shared.value if self.w3_shared is not None else None
        w2_shared_val = self.w2_shared.value if self.w2_shared is not None else None
        w1_scale = self.w1_scale.value if self.w1_scale is not None else None
        w3_scale = self.w3_scale.value if self.w3_scale is not None else None
        w2_scale = self.w2_scale.value if self.w2_scale is not None else None
        w1_shared_scale = self.w1_shared_scale.value if self.w1_shared_scale is not None else None
        w3_shared_scale = self.w3_shared_scale.value if self.w3_shared_scale is not None else None
        w2_shared_scale = self.w2_shared_scale.value if self.w2_shared_scale is not None else None

        quant_block_k = self.quant_block_k if self.quant_block_k is not None else None

        output = fused_ep_moe(
            mesh=self.mesh,
            tokens=hidden_states,
            w1=self.w1.value,
            w2=self.w2.value,
            w3=self.w3.value,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=self.num_experts_per_tok,
            use_grouped_topk=self.use_grouped_topk,
            num_groups=self.num_groups,
            top_k_groups=self.top_k_groups,
            renormalize_topk_logits=self.renormalize_topk_logits,
            routed_scaling_factor=self.routed_scaling_factor,
            act_fn=self.activation,
            block_config=block_config,
            disable_a2a=self.disable_a2a,
            disable_dynamic_ffn1=self.disable_dynamic_ffn1,
            disable_dynamic_ffn2=self.disable_dynamic_ffn2,
            disable_weight_load=self.disable_weight_load,
            disable_a2a_s_tile_read=self.disable_a2a_s_tile_read,
            disable_a2a_s_acc_tile_write=self.disable_a2a_s_acc_tile_write,
            disable_shared_expert=self.disable_shared_expert,
            disable_all_reduce_metadata=self.disable_all_reduce_metadata,
            disable_sync_barrier=self.disable_sync_barrier,
            use_jax_allreduce_metadata=self.use_jax_allreduce_metadata,
            # Optional parameters (not used in basic case)
            quant_block_k=quant_block_k,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=w1_shared_val,
            w2_shared=w2_shared_val,
            w3_shared=w3_shared_val,
            w1_shared_scale=w1_shared_scale,
            w2_shared_scale=w2_shared_scale,
            w3_shared_scale=w3_shared_scale,
            b1=None,
            b2=None,
            b3=None,
            dp_axis_name="data",
            tp_axis_name="tensor",
        )

        if out_sharding is None:
            out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * output.ndim)))
        output = jax.sharding.reshard(output, out_sharding)
        return output


class FusedEPMoEV2(FusedEPMoE):
    """V2 fused EP-MoE layer using the Strix-style double-buffer kernel.

    Inherits weight init and quantization from FusedEPMoE. Overrides __call__
    to dispatch to fused_ep_moe_v2 with v2-specific flags.
    """

    def __call__(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        block_config=None,
        out_sharding: jax.sharding.Sharding | None = None,
        swiglu_limit: float | None = None,
        shared_swiglu_limit: float | None = None,
    ) -> jax.Array:
        from sgl_jax.srt.kernels.fused_moe.v2.kernel import fused_ep_moe_v2
        from sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs import (
            get_tuned_fused_moe_v2_block_config,
        )

        assert hidden_states.ndim == 2

        # ── Same ep_size * t_packing padding as FusedEPMoE.__call__ ──
        # The v2 kernel requires num_tokens divisible by ep_size*t_packing
        # (kernel.py:59-60 assert num_tokens % ep_size == 0). Without this
        # bs=1 with ep=4 produces local_num_tokens=0 and the kernel raises.
        # Pad with zero-weight rows, run the kernel, then slice back.
        num_tokens = hidden_states.shape[0]
        t_packing = get_dtype_packing(self.dtype)
        kernel_ep_size = get_ep_size(self.mesh, "data", "tensor")
        token_multiple = kernel_ep_size * t_packing
        pad = (-num_tokens) % token_multiple
        if pad:
            hidden_states = jnp.concatenate(
                [hidden_states, jnp.zeros((pad, hidden_states.shape[1]), hidden_states.dtype)],
                axis=0,
            )
            topk_weights = jnp.concatenate(
                [topk_weights, jnp.zeros((pad, topk_weights.shape[1]), topk_weights.dtype)],
                axis=0,
            )
            topk_ids = jnp.concatenate(
                [topk_ids, jnp.zeros((pad, topk_ids.shape[1]), topk_ids.dtype)],
                axis=0,
            )

        w1_scale = self.w1_scale.value if self.w1_scale is not None else None
        w3_scale = self.w3_scale.value if self.w3_scale is not None else None
        w2_scale = self.w2_scale.value if self.w2_scale is not None else None

        w1_shared_val = self.w1_shared.value if self.w1_shared is not None else None
        w3_shared_val = self.w3_shared.value if self.w3_shared is not None else None
        w2_shared_val = self.w2_shared.value if self.w2_shared is not None else None

        # SE per-channel scales are stored 3D (1, 1, out); the v2 kernel reads them
        # 2D (1, out). Squeeze here (not in quantize_weights, which the v1 path and
        # the weight mapping consume in 3D form). None for bf16 SE weights (Mode 3).
        w1_shared_scale = (
            self.w1_shared_scale.value[:, 0, :] if self.w1_shared_scale is not None else None
        )
        w3_shared_scale = (
            self.w3_shared_scale.value[:, 0, :] if self.w3_shared_scale is not None else None
        )
        w2_shared_scale = (
            self.w2_shared_scale.value[:, 0, :] if self.w2_shared_scale is not None else None
        )
        # In-kernel act-quant (fp8 token, Mode 1) needs both a quant-config
        # activation dtype and fp8 weights. Weight-only fp8 checkpoints stay in
        # Mode 2 (bf16 token x fp8 weight).
        enable_act_quant = (
            self.activation_quantized_dtype is not None and self.enable_act_quant_cfg is not False
        ) and (w1_scale is not None)

        if block_config is None:
            block_config = get_tuned_fused_moe_v2_block_config(
                num_tokens=hidden_states.shape[0],
                num_experts=self.num_experts,
                top_k=self.num_experts_per_tok,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_dim,
                dtype=hidden_states.dtype,
                weight_dtype=self.w1.value.dtype,
                # Tune lookup must key on the ACTUAL EP (full mesh = data*tensor),
                # not self.ep_size: the fused kernel always treats the whole 2D
                # mesh as its EP group, but self.ep_size comes from config.ep_size
                # (often left at 1, e.g. Ling3 via the long-ctx bench), which made
                # the lookup miss every tuned entry and silently fall back to the
                # untuned _large_expert_default. For models that set config.ep_size
                # == mesh size (e.g. MiMo) this is a no-op.
                ep_size=get_ep_size(self.mesh, "data", "tensor"),
                use_shared_expert=self.w1_shared is not None,
                use_grouped_topk=self.use_grouped_topk,
                enable_act_quant=enable_act_quant,
            )

        direct_scaled_dot = w1_scale is not None

        output = fused_ep_moe_v2(
            self.mesh,
            hidden_states,
            self.w1.value,
            self.w2.value,
            self.w3.value,
            topk_weights,
            topk_ids,
            self.num_experts_per_tok,
            act_fn=self.activation,
            swiglu_limit=swiglu_limit,
            shared_swiglu_limit=shared_swiglu_limit,
            block_config=block_config,
            quant_block_k=self.quant_block_k if hasattr(self, "quant_block_k") else None,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=w1_shared_val,
            w2_shared=w2_shared_val,
            w3_shared=w3_shared_val,
            w1_shared_scale=w1_shared_scale,
            w2_shared_scale=w2_shared_scale,
            w3_shared_scale=w3_shared_scale,
            enable_act_quant=enable_act_quant,
            direct_scaled_dot=direct_scaled_dot,
            dp_axis_name="data",
            tp_axis_name="tensor",
            # When local_num_experts >= 64, the v2 kernel's sflag budget
            # (16 KB) overflows because bt_scatter_overlap and interleave_bt
            # allocate semaphores that scale with num_bt and expert count.
            # Disable these optimizations for large-expert models.
            # See: sflag allocation 32KB > 16KB with 512/4=128 local experts.
            cross_expert_prefetch_mode=(
                "none"
                if self.num_experts // self.ep_size >= 64
                else "full"
            ),
            enable_bt_scatter_overlap=(
                False if self.num_experts // self.ep_size >= 64 else True
            ),
            interleave_bt=(
                False if self.num_experts // self.ep_size >= 64 else True
            ),
        )

        # Reshard the MoE output to the caller-requested layout. Under sequence
        # parallelism out_sharding carries the SP-aware reduce_sharding
        # (('data','tensor') on the scatter dim); without it we fall back to the
        # plain DP layout P('data', None). b79f9951 dropped this arg and hardcoded
        # the DP layout, which silently broke SP for MoE layers — restored here.
        if out_sharding is not None:
            output = jax.sharding.reshard(output, out_sharding)
        else:
            output = jax.sharding.reshard(
                output, jax.sharding.NamedSharding(self.mesh, P("data", None))
            )
        if pad:
            output = output[:num_tokens]
        return output


class FusedTPMoEV4(FusedEPMoE):
    """Tensor-Parallel MoE layer (v4 kernel).

    Holds ALL experts on every chip but with the intermediate dim TP-sharded
    1/tp. Replaces the EP a2a / barrier with a single psum across the tp axis.
    bf16-only; no quantization, no in-kernel shared expert, incompatible with
    EPLB (TP replicates every expert).

    Weight load reuses the FusedEPMoE EP-sharded layout. After
    ``WeightLoader.load_weights_from_safetensors`` completes, the model's
    ``_post_load_weights`` hook must call ``reshape_weights_for_tp()`` on each
    instance — this performs a one-time EP -> TP reshard and drops the EP-layout
    tensors. The kernel reads ``w1_tp / w2_tp / w3_tp`` thereafter.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        *,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        use_grouped_topk: bool = False,
        num_groups: int = 1,
        top_k_groups: int = 1,
        renormalize_topk_logits: bool = False,
        routed_scaling_factor: float | None = None,
        num_shared_experts: int = 0,
        moe_shared_expert_intermediate_size: int | None = None,
        quantization_config=None,
        **kwargs,
    ) -> None:
        # Guard 1: EPLB metadata is incompatible — TP holds every expert on every
        # chip; physical->logical mapping would silently mis-route. Hard-fail at
        # init so the error surfaces before weight load instead of as numerical
        # garbage at runtime.
        if get_global_expert_location_metadata() is not None:
            raise NotImplementedError(
                "MoEBackend.FUSED_V4 is incompatible with EPLB. "
                "TP replicates every expert on every chip; physical->logical "
                "expert remapping has no meaning under this backend. Disable "
                "EPLB before selecting fused_v4."
            )
        # Guard 2: shared expert must be external (matches sglang-jax bailing_moe_v3
        # convention; AInfer v4's decode top_k+1-slot fusion is not ported).
        if num_shared_experts > 0:
            raise NotImplementedError(
                "FUSED_V4 expects external shared expert (num_shared_experts=0). "
                f"Got num_shared_experts={num_shared_experts}. The caller "
                "(e.g. BailingMoeV3 decoder layer) must compute the shared "
                "expert separately and add it after the MoE call."
            )
        # Guard 3: bf16-only. Reject any quantization config that asks for
        # quantized weights or activations.
        if quantization_config is not None and (
            quantization_config.get_moe_weight_dtype() is not None
            or quantization_config.get_moe_activation_dtype() is not None
        ):
            raise NotImplementedError(
                "FUSED_V4 is bf16-only; quantization is not supported. "
                f"Got weight_dtype={quantization_config.get_moe_weight_dtype()}, "
                f"activation_dtype={quantization_config.get_moe_activation_dtype()}."
            )
        super().__init__(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=intermediate_dim,
            weight_dtype=weight_dtype,
            dtype=dtype,
            activation=activation,
            layer_id=layer_id,
            use_grouped_topk=use_grouped_topk,
            num_groups=num_groups,
            top_k_groups=top_k_groups,
            renormalize_topk_logits=renormalize_topk_logits,
            routed_scaling_factor=routed_scaling_factor,
            num_shared_experts=0,  # enforced by Guard 2
            moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
            quantization_config=quantization_config,
            **kwargs,
        )
        self._tp_weights_ready = False

    def quantize_weights(self, is_static: bool = False) -> None:
        # Defensive: the inherited path is a no-op when quantized_dtype is None
        # (Guard 3 already enforces that), but make doubly sure quantize is never
        # invoked silently on v4 weights — caller bugs surface as a clear error
        # instead of corrupted bf16 storage.
        if self.quantized_dtype is not None:
            raise RuntimeError(
                "FusedTPMoEV4.quantize_weights called with quantized_dtype="
                f"{self.quantized_dtype}; FUSED_V4 must remain bf16."
            )

    def reshape_weights_for_tp(self) -> None:
        """One-time EP -> TP reshard. Called from the model's _post_load_weights.

        Per AInfer prepare_v4_weights (ainfer/layers/tpu/moe.py:883-945) the path
        is EP-sharded -> fully replicated -> TP-sharded. The transient replicated
        intermediate is ~1 GiB/weight for ling_v3_flash (E=256-512, H=2560, I=768);
        on a 4-chip v7x the HBM headroom is tight, so strictly free the
        intermediate AND the old EP buffer before allocating the next layer's
        weights:

          full = device_put(self.wX.value, replicated)
          full.block_until_ready()
          tp   = device_put(full, tp_sharding)
          full.delete()           # release the replicated intermediate
          self.wX_tp = nnx.Param(tp)
          delattr(self, "wX")     # drop the old EP-sharded buffer

        Skipping any of these three steps doubles the HBM peak.
        """
        if self._tp_weights_ready:
            return

        logger.info(
            "FusedTPMoEV4 layer %d EP->TP reshard starting (E=%d, H=%d, I=%d, mesh=%s)",
            self.layer_id, self.num_experts, self.hidden_size, self.intermediate_dim,
            self.mesh.shape,
        )

        rep_sharding = NamedSharding(self.mesh, P(None, None, None))
        w13_sharding = NamedSharding(self.mesh, P(None, None, "tensor"))
        w2_sharding = NamedSharding(self.mesh, P(None, "tensor", None))

        for src_name, tgt_name, target_sharding in (
            ("w1", "w1_tp", w13_sharding),
            ("w3", "w3_tp", w13_sharding),
            ("w2", "w2_tp", w2_sharding),
        ):
            src_param = getattr(self, src_name)
            full = jax.device_put(src_param.value, rep_sharding)
            full.block_until_ready()
            tp = jax.device_put(full, target_sharding)
            tp.block_until_ready()
            try:
                full.delete()
            except Exception:
                # Some JAX builds raise if delete() runs after the buffer was
                # consumed by the next device_put. Best-effort; the GC will
                # collect when references drop.
                pass
            setattr(self, tgt_name, nnx.Param(tp))
            try:
                # Drop the old EP-sharded buffer aggressively. Without this the
                # 40+ layer model peaks at 2x weight bytes during reshard.
                getattr(self, src_name).value.delete()
            except Exception:
                pass
            delattr(self, src_name)

        self._tp_weights_ready = True
        logger.info("FusedTPMoEV4 layer %d EP->TP reshard done", self.layer_id)

    def __call__(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        block_config: FusedMoEBlockConfig | None = None,  # accepted, unused
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """TP MoE forward.

        v4 does NOT pad tokens on EP boundaries (the FusedEPMoE input padding is
        skipped). Tokens are conceptually replicated across the tp axis and the
        kernel does a final psum to combine partial down-projections.
        """
        try:
            from jax import shard_map
        except ImportError:
            from jax.experimental.shard_map import shard_map

        from sgl_jax.srt.kernels.fused_moe.v4.kernel import tp_moe_per_device

        assert hidden_states.ndim == 2, hidden_states.shape
        if not self._tp_weights_ready:
            raise RuntimeError(
                f"FusedTPMoEV4 layer {self.layer_id}: reshape_weights_for_tp() "
                "must be called from the model's _post_load_weights hook before "
                "forward."
            )

        # Sanity-check the kernel's TP slicing assumption up front so a bad
        # config fails with a readable message instead of a downstream shape
        # error inside ragged_dot. The kernel still works when I/tp is not a
        # multiple of 128 (it just wastes MXU cycles), so this is a soft warn
        # rather than a hard error.
        i_local = self.w2_tp.value.shape[1]
        if i_local % 128 != 0:
            logger.warning(
                "FusedTPMoEV4 layer %d: I_local=%d is not a multiple of 128; "
                "expect ~%.0f%% MXU padding waste in prefill.",
                self.layer_id, i_local, 100.0 * (1 - i_local / (((i_local + 127) // 128) * 128)),
            )

        topk_ids = topk_ids.astype(jnp.int32)
        num_experts = self.num_experts

        def _body(toks, w1l, w2l, w3l, ids, wts):
            return tp_moe_per_device(
                toks, w1l, w2l, w3l, ids, wts,
                num_experts=num_experts, tp_axis_name="tensor",
            )

        in_specs = (
            P("data", None),            # tokens (replicated across tensor axis)
            P(None, None, "tensor"),    # w1 [E, H, I/tp]
            P(None, "tensor", None),    # w2 [E, I/tp, H]
            P(None, None, "tensor"),    # w3 [E, H, I/tp]
            P("data", None),            # topk_ids
            P("data", None),            # topk_weights
        )
        out_specs = P("data", None)

        try:
            output = shard_map(
                _body, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs,
                check_rep=False,
            )(
                hidden_states,
                self.w1_tp.value,
                self.w2_tp.value,
                self.w3_tp.value,
                topk_ids,
                topk_weights,
            )
        except TypeError:
            output = shard_map(
                _body, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs,
            )(
                hidden_states,
                self.w1_tp.value,
                self.w2_tp.value,
                self.w3_tp.value,
                topk_ids,
                topk_weights,
            )

        # Same output contract as FusedEPMoE.
        if out_sharding is not None:
            output = jax.sharding.reshard(output, out_sharding)
        else:
            output = jax.sharding.reshard(
                output, jax.sharding.NamedSharding(self.mesh, P("data", None))
            )
        return output
