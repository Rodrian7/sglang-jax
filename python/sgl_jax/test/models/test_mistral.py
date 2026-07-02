import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from transformers import MistralConfig

from sgl_jax.srt.models.mistral import MistralForCausalLM, MistralForCausalLMMistralFormat
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


mesh = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(mesh)


def _make_config(num_layers: int = 1):
    return MistralConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=1024,
        sliding_window=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


class TestMistralModel(unittest.TestCase):
    def test_patch_model_config_fills_missing_head_dim(self):
        config = SimpleNamespace(hidden_size=4096, num_attention_heads=32, head_dim=None)
        model_config = SimpleNamespace(
            hf_text_config=config,
            hf_config=config,
            head_dim=None,
            v_head_dim=None,
        )

        MistralForCausalLM.patch_model_config(model_config)

        self.assertEqual(config.head_dim, 128)
        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.v_head_dim, 128)

    def test_mistral_architectures_are_registered(self):
        cls, arch = ModelRegistry.resolve_model_cls(["MistralForCausalLM"])
        self.assertIs(cls, MistralForCausalLM)
        self.assertEqual(arch, "MistralForCausalLM")

        cls, arch = ModelRegistry.resolve_model_cls(["MistralForCausalLMMistralFormat"])
        self.assertIs(cls, MistralForCausalLMMistralFormat)
        self.assertEqual(arch, "MistralForCausalLMMistralFormat")

    def test_sliding_window_is_attached_to_attention_layer(self):
        with jax.set_mesh(mesh):
            model = MistralForCausalLM(config=_make_config(), mesh=mesh, dtype=jnp.bfloat16)

        layer = model.model.layers[0]
        self.assertEqual(layer.self_attn.attn.sliding_window_size, 128)

    def test_native_mistral_weight_mapping_targets_llama_modules(self):
        with jax.set_mesh(mesh):
            model = MistralForCausalLM(
                config=_make_config(num_layers=2), mesh=mesh, dtype=jnp.bfloat16
            )

        hf_mappings = model._create_llama_weight_mappings()
        native_mappings = model._create_mistral_native_weight_mappings(hf_mappings)

        self.assertEqual(
            native_mappings["tok_embeddings.weight"].target_path,
            "model.embed_tokens.embedding",
        )
        self.assertEqual(native_mappings["output.weight"].target_path, "lm_head.embedding")
        self.assertEqual(
            native_mappings["layers.1.attention_norm.weight"].target_path,
            "model.layers.1.input_layernorm.scale",
        )
        self.assertEqual(
            native_mappings["layers.1.attention.wq.weight"].target_path,
            "model.layers.1.self_attn.q_proj.weight",
        )
        self.assertEqual(
            native_mappings["layers.1.feed_forward.w1.weight"].target_path,
            "model.layers.1.mlp.gate_proj.weight",
        )
        self.assertEqual(
            native_mappings["layers.1.feed_forward.w2.weight"].target_path,
            "model.layers.1.mlp.down_proj.weight",
        )
        self.assertEqual(
            native_mappings["layers.1.feed_forward.w3.weight"].target_path,
            "model.layers.1.mlp.up_proj.weight",
        )


if __name__ == "__main__":
    unittest.main()
