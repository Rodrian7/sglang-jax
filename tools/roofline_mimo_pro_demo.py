"""Standalone roofline demo for MiMo-V2-Pro at the validated EP layout
(tp=32, dp=8, devices=32, +SP). Prints all views (B/C cost, D/E critical path +
fusion, F jaxpr auto-graph) for decode and prefill, with NO --model-path needed.

Run:  python tools/roofline_mimo_pro_demo.py
      python tools/roofline_mimo_pro_demo.py > /tmp/roofline_mimo_pro.txt
"""

import sys

sys.path.insert(0, "python")

from sgl_jax.srt.utils.roofline import descriptors  # noqa: E402
from sgl_jax.srt.utils.roofline import interp  # noqa: E402
from sgl_jax.srt.utils.roofline import graph_from_jaxpr as gjax  # noqa: E402
from sgl_jax.srt.utils.roofline.report import (  # noqa: E402
    HardwarePeaks,
    render_cost_views,
    render_graph_views,
)

# MiMo-V2-Pro dims (config.json-equivalent)
CFG = dict(
    hidden_size=6144,
    num_hidden_layers=70,
    vocab_size=152576,
    num_attention_heads=128,
    num_key_value_heads=8,
    head_dim=192,
    v_head_dim=128,
    n_routed_experts=384,
    num_experts_per_tok=8,
    moe_intermediate_size=2048,
    intermediate_size=16384,
    sliding_window_size=128,
    hybrid_layer_pattern=[1] * 60 + [0] * 10,  # 60 SWA + 10 full
    moe_layer_freq=[1] * 69 + [0],  # 69 MoE + 1 dense
)
# Validated MiMo-V2-Pro EP layout (matches the server launch flags)
PAR = dict(
    tp=32,
    dp=8,
    ep=32,
    devices=32,
    enable_sp=True,
    moe_backend="fused_v2",
    batch=64,
    seq_len=4096,
    chunk=16384,
)
ARCH = "MiMoV2ForCausalLM"
PEAKS = HardwarePeaks()


def banner(s):
    print("\n" + "#" * 100 + f"\n# {s}\n" + "#" * 100)


for phase in ("decode", "prefill"):
    banner(f"PHASE = {phase}   (MiMo-V2-Pro, tp=32 dp=8 devices=32 +SP)")
    model = descriptors.build(ARCH, CFG, phase, PAR, PEAKS)
    print(render_cost_views(model))
    print("\n" + render_graph_views(interp.graph_analysis(CFG, phase, PAR, PEAKS)))
    print("\n" + gjax.render_auto_graph(gjax.analyze_reference(ARCH, CFG, phase, PAR, PEAKS)))
