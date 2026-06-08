"""Cross-validate the auto-derived (jaxpr) layer graph against the hand-written
build_layer_graph. Run: python tools/roofline_jaxpr_validate.py
"""

import sys

sys.path.insert(0, "python")

import jax  # noqa: E402

from sgl_jax.srt.utils.roofline import references  # noqa: E402
from sgl_jax.srt.utils.roofline import critical_path, descriptors  # noqa: E402
from sgl_jax.srt.utils.roofline import graph as G  # noqa: E402
from sgl_jax.srt.utils.roofline import graph_from_jaxpr as GJ  # noqa: E402
from sgl_jax.srt.utils.roofline.report import HardwarePeaks  # noqa: E402

cfg = dict(
    hidden_size=4096,
    num_hidden_layers=48,
    vocab_size=152576,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=192,
    v_head_dim=128,
    n_routed_experts=256,
    num_experts_per_tok=8,
    moe_intermediate_size=1536,
    intermediate_size=16384,
)
# tp=1, ep=1 => unsharded, apples-to-apples with the unsharded jaxpr trace
par = dict(tp=1, ep=1, batch=8, seq_len=4096, chunk=2048, devices=1)
phase = "decode"
peaks = HardwarePeaks()

H = cfg["hidden_size"]
nh, nkv, hd, vhd = 64, 4, 192, 128
NEXP, TOPK, MOEF = 256, 8, 1536
tokens = par["batch"]
ctx = par["seq_len"]


def mimo_pallas_coster(eqn, occ):
    """occ 0 = attention, occ 1 = experts (reference forward order)."""
    if occ == 0:
        c = references.attention_cost(
            num_q_heads=nh,
            num_kv_heads=nkv,
            head_dim=hd,
            v_head_dim=vhd,
            q_tokens=tokens,
            kv_tokens=ctx,
            total_interactions=tokens * ctx,
        )
        return {
            "flops": c["flops"],
            "hbm_bytes": c["hbm_bytes"],
            "category": "attention",
            "label": "attention[PALLAS]",
        }
    c = references.moe_experts_cost(
        tokens_per_device=tokens * TOPK, local_experts=NEXP, d=H, f=MOEF
    )
    return {
        "flops": c["flops"],
        "hbm_bytes": c["hbm_bytes"],
        "category": "moe",
        "label": "experts[PALLAS]",
    }


# --- auto-derived graph from jaxpr ---
fn, args = descriptors.reference_forward("MiMoV2ForCausalLM", cfg, phase, par)
jaxpr = jax.make_jaxpr(fn)(*args).jaxpr
ag = GJ.build_graph_from_jaxpr(jaxpr, pallas_coster=mimo_pallas_coster)
ag_fused = GJ.fuse(ag)

# --- hand-written graph (full attention + MoE layer) ---
hw = G.build_layer_graph(cfg, phase, par, swa=False, moe=True)


def totals(g):
    sc = sum(o.roofline().compute_ms(peaks) for o in g.ops)
    sh = sum(o.roofline().hbm_ms(peaks) for o in g.ops)
    fl = sum(o.flops for o in g.ops)
    hb = sum(o.hbm_bytes for o in g.ops)
    cp = critical_path.analyze(g, peaks)
    return sc, sh, fl, hb, cp


def show(name, g):
    sc, sh, fl, hb, cp = totals(g)
    print(f"\n### {name}: {len(g.ops)} ops, {len(g.tensors)} tensors")
    print(f"  FLOPs={fl/1e9:.3f} GFLOP   HBM={hb/1e6:.1f} MB")
    print(f"  sum_compute={sc:.4f}ms  sum_hbm={sh:.4f}ms  t_critical={cp['t_critical_ms']:.4f}ms")
    print("  category breakdown (compute_ms / hbm_ms / GFLOP):")
    from collections import defaultdict

    cat = defaultdict(lambda: [0.0, 0.0, 0.0])
    for o in g.ops:
        r = o.roofline()
        cat[o.category][0] += r.compute_ms(peaks)
        cat[o.category][1] += r.hbm_ms(peaks)
        cat[o.category][2] += o.flops / 1e9
    for c, (cm, hm, gf) in sorted(cat.items(), key=lambda x: -x[1][0] - x[1][1]):
        print(f"    {c:12s} compute={cm:8.4f}ms  hbm={hm:8.4f}ms  {gf:8.3f} GFLOP")


show("AUTO (jaxpr, unfused)", ag)
show("AUTO (jaxpr, fused)", ag_fused)
show("HAND-WRITTEN build_layer_graph", hw)

print("\n=== source attribution sample (auto, top compute ops) ===")
for o in sorted(ag.ops, key=lambda o: -o.flops)[:8]:
    print(f"  {o.flops/1e9:7.3f} GFLOP  {o.category:10s} {o.label:16s} <- {o.source}")
