"""Dump the compiled HLO for grouped_topk_pallas_v3 to locate the layout `copy` ops."""
import argparse
import jax
import jax.numpy as jnp
from sgl_jax.srt.kernels.grouped_topk.v2.kernel3 import grouped_topk_pallas_v3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=16384)
    ap.add_argument("--E", type=int, default=256)
    a = ap.parse_args()
    G, Gtop, k = 8, 4, 8
    lg = jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(0), (a.T, a.E), jnp.float32))
    b = jax.random.normal(jax.random.PRNGKey(1), (a.E,), jnp.float32) * 0.1
    fn = jax.jit(lambda l, bb: grouped_topk_pallas_v3(
        l, bb, num_expert_group=G, topk_group=Gtop, topk=k))
    txt = fn.lower(lg, b).compile().as_text()
    print(f"=== compiled HLO for T={a.T} E={a.E} (entry only) ===")
    in_entry = False
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("ENTRY"):
            in_entry = True
        if in_entry and any(x in s for x in (
            "copy", "transpose", "custom-call", "bitcast", "fusion(", "ROOT",
            "parameter(", "-> ")):
            print(s[:220])

if __name__ == "__main__":
    main()
