"""On-TPU tie-break check: does grouped_topk_pallas_v3 match the sort-based reference id-for-id,
including exact-tie inputs? (Interpret can't tell — numpy argmax is always lowest-index.)"""
import jax
import jax.numpy as jnp
import numpy as np
from sgl_jax.srt.kernels.grouped_topk.v2.kernel3 import grouped_topk_pallas_v3

G, Gtop, k, E = 8, 4, 8, 256

def ref(rl, cb):
    rl = rl.astype(jnp.float32); n = rl.shape[0]
    s = rl + cb[None, :]; sg = s.reshape(n, G, -1)
    gs = jnp.sum(jax.lax.top_k(sg, k=2)[0], axis=-1)
    gi = jax.lax.top_k(gs, k=Gtop)[1]
    gm = jnp.clip(jax.nn.one_hot(gi, G).sum(axis=1), 0, 1)
    epg = rl.shape[-1] // G
    sm = jnp.broadcast_to(gm[..., None], (n, G, epg)).reshape(n, -1)
    tmp = jnp.where(sm, s, float("-inf"))
    ids = jax.lax.top_k(tmp, k=k)[1]
    return jnp.take_along_axis(rl, ids, axis=1), ids

def run(name, lg, b):
    wr, ir = ref(lg, b)
    wv, iv = grouped_topk_pallas_v3(lg, b, num_expert_group=G, topk_group=Gtop, topk=k)
    jax.block_until_ready((wv, iv))
    ir, iv = np.array(ir), np.array(iv)
    idfor = np.array_equal(iv, ir)
    seteq = np.array_equal(np.sort(iv, 1), np.sort(ir, 1))
    print(f"[{name}] id-for-id={idfor}  set-equal={seteq}")
    print(f"   ref row0: {ir[0]}")
    print(f"   v3  row0: {iv[0]}")

print(f"device={jax.devices()[0].device_kind}")
# 1) random (no ties)
lg = jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(0), (512, E), jnp.float32))
b = jax.random.normal(jax.random.PRNGKey(9), (E,), jnp.float32) * 0.1
run("random", lg, b)
# 2) all-equal flat tie
run("flat-tie", jnp.full((8, E), 0.5, jnp.float32), jnp.zeros((E,), jnp.float32))
# 3) partial tie: experts 3 and 5 identical
lg2 = jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(3), (512, E), jnp.float32))
lg2 = lg2.at[:, 5].set(lg2[:, 3])
run("partial-tie(3==5)", lg2, b)
print("=== ties check exit: 0 ===")
