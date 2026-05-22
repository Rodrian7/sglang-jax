"""EP8 correctness: fused_ep_moe_v2 (ilv_bt=0/1) vs ref_moe, fp8 MiMo V2."""
from __future__ import annotations
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax import lax

t0 = time.time()
def log(msg):
    if jax.process_index() == 0:
        print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)

jax.distributed.initialize()
num_devices = jax.device_count()
log(f"initialized: {num_devices} devices")

sys.path.insert(0, "/tmp/tpu_logs/sglang-jax/python")
from sgl_jax.srt.kernels.fused_moe.v2.kernel import (
    fused_ep_moe_v2, ref_moe, FusedMoEBlockConfig,
)

P = jax.sharding.PartitionSpec
devices = np.array(jax.devices()).reshape(1, num_devices)
mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

E, d, f, top_k = 384, 6144, 2048, 8
num_tokens = int(os.environ.get("TEST_TOKENS", "512"))
qbk = 128

log(f"config: E={E} d={d} f={f} k={top_k} ep={num_devices} tokens={num_tokens}")

key = jax.random.key(42)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)

log("creating arrays...")
tokens = jax.random.normal(k1, (num_tokens, d), dtype=jnp.bfloat16) * 0.01
w1 = jax.random.normal(k2, (E, d, f), dtype=jnp.bfloat16) * 0.01
w2 = jax.random.normal(k3, (E, f, d), dtype=jnp.bfloat16) * 0.01
w3 = jax.random.normal(k4, (E, d, f), dtype=jnp.bfloat16) * 0.01

gating = jax.random.normal(k5, (num_tokens, E), dtype=jnp.float32)
_, topk_idx = lax.top_k(gating, top_k)
topk_wts = jax.nn.softmax(
    jnp.take_along_axis(gating, topk_idx, axis=-1), axis=-1)

log("quantizing to fp8...")

def quantize_weight(w):
    E_loc, K, N = w.shape
    w_f32 = w.astype(jnp.float32).reshape(E_loc, K // qbk, qbk, N)
    amax = jnp.max(jnp.abs(w_f32), axis=2, keepdims=True)
    scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
    w_q = (w_f32 / scale).astype(jnp.float8_e4m3fn).reshape(E_loc, K, N)
    return w_q, scale.astype(jnp.float32)

w1_q, w1_scale = quantize_weight(w1)
w2_q, w2_scale = quantize_weight(w2)
w3_q, w3_scale = quantize_weight(w3)

log("sharding...")
tokens_s = jax.device_put(tokens, ep_sharding)
w1_s = jax.device_put(w1_q, ep_sharding)
w2_s = jax.device_put(w2_q, ep_sharding)
w3_s = jax.device_put(w3_q, ep_sharding)
w1_scale_s = jax.device_put(w1_scale, ep_sharding)
w2_scale_s = jax.device_put(w2_scale, ep_sharding)
w3_scale_s = jax.device_put(w3_scale, ep_sharding)
topk_wts_s = jax.device_put(topk_wts, ep_sharding)
topk_idx_s = jax.device_put(topk_idx, ep_sharding)

bc = FusedMoEBlockConfig(bt=16, bf=1024, btc=32, bse=256, bts=32)

common = dict(
    block_config=bc,
    quant_block_k=qbk,
    w1_scale=w1_scale_s, w2_scale=w2_scale_s, w3_scale=w3_scale_s,
    direct_scaled_dot=True,
    skip_inter_bt_sync=True,
)

log("computing ref_moe (ground truth)...")
ref = ref_moe(
    tokens, w1_q, w2_q, w3_q, topk_wts, topk_idx, top_k,
    quant_block_k=qbk, w1_scale=w1_scale, w2_scale=w2_scale, w3_scale=w3_scale,
)
ref_f32 = np.asarray(ref).astype(np.float32)
log(f"ref done, range=[{np.min(ref_f32):.4f}, {np.max(ref_f32):.4f}]")

def run_and_compare(label, **extra):
    log(f"running {label}...")
    out = fused_ep_moe_v2(
        mesh, tokens_s, w1_s, w2_s, w3_s,
        topk_wts_s, topk_idx_s, top_k,
        **common, **extra,
    )
    out_np = jax.device_get(
        jax.device_put(out, jax.sharding.NamedSharding(mesh, P()))
    ).astype(np.float32)
    max_abs = float(np.max(np.abs(out_np - ref_f32)))
    denom = float(np.max(np.abs(ref_f32))) + 1e-6
    rel_err = max_abs / denom
    log(f"  {label}: max_abs={max_abs:.6f} rel_err={rel_err:.8f}")
    if rel_err < 1e-2:
        log(f"  {label}: PASS")
    else:
        log(f"  {label}: FAIL")
    return rel_err

if jax.process_index() == 0:
    log("")

e0 = run_and_compare("ilv_bt=0", interleave_bt=False)
e1 = run_and_compare("ilv_bt=1", interleave_bt=True)

if jax.process_index() == 0:
    cross = float(np.max(np.abs(
        jax.device_get(jax.device_put(
            fused_ep_moe_v2(mesh, tokens_s, w1_s, w2_s, w3_s,
                            topk_wts_s, topk_idx_s, top_k,
                            interleave_bt=False, **common),
            jax.sharding.NamedSharding(mesh, P()))).astype(np.float32)
        - jax.device_get(jax.device_put(
            fused_ep_moe_v2(mesh, tokens_s, w1_s, w2_s, w3_s,
                            topk_wts_s, topk_idx_s, top_k,
                            interleave_bt=True, **common),
            jax.sharding.NamedSharding(mesh, P()))).astype(np.float32)
    )))
    log(f"\nilv_bt=0 vs ilv_bt=1 max_abs_diff={cross:.6f}")
    ok = e0 < 1e-2 and e1 < 1e-2
    log(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
    if not ok:
        sys.exit(1)
