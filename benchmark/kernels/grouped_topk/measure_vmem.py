"""Measure the real peak scoped-VMEM of `grouped_topk_pallas` and verify what scales with topk.

Method: compile each (BT, E, topk, unroll) ONCE at the compiler's default VMEM limit (fast -- no
near-threshold spilling). If it fits, we only learn "<= default". If it OOMs, the Mosaic/XLA error
states the exact bytes it tried to allocate -> that's the config's peak VMEM need. We push topk high
enough that full-unroll OOMs at several points, then read the slope off those exact byte counts:

    full-unroll  peak grows ~ topk * (BT*E*4)   slope == BT*E*4   (the replicated [BT,E] `cur`)
    rolled       peak ~ const (one live [BT,E])  -> always fits, never OOMs

If the full-unroll OOM bytes rise by ~BT*E*4 per unit topk (NOT BT*128*4), the variable that gets
topk live copies is `cur`, confirming the kernel comment. Run on a TPU host:

    python -m benchmark.kernels.grouped_topk.measure_vmem \
        --T 2048 --E 256 --G 8 --Gtop 4 --topks 16,32,64,128 --unroll full,rolled
"""

import argparse
import re

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.grouped_topk.v1.kernel import grouped_topk_pallas

MiB = 1 << 20

# Mosaic/XLA VMEM-OOM messages vary by version; pull the largest byte count out of any of them, e.g.
#   "Failed to allocate request for 33.55M (35192832B) on device ... (VMEM)"
#   "...VMEM... requested 35192832 bytes ... limit 33554432 bytes"
_BYTES_RE = re.compile(r"(\d[\d,]*)\s*(?:B\b|bytes)")


def _oom_bytes(msg):
    """Largest byte count mentioned in an OOM message (the requested allocation), or None."""
    nums = [int(m.replace(",", "")) for m in _BYTES_RE.findall(msg)]
    return max(nums) if nums else None


def compile_vmem(bt, e, G, Gtop, topk, full_unroll):
    """Compile once at the default VMEM limit. Returns ('ok', None) or ('oom', bytes|None)."""
    x = jnp.zeros((bt, e), jnp.float32)
    bias = jnp.zeros((e,), jnp.float32)
    f = jax.jit(
        lambda x, b: grouped_topk_pallas(
            x,
            b,
            num_expert_group=G,
            topk_group=Gtop,
            topk=topk,
            block_tokens=bt,  # single block: one [BT,E] tile, no grid noise
            unroll=full_unroll,
        )
    )
    try:
        f.lower(x, bias).compile()
        return "ok", None
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "resource_exhausted" in msg or "vmem" in msg or "exceeds" in msg or "allocate" in msg:
            return "oom", _oom_bytes(str(exc))
        raise


def _parse_csv_int(s):
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2048, help="block_tokens (BT), a single block")
    ap.add_argument("--E", type=int, default=256, help="num experts")
    ap.add_argument("--G", type=int, default=8, help="num expert groups")
    ap.add_argument("--Gtop", type=int, default=4, help="topk_group")
    ap.add_argument("--topks", type=_parse_csv_int, default=[16, 32, 64, 128])
    ap.add_argument("--unroll", type=str, default="full,rolled", help="full,rolled")
    args = ap.parse_args()

    bt, e = args.T, args.E
    cur_step = bt * e * 4  # bytes added per +1 topk IF `cur` [BT,E] is what replicates
    buf_step = bt * 128 * 4  # bytes added per +128 topk IF ids/w_buf [BT,padded_topk] dominates
    print(f"BT={bt} E={e} G={args.G} Gtop={args.Gtop}")
    print(f"  predicted per-topk slope if `cur`:        BT*E*4   = {cur_step/MiB:.3f} MiB")
    print(f"  predicted per-topk slope if `ids/w_buf`:  BT*128*4 = {buf_step/MiB:.3f} MiB")

    for mode in args.unroll.split(","):
        full = mode == "full"
        print(f"\n[{mode}]")
        prev_k = prev_v = None
        for k in args.topks:
            status, nbytes = compile_vmem(bt, e, args.G, args.Gtop, k, full)
            if status == "ok":
                print(f"  topk={k:4d}  fits under default VMEM")
                prev_k = prev_v = None  # reset slope baseline; OK rows have no exact number
                continue
            if nbytes is None:
                print(f"  topk={k:4d}  OOM (bytes not parsed from message)")
                continue
            line = f"  topk={k:4d}  peak={nbytes/MiB:8.3f} MiB"
            if prev_v is not None:
                slope = (nbytes - prev_v) / (k - prev_k)
                line += f"  slope={slope/MiB:6.3f} MiB/topk  (cur:{cur_step/MiB:.3f})"
            print(line)
            prev_k, prev_v = k, nbytes


if __name__ == "__main__":
    main()
