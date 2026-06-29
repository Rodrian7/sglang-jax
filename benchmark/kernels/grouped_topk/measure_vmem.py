"""Measure the real peak scoped-VMEM of `grouped_topk_pallas` and verify what scales with topk.

Method: compile each (BT, E, topk, unroll) under a deliberately tiny `vmem_limit_bytes` so Mosaic's
stack allocator immediately reports the kernel's total scoped-VMEM need and bails -- the message

    "Scoped allocation with size 58.33M and limit 32.00M exceeded scoped vmem limit by 26.33M"

states the exact size. This stack-size check is fast at any limit (unlike a near-threshold limit,
which makes Mosaic spill/retry for ~30s). We parse that size for every config and read the slope:

    full-unroll  peak grows with topk  (XLA keeps several live [BT,E] `cur` copies, not all topk)
    rolled       peak ~ const          (one live [BT,E])

If full-unroll's size rises with topk while rolled stays flat, the variable that replicates is `cur`
([BT,E]), confirming the kernel comment. Run on a TPU host:

    python -m benchmark.kernels.grouped_topk.measure_vmem \
        --T 2048 --E 256 --G 8 --Gtop 4 --topks 8,16,32,64,128 --unroll full,rolled
"""

import argparse
import re

import jax
import jax.numpy as jnp

from python.sgl_jax.srt.kernels.grouped_topk.v1.kernel2 import grouped_topk_pallas

MiB = 1 << 20

# Force OOM with a tiny limit so the stack allocator prints the total need immediately.
TINY_LIMIT = 1 << 10  # 1 KB -- below any real config, fails fast

# "Scoped allocation with size 58.33M and limit 32.00M exceeded ..." -> 58.33 (MiB)
_SIZE_RE = re.compile(r"Scoped allocation with size ([\d.]+)M")


def scoped_vmem_mib(bt, e, G, Gtop, topk, full_unroll):
    """Total scoped-VMEM (MiB) the kernel needs, from the forced-OOM stack message. None if unparsed."""
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
            vmem_limit_bytes=TINY_LIMIT,
        )
    )
    try:
        f.lower(x, bias).compile()
        return 0.0  # compiled even under 1 MiB (shouldn't happen for these sizes)
    except Exception as exc:  # noqa: BLE001
        m = _SIZE_RE.search(str(exc))
        if m:
            return float(m.group(1))
        msg = str(exc).lower()
        if "resource_exhausted" in msg or "vmem" in msg:
            return None  # OOM but size not parsed
        raise


def _parse_csv_int(s):
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2048, help="block_tokens (BT), a single block")
    ap.add_argument("--E", type=int, default=256, help="num experts")
    ap.add_argument("--G", type=int, default=8, help="num expert groups")
    ap.add_argument("--Gtop", type=int, default=4, help="topk_group")
    ap.add_argument("--topks", type=_parse_csv_int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--unroll", type=str, default="full,rolled", help="full,rolled")
    args = ap.parse_args()

    bt, e = args.T, args.E
    tile_mib = bt * e * 4 / MiB  # one [BT,E] f32 tile
    print(f"BT={bt} E={e} G={args.G} Gtop={args.Gtop}  | one [BT,E] tile = {tile_mib:.3f} MiB")

    for mode in args.unroll.split(","):
        full = mode == "full"
        print(f"\n[{mode}]")
        prev_k = prev_v = None
        for k in args.topks:
            v = scoped_vmem_mib(bt, e, args.G, args.Gtop, k, full)
            if v is None:
                print(f"  topk={k:4d}  OOM (size not parsed)")
                continue
            line = f"  topk={k:4d}  scoped_vmem={v:8.3f} MiB"
            if prev_v is not None and k != prev_k:
                slope = (v - prev_v) / (k - prev_k)  # MiB per unit topk
                line += f"  slope={slope:6.3f} MiB/topk  ([BT,E]={tile_mib:.3f})"
            print(line)
            prev_k, prev_v = k, v


if __name__ == "__main__":
    main()

