"""Measure the real peak scoped-VMEM of `grouped_topk_pallas` and verify what scales with topk.

Method (black-box): for each (BT, E, topk, unroll) binary-search the smallest `vmem_limit_bytes`
that still COMPILES (lower+compile only -- no device run needed). That threshold is the config's
peak VMEM need. Then look at how peak grows with topk at fixed (BT, E):

    full-unroll  -> peak ~= C + topk * (BT*E*4)      slope == BT*E*4   (the `cur` [BT,E] copies)
    rolled       -> peak ~= const                    slope ~= 0        (one live [BT,E])

If the measured full-unroll slope matches BT*E*4 (and NOT BT*padded_topk*4), the variable that gets
topk live copies is `cur`, confirming the kernel comment. The script prints both the raw thresholds
and the per-step slope next to the BT*E*4 vs BT*128*4 predictions so the two are easy to tell apart.

Run on a TPU host:

    python -m benchmark.kernels.grouped_topk.measure_vmem \
        --T 2048 --E 256 --G 8 --Gtop 4 --topks 8,16,32,64 --unroll full,rolled
"""

import argparse

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.grouped_topk.v1.kernel import grouped_topk_pallas

MiB = 1 << 20


def _compiles(bt, e, G, Gtop, topk, full_unroll, limit_bytes):
    """True iff the kernel lowers+compiles under the given VMEM cap."""
    x = jnp.zeros((bt, e), jnp.float32)
    bias = jnp.zeros((e,), jnp.float32)
    f = jax.jit(
        lambda x, b: grouped_topk_pallas(
            x,
            b,
            num_expert_group=G,
            topk_group=Gtop,
            topk=topk,
            block_tokens=bt,  # single block: measure one [BT,E] tile, no grid noise
            unroll=full_unroll,
            vmem_limit_bytes=limit_bytes,
        )
    )
    try:
        f.lower(x, bias).compile()
        return True
    except Exception as exc:  # noqa: BLE001
        # Only a VMEM/resource-exhaustion failure means "limit too small". Anything else
        # (bad signature, missing attr, ...) is a real bug -- re-raise so it isn't silently
        # reported as OOM across the whole sweep.
        msg = str(exc).lower()
        if "resource_exhausted" in msg or "vmem" in msg or "exceeds" in msg or "allocate" in msg:
            return False
        raise


def peak_vmem(bt, e, G, Gtop, topk, full_unroll, lo=64 << 10, hi=128 * MiB):
    """Smallest vmem_limit_bytes (within [lo, hi]) that still compiles, by binary search."""
    if not _compiles(bt, e, G, Gtop, topk, full_unroll, hi):
        return None  # doesn't fit even at hi
    grain = 64 << 10  # 64 KiB resolution
    while hi - lo > grain:
        mid = (lo + hi) // 2
        if _compiles(bt, e, G, Gtop, topk, full_unroll, mid):
            hi = mid
        else:
            lo = mid
    return hi


def _parse_csv_int(s):
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2048, help="block_tokens (BT), a single block")
    ap.add_argument("--E", type=int, default=256, help="num experts")
    ap.add_argument("--G", type=int, default=8, help="num expert groups")
    ap.add_argument("--Gtop", type=int, default=4, help="topk_group")
    ap.add_argument("--topks", type=_parse_csv_int, default=[8, 16, 32, 64])
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
            v = peak_vmem(bt, e, args.G, args.Gtop, k, full)
            if v is None:
                print(f"  topk={k:4d}  peak=OOM (>128MiB)")
                continue
            line = f"  topk={k:4d}  peak={v/MiB:7.3f} MiB"
            if prev_v is not None:
                slope = (v - prev_v) / (k - prev_k)  # bytes per unit topk
                line += f"  slope={slope/MiB:6.3f} MiB/topk  (cur:{cur_step/MiB:.3f})"
            print(line)
            prev_k, prev_v = k, v


if __name__ == "__main__":
    main()
