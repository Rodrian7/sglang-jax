#!/usr/bin/env python3
"""Dump the optimized, scheduled HLO of the real forward — for the Overlap
analysis (which collectives XLA made async + the compute scheduled in their
shadow). Compiles only (no weights, no run). The async-collective schedule is
TPU-target-specific, so this runs on the real device mesh.

Multi-host (e.g. 4-pod tp=32): launch on every node with matching --nnodes /
--dist-init-addr and a distinct --node-rank; rank 0 writes the HLO.

    PYTHONPATH=python python tools/dump_forward_hlo.py \
        --model-path /models/MiMo-V2-Pro-Private --tp 32 --dp 8 --phase prefill \
        --nnodes 4 --node-rank $R --dist-init-addr 10.116.17.6:30571 \
        --out /tmp/fwd_prefill.hlo
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--phase", choices=["extend", "decode", "prefill"], default="prefill")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--moe-backend", default="fused_v2")
    ap.add_argument(
        "--layers",
        type=int,
        default=None,
        help="keep first-N layers instead of the default representative set "
        "(one layer per distinct swa/moe type — covers all collectives even for "
        "dense-first/MoE-later models; avoids the full-70-layer XLA MSA crash)",
    )
    ap.add_argument("--nnodes", type=int, default=1)
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--dist-init-addr", default=None)
    ap.add_argument("--out", default="/tmp/fwd.hlo")
    args = ap.parse_args()

    import jax

    if args.nnodes > 1:
        jax.distributed.initialize(
            coordinator_address=args.dist_init_addr,
            num_processes=args.nnodes,
            process_id=args.node_rank,
        )
    print(
        f"process {jax.process_index()}/{jax.process_count()} devices={jax.device_count()}",
        file=sys.stderr,
        flush=True,
    )

    from sgl_jax.srt.utils.roofline.standalone_trace import compile_forward_hlo

    phase = "extend" if args.phase == "prefill" else args.phase
    hlo, meta = compile_forward_hlo(
        args.model_path,
        args.tp,
        args.dp,
        phase=phase,
        num_tokens=args.tokens,
        moe_backend=args.moe_backend,
        layers=args.layers,
        representative=args.layers is None,
    )
    if jax.process_index() == 0:
        import json

        with open(args.out, "w") as f:
            f.write(hlo)
        with open(args.out + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"HLO -> {args.out}  ({len(hlo):,} chars)", flush=True)
        print(
            f"  compiled: {meta['n_layers_compiled']} layers {meta['layer_types']} · "
            f"{meta['tokens_global']} tokens · SP {'on' if meta['sp_triggered'] else 'off'}"
            f" (threshold {meta['sp_threshold_tokens']})",
            flush=True,
        )
    else:
        print(f"rank {jax.process_index()} compiled OK", flush=True)


if __name__ == "__main__":
    main()
