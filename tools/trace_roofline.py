#!/usr/bin/env python3
"""Standalone, server-free forward tracer for the roofline tool.

Traces the REAL forward of ANY registered sglang-jax model on CPU (fake device
mesh, abstract weights via eval_shape) -- no TPU, no checkpoint, no server. The
emitted JSON (per-device ops + real models/*.py source + Pallas kernels) is the
generic, per-model-code-free input the roofline analyzer consumes.

This wrapper sets JAX_PLATFORMS=cpu + the fake-device count BEFORE importing jax,
so you don't have to remember the env. On a TPU host it traces on TPU instead
(omit nothing; it auto-detects). Example (CPU pod / laptop):

    PYTHONPATH=python python tools/trace_roofline.py \
        --model-path /models/MiMo-V2-Pro-Private --tp 32 --dp 8 --phase extend \
        --out /tmp/fwd.json
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--phase", choices=["extend", "decode"], default="extend")
    ap.add_argument("--tokens", type=int, default=512, help="global extend tokens (chunk)")
    ap.add_argument("--moe-backend", default="fused_v2")
    ap.add_argument(
        "--devices",
        type=int,
        default=None,
        help="fake CPU device count (default = tp). Ignored on a real TPU host.",
    )
    ap.add_argument("--out", default="/tmp/fwd_jaxpr_cpu.json")
    args = ap.parse_args()

    # Set the CPU/fake-mesh env BEFORE importing jax (jax reads these at import).
    devices = args.devices or args.tp
    if "JAX_PLATFORMS" not in os.environ and not os.environ.get("RL_FORCE_TPU"):
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={devices}"
        ).strip()

    import jax

    from sgl_jax.srt.utils.roofline.forward_jaxpr_dump import dump_closed_jaxpr
    from sgl_jax.srt.utils.roofline.standalone_trace import (
        patch_for_cpu,
        trace_model_forward,
    )

    if jax.default_backend() != "tpu":
        patch_for_cpu(7)
    print(
        f"platform={jax.default_backend()} devices={len(jax.devices())} "
        f"tp={args.tp} dp={args.dp} phase={args.phase}",
        file=sys.stderr,
    )
    res = trace_model_forward(
        args.model_path,
        args.tp,
        args.dp,
        phase=args.phase,
        num_tokens=args.tokens,
        moe_backend=args.moe_backend,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dump_closed_jaxpr(res.jaxpr, args.out)
    print(
        f"traced arch={res.arch} phase={res.phase} tokens_global={res.tokens_global} "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
