#!/usr/bin/env python3
"""Theoretical whole-model roofline for sglang-jax models.

Given a checkpoint dir (``--model-path`` with ``config.json``), resolves the
architecture through the framework model registry (``scan models/``), composes a
per-device theoretical roofline for the requested phase(s), and prints three
views:

  View A  structure   -- op histogram + source attribution (jax jaxpr_util)
  View B  cost         -- FLOPs / HBM / ICI / AI / bound by op category
  View C  fused        -- the same cost attributed back to source lines

Pure theory (no profiling/trace): runs on CPU, no TPU init required.

Example:
  python -m sgl_jax.tools.model_roofline --model-path /models/MiMo-V2-Flash \\
      --phase both --batch 64 --seq-len 4096 --tp 8 --ep 32 --devices 32 \\
      --peak-ici-gbps 40 --json-out roofline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Hard dependency: resolve the architecture through the framework registry
# ("scan models/"). Fails loudly if the model is not a supported architecture.
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.utils.roofline import descriptors, interp
from sgl_jax.srt.utils.roofline.report import (
    HardwarePeaks,
    render_cost_views,
    render_graph_views,
)


def _load_config(model_path: str) -> dict:
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found under {model_path}")
    with open(cfg_path) as f:
        return json.load(f)


def _resolve_arch(config: dict) -> str:
    archs = config.get("architectures")
    if not archs:
        raise ValueError("config.json has no 'architectures' field")
    _, arch_name = ModelRegistry.resolve_model_cls(archs)  # hard dep: must be supported
    return arch_name


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", help="checkpoint dir containing config.json")
    ap.add_argument("--phase", choices=["decode", "prefill", "both"], default="both")
    ap.add_argument(
        "--batch",
        type=int,
        default=512,
        help="decode batch (tokens) / #seqs emitting logits",
    )
    ap.add_argument("--seq-len", type=int, default=4096, help="decode KV context length")
    ap.add_argument("--chunk", type=int, default=16384, help="prefill chunk tokens")
    # parallelism
    ap.add_argument("--tp", type=int, default=8, help="attention/linear tensor-parallel degree")
    ap.add_argument("--ep", type=int, default=32, help="expert-parallel degree")
    ap.add_argument("--devices", type=int, default=32)
    ap.add_argument("--dp", type=int, default=None)
    # peaks (per-device); defaults are v7x
    ap.add_argument("--peak-bf16-tflops", type=float, default=None)
    ap.add_argument("--peak-fp8-tflops", type=float, default=None)
    ap.add_argument("--peak-hbm-gbps", type=float, default=None)
    ap.add_argument(
        "--peak-ici-gbps",
        type=float,
        default=None,
        help="effective collective BW; ~40 measured, 100 hw",
    )
    # output
    ap.add_argument("--view", choices=["a", "b", "c", "d", "e", "f", "all"], default="all")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--pprof", default=None, help="write jaxpr_util pprof profile here (.pb.gz)")
    ap.add_argument("--list", action="store_true", help="list supported architectures and exit")
    args = ap.parse_args()

    if args.list:
        print("Supported architectures (registry):")
        for a in sorted(ModelRegistry.get_supported_archs()):
            mark = " [roofline]" if a in descriptors.DESCRIPTORS else ""
            print(f"  {a}{mark}")
        return 0

    if not args.model_path:
        ap.error("--model-path is required (or use --list)")

    config = _load_config(args.model_path)
    arch = _resolve_arch(config)
    if arch not in descriptors.DESCRIPTORS:
        raise ValueError(
            f"Architecture '{arch}' is registered in the model registry but has no roofline "
            f"descriptor yet. Available: {sorted(descriptors.DESCRIPTORS)}"
        )

    overrides = {
        k: v
        for k, v in {
            "bf16_tflops": args.peak_bf16_tflops,
            "fp8_tflops": args.peak_fp8_tflops,
            "hbm_gbps": args.peak_hbm_gbps,
            "ici_gbps": args.peak_ici_gbps,
        }.items()
        if v is not None
    }
    peaks = (
        HardwarePeaks(**{**HardwarePeaks().__dict__, **overrides}) if overrides else HardwarePeaks()
    )

    par = {
        "tp": args.tp,
        "ep": args.ep,
        "devices": args.devices,
        "dp": args.dp if args.dp is not None else max(1, args.devices // args.tp),
        "batch": args.batch,
        "seq_len": args.seq_len,
        "chunk": args.chunk,
    }

    phases = ["decode", "prefill"] if args.phase == "both" else [args.phase]
    out_json: dict = {"arch": arch, "model_path": args.model_path, "phases": {}}

    for phase in phases:
        model = descriptors.build(arch, config, phase, par, peaks)
        print("\n" + "=" * 100)
        if args.view in ("a", "all"):
            ref = descriptors.reference_forward(arch, config, phase, par)
            view_a = interp.structure_view(ref)
            print(interp.render_structure(view_a))
            print()
            if args.pprof and view_a is not None:
                p = f"{args.pprof}.{phase}" if len(phases) > 1 else args.pprof
                n = interp.write_pprof(view_a, p)
                print(f"[pprof] wrote {n} bytes -> {p}  (view with: pprof -http=: {p})")
        if args.view in ("b", "c", "all"):
            print(render_cost_views(model))
        if args.view in ("d", "e", "all"):
            ga = interp.graph_analysis(config, phase, par, peaks)
            print("\n" + render_graph_views(ga))
            out_json.setdefault("graph", {})[phase] = ga
        if args.view in ("f", "all"):
            from sgl_jax.srt.utils.roofline import graph_from_jaxpr as gjax

            fa = gjax.analyze_reference(arch, config, phase, par, peaks)
            print("\n" + gjax.render_auto_graph(fa))
            if fa is not None:
                out_json.setdefault("auto_graph", {})[phase] = {
                    k: v for k, v in fa.items() if k != "path"
                }
        out_json["phases"][phase] = model.to_dict()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"\n[json] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
