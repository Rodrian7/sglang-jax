#!/usr/bin/env python3
"""Theoretical whole-model roofline for sglang-jax models.

Given a checkpoint dir (``--model-path`` with ``config.json``), resolves the
architecture through the framework model registry (``scan models/``), composes a
per-device theoretical roofline for the requested phase(s), and prints views:

  View A  structure   -- op histogram + source attribution (jax jaxpr_util)
  View B  cost         -- FLOPs / HBM / ICI / AI / bound by op category
  View C  fused        -- the same cost attributed back to source lines
  View D/E graph        -- hand-written layer dataflow + critical path + fusion
  View F  jaxpr graph   -- dataflow auto-derived from the traced reference jaxpr

Parallelism mirrors the server flags: ``--tp`` is tp_size = mesh total = device
count; the real tensor-parallel degree for attention/linears is ``tp//dp``; the
fused MoE expert-parallel group is the full mesh (= devices). The layout is
validated (``--devices`` must equal ``--tp``, ``tp % dp == 0``, heads/experts
divisibility) before simulating. Pure theory (no profiling/trace): runs on CPU.

Example (the validated MiMo-V2-Pro EP layout):
  python -m sgl_jax.tools.model_roofline --model-path /models/MiMo-V2-Pro \\
      --phase both --batch 64 --seq-len 4096 --tp 32 --dp 8 --devices 32 \\
      --enable-sp --peak-ici-gbps 40 --json-out roofline.json
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
    ap.add_argument(
        "--tp",
        type=int,
        default=8,
        help="tp_size = mesh total = device count (same as the server --tp-size). "
        "The real tensor-parallel degree for attention/linears is tp//dp.",
    )
    ap.add_argument(
        "--ep",
        type=int,
        default=None,
        help="ep_size; fused MoE ignores it (EP=devices)",
    )
    ap.add_argument(
        "--devices",
        type=int,
        default=None,
        help="total devices; must equal tp (default = tp)",
    )
    ap.add_argument("--dp", type=int, default=1, help="data-parallel degree; tensor axis = tp//dp")
    ap.add_argument(
        "--enable-sp",
        action="store_true",
        help="model sequence parallelism (reduce-scatter + all-gather above the scatter threshold)",
    )
    ap.add_argument(
        "--moe-backend",
        default="fused_v2",
        help="fused_v2|fused|... (affects EP semantics)",
    )
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
    ap.add_argument(
        "--chart-dir",
        default=None,
        help="write a roofline PNG per phase to this dir (needs matplotlib)",
    )
    ap.add_argument(
        "--html",
        default=None,
        help="write a self-contained INTERACTIVE roofline HTML here (live parallelism knobs)",
    )
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

    devices = args.devices if args.devices is not None else args.tp
    par = {
        "tp": args.tp,
        "ep": args.ep if args.ep is not None else devices,
        "devices": devices,
        "dp": args.dp,
        "enable_sp": args.enable_sp,
        "moe_backend": args.moe_backend,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "chunk": args.chunk,
    }
    # Fail loudly on an impossible parallelism layout before simulating it.
    from sgl_jax.srt.utils.roofline import parallelism as _para

    try:
        _lp, _warns = _para.resolve(
            config, par, moe_backend=args.moe_backend, enable_sp=args.enable_sp
        )
    except ValueError as e:
        ap.error(str(e))
    for w in _warns:
        print(f"[warn] {w}")

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
        if args.chart_dir:
            import os as _os

            from sgl_jax.srt.utils.roofline import chart as _chart

            _os.makedirs(args.chart_dir, exist_ok=True)
            p = _os.path.join(args.chart_dir, f"{arch}_{phase}_roofline.png")
            _chart.roofline_chart(model, peaks, p)
            print(f"[chart] wrote {p}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"\n[json] wrote {args.json_out}")
    if args.html:
        from sgl_jax.srt.utils.roofline import report_html

        report_html.build_html_report(
            arch,
            config,
            peaks,
            {
                "tp": args.tp,
                "dp": args.dp,
                "batch": args.batch,
                "seq_len": args.seq_len,
                "chunk": args.chunk,
                "enable_sp": args.enable_sp,
            },
            args.html,
        )
        print(f"[html] wrote interactive report -> {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
