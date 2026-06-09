#!/usr/bin/env python3
"""End-to-end, server-free roofline for ANY registered sglang-jax model.

Traces the REAL forward on CPU (fake device mesh, abstract weights via
eval_shape -- no TPU, no checkpoint, no server), turns the per-device jaxpr into
a roofline (costs validated to match the analytic descriptor), and emits a text
summary + a self-contained interactive HTML report grounded in the real trace
(roofline chart, cost table, real models/*.py code-path index, Pallas kernels).

Sets JAX_PLATFORMS=cpu + the fake-device count before importing jax, so you need
no env. On a TPU host it traces on TPU instead. Example (CPU pod / laptop):

    PYTHONPATH=python python tools/trace_roofline.py \
        --model-path /models/MiMo-V2-Pro-Private --tp 32 --dp 8 \
        --seq-len 4096 --html /tmp/roofline.html
"""

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument(
        "--seq-len", type=int, default=4096, help="decode KV context / workload seq len"
    )
    ap.add_argument("--tokens", type=int, default=512, help="global extend (prefill chunk) tokens")
    ap.add_argument("--moe-backend", default="fused_v2")
    ap.add_argument("--enable-sp", action="store_true", default=True)
    ap.add_argument("--devices", type=int, default=None, help="fake CPU device count (default=tp)")
    ap.add_argument("--phases", default="prefill,decode", help="comma list: prefill,decode")
    ap.add_argument("--html", default=None, help="write self-contained HTML report")
    ap.add_argument("--dump", default=None, help="also write the raw jaxpr JSON (prefill)")
    args = ap.parse_args()

    devices = args.devices or args.tp
    if "JAX_PLATFORMS" not in os.environ and not os.environ.get("RL_FORCE_TPU"):
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={devices}"
        ).strip()

    import jax

    from sgl_jax.srt.utils.roofline import parallelism, report_html, trace_analyze
    from sgl_jax.srt.utils.roofline.forward_jaxpr_dump import (
        dump_closed_jaxpr,
        extract_jaxpr_records,
    )
    from sgl_jax.srt.utils.roofline.report import HardwarePeaks
    from sgl_jax.srt.utils.roofline.standalone_trace import (
        patch_for_cpu,
        trace_model_forward,
    )

    if jax.default_backend() != "tpu":
        patch_for_cpu(7)

    config = json.load(open(os.path.join(args.model_path, "config.json")))
    arch = (config.get("architectures") or ["traced"])[0]
    peaks = HardwarePeaks()
    par = dict(tp=args.tp, dp=args.dp, devices=devices, moe_backend=args.moe_backend)
    layout, warns = parallelism.resolve(
        config, par, moe_backend=args.moe_backend, enable_sp=args.enable_sp
    )
    for w in warns:
        print(f"[warn] {w}", file=sys.stderr)
    print(
        f"platform={jax.default_backend()} devices={len(jax.devices())} arch={arch} "
        f"mesh data={layout.dp}x tensor={layout.t} ep={layout.ep}",
        file=sys.stderr,
    )

    phase_to_mode = {"prefill": "extend", "decode": "decode"}
    results, records_by = {}, {}
    for ph in [p.strip() for p in args.phases.split(",") if p.strip()]:
        res = trace_model_forward(
            args.model_path,
            args.tp,
            args.dp,
            phase=phase_to_mode[ph],
            num_tokens=args.tokens,
            moe_backend=args.moe_backend,
        )
        records = extract_jaxpr_records(res.jaxpr)
        records_by[ph] = records
        model = trace_analyze.analyze_trace(
            records, config, layout, peaks, phase=phase_to_mode[ph], seq_len=args.seq_len, arch=arch
        )
        results[ph] = model
        if args.dump and ph == "prefill":
            dump_closed_jaxpr(res.jaxpr, args.dump)
        _print_phase(ph, model, peaks)

    if args.html:
        records = records_by.get("prefill") or next(iter(records_by.values()))
        codepath = trace_analyze.code_path_index(records, config)
        # interactive report (live knobs + Dataflow + Fusion) grounded in the real
        # trace's code-path + kernel tabs; defaults seed the knobs at this layout.
        defaults = dict(
            tp=args.tp,
            dp=args.dp,
            batch=256,
            seq_len=args.seq_len,
            chunk=256,
            enable_sp=args.enable_sp,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.html)), exist_ok=True)
        report_html.build_html_report(arch, config, peaks, defaults, args.html, codepath=codepath)
        print(f"\nHTML report -> {args.html}")


def _print_phase(ph, model, peaks):
    print(
        f"\n===== {ph}  (tokens/dp={model.meta['tokens_per_dp']}, "
        f"global={model.meta['global_tokens']}) ====="
    )
    print(
        f"{'category':12s} {'TFLOP':>8s} {'HBM GB':>8s} {'ICI GB':>8s} {'ideal ms':>9s} {'bound':>8s}"
    )
    for r in model.by_category():
        print(
            f"{r.category:12s} {r.flops/1e12:8.3f} {r.hbm_bytes/1e9:8.3f} "
            f"{r.ici_bytes/1e9:8.3f} {r.ideal_ms(peaks):9.4f} {r.bound(peaks):>8s}"
        )
    t = model.total()
    print(
        f"{'TOTAL':12s} {t.flops/1e12:8.3f} {t.hbm_bytes/1e9:8.3f} {t.ici_bytes/1e9:8.3f} "
        f"{sum(r.ideal_ms(peaks) for r in model.by_category()):9.4f} {t.bound(peaks):>8s}"
    )


if __name__ == "__main__":
    main()
