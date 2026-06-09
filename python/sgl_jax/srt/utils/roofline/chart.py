"""Roofline scatter chart (matplotlib) for a ``ModelRoofline``.

A classic log-log roofline: x = operational intensity (FLOP per HBM byte),
y = attainable performance (TFLOP/s). The roof = the HBM-bandwidth diagonal
capped by the compute ceiling (bf16, and fp8 if any op is fp8); the ridge point
is where they meet. Each op-category is a point ON the roof when it is
compute/HBM-bound, and FALLS BELOW the roof when it is ICI-bound (its time is set
by a collective, not HBM/compute) -- those are drawn with an 'X' so a
communication bottleneck is visible at a glance. Point size scales with the op's
ideal time, so the eye lands on what dominates.

Pure theory (uses the same per-op compute/HBM/ICI model as the text views); only
needs matplotlib. No VMEM roof (VMEM bandwidth is high and tiling-dependent --
essentially never the model-level bound).
"""

from __future__ import annotations

from .report import HardwarePeaks, ModelRoofline

# stable per-category colors
_CAT_COLORS = {
    "moe": "#d62728",
    "linear": "#1f77b4",
    "attention": "#2ca02c",
    "router": "#9467bd",
    "lm_head": "#8c564b",
    "norm": "#e377c2",
    "rope": "#7f7f7f",
    "other": "#bcbd22",
    "embedding": "#17becf",
}


def roofline_chart(model: ModelRoofline, peaks: HardwarePeaks, out_path: str, *, title=None):
    """Write a roofline PNG for ``model`` to ``out_path``; returns the path."""
    import matplotlib.pyplot as plt

    fig = roofline_figure(model, peaks, title=title)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def roofline_figure(model: ModelRoofline, peaks: HardwarePeaks, *, title=None):
    """Build and return the roofline ``matplotlib`` Figure (not saved/closed), so
    it can be embedded in a multi-page PDF report. Raises ImportError with a clear
    message if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:  # pragma: no cover
        raise ImportError("roofline chart needs matplotlib (`pip install matplotlib`)") from e

    rows = [r for r in model.rows if r.flops > 0 and r.hbm_bytes > 0]
    if not rows:
        raise ValueError("no costed ops with FLOPs+HBM to plot")

    bf16 = peaks.flops_per_s("bf16") / 1e12  # TFLOP/s
    fp8 = peaks.flops_per_s("fp8") / 1e12
    hbm_bw = peaks.hbm_bytes_per_s()  # byte/s
    has_fp8 = any(r.peak_kind == "fp8" for r in rows)
    ceil = fp8 if has_fp8 else bf16

    pts = []
    for r in rows:
        oi = r.flops / r.hbm_bytes  # FLOP / HBM-byte
        perf = r.flops / (r.ideal_ms(peaks) / 1e3) / 1e12  # TFLOP/s
        pts.append((oi, perf, r.ideal_ms(peaks), r.category, r.bound(peaks), r.label))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    xmin = min(p[0] for p in pts) / 3.0
    xmax = max(p[0] for p in pts) * 3.0
    xx = [xmin * (xmax / xmin) ** (i / 240.0) for i in range(241)]
    # HBM diagonal capped by compute ceiling
    roof = [min(x * hbm_bw / 1e12, ceil) for x in xx]
    ax.plot(
        xx,
        roof,
        "k-",
        lw=1.6,
        zorder=2,
        label=f"HBM {peaks.hbm_bytes_per_s()/1e9:.0f} GB/s",
    )
    ax.axhline(bf16, ls="--", color="gray", lw=1, label=f"bf16 {bf16:.0f} TFLOP/s")
    if has_fp8:
        ax.axhline(fp8, ls=":", color="gray", lw=1, label=f"fp8 {fp8:.0f} TFLOP/s")
    ridge = ceil / (hbm_bw / 1e12)
    ax.axvline(ridge, ls=":", color="lightgray", lw=0.8, zorder=1)
    ax.annotate(
        f"ridge OI={ridge:.0f}",
        (ridge, ceil),
        fontsize=7,
        color="gray",
        xytext=(2, -10),
        textcoords="offset points",
    )

    smax = max(p[2] for p in pts) or 1.0
    for oi, perf, ms, cat, bound, label in pts:
        size = 40 + 420 * (ms / smax)
        marker = "X" if bound == "ICI" else "o"
        ax.scatter(
            [oi],
            [perf],
            s=size,
            c=[_CAT_COLORS.get(cat, "#333333")],
            marker=marker,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            zorder=4,
        )
        if ms > 0.02 * smax:  # label the ops that matter
            ax.annotate(
                label.split("[")[0],
                (oi, perf),
                fontsize=6.5,
                xytext=(4, 3),
                textcoords="offset points",
                zorder=5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("operational intensity  (FLOP / HBM byte)")
    ax.set_ylabel("attainable performance  (TFLOP/s)")
    m = model.meta
    ax.set_title(
        title
        or f"{model.arch}  phase={model.phase}  roofline  "
        f"(per-device v7x; mesh data={m.get('dp')} x tensor={m.get('attention_tp')}"
        f"{', +SP' if m.get('enable_sp') else ''})",
        fontsize=9,
    )
    ax.grid(True, which="both", ls=":", alpha=0.3)

    handles, seen = [], set()
    for _, _, _, cat, _, _ in pts:
        if cat not in seen:
            seen.add(cat)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markersize=7,
                    markerfacecolor=_CAT_COLORS.get(cat, "#333333"),
                    label=cat,
                )
            )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markersize=8,
            markerfacecolor="gray",
            markeredgecolor="k",
            label="ICI-bound (below roof)",
        )
    )
    ax.legend(handles=handles, fontsize=6.5, loc="lower right", ncol=2)
    fig.tight_layout()
    return fig
