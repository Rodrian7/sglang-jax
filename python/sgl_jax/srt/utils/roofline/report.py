"""Roofline math and reporting primitives (hardware-independent of JAX).

This module holds the peak-hardware model, the per-op roofline record, and the
aggregation / formatting logic. It deliberately does NOT import jax so it can be
unit-tested standalone; callers convert a ``RooflineResult`` into ``OpRoofline``
rows before handing them here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

PeakKind = Literal["bf16", "fp8"]
BoundKind = Literal["compute", "HBM", "ICI", "none"]


@dataclass(frozen=True)
class HardwarePeaks:
    """Per-device (chiplet) peak rates for v7x. These are the theoretical
    hardware ceilings; ``ideal_ms = work / peak`` is compared against measured
    time to read off efficiency.

    NOTE on ICI: ``ici_gbps`` defaults to the hardware unidirectional link BW
    (~100 GB/s). A collective (all-to-all / all-reduce) realistically attains
    only a fraction of that (measured ~40 GB/s); for a *practical* roofline pass
    ``--peak-ici-gbps 40`` to use the achievable collective bandwidth instead.
    """

    bf16_tflops: float = 1153.5  # v7x per-chip
    fp8_tflops: float = 2307.0
    hbm_gbps: float = 3690.0  # 3.69 TB/s (decimal GB)
    ici_gbps: float = 100.0  # unidirectional link BW; collective-achievable ~40
    hbm_capacity_gb: float = 96.0  # informational (HBM3e)
    vmem_mb: float = 64.0  # informational

    def flops_per_s(self, kind: PeakKind) -> float:
        return (self.fp8_tflops if kind == "fp8" else self.bf16_tflops) * 1e12

    def hbm_bytes_per_s(self) -> float:
        return self.hbm_gbps * 1e9

    def ici_bytes_per_s(self) -> float:
        return self.ici_gbps * 1e9


@dataclass
class OpRoofline:
    """Cost of one op (or a group of identical ops) plus derived roofline metrics.

    All counts/bytes are already scaled to the *whole-model, per-device* total
    for the analyzed phase (i.e. multiplied by layer repetition and divided by
    the relevant parallel degree by the descriptor).
    """

    label: str
    category: str  # linear | attention | moe | router | norm | rope | embedding | lm_head | collective | other
    source: str = ""  # "file:line (fn)" attribution from jaxpr_util, when available
    count: int = 1
    flops: int = 0
    hbm_bytes: int = 0
    ici_bytes: int = 0
    peak_kind: PeakKind = "bf16"

    # --- derived metrics (ms) ---
    def compute_ms(self, peaks: HardwarePeaks) -> float:
        return self.flops / peaks.flops_per_s(self.peak_kind) * 1e3

    def hbm_ms(self, peaks: HardwarePeaks) -> float:
        return self.hbm_bytes / peaks.hbm_bytes_per_s() * 1e3

    def ici_ms(self, peaks: HardwarePeaks) -> float:
        return self.ici_bytes / peaks.ici_bytes_per_s() * 1e3

    def arithmetic_intensity(self) -> float:
        return self.flops / self.hbm_bytes if self.hbm_bytes else float("inf")

    def ideal_ms(self, peaks: HardwarePeaks) -> float:
        # Roofline lower bound assuming perfect compute/HBM/ICI overlap.
        return max(self.compute_ms(peaks), self.hbm_ms(peaks), self.ici_ms(peaks))

    def bound(self, peaks: HardwarePeaks) -> BoundKind:
        ms = {
            "compute": self.compute_ms(peaks),
            "HBM": self.hbm_ms(peaks),
            "ICI": self.ici_ms(peaks),
        }
        top = max(ms, key=ms.get)
        return top if ms[top] > 0 else "none"  # type: ignore[return-value]

    def hidden(self, peaks: HardwarePeaks) -> str:
        """Resources overlapped under the bound (ideal_ms=max => the non-bound,
        nonzero resources are hidden). C=compute, H=HBM, I=ICI."""
        b = self.bound(peaks)
        if b == "none":
            return "-"
        ms = {
            "compute": ("C", self.compute_ms(peaks)),
            "HBM": ("H", self.hbm_ms(peaks)),
            "ICI": ("I", self.ici_ms(peaks)),
        }
        hid = [sym for k, (sym, v) in ms.items() if k != b and v > 0]
        return "+".join(hid) if hid else "-"

    def scaled(self, factor: float) -> OpRoofline:
        """Return a copy with count/flops/bytes multiplied (e.g. by #layers)."""
        return dataclasses.replace(
            self,
            count=int(round(self.count * factor)),
            flops=int(round(self.flops * factor)),
            hbm_bytes=int(round(self.hbm_bytes * factor)),
            ici_bytes=int(round(self.ici_bytes * factor)),
        )


def _merge(rows: list[OpRoofline], key) -> dict[str, OpRoofline]:
    """Sum rows into buckets keyed by ``key(row)`` (category or source)."""
    out: dict[str, OpRoofline] = {}
    for r in rows:
        k = key(r)
        if k not in out:
            out[k] = OpRoofline(
                label=k, category=r.category, source=r.source, peak_kind=r.peak_kind, count=0
            )
        acc = out[k]
        acc.count += r.count
        acc.flops += r.flops
        acc.hbm_bytes += r.hbm_bytes
        acc.ici_bytes += r.ici_bytes
    return out


@dataclass
class ModelRoofline:
    """Whole-model, per-device roofline for one phase, plus the three views."""

    arch: str
    phase: str
    peaks: HardwarePeaks
    rows: list[OpRoofline] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # batch, seq_len, tp, ep, devices, num_layers, ...

    # --- totals / aggregation ---
    def total(self) -> OpRoofline:
        t = OpRoofline(label="TOTAL", category="total", count=0)
        for r in self.rows:
            t.flops += r.flops
            t.hbm_bytes += r.hbm_bytes
            t.ici_bytes += r.ici_bytes
            t.count += r.count
        return t

    def by_category(self) -> list[OpRoofline]:
        # sort by ideal_ms (time) descending -- the apples-to-apples "expensive" metric
        rows = _merge(self.rows, lambda r: r.category).values()
        return sorted(rows, key=lambda r: -r.ideal_ms(self.peaks))

    def by_source(self) -> list[OpRoofline]:
        rows = _merge(self.rows, lambda r: r.source or r.label).values()
        return sorted(rows, key=lambda r: -r.ideal_ms(self.peaks))

    def utilization(self) -> dict:
        """At the ideal (bound) step time, what % of each hardware peak is used.
        The bound resource = 100%; the rest is headroom. (Theoretical balance,
        NOT measured efficiency -- the latter needs real wall time.)"""
        p = self.peaks
        t = self.total()
        ideal = t.ideal_ms(p)
        if ideal <= 0:
            return {}
        return {
            "compute_pct": t.compute_ms(p) / ideal * 100,
            "hbm_pct": t.hbm_ms(p) / ideal * 100,
            "ici_pct": t.ici_ms(p) / ideal * 100,
            "eff_tflops": t.flops / (ideal / 1e3) / 1e12,
            "eff_hbm_gbps": t.hbm_bytes / (ideal / 1e3) / 1e9,
            "eff_ici_gbps": t.ici_bytes / (ideal / 1e3) / 1e9,
        }

    # --- JSON ---
    def to_dict(self) -> dict:
        p = self.peaks

        def row_dict(r: OpRoofline) -> dict:
            return {
                "label": r.label,
                "category": r.category,
                "source": r.source,
                "count": r.count,
                "flops": r.flops,
                "hbm_bytes": r.hbm_bytes,
                "ici_bytes": r.ici_bytes,
                "arithmetic_intensity": r.arithmetic_intensity(),
                "compute_ms": r.compute_ms(p),
                "hbm_ms": r.hbm_ms(p),
                "ici_ms": r.ici_ms(p),
                "ideal_ms": r.ideal_ms(p),
                "bound": r.bound(p),
                "hidden": r.hidden(p),
                "peak_kind": r.peak_kind,
            }

        tot = self.total()
        return {
            "arch": self.arch,
            "phase": self.phase,
            "meta": self.meta,
            "peaks": dataclasses.asdict(p),
            "utilization": self.utilization(),
            "total": row_dict(tot),
            "by_category": [row_dict(r) for r in self.by_category()],
            "by_source": [row_dict(r) for r in self.by_source()],
        }


def _g(x: float) -> str:  # GFLOP / GB formatting helper
    return f"{x:>9.2f}"


def _render_rows(
    rows: list[OpRoofline],
    peaks: HardwarePeaks,
    title: str,
    subtitle: str = "",
    name_w: int = 40,
    total_ms: float = 0.0,
) -> str:
    head = (
        f"{'op / source':<{name_w}}{'cnt':>5}{'GFLOP':>10}{'HBM_MB':>10}"
        f"{'ICI_MB':>10}{'AI':>9}{'ideal_ms':>10}{'%step':>7}{'bound':>9}{'hidden':>8}"
    )
    lines = [f"===== {title} ====="]
    if subtitle:
        lines.append(subtitle)
    lines += [head, "-" * len(head)]
    for r in rows:
        ai = r.arithmetic_intensity()
        im = r.ideal_ms(peaks)
        pct = f"{im/total_ms*100:.0f}%" if total_ms > 0 else "-"
        lines.append(
            f"{r.label[:name_w]:<{name_w}}{r.count:>5}{r.flops/1e9:>10.2f}"
            f"{r.hbm_bytes/1e6:>10.2f}{r.ici_bytes/1e6:>10.2f}"
            f"{('inf' if ai==float('inf') else f'{ai:.1f}'):>9}"
            f"{im:>10.4f}{pct:>7}{r.bound(peaks):>9}{r.hidden(peaks):>8}"
        )
    return "\n".join(lines)


_LEGEND = (
    "cols: GFLOP/HBM_MB/ICI_MB = per-device work | AI = FLOP/HBM-byte | "
    "ideal_ms = max(compute,HBM,ICI)/peak | %step = ideal_ms / total | "
    "bound = binding resource | hidden = resources overlapped under bound (C=compute,H=HBM,I=ICI)"
)


def render_cost_views(model: ModelRoofline) -> str:
    """View B (by op category) + View C (by source attribution), human-readable."""
    p = model.peaks
    tot = model.total()
    m = model.meta
    header = (
        f"# {model.arch}  phase={model.phase}  "
        f"(per-device; batch={m.get('batch')} seq_len={m.get('seq_len')} "
        f"layers={m.get('num_layers')} tp={m.get('tp')} ep={m.get('ep')} devices={m.get('devices')})"
    )
    if m.get("quant"):
        header += f"\n# quant: {m['quant']}"
    tot = model.total()
    ideal = tot.ideal_ms(p)
    b = _render_rows(model.by_category(), p, "View B: cost by op category", total_ms=ideal)
    c = _render_rows(model.by_source(), p, "View C: cost attributed to source", total_ms=ideal)
    u = model.utilization()
    util_line = (
        f"util@ideal vs v7x peak:  compute {u.get('compute_pct',0):.0f}% ({u.get('eff_tflops',0):.0f} TFLOP/s)  "
        f"HBM {u.get('hbm_pct',0):.0f}% ({u.get('eff_hbm_gbps',0):.0f} GB/s)  "
        f"ICI {u.get('ici_pct',0):.0f}% ({u.get('eff_ici_gbps',0):.0f} GB/s)   [bound=100%, rest=headroom]"
    )
    summary = (
        f"TOTAL  GFLOP={tot.flops/1e9:.2f}  HBM_MB={tot.hbm_bytes/1e6:.2f}  "
        f"ICI_MB={tot.ici_bytes/1e6:.2f}  ideal_compute_ms={tot.compute_ms(p):.4f}  "
        f"ideal_hbm_ms={tot.hbm_ms(p):.4f}  ideal_ici_ms={tot.ici_ms(p):.4f}  "
        f"overall_bound={tot.bound(p)}\n"
        f"{util_line}"
    )
    return "\n\n".join([header, _LEGEND, b, c, summary])


def render_graph_views(analysis: dict) -> str:
    """View D (critical path / CPM) + View E (fusion opportunities)."""
    a = analysis
    d = [
        f"===== View D: critical path (CPM, phase={a['phase']}) =====",
        f"t_critical={a['t_critical_ms']:.3f}ms (serial dependency chain)  "
        f"t_resource={a['t_resource_ms']:.3f}ms (perfect overlap)  "
        f"gap={a['gap_ms']:.3f}ms (= fusion/pipeline headroom)",
        f"  resource totals: compute={a['sum_compute_ms']:.3f}ms  HBM={a['sum_hbm_ms']:.3f}ms  ICI={a['sum_ici_ms']:.3f}ms",
    ]
    for pt in a["per_type"][:2]:
        d.append(f"  critical path of one {pt['type']} layer (x{pt['count']} layers):")
        for lbl, ms in pt["path"]:
            d.append(f"      {ms:8.4f}ms  {lbl}")
    e = [
        "===== View E: fusion opportunities (single-producer/single-consumer, ranked by HBM saved) =====",
        f"{'producer + consumer':<52}{'layers':>7}{'MB/layer':>10}{'total_MB':>10}{'reason':>26}",
        "-" * 105,
    ]
    for f in a["fusions"]:
        pc = f"{f['producer']} + {f['consumer']}"
        e.append(
            f"{pc[:52]:<52}{f['layers']:>7}{f['saved_hbm_bytes']/1e6:>10.2f}"
            f"{f['total_saved_mb']:>10.1f}{f['reason']:>26}"
        )
    if not a["fusions"]:
        e.append("  (no fusable single-producer/single-consumer pairs found)")
    e.append(
        "注: 仅列结构上可融的候选 + 省的 HBM 往返字节; 本工具不知道 XLA 是否已自动融合, 也不保证省字节=省时间, 需 profile 核实."
    )
    return "\n\n".join(["\n".join(d), "\n".join(e)])
