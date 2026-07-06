#!/usr/bin/env python3
"""Compare two evalscope runs (grouped-topk kernel A/B) and print a metric diff.

Usage:
    python compare_gtopk_eval.py <disabled_dir> <enabled_dir>

Each *_dir is the evalscope --work-dir passed to a run (e.g. eval-out/disabled).
The script walks the run's reports/ tree, pulls every numeric metric it can
find per dataset, and prints a side-by-side table with the delta.

It is deliberately schema-tolerant: evalscope's report JSON layout has shifted
across versions, so instead of hard-coding keys we recursively collect any
{name/metric: str, score/value: number} pairs plus top-level "metrics" lists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _find_report_jsons(root: Path) -> list[Path]:
    """Return report JSONs under a run dir, preferring a reports/ subtree."""
    candidates = sorted(root.rglob("reports/**/*.json"))
    if candidates:
        return candidates
    # Fallback: any json that is not an obvious config/log artifact.
    return [
        p
        for p in sorted(root.rglob("*.json"))
        if "config" not in p.parts and "logs" not in p.parts
    ]


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _extract_metrics(obj, prefix: str = "") -> dict[str, float]:
    """Recursively pull metric_name -> score from an evalscope report object."""
    out: dict[str, float] = {}

    if isinstance(obj, dict):
        # Shape A: {"metric"/"name": "AveragePass@1", "score"/"value": 0.5}
        name = obj.get("name") or obj.get("metric") or obj.get("metric_name")
        score = obj.get("score")
        if score is None:
            score = obj.get("value")
        if isinstance(name, str) and _num(score):
            out[f"{prefix}{name}"] = float(score)

        for k, v in obj.items():
            if k in ("name", "metric", "metric_name", "score", "value"):
                continue
            if _num(v):
                # Shape B: {"AveragePass@1": 0.5, "Pass@4": 0.7}
                out[f"{prefix}{k}"] = float(v)
            else:
                out.update(_extract_metrics(v, prefix))
    elif isinstance(obj, list):
        for item in obj:
            out.update(_extract_metrics(item, prefix))
    return out


def _collect(run_dir: Path) -> dict[str, dict[str, float]]:
    """dataset -> {metric: score} for one run."""
    result: dict[str, dict[str, float]] = {}
    for jp in _find_report_jsons(run_dir):
        try:
            data = json.loads(jp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        dataset = data.get("dataset_name") or data.get("name") or jp.stem
        metrics = _extract_metrics(data)
        if metrics:
            result.setdefault(str(dataset), {}).update(metrics)
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    dis_dir, en_dir = Path(argv[1]), Path(argv[2])
    for d in (dis_dir, en_dir):
        if not d.exists():
            print(f"!! missing dir: {d}")
            return 1

    dis = _collect(dis_dir)
    en = _collect(en_dir)

    datasets = sorted(set(dis) | set(en))
    if not datasets:
        print("No report metrics found. Point me at the run --work-dir(s).")
        print(f"  disabled: {dis_dir}")
        print(f"  enabled : {en_dir}")
        return 1

    print("=" * 78)
    print("grouped-topk kernel A/B  —  disabled (JAX)  vs  enabled (Pallas)")
    print("=" * 78)
    any_diff = False
    for ds in datasets:
        dm, em = dis.get(ds, {}), en.get(ds, {})
        metrics = sorted(set(dm) | set(em))
        print(f"\n### {ds}")
        print(f"{'metric':<34}{'disabled':>12}{'enabled':>12}{'delta':>12}")
        print("-" * 70)
        for m in metrics:
            dv, ev = dm.get(m), em.get(m)
            ds_s = f"{dv:.4f}" if dv is not None else "-"
            en_s = f"{ev:.4f}" if ev is not None else "-"
            if dv is not None and ev is not None:
                delta = ev - dv
                dl_s = f"{delta:+.4f}"
                if abs(delta) > 1e-9:
                    any_diff = True
            else:
                dl_s = "-"
            print(f"{m:<34}{ds_s:>12}{en_s:>12}{dl_s:>12}")

    print("\n" + "=" * 78)
    print(
        "No metric difference — kernel matches JAX path."
        if not any_diff
        else "Metrics differ — inspect deltas above (temp=0.6 sampling adds noise; "
        "repeats=4 pass@k helps but small deltas may be sampling variance)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
