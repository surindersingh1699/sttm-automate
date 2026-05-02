"""Baseline KPI diff and regression gate.

  python -m tests.eval.diff current.json baseline.json

Compares each KPI in the current run against the baseline and prints a
human-readable table. Exits non-zero if any *gate* KPI has regressed past
its allowed tolerance.

Gate thresholds (tune after first baseline run):
  lock_accuracy           ≥ 90 %       (delta tolerance −5 pp)
  p90_ttfcl_s             ≤ 12 s       (delta tolerance +3 s)
  transition_detection    ≥ 85 %       (delta tolerance −5 pp)
  line_accuracy_pm1       ≥ 80 %       (delta tolerance −5 pp)
  p90_line_lag_s          ≤ 5 s        (delta tolerance +2 s)
  composite_pct_correct   ≥ 70 %       (delta tolerance −5 pp)
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tests.eval.scorer import AggregateKPIs


# (field_name, direction, absolute_threshold, delta_tolerance)
# direction: "higher_is_better" | "lower_is_better"
_GATE_RULES: list[tuple[str, str, float, float]] = [
    ("median_lock_accuracy_pct",          "higher_is_better",  90.0,  -5.0),
    ("p90_ttfcl_s",                        "lower_is_better",   12.0,  +3.0),
    ("overall_detection_rate_pct",         "higher_is_better",  85.0,  -5.0),
    ("median_line_accuracy_pm1_pct",       "higher_is_better",  80.0,  -5.0),
    ("p90_line_lag_s",                     "lower_is_better",    5.0,  +2.0),
    ("composite_pct_time_correct",         "higher_is_better",  70.0,  -5.0),
]


def diff_kpis(current: dict, baseline: dict) -> list[dict]:
    """Return per-field diff records."""
    records = []
    for key, cur_val in current.items():
        if not isinstance(cur_val, (int, float)) or cur_val is None:
            continue
        base_val = baseline.get(key)
        if base_val is None or not isinstance(base_val, (int, float)):
            continue
        delta = cur_val - base_val
        records.append({
            "field": key,
            "current": cur_val,
            "baseline": base_val,
            "delta": round(delta, 3),
        })
    return records


def check_gate(kpis: AggregateKPIs, baseline: dict, verbose: bool = True) -> bool:
    """Return True if all gate KPIs pass. Prints diff table if verbose."""
    current = asdict(kpis)
    baseline_agg = baseline.get("aggregate", baseline)

    diffs = diff_kpis(current, baseline_agg)
    by_field = {d["field"]: d for d in diffs}

    if verbose:
        _print_diff_table(diffs)

    passed = True
    failures = []
    for field, direction, threshold, tol in _GATE_RULES:
        cur = current.get(field)
        base = baseline_agg.get(field)
        if cur is None:
            continue

        # Check absolute threshold
        if direction == "higher_is_better" and cur < threshold:
            failures.append(
                f"  ✗ {field}: {cur:.1f} < gate threshold {threshold:.1f}"
            )
            passed = False
            continue
        if direction == "lower_is_better" and cur > threshold:
            failures.append(
                f"  ✗ {field}: {cur:.1f} > gate threshold {threshold:.1f}"
            )
            passed = False
            continue

        # Check delta regression
        if base is not None:
            delta = cur - base
            if direction == "higher_is_better" and delta < tol:
                failures.append(
                    f"  ✗ {field}: regressed {delta:+.1f} (tolerance {tol:+.1f})"
                )
                passed = False
            elif direction == "lower_is_better" and delta > tol:
                failures.append(
                    f"  ✗ {field}: regressed {delta:+.1f} (tolerance {tol:+.1f})"
                )
                passed = False

    if verbose:
        if failures:
            print("\n[Gate] FAILED — regressions detected:")
            for f in failures:
                print(f)
        else:
            print("\n[Gate] PASSED — all KPIs within tolerance.")

    return passed


def _print_diff_table(diffs: list[dict]):
    bar = "─" * 68
    print(f"\n{'Field':<42} {'Current':>9} {'Baseline':>9} {'Delta':>8}")
    print(bar)
    for d in sorted(diffs, key=lambda x: x["field"]):
        delta_str = f"{d['delta']:+.3f}"
        print(f"  {d['field']:<40} {d['current']:>9.3f} {d['baseline']:>9.3f} {delta_str:>8}")
    print(bar)


def _main():
    import argparse
    p = argparse.ArgumentParser(description="Diff two eval report JSON files")
    p.add_argument("current", help="Current run report JSON")
    p.add_argument("baseline", help="Baseline report JSON")
    p.add_argument("--gate", action="store_true", help="Exit non-zero on regression")
    args = p.parse_args()

    current = json.loads(Path(args.current).read_text())
    baseline = json.loads(Path(args.baseline).read_text())

    current_agg = current.get("aggregate", current)
    base_agg = baseline.get("aggregate", baseline)
    diffs = diff_kpis(current_agg, base_agg)
    _print_diff_table(diffs)

    if args.gate:
        from tests.eval.scorer import AggregateKPIs
        import dataclasses

        # Reconstruct AggregateKPIs from dict for gate check
        fields = {f.name for f in dataclasses.fields(AggregateKPIs)}
        kpis_kwargs = {k: v for k, v in current_agg.items() if k in fields}
        kpis = AggregateKPIs(**kpis_kwargs)
        passed = check_gate(kpis, baseline, verbose=True)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    _main()
