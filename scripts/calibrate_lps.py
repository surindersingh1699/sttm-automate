"""Calibrate per-engine speech-rate (LPS) thresholds from benchmark event logs.

Background
----------
``MatcherConfig`` has three LPS knobs that decide which audio-window length the
orchestrator picks:

  * ``slow_speech_letters_per_second``      (longer window when below)
  * ``fast_speech_letters_per_second``      (shorter window when above)
  * ``very_fast_speech_letters_per_second`` (locked-very-fast tier)

These were tuned against Whisper output and don't necessarily match the
distribution IndicConformer (RNN-T or CTC) produces on the same audio. This
script reads ``transcription`` events from any directory of ``runs/*.jsonl``
event logs, computes the LPS distribution per ``listening_mode`` (a useful
proxy for per-engine slices when you run the benchmark separately per engine),
and prints recommended threshold values.

Recommended thresholds use the standard ``balanced``-profile percentiles:

  * slow      = 25th percentile  (a quarter of windows are below this)
  * fast      = 75th percentile  (top quartile = "fast")
  * very_fast = 90th percentile  (top decile = "very fast")

Usage
-----
    .venv/bin/python scripts/calibrate_lps.py \\
        --runs tests/eval/runs/kirtan_benchmark_20260503_120000

Run this once per engine (rerun the benchmark with each engine pinned, then
point the script at the new runs dir). Compare the printed values against the
current defaults in ``src/config.py`` and update if they differ materially.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _iter_transcription_events(root: Path):
    """Yield ``(file_stem, event)`` for every transcription record under ``root``."""
    for path in sorted(root.rglob("*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("type") != "transcription":
                    continue
                # Skip vocal-break / silence-suppressed windows — they have
                # no first_letters and would skew toward zero LPS.
                if not ev.get("first_letters"):
                    continue
                window_s = float(ev.get("window_seconds") or 0.0)
                if window_s < 0.3:
                    continue
                yield path.stem, ev
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] skipping {path}: {e}")


def _percentiles(values: list[float], pcts: tuple[float, ...]) -> dict[float, float]:
    """Return {percentile: value} for the requested percentiles (linear interp)."""
    if not values:
        return {p: float("nan") for p in pcts}
    s = sorted(values)
    n = len(s)
    out: dict[float, float] = {}
    for p in pcts:
        if n == 1:
            out[p] = s[0]
            continue
        # Linear interpolation between the two nearest ranks.
        rank = (p / 100.0) * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        out[p] = s[lo] * (1 - frac) + s[hi] * frac
    return out


def calibrate(runs_dir: Path) -> dict:
    """Compute LPS distributions and recommended thresholds.

    Buckets by ``listening_mode`` so locked vs searching windows are scored
    separately — the thresholds matter most in the locked-state window picker
    (``_select_transcription_audio``).
    """
    by_mode: dict[str, list[float]] = defaultdict(list)
    all_lps: list[float] = []
    for _stem, ev in _iter_transcription_events(runs_dir):
        first_letters = ev["first_letters"]
        window_s = float(ev["window_seconds"])
        # Use instantaneous LPS, not the EMA-smoothed value, so the calibration
        # reflects the raw output distribution rather than the smoother's lag.
        lps = len(first_letters) / max(window_s, 0.1)
        mode = ev.get("listening_mode") or "unknown"
        by_mode[mode].append(lps)
        all_lps.append(lps)

    pcts = (10.0, 25.0, 50.0, 75.0, 90.0)
    summary = {
        "total_windows": len(all_lps),
        "modes": {},
        "all": {},
        "recommended": {},
    }

    if all_lps:
        all_p = _percentiles(all_lps, pcts)
        summary["all"] = {
            "n": len(all_lps),
            "mean": round(statistics.fmean(all_lps), 3),
            "stdev": round(statistics.pstdev(all_lps), 3),
            "percentiles": {f"p{int(p)}": round(v, 3) for p, v in all_p.items()},
        }
        summary["recommended"] = {
            "slow_speech_letters_per_second": round(all_p[25.0], 2),
            "fast_speech_letters_per_second": round(all_p[75.0], 2),
            "very_fast_speech_letters_per_second": round(all_p[90.0], 2),
        }

    for mode, values in sorted(by_mode.items()):
        ps = _percentiles(values, pcts)
        summary["modes"][mode] = {
            "n": len(values),
            "mean": round(statistics.fmean(values), 3),
            "percentiles": {f"p{int(p)}": round(v, 3) for p, v in ps.items()},
        }

    return summary


def _print_human(summary: dict) -> None:
    print(f"\nTotal transcription windows: {summary['total_windows']}\n")
    if not summary.get("all"):
        print("No transcription events with non-empty first_letters found.")
        print("Run the benchmark with vocal-content audio first.")
        return

    a = summary["all"]
    print("Overall LPS distribution:")
    print(f"  n={a['n']}, mean={a['mean']}, stdev={a['stdev']}")
    print(f"  percentiles: {a['percentiles']}\n")

    print("Per listening_mode (proxies engine slices when rerun per engine):")
    for mode, m in summary["modes"].items():
        print(f"  {mode:>22s}  n={m['n']:>4d}  mean={m['mean']:.2f}  {m['percentiles']}")

    rec = summary["recommended"]
    print("\nRecommended thresholds (paste into MatcherConfig defaults):")
    for key, value in rec.items():
        print(f"  {key:>40s} = {value}")
    print(
        "\nNote: rerun this script per engine for engine-specific thresholds. "
        "Current MatcherConfig defaults: slow=0.65, fast=1.50, very_fast=2.20."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, type=Path, help="Directory of *.jsonl event logs")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable summary")
    args = p.parse_args()

    if not args.runs.exists():
        raise SystemExit(f"runs dir not found: {args.runs}")

    summary = calibrate(args.runs)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_human(summary)


if __name__ == "__main__":
    main()
