"""Turn an eval.py --output JSON into a markdown report for Linear/GitHub.

Usage:
    .venv/bin/python scripts/format_kirtan_benchmark_report.py \\
        --results /tmp/kirtan_results.json \\
        --baselines /tmp/empty_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(stem: str, acc: float, correct: int, total: int, n_segs: int) -> str:
    return f"| `{stem}` | {acc:.1f}% | {correct}/{total} | {n_segs} |"


def _per_video_table(per_video: list[dict]) -> str:
    rows = ["| Case | Frame acc | Correct/Total | Pred segs |", "|---|---|---|---|"]
    for v in per_video:
        rows.append(_row(v["stem"], v["frame_accuracy"], v["correct"], v["total"], v["n_pred_segments"]))
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="eval.py --output JSON for our system")
    p.add_argument("--baselines", help="eval.py --output JSON for empty baseline (optional)")
    args = p.parse_args()

    ours = json.loads(Path(args.results).read_text())

    print(f"## Frame accuracy at collar=1s")
    print()
    print(f"**Overall: {ours['overall_accuracy']:.1f}%** ({ours['total_correct']}/{ours['total_frames']} frames across {ours['n_videos']} cases)")
    print()

    if args.baselines:
        base = json.loads(Path(args.baselines).read_text())
        print(f"Reference points:")
        print(f"- Empty baseline (all `null`): **{base['overall_accuracy']:.1f}%**")
        print(f"- Perfect (GT copy): **100.0%**")
        print(f"- **Our pipeline: {ours['overall_accuracy']:.1f}%**")
        print()

    print("### Per-case scores")
    print()
    print(_per_video_table(ours["per_video"]))


if __name__ == "__main__":
    main()
