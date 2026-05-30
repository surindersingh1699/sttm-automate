"""CLI entry point for the STTM-automate integrated eval framework.

Usage examples:

  # Headless: test 3 videos, save report
  python -m tests.eval --mode headless --limit 3 --report eval_out.json

  # Headless: specific video
  python -m tests.eval --mode headless --video-id dQw4w9WgXcQ

  # Headless: gate against baseline (exits non-zero if regression)
  python -m tests.eval --mode headless --limit 5 --baseline tests/eval/baseline.json --gate

  # Live: play YouTube + watch STTM + score (requires STTM + BlackHole)
  python -m tests.eval --mode live --video-id dQw4w9WgXcQ

  # Generate / update baseline from current run
  python -m tests.eval --mode headless --limit 20 --report tests/eval/baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def main(args: argparse.Namespace):
    from tests.eval.dataset import _DATASET, load_eval_sessions
    from tests.eval.metrics import compute_aggregate, print_session_summary, save_json
    from tests.eval.runner import HeadlessSessionDriver, MicSessionDriver, SessionResult
    from tests.eval.scorer import print_kpis

    if args.mode == "mic":
        from tests.eval.preflight import print_preflight
        if not print_preflight("mic"):
            sys.exit(1)

    dataset = args.dataset or _DATASET
    video_ids = [args.video_id] if args.video_id else None
    print(f"[Eval] Loading sessions from: {dataset}")

    sessions = load_eval_sessions(
        dataset_name=dataset,
        split=args.split,
        limit_videos=args.limit,
        min_match_score=args.min_score,
        video_ids=video_ids,
    )
    if not sessions:
        print("[Eval] No sessions loaded. Check dataset name / video-id / min-score.")
        sys.exit(1)

    run_id = str(uuid.uuid4())[:8]
    print(f"[Eval] run_id={run_id}  sessions={len(sessions)}  mode={args.mode.upper()}")

    if args.mode == "headless":
        driver = HeadlessSessionDriver(run_id=run_id)
    else:
        driver = MicSessionDriver(run_id=run_id)

    results: list[SessionResult] = []
    for i, session in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] {session.session_id} ({session.duration_s:.0f}s)")
        try:
            result = await driver.run_session(session)
            results.append(result)
            print_session_summary(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    if not results:
        print("[Eval] No results.")
        sys.exit(1)

    kpis = compute_aggregate(results, mode=args.mode)
    print_kpis(kpis)

    if args.report:
        save_json(kpis, [r.metrics for r in results], args.report)

    if args.gate and args.baseline:
        from tests.eval.diff import check_gate
        import json
        baseline = json.loads(Path(args.baseline).read_text())
        passed = check_gate(kpis, baseline, verbose=True)
        sys.exit(0 if passed else 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate sttm-automate integrated pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", help="HuggingFace dataset ID")
    p.add_argument("--split", default="train")
    p.add_argument("--mode", choices=["headless", "mic"], default="headless",
                   help="headless=yt-dlp+mock (CI), mic=play through speakers + capture (observable)")
    p.add_argument("--video-id", dest="video_id",
                   help="Run only this YouTube video ID")
    p.add_argument("--limit", type=int, help="Max videos to load from dataset")
    p.add_argument("--min-score", type=float, default=60.0, dest="min_score",
                   help="Minimum OCR match confidence 0-100 (default 60)")
    p.add_argument("--report", help="Save JSON report to this path")
    p.add_argument("--baseline", help="Path to baseline JSON for regression gating")
    p.add_argument("--gate", action="store_true",
                   help="Exit non-zero if any KPI regresses past tolerance (needs --baseline)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
