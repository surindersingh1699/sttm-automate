#!/usr/bin/env python3
"""Train a small learned shabad lock ranker from eval event logs.

Example:
  python scripts/train_lock_ranker.py tests/eval/runs/my-run \
    --out models/lock_ranker.joblib --limit-videos 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.matcher.learned_ranker import FEATURE_NAMES, candidate_features
from tests.eval.dataset import gt_at, load_eval_sessions


def _load_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _iter_candidate_rows(run_dir: Path, sessions: dict[str, object]):
    for path in sorted(run_dir.rglob("*.jsonl")):
        session_id = path.stem
        session = sessions.get(session_id)
        if session is None:
            continue
        for event in _load_jsonl(path):
            if event.get("type") != "candidates":
                continue
            gt = gt_at(session.gt_timeline, float(event.get("t", 0.0)))
            if gt is None or gt.shabad_id is None:
                continue
            candidates = event.get("matches") or []
            if not candidates:
                continue
            scores = [float(c.get("score") or 0.0) for c in candidates]
            top_score = scores[0]
            second_score = scores[1] if len(scores) > 1 else 0.0
            for candidate in candidates:
                y = 1 if candidate.get("shabad_id") == gt.shabad_id else 0
                x = candidate_features(
                    candidate,
                    top_score=top_score,
                    second_score=second_score,
                    speech_rate_lps=float(event.get("speech_rate_lps") or 0.0),
                    window_seconds=float(event.get("window_seconds") or 0.0),
                )
                yield x, y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Directory containing eval JSONL logs")
    parser.add_argument("--out", type=Path, default=Path("models/lock_ranker.joblib"))
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    args = parser.parse_args()

    try:
        import joblib  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Install training deps first: pip install scikit-learn joblib"
        ) from exc

    sessions_list = load_eval_sessions(
        limit_videos=args.limit_videos,
        video_ids=args.video_ids,
    )
    sessions = {s.session_id: s for s in sessions_list}
    rows = list(_iter_candidate_rows(args.run_dir, sessions))
    if not rows:
        raise SystemExit("No labeled candidate rows found.")

    x = [row[0] for row in rows]
    y = [row[1] for row in rows]
    positives = sum(y)
    if positives == 0 or positives == len(y):
        raise SystemExit(
            f"Need both positive and negative examples; got {positives}/{len(y)} positives."
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    )
    model.fit(x, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved {args.out} with {len(rows)} rows, {positives} positives.")
    print("Features:", ", ".join(FEATURE_NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
