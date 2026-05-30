"""Run the headless eval across every cached audio file and aggregate.

Defaults to the LM-OFF baseline config (the proven-safe one). Pass --lm
to also run with LM on per-video for an apples-to-apples per-video
comparison.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _wire(lm_on: bool):
    from src.config import config  # noqa: PLC0415
    from src.transcription.factory import pin_indic_best_settings  # noqa: PLC0415

    rt = json.loads((ROOT / ".runtime_settings.json").read_text(encoding="utf-8"))
    if (mid := rt.get("hf_model_id")) in config.whisper.available_models:
        config.whisper.apply_model_id(mid)
    config.whisper.engine = rt.get("engine", "indicconformer")
    if (prec := rt.get("onnx_precision")) in config.whisper.available_precisions:
        config.whisper.onnx_precision = prec
    config.whisper.lm_enabled = lm_on
    streaming = rt.get("streaming", {}) or {}
    for k, v in streaming.items():
        if hasattr(config.streaming, k):
            setattr(config.streaming, k, v)
    pin_indic_best_settings()
    config.streaming.streaming_mode = "naive"


async def _run_one(video_id: str, run_id: str):
    from tests.eval.dataset import load_eval_sessions  # noqa: PLC0415
    from tests.eval.runner import HeadlessSessionDriver  # noqa: PLC0415

    sessions = load_eval_sessions(video_ids=[video_id])
    if not sessions:
        return None
    driver = HeadlessSessionDriver(run_id=run_id)
    t0 = time.monotonic()
    result = await driver.run_session(sessions[0])
    wall = time.monotonic() - t0
    m = result.metrics
    return {
        "video_id": video_id,
        "duration_s": round(m.duration_s, 1),
        "wall_s": round(wall, 1),
        # The 70% target & its penalty companions
        "locked_correct_pct": m.lock.locked_correct_pct,
        "locked_wrong_pct": m.lock.locked_wrong_pct,
        "unlocked_pct": m.lock.unlocked_pct,
        "net_lock_score_pct": m.lock.net_lock_score_pct,
        "wrong_line_pct": m.lock.wrong_line_pct,
        # Traditional accuracy/coverage view
        "lock_accuracy_pct": m.lock.lock_accuracy_pct,
        "lock_coverage_pct": m.lock.lock_coverage_pct,
        "ttfcl_s": m.lock.ttfcl_s,
        "wrong_first_lock": m.lock.wrong_first_lock,
        "line_accuracy_exact_pct": m.line.line_accuracy_exact_pct,
        "line_accuracy_pm1_pct": m.line.line_accuracy_pm1_pct,
        "pct_time_correct": m.disruption.pct_time_correct,
        "line_skip_count": m.line.line_skip_count,
        "line_flicker_count": m.line.line_flicker_count,
        "total_shabads": m.lock.total_shabads,
    }


async def _main(videos: list[str], label: str, lm_on: bool):
    _wire(lm_on=lm_on)
    from src.config import config  # noqa: PLC0415

    print(
        f"[Multi-eval] label={label} lm={config.whisper.lm_enabled} "
        f"mode={config.streaming.streaming_mode} videos={len(videos)}"
    )
    run_id = f"{label}_{uuid.uuid4().hex[:6]}"
    results = []
    for vid in videos:
        print(f"\n[Multi-eval] === {vid} ===")
        try:
            r = await _run_one(vid, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            r = None
        if r:
            results.append(r)
            print(f"  lock_acc={r['lock_accuracy_pct']}  composite={r['pct_time_correct']}  ttfcl={r['ttfcl_s']}")

    def _med(field):
        vals = [r[field] for r in results if r.get(field) is not None]
        return round(median(vals), 1) if vals else None

    agg = {
        "label": label,
        "lm_on": lm_on,
        "videos": len(results),
        # ★ User's 70% target metrics
        "median_locked_correct_pct": _med("locked_correct_pct"),
        "median_locked_wrong_pct": _med("locked_wrong_pct"),
        "median_unlocked_pct": _med("unlocked_pct"),
        "median_net_lock_score_pct": _med("net_lock_score_pct"),
        "median_wrong_line_pct": _med("wrong_line_pct"),
        # Traditional metrics
        "median_lock_accuracy_pct": _med("lock_accuracy_pct"),
        "median_lock_coverage_pct": _med("lock_coverage_pct"),
        "median_ttfcl_s": _med("ttfcl_s"),
        "median_line_accuracy_exact_pct": _med("line_accuracy_exact_pct"),
        "median_line_accuracy_pm1_pct": _med("line_accuracy_pm1_pct"),
        "median_pct_time_correct": _med("pct_time_correct"),
        "total_wall_s": round(sum(r["wall_s"] for r in results), 1),
        "total_audio_s": round(sum(r["duration_s"] for r in results), 1),
        "per_video": results,
    }
    out = ROOT / "tests" / "eval" / "runs" / f"{label}.json"
    out.write_text(json.dumps(agg, indent=2))
    print(f"\n[Multi-eval] AGGREGATE (median across {len(results)} videos):")
    for k, v in agg.items():
        if k.startswith("median_") or k.startswith("total_"):
            print(f"  {k:34s} {v}")
    print(f"[Saved] {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="multi_naive_lm_off")
    p.add_argument("--lm", action="store_true", help="Enable LM in-beam fusion")
    p.add_argument("videos", nargs="*", help="Video IDs (default: all cached)")
    args = p.parse_args()
    videos = args.videos or sorted(
        f.stem for f in (ROOT / "tests" / "eval" / "cache" / "audio").glob("*.opus")
    )
    asyncio.run(_main(videos, args.label, lm_on=args.lm))
