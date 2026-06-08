"""Run the headless eval with production-style settings.

Mirrors what `src.api.server.load_runtime_settings` does at dashboard
startup: pin the IndicConformer model + engine + LM config, then drive
the cached session through PipelineOrchestrator and score it.

Usage:
    python scripts/eval_indic_lm.py [VIDEO_ID]

Defaults to `-Dyi8-Qyx4I`, which is already in tests/eval/cache/audio.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config  # noqa: E402
from src.transcription.factory import pin_indic_best_settings  # noqa: E402


def _wire_runtime():
    """Mirror server.load_runtime_settings for the eval process."""
    rt = json.loads((ROOT / ".runtime_settings.json").read_text(encoding="utf-8"))
    if (mid := rt.get("hf_model_id")) in config.whisper.available_models:
        config.whisper.apply_model_id(mid)
    config.whisper.engine = rt.get("engine", "indicconformer")
    if (prec := rt.get("onnx_precision")) in config.whisper.available_precisions:
        config.whisper.onnx_precision = prec
    config.whisper.lm_enabled = bool(rt.get("lm_enabled", True))
    streaming = rt.get("streaming", {}) or {}
    for k, v in streaming.items():
        if hasattr(config.streaming, k):
            setattr(config.streaming, k, v)
    pin_indic_best_settings()
    # Match the existing `engine_indic_naive.json` baseline exactly — naive
    # rolling-window mode — so the only varying knobs are the LM fusion +
    # wider engine audio cap. Has to come AFTER pin_indic_best_settings,
    # which would otherwise force vad_segmented.
    config.streaming.streaming_mode = "naive"


async def _main(video_id: str, label: str):
    _wire_runtime()
    print(
        f"[Eval] engine={config.whisper.engine} model={config.whisper.hf_model_id} "
        f"precision={config.whisper.onnx_precision} lm={config.whisper.lm_enabled} "
        f"beam={config.whisper.lm_beam_width} mode={config.streaming.streaming_mode} "
        f"vad_max_utt={config.streaming.vad_max_utterance_ms}ms "
        f"nemo_chunk={config.whisper.nemo_chunk_len_s}s"
    )

    from tests.eval.dataset import load_eval_sessions  # noqa: PLC0415
    from tests.eval.runner import HeadlessSessionDriver  # noqa: PLC0415

    sessions = load_eval_sessions(video_ids=[video_id])
    if not sessions:
        raise SystemExit(f"No session built for video_id={video_id}")

    run_id = f"{label}_{uuid.uuid4().hex[:6]}"
    driver = HeadlessSessionDriver(run_id=run_id)

    t0 = time.monotonic()
    result = await driver.run_session(sessions[0])
    wall = time.monotonic() - t0
    m = result.metrics

    summary = {
        "label": label,
        "run_id": run_id,
        "video_id": video_id,
        "duration_s": round(m.duration_s, 1),
        "wall_s": round(wall, 1),
        "lock_accuracy_pct": m.lock.lock_accuracy_pct,
        "lock_coverage_pct": m.lock.lock_coverage_pct,
        "ttfcl_s": m.lock.ttfcl_s,
        "wrong_first_lock": m.lock.wrong_first_lock,
        "line_accuracy_exact_pct": m.line.line_accuracy_exact_pct,
        "line_accuracy_pm1_pct": m.line.line_accuracy_pm1_pct,
        "pct_time_correct": m.disruption.pct_time_correct,
        "line_skip_count": m.line.line_skip_count,
        "line_flicker_count": m.line.line_flicker_count,
    }
    out_path = ROOT / "tests" / "eval" / "runs" / f"{label}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Result] {json.dumps(summary, indent=2)}")
    print(f"[Saved]  {out_path}")


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "-Dyi8-Qyx4I"
    label = sys.argv[2] if len(sys.argv) > 2 else "indic_lm_wider_window"
    asyncio.run(_main(vid, label))
