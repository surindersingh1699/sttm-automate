"""Run the cached-audio eval against multiple streaming modes back-to-back and
print a side-by-side comparison.

Each mode shares the same ``-Dyi8-Qyx4I`` cached opus file so the audio input
is identical — any KPI delta is attributable to the streaming-mode change,
not to randomness in playback or audio source.

Captures both:
  * Accuracy KPIs from the scorer (lock acc, line acc ±1, time-correct, etc.)
  * Decode cost from the JSONL event log (number of Whisper calls, total
    decode ms, mean per-call decode ms, decode wall-time per minute of audio)

Run via:
    .venv/bin/python scripts/compare_streaming_modes.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import here so we can mutate config before the runner instantiates the pipeline.
from src.config import config


MODES_TO_TEST = [
    {
        "label": "naive",
        "streaming_mode": "naive",
        "dedup_strategy": "text",
    },
    {
        "label": "vad_segmented",
        "streaming_mode": "vad_segmented",
        "dedup_strategy": "none",
    },
    {
        "label": "local_agreement",
        "streaming_mode": "local_agreement",
        "dedup_strategy": "none",
    },
]


def _apply_mode(spec: dict) -> None:
    config.streaming.streaming_mode = spec["streaming_mode"]
    config.streaming.dedup_strategy = spec["dedup_strategy"]
    # Defaults for everything else — mirror what the dashboard would set.
    config.streaming.locked_prompt_anchor = False


def _summarize_jsonl(log_path: Path) -> dict:
    """Pull decode-cost stats out of the headless run's JSONL event log.

    The orchestrator emits ``transcription`` events that carry
    ``transcribe_ms`` (Whisper wall time) and ``window_seconds`` (audio
    duration). We sum those to compute total decode cost. RTF is the ratio
    of decode time to audio time — lower = better.
    """
    if not log_path.exists():
        return {"error": f"missing log {log_path}"}
    n_calls = 0
    total_decode_ms = 0
    total_audio_s = 0.0
    transcripts_with_text = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Event shape varies by writer; the common fields used here are
        # `event_type` (the broadcast wrapper's `type`) or directly `type`.
        evt_type = evt.get("type") or evt.get("event_type") or evt.get("event", {}).get("type")
        evt_data = evt.get("event") or evt
        if evt_type != "transcription":
            continue
        if "transcribe_ms" in evt_data:
            n_calls += 1
            total_decode_ms += int(evt_data.get("transcribe_ms") or 0)
            total_audio_s += float(evt_data.get("window_seconds") or 0.0)
            if (evt_data.get("text") or "").strip():
                transcripts_with_text += 1
    return {
        "n_decode_calls": n_calls,
        "total_decode_ms": total_decode_ms,
        "total_audio_decoded_s": round(total_audio_s, 2),
        "mean_decode_ms": round(total_decode_ms / max(n_calls, 1), 1),
        "transcripts_nonempty": transcripts_with_text,
    }


async def _run_one_mode(spec: dict) -> dict:
    label = spec["label"]
    print()
    print("=" * 70)
    print(f" Running mode: {label}")
    print("=" * 70)
    _apply_mode(spec)

    # Imported lazily so config mutations apply before the orchestrator constructs.
    from tests.eval.runner import HeadlessSessionDriver
    from tests.eval.dataset import load_eval_sessions

    sessions = load_eval_sessions(video_ids=["-Dyi8-Qyx4I"])
    if not sessions:
        return {"label": label, "error": "no eval session found"}
    session = sessions[0]

    run_id = f"compare_{label}_{int(time.time())}"
    driver = HeadlessSessionDriver(run_id=run_id)
    t_wall_start = time.monotonic()
    try:
        result = await driver.run_session(session)
    except Exception as e:
        return {"label": label, "error": f"run failed: {e}"}
    wall_s = time.monotonic() - t_wall_start

    metrics = result.metrics
    log_path = result.event_log_path
    decode_stats = _summarize_jsonl(log_path) if log_path else {}

    out = {
        "label": label,
        "streaming_mode": spec["streaming_mode"],
        "dedup_strategy": spec["dedup_strategy"],
        "wall_s": round(wall_s, 1),
        "audio_duration_s": getattr(metrics, "duration_s", None),
        "session_id": getattr(metrics, "session_id", session.video_id),
        "log_path": str(log_path) if log_path else None,
    }
    # The metrics dataclass nests sub-metrics; pull common headline fields.
    for attr_path in (
        "lock.lock_accuracy_pct",
        "lock.lock_coverage_pct",
        "lock.ttfcl_p50_s",
        "lock.ttfcl_p90_s",
        "lock.never_locked_pct",
        "transition.detection_rate_pct",
        "transition.spurious_per_hour",
        "line.line_exact_pct",
        "line.line_pm1_pct",
        "line.lag_p50_s",
        "line.lag_p90_s",
        "disruption.recovery_p50_s",
        "disruption.disruptions_per_hour",
        "composite.pct_time_correct",
    ):
        cur = metrics
        for attr in attr_path.split("."):
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if cur is not None:
            out[attr_path.replace(".", "_")] = cur
    out.update(decode_stats)
    return out


def _print_comparison(rows: list[dict]) -> None:
    print()
    print("=" * 70)
    print(" Comparison")
    print("=" * 70)

    def get(d, k, default="—"):
        v = d.get(k, default)
        return v if v is not None else default

    fields = [
        ("Mode",                 "label"),
        ("streaming_mode",       "streaming_mode"),
        ("dedup_strategy",       "dedup_strategy"),
        ("Wall-clock (s)",       "wall_s"),
        ("Audio duration (s)",   "audio_duration_s"),
        ("",                     None),  # blank
        ("Decode calls",         "n_decode_calls"),
        ("Total decode (ms)",    "total_decode_ms"),
        ("Mean per call (ms)",   "mean_decode_ms"),
        ("Audio decoded (s)",    "total_audio_decoded_s"),
        ("Decode RTF",           None),  # computed below
        ("Non-empty transcripts","transcripts_nonempty"),
        ("",                     None),
        ("Lock accuracy %",      "lock_lock_accuracy_pct"),
        ("Lock coverage %",      "lock_lock_coverage_pct"),
        ("TTFCL p50 (s)",        "lock_ttfcl_p50_s"),
        ("TTFCL p90 (s)",        "lock_ttfcl_p90_s"),
        ("Never locked %",       "lock_never_locked_pct"),
        ("Transition detect %",  "transition_detection_rate_pct"),
        ("Spurious / hr",        "transition_spurious_per_hour"),
        ("Line exact %",         "line_line_exact_pct"),
        ("Line ±1 %",            "line_line_pm1_pct"),
        ("% time correct",       "composite_pct_time_correct"),
    ]

    # Header
    headers = ["Metric"] + [r["label"] for r in rows]
    widths = [22] + [22] * len(rows)
    line = lambda cells: "  " + " ".join(f"{c:<{w}}" for c, w in zip(cells, widths))
    print(line(headers))
    print(line(["-" * (w - 1) for w in widths]))
    for label, key in fields:
        if key is None and label == "":
            print(line([""] * len(headers)))
            continue
        cells = [label]
        for r in rows:
            if key == "Decode RTF" or label == "Decode RTF":
                # CPU-cost summary: total decode wall-time / total audio duration.
                decode_ms = r.get("total_decode_ms") or 0
                audio_s = r.get("total_audio_decoded_s") or 0.0
                if audio_s > 0:
                    cells.append(f"{(decode_ms / 1000.0) / audio_s:.3f}")
                else:
                    cells.append("—")
            elif key is None:
                cells.append(str(get(r, "label")))
            else:
                v = get(r, key)
                if isinstance(v, float):
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(str(v))
        print(line(cells))

    # CPU-savings note vs naive
    naive_audio = next((r.get("total_audio_decoded_s") for r in rows if r["label"] == "naive"), None)
    if naive_audio:
        print()
        print("  Notes:")
        print("  - 'Audio decoded (s)' counts the SAME audio multiple times if windows overlap.")
        print(f"  - naive decoded {naive_audio:.0f}s of audio across overlapping windows.")
        for r in rows:
            if r["label"] == "naive":
                continue
            other_audio = r.get("total_audio_decoded_s") or 0
            if other_audio > 0:
                ratio = naive_audio / other_audio
                print(f"  - {r['label']}: {other_audio:.0f}s decoded → {ratio:.2f}× less audio than naive")


async def main() -> int:
    print(f"Comparing {len(MODES_TO_TEST)} streaming modes on cached session -Dyi8-Qyx4I")
    print(f"Each run is independent. Total expected wall-clock: ~{6 * len(MODES_TO_TEST)} min.")

    rows = []
    overall_start = time.monotonic()
    for spec in MODES_TO_TEST:
        result = await _run_one_mode(spec)
        rows.append(result)
        # Progress dump after each mode so we always have partial results
        # even if a later mode crashes.
        snap_path = ROOT / f"/tmp/streaming_compare_partial.json"
        snap_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        elapsed = time.monotonic() - overall_start
        print(f"  → {spec['label']} done in {result.get('wall_s', '?')}s "
              f"(total elapsed: {elapsed:.0f}s)")

    _print_comparison(rows)

    out = ROOT / "tests" / "eval" / "runs" / "streaming_compare_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
