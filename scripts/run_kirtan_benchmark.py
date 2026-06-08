"""Run our end-to-end pipeline against the live-gurbani-captioning-benchmark-v1.

Usage:
    .venv/bin/python scripts/run_kirtan_benchmark.py \\
        --gt /tmp/live-gurbani-captioning-benchmark-v1/test \\
        --out /tmp/sttm_kirtan_submission \\
        [--only zOtIpxMT9hU]                  # one case
        [--cases zOtIpxMT9hU IZOsmkdmmcg ...] # subset (filename stems)

For each ground-truth case in `--gt`:

  1. yt-dlp downloads the audio (cached) at 16 kHz mono.
  2. PipelineOrchestrator boots with a MockSTTMController and an
     EventLogger broadcast.
  3. Audio is fed at 1× wall-clock starting at the cold-start offset
     (UEM start), so the pipeline behaves "live" — predictions at time t
     only depend on audio up to t.
  4. The captured event stream is converted to a benchmark submission
     JSON: a list of (start, end, line_idx) segments using the line index
     from `line_aligned` events while the pipeline is locked. Timestamps
     are absolute seconds into the source audio file (offset added back).

Per-case event logs are saved to runs/kirtan_benchmark_<run_id>/<stem>.jsonl
for later inspection.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _events_to_segments(events: list[dict], offset_s: float) -> list[dict]:
    """Walk the event log and emit one segment per contiguous (shabad, line) state.

    `offset_s` is added to every timestamp so segments are absolute audio-file
    seconds (the benchmark's submission convention), not virtual feed time.
    """
    segments: list[dict] = []
    locked_shabad: int | None = None
    current_line: int | None = None
    seg_start: float | None = None

    def _close(end_t: float) -> None:
        nonlocal seg_start, current_line, locked_shabad
        if seg_start is not None and current_line is not None and locked_shabad is not None:
            end_abs = round(end_t + offset_s, 3)
            start_abs = round(seg_start + offset_s, 3)
            if end_abs > start_abs:
                segments.append(
                    {"start": start_abs, "end": end_abs, "line_idx": int(current_line)}
                )
        seg_start = None

    last_t = 0.0
    for ev in events:
        t = float(ev.get("t", 0.0))
        last_t = max(last_t, t)
        tp = ev.get("type", "")

        if tp == "shabad_locked":
            new_sid = ev.get("shabad_id")
            if new_sid != locked_shabad:
                _close(t)
                locked_shabad = new_sid
                current_line = None
                seg_start = None
        elif tp == "shabad_switched":
            new_sid = ev.get("new_shabad_id")
            if new_sid != locked_shabad:
                _close(t)
                locked_shabad = new_sid
                current_line = None
                seg_start = None
        elif tp in ("force_unlock", "context_flushed"):
            _close(t)
            locked_shabad = None
            current_line = None
        elif tp in ("line_aligned", "line_update"):
            if locked_shabad is None:
                continue
            li = ev.get("line_index")
            if li is None:
                continue
            if li != current_line:
                _close(t)
                current_line = int(li)
                seg_start = t

    _close(last_t)
    return segments


async def _run_one(gt: dict, out_dir: Path, runs_dir: Path, stem: str) -> dict:
    """Run pipeline against one GT case and write the submission JSON."""
    from src.pipeline.orchestrator import PipelineOrchestrator
    from src.config import config
    from tests.eval.event_log import EventLogger
    from tests.eval.mock_controller import MockSTTMController
    from tests.eval.playback import YtDlpAudioFeeder

    video_id: str = gt["video_id"]
    total_dur: float = float(gt["total_duration"])
    uem_start: float = float(gt["uem"]["start"])
    uem_end: float = float(gt["uem"]["end"])

    # Cold-start variants: feed from UEM start onwards. For the canonical
    # case (uem.start = 0) we feed the whole file. We always feed up to
    # the end of the audio file so the pipeline can behave normally.
    audio_t0 = uem_start
    audio_t_end = total_dur

    print(
        f"[{stem}] video={video_id} feed=[{audio_t0:.1f}s..{audio_t_end:.1f}s] "
        f"({audio_t_end - audio_t0:.1f}s of audio)"
    )

    feeder = YtDlpAudioFeeder(
        video_id=video_id,
        audio_t0=audio_t0,
        audio_t_end=audio_t_end,
    )
    audio = await feeder.load()
    duration_s = len(audio) / 16_000
    print(f"[{stem}] audio loaded: {duration_s:.1f}s")

    log_path = runs_dir / f"{stem}.jsonl"
    event_log = EventLogger(out_path=log_path)

    controller = MockSTTMController()
    controller.reset()
    orchestrator = PipelineOrchestrator(
        controller=controller,
        broadcast=event_log.make_broadcast(),
        audio_device=-1,
    )

    print(f"[{stem}] loading transcription engine ({config.whisper.engine})…")
    await asyncio.to_thread(orchestrator.transcriber.load)
    await orchestrator.controller.connect()

    orchestrator._audio_source = "remote"
    orchestrator.running = True

    streaming_mode = getattr(config.streaming, "streaming_mode", "naive")
    if streaming_mode == "vad_segmented":
        tasks = [
            asyncio.create_task(orchestrator._meter_tick_task()),
            asyncio.create_task(orchestrator._vad_streaming_loop()),
        ]
    elif streaming_mode == "local_agreement":
        tasks = [
            asyncio.create_task(orchestrator._meter_tick_task()),
            asyncio.create_task(orchestrator._local_agreement_loop()),
        ]
    elif streaming_mode == "hybrid":
        tasks = [
            asyncio.create_task(orchestrator._meter_tick_task()),
            asyncio.create_task(orchestrator._hybrid_streaming_loop()),
        ]
    else:
        tasks = [
            asyncio.create_task(orchestrator._capture_tick_task()),
            asyncio.create_task(orchestrator._decode_loop()),
        ]

    t_start = time.monotonic()
    event_log.open(t0=t_start)

    stop_event = asyncio.Event()
    feed_task = asyncio.create_task(feeder.feed(orchestrator.audio, stop_event))

    async def _progress():
        while not stop_event.is_set():
            elapsed = time.monotonic() - t_start
            print(f"  [{stem}] {min(elapsed, duration_s):6.1f}/{duration_s:.1f}s")
            await asyncio.sleep(30)

    progress_task = asyncio.create_task(_progress())

    try:
        await feed_task
    finally:
        stop_event.set()
        orchestrator.running = False
        for t in (*tasks, progress_task):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await orchestrator.stop()
        except Exception:
            pass
        event_log.close()
        event_log.save(log_path)

    wall = time.monotonic() - t_start

    # Convert to submission format. Event "t" times are relative to feed start
    # (= audio_t0); the benchmark expects absolute audio-file seconds.
    segments = _events_to_segments(event_log.events, offset_s=audio_t0)
    # Clip to UEM end + small tail (predictions outside UEM are ignored anyway)
    segments = [s for s in segments if s["start"] < uem_end + 5]
    for s in segments:
        s["end"] = min(s["end"], uem_end)
        s["start"] = min(s["start"], s["end"] - 0.001 if s["end"] > 0 else 0)
    segments = [s for s in segments if s["end"] > s["start"]]

    submission = {"video_id": video_id, "segments": segments}
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(submission, indent=2, ensure_ascii=False))

    print(
        f"[{stem}] DONE wall={wall:.1f}s segments={len(segments)} → {out_path}"
    )
    return {
        "stem": stem,
        "wall_s": round(wall, 1),
        "audio_s": round(duration_s, 1),
        "segments": len(segments),
        "events": len(event_log.events),
        "out_path": str(out_path),
    }


async def _main_async(args):
    gt_dir = Path(args.gt).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    runs_dir = _REPO / "tests" / "eval" / "runs" / f"kirtan_benchmark_{run_id}"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(gt_dir.glob("*.json"))
    if args.only:
        cases = [c for c in cases if c.stem == args.only]
    elif args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c.stem in wanted]

    if not cases:
        print(f"No cases match in {gt_dir}")
        return

    print(f"Running {len(cases)} case(s); event logs at {runs_dir}")

    summary = []
    for gt_file in cases:
        gt = json.loads(gt_file.read_text())
        try:
            result = await _run_one(gt, out_dir, runs_dir, gt_file.stem)
            summary.append(result)
        except Exception as e:
            print(f"[{gt_file.stem}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            summary.append({"stem": gt_file.stem, "error": str(e)})

    summary_path = runs_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {summary_path}")
    print(f"Submission dir → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="Ground-truth dir (test/ from benchmark repo)")
    p.add_argument("--out", required=True, help="Submission output dir")
    p.add_argument("--only", help="Run a single GT case by filename stem (e.g. zOtIpxMT9hU_cold66)")
    p.add_argument("--cases", nargs="+", help="Subset of stems to run")
    args = p.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
