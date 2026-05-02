"""Session drivers for the STTM-automate integrated eval.

Both drivers wrap the *real* PipelineOrchestrator — no reimplemented dispatch.

HeadlessSessionDriver  – downloads audio via yt-dlp (cached), feeds at 1×
                         speed into AudioCapture.push_external, uses a
                         MockSTTMController. Suitable for CI / regression gating.

MicSessionDriver       – no Playwright. User plays audio through speakers, mic
                         captures it. Pipeline listens for session.duration_s
                         and scores the result.

Both emit a JSONL event log via EventLogger and score via scorer.score_session().
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tests.eval.dataset import SessionDescriptor
from tests.eval.event_log import EventLogger
from tests.eval.mock_controller import MockSTTMController
from tests.eval.scorer import SessionMetrics, score_session

_RUNS_DIR = Path(__file__).parent / "runs"


@dataclass
class SessionResult:
    session: SessionDescriptor
    metrics: SessionMetrics
    event_log_path: Path | None
    mode: Literal["headless", "mic"]
    wall_duration_s: float   # actual wall-clock time the run took


# ── shared orchestrator bootstrap ────────────────────────────────────────────

async def _run_with_orchestrator(
    session: SessionDescriptor,
    mode: Literal["headless", "mic"],
    event_log: EventLogger,
    run_id: str,
    blackhole_device: int | None = None,
    progress_cb=None,
) -> SessionResult:
    """Instantiate real PipelineOrchestrator, run it, score the events."""
    from src.config import config
    from src.pipeline.orchestrator import PipelineOrchestrator

    log_path = _RUNS_DIR / run_id / f"{session.session_id}.jsonl"

    if mode == "headless":
        controller = MockSTTMController()
        controller.reset()
        # In headless mode, force no local audio device — we push_external manually.
        audio_device = -1  # invalid index → AudioCapture.start() will fail gracefully
        orchestrator = PipelineOrchestrator(
            controller=controller,
            broadcast=event_log.make_broadcast(),
            audio_device=audio_device,
        )
    else:
        # Live/mic mode: use real STTMHttpController + specified audio device
        from src.controller.sttm_http import STTMHttpController
        controller = STTMHttpController()
        orchestrator = PipelineOrchestrator(
            controller=controller,
            broadcast=event_log.make_broadcast(),
            audio_device=blackhole_device,
        )

    t_wall_start = time.monotonic()
    event_log.open(t0=t_wall_start)

    try:
        if mode == "headless":
            await _run_headless(session, orchestrator, progress_cb)
        else:
            await _run_mic(session, orchestrator, progress_cb)
    finally:
        try:
            await orchestrator.stop()
        except Exception:
            pass
        event_log.close()
        if log_path:
            event_log.save(log_path)

    wall_duration = time.monotonic() - t_wall_start
    metrics = score_session(event_log.events, session)

    return SessionResult(
        session=session,
        metrics=metrics,
        event_log_path=log_path,
        mode=mode,
        wall_duration_s=round(wall_duration, 1),
    )


# ── headless run ─────────────────────────────────────────────────────────────

async def _run_headless(session: SessionDescriptor, orchestrator, progress_cb=None):
    """Feed yt-dlp cached audio at 1× speed; orchestrator processes it."""
    from src.config import config
    from tests.eval.playback import YtDlpAudioFeeder

    feeder = YtDlpAudioFeeder(
        video_id=session.video_id,
        audio_t0=session.audio_t0,
        audio_t_end=session.audio_t_end,
    )

    # Load audio first so we know the duration
    audio = await feeder.load()
    duration_s = len(audio) / 16_000

    stop_event = asyncio.Event()

    # Load transcription model
    print(f"[Headless] Loading transcription engine for {session.session_id}…")
    await asyncio.to_thread(orchestrator.transcriber.load)

    # Connect controller (MockSTTMController always succeeds)
    await orchestrator.controller.connect()

    # Skip real audio hardware — set up for push_external feeding
    orchestrator._audio_source = "remote"
    orchestrator.running = True

    # Start orchestrator internal tasks. Honor streaming_mode (REA-10) — the
    # naive path uses capture_tick + decode_loop; vad_segmented uses
    # meter_tick + vad_streaming_loop. Modes that aren't wired yet
    # (local_agreement, hybrid) fall back to naive — matches start().
    streaming_mode = getattr(config.streaming, "streaming_mode", "naive")
    if streaming_mode == "vad_segmented":
        print(f"[Headless] streaming_mode=vad_segmented")
        orchestrator_tasks = [
            asyncio.create_task(orchestrator._meter_tick_task()),
            asyncio.create_task(orchestrator._vad_streaming_loop()),
        ]
    elif streaming_mode == "local_agreement":
        print(f"[Headless] streaming_mode=local_agreement")
        orchestrator_tasks = [
            asyncio.create_task(orchestrator._meter_tick_task()),
            asyncio.create_task(orchestrator._local_agreement_loop()),
        ]
    elif streaming_mode == "hybrid":
        print(f"[Headless] streaming_mode=hybrid")
        orchestrator_tasks = [
            asyncio.create_task(orchestrator._meter_tick_task()),
            asyncio.create_task(orchestrator._hybrid_streaming_loop()),
        ]
    else:
        orchestrator_tasks = [
            asyncio.create_task(orchestrator._capture_tick_task()),
            asyncio.create_task(orchestrator._decode_loop()),
        ]

    # Feed audio at 1× speed
    feed_task = asyncio.create_task(
        feeder.feed(orchestrator.audio, stop_event)
    )

    # Progress reporting task
    async def _progress():
        while not stop_event.is_set():
            elapsed = time.monotonic()
            if progress_cb:
                await progress_cb(
                    min(elapsed, duration_s),
                    duration_s,
                    session.session_id,
                )
            await asyncio.sleep(3)

    progress_task = asyncio.create_task(_progress())

    try:
        await feed_task
    finally:
        stop_event.set()
        orchestrator.running = False
        for task in (*orchestrator_tasks, progress_task):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ── mic run ──────────────────────────────────────────────────────────────────

async def _run_mic(session: SessionDescriptor, orchestrator, progress_cb=None):
    """Mic-only live eval — no Playwright. User plays the video manually.
    Pipeline listens via mic for session.duration_s and scores the result."""
    print(
        f"[Mic] ▶ Play this video NOW and seek to {session.audio_t0:.0f}s:\n"
        f"        https://www.youtube.com/watch?v={session.video_id}"
        f"&t={int(session.audio_t0)}s\n"
        f"[Mic] Recording for {session.duration_s:.0f}s…"
    )
    pipeline_task = asyncio.create_task(orchestrator.start())
    await asyncio.sleep(1)  # let pipeline warm up

    elapsed = 0.0
    step = 3.0
    while elapsed < session.duration_s:
        await asyncio.sleep(step)
        elapsed += step
        if progress_cb:
            await progress_cb(elapsed, session.duration_s, session.session_id)

    pipeline_task.cancel()
    try:
        await pipeline_task
    except (asyncio.CancelledError, Exception):
        pass


# ── public API ───────────────────────────────────────────────────────────────

class HeadlessSessionDriver:
    """Run one or more sessions in headless mode (yt-dlp audio, no STTM)."""

    def __init__(self, run_id: str):
        self.run_id = run_id

    async def run_session(
        self,
        session: SessionDescriptor,
        progress_cb=None,
    ) -> SessionResult:
        log = EventLogger(
            out_path=_RUNS_DIR / self.run_id / f"{session.session_id}.jsonl"
        )
        return await _run_with_orchestrator(
            session, mode="headless",
            event_log=log, run_id=self.run_id,
            progress_cb=progress_cb,
        )


class MicSessionDriver:
    """Mic-only eval — no Playwright. Prints a prompt to play the video manually."""

    def __init__(self, run_id: str, audio_device: int | None = None):
        self.run_id = run_id
        self.audio_device = audio_device

    async def run_session(self, session: SessionDescriptor, progress_cb=None) -> SessionResult:
        log = EventLogger(out_path=_RUNS_DIR / self.run_id / f"{session.session_id}.jsonl")
        return await _run_with_orchestrator(
            session, mode="mic",
            event_log=log, run_id=self.run_id,
            blackhole_device=self.audio_device,
            progress_cb=progress_cb,
        )


