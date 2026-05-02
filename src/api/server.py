"""FastAPI application with WebSocket for real-time dashboard."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import config
from src.controller.sttm_http import STTMHttpController
from src.pipeline.orchestrator import PipelineOrchestrator
from src.transcription.factory import SUPPORTED_ENGINES

# Connected WebSocket clients
clients: list[WebSocket] = []

# Pipeline instance (initialized on startup)
pipeline: PipelineOrchestrator | None = None

# Eval job manager (initialized on startup)
eval_manager = None
runtime_settings_path = Path(__file__).parent.parent.parent / ".runtime_settings.json"
confidence_mode = "balanced"
mic_muted_pref = False


# Maps each wire toggle key to the config section that owns it. Keeps the
# dashboard payload flat while fields can live in whatever config class fits.
TOGGLE_SECTIONS: dict[str, str] = {
    "greedy_decode": "whisper",
    "single_temperature": "whisper",
    "allow_repetition": "whisper",
    "independent_windows": "whisper",
    "cap_decode_length": "whisper",
    "skip_slow_windows": "whisper",
    "hallucination_guards": "whisper",
    "multi_line_search": "matcher",
    "multi_line_locked_align": "matcher",
    "fast_response_enabled": "matcher",
    "zero_overlap_window": "audio",
}
DECODER_TOGGLE_KEYS = tuple(TOGGLE_SECTIONS.keys())


# Streaming-pipeline settings (REA-10). Non-boolean values, separate plumbing
# from the boolean decoder toggles above. Each entry maps to a field on
# ``config.streaming``. Validation is done in ``_apply_streaming_setting``.
STREAMING_KEYS: tuple[str, ...] = (
    "streaming_mode",
    "vad_backend",
    "vad_threshold",
    "vad_min_silence_ms",
    "vad_min_speech_ms",
    "vad_speech_pad_ms",
    "vad_max_utterance_ms",
    "local_agreement_n",
    "local_agreement_decode_interval_ms",
    "local_agreement_max_buffer_ms",
    "dedup_strategy",
    "locked_prompt_anchor",
)
_STREAMING_MODE_VALUES = ("naive", "vad_segmented", "local_agreement", "hybrid")
_VAD_BACKEND_VALUES = ("kirtan", "silero")
_DEDUP_STRATEGY_VALUES = ("text", "audio_time", "none")


def get_streaming_settings() -> dict:
    return {key: getattr(config.streaming, key) for key in STREAMING_KEYS}


def _apply_streaming_setting(key: str, value) -> None:
    if key not in STREAMING_KEYS:
        return
    # Enum-typed fields — reject unknown values rather than silently corrupting state.
    if key == "streaming_mode" and value not in _STREAMING_MODE_VALUES:
        return
    if key == "vad_backend" and value not in _VAD_BACKEND_VALUES:
        return
    if key == "dedup_strategy" and value not in _DEDUP_STRATEGY_VALUES:
        return
    # Numeric / boolean coercion. Bool check first because Python treats
    # bool as a subclass of int.
    cur = getattr(config.streaming, key)
    if isinstance(cur, bool):
        value = bool(value)
    elif isinstance(cur, int):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return
    elif isinstance(cur, float):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
    setattr(config.streaming, key, value)


def get_decoder_toggles() -> dict:
    return {
        key: bool(getattr(getattr(config, TOGGLE_SECTIONS[key]), key))
        for key in DECODER_TOGGLE_KEYS
    }


def _apply_decoder_toggle(key: str, value: bool) -> None:
    section = TOGGLE_SECTIONS.get(key)
    if section is None:
        return
    setattr(getattr(config, section), key, bool(value))


def load_runtime_settings():
    """Load persisted runtime settings (if present)."""
    global confidence_mode, mic_muted_pref
    if not runtime_settings_path.exists():
        return
    try:
        data = json.loads(runtime_settings_path.read_text(encoding="utf-8"))
        if "controller_pin" in data:
            value = data["controller_pin"]
            config.sttm.controller_pin = int(value) if value not in (None, "") else None
        mode = data.get("confidence_mode", "balanced")
        confidence_mode = mode if mode in ("conservative", "balanced", "fast", "gurudwara") else "balanced"
        if "audio_device" in data:
            dev = data["audio_device"]
            config.audio.device = int(dev) if dev is not None else None
        toggles = data.get("decoder_toggles") or {}
        for key in DECODER_TOGGLE_KEYS:
            if key in toggles:
                _apply_decoder_toggle(key, toggles[key])
        # Streaming settings (REA-10) — same pattern as decoder_toggles but
        # values include enums and ints, not just bools.
        streaming = data.get("streaming") or {}
        for key in STREAMING_KEYS:
            if key in streaming:
                _apply_streaming_setting(key, streaming[key])
        if "engine" in data:
            eng = data["engine"]
            if eng in SUPPORTED_ENGINES:
                config.whisper.engine = eng
        if "hf_model_id" in data:
            mid = data["hf_model_id"]
            if isinstance(mid, str) and mid in config.whisper.available_models:
                config.whisper.apply_model_id(mid)
        if "mic_muted" in data:
            mic_muted_pref = bool(data["mic_muted"])
    except Exception as e:
        print(f"[Server] Could not load runtime settings: {e}")


def save_runtime_settings():
    """Persist runtime settings so they survive app restarts."""
    payload = {
        "controller_pin": config.sttm.controller_pin,
        "confidence_mode": confidence_mode,
        "audio_device": config.audio.device,
        "decoder_toggles": get_decoder_toggles(),
        "streaming": get_streaming_settings(),
        "engine": config.whisper.engine,
        "hf_model_id": config.whisper.hf_model_id,
        "mic_muted": bool(pipeline.mic_muted) if pipeline else mic_muted_pref,
    }
    try:
        runtime_settings_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[Server] Could not save runtime settings: {e}")


async def broadcast(data: dict):
    """Send data to all connected dashboard clients."""
    if not clients:
        return
    message = json.dumps(data, ensure_ascii=False)
    disconnected = []
    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        clients.remove(client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start pipeline on app startup, stop on shutdown."""
    global pipeline, eval_manager
    load_runtime_settings()

    # Try HTTP controller — if STTM isn't running, pipeline still works in monitor mode
    controller = STTMHttpController()
    connected = await controller.connect()
    if not connected:
        print("[Server] STTM not detected. Running in monitor-only mode.")
        print("[Server] Start STTM Desktop to enable auto-display.")

    pipeline = PipelineOrchestrator(
        controller=controller,
        broadcast=broadcast,
    )
    pipeline.set_confidence_mode(confidence_mode)
    if mic_muted_pref:
        pipeline.mic_muted = True

    from tests.eval.job_manager import EvalJobManager
    eval_manager = EvalJobManager(broadcast)

    # Start pipeline in background
    task = asyncio.create_task(pipeline.start())

    yield

    # Shutdown
    if pipeline:
        await pipeline.stop()
    task.cancel()


app = FastAPI(title="STTM Automate", lifespan=lifespan)

# Serve static files
static_dir = Path(__file__).parent.parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    """Serve the dashboard."""
    return FileResponse(str(static_dir / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard communication."""
    global confidence_mode
    await websocket.accept()
    clients.append(websocket)

    # Send initial state (include verses so pangati panel populates immediately)
    if pipeline:
        current = pipeline.tracker.current
        init_state = {
            "type": "state",
            "pipeline_state": pipeline.tracker.state.value,
            "current": current.to_dict() if current else None,
            "history": pipeline.tracker.get_history_list(),
            "controller_pin": config.sttm.controller_pin,
            "confidence_mode": pipeline.confidence_mode,
            "audio_source": pipeline._audio_source,
            "audio_device": config.audio.device,
            "mic_muted": bool(pipeline.mic_muted),
            "hypotheses": pipeline.tracker.get_hypotheses(),
            "decoder_toggles": get_decoder_toggles(),
            "streaming_settings": get_streaming_settings(),
            "engine": config.whisper.engine,
            "engines": list(SUPPORTED_ENGINES),
        }
        if current and current.verses:
            init_state["verses"] = [
                {"unicode": v.unicode, "english": v.english}
                for v in current.verses
            ]
        await websocket.send_text(json.dumps(init_state, ensure_ascii=False))

    try:
        while True:
            try:
                data = await websocket.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                break

            msg = json.loads(data)

            if not pipeline:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "manual_select":
                shabad_id = msg.get("shabad_id")
                if shabad_id:
                    await pipeline.manual_select(int(shabad_id))

            elif msg_type == "navigate":
                direction = msg.get("direction", "next")
                await pipeline.manual_navigate(direction)

            elif msg_type == "recall":
                shabad_id = msg.get("shabad_id")
                if shabad_id:
                    await pipeline.recall_shabad(int(shabad_id))

            elif msg_type == "pause":
                pipeline.pause()
                await broadcast({"type": "status", "paused": True})

            elif msg_type == "resume":
                pipeline.resume()
                await broadcast({"type": "status", "paused": False})

            elif msg_type == "set_controller_pin":
                pin = msg.get("controller_pin")
                if pin in (None, ""):
                    config.sttm.controller_pin = None
                else:
                    config.sttm.controller_pin = int(pin)
                save_runtime_settings()
                await broadcast({
                    "type": "controller_pin_updated",
                    "controller_pin": config.sttm.controller_pin,
                })

            elif msg_type == "force_unlock":
                await pipeline.force_unlock()

            elif msg_type == "flush_context":
                await pipeline.flush_context()

            elif msg_type == "set_confidence_mode":
                mode = msg.get("mode", "balanced")
                if mode not in ("conservative", "balanced", "fast", "gurudwara"):
                    mode = "balanced"
                confidence_mode = mode
                pipeline.set_confidence_mode(mode)
                save_runtime_settings()
                await broadcast({
                    "type": "confidence_mode_updated",
                    "mode": confidence_mode,
                })

            elif msg_type == "set_mic_muted":
                muted = bool(msg.get("muted", False))
                ok = pipeline.set_mic_muted(muted)
                save_runtime_settings()
                await broadcast({
                    "type": "mic_muted_updated",
                    "muted": bool(pipeline.mic_muted),
                    "ok": ok,
                })

            elif msg_type == "set_audio_source":
                source = msg.get("source", "local")
                if source in ("local", "remote"):
                    pipeline.set_audio_source(source)
                    await broadcast({
                        "type": "audio_source_updated",
                        "source": source,
                    })

            elif msg_type == "set_audio_device":
                dev = msg.get("device")
                device_index = int(dev) if dev is not None else None
                ok = pipeline.switch_audio_device(device_index)
                if ok:
                    save_runtime_settings()
                    await broadcast({
                        "type": "audio_device_updated",
                        "device": device_index,
                    })

            elif msg_type == "set_decoder_toggles":
                toggles = msg.get("toggles") or {}
                for key in DECODER_TOGGLE_KEYS:
                    if key in toggles:
                        _apply_decoder_toggle(key, toggles[key])
                save_runtime_settings()
                await broadcast({
                    "type": "decoder_toggles_updated",
                    "toggles": get_decoder_toggles(),
                })

            elif msg_type == "set_streaming_settings":
                # REA-10: streaming-mode + VAD + dedup + LocalAgreement toggles.
                # Validation (enum values, type coercion) happens inside
                # _apply_streaming_setting; unknown keys are silently dropped.
                settings = msg.get("settings") or {}
                for key, value in settings.items():
                    _apply_streaming_setting(key, value)
                save_runtime_settings()
                await broadcast({
                    "type": "streaming_settings_updated",
                    "settings": get_streaming_settings(),
                })

            elif msg_type == "reconnect_sttm":
                # STTM Desktop might have been started (or restarted) after the
                # Python server; re-probe its ports on demand.
                await broadcast({"type": "sttm_reconnecting"})
                ok = await pipeline.controller.connect()
                await broadcast({
                    "type": "sttm_reconnect_result",
                    "connected": bool(ok),
                    "base_url": pipeline.controller.base_url,
                })

            elif msg_type == "set_model":
                # Switch the active Whisper model. Rewrites all engine-specific
                # cache paths (CT2/MLX/GGML/GGML-q8) via apply_model_id() and
                # reloads the current engine with the new paths so the swap is
                # live without a server restart.
                model_id = msg.get("model_id")
                if not isinstance(model_id, str) or model_id not in config.whisper.available_models:
                    await broadcast({
                        "type": "model_update_failed",
                        "model_id": model_id,
                        "error": f"Unknown model '{model_id}'.",
                        "current_model_id": config.whisper.hf_model_id,
                    })
                else:
                    await broadcast({
                        "type": "model_loading",
                        "model_id": model_id,
                    })
                    previous_model_id = config.whisper.hf_model_id
                    config.whisper.apply_model_id(model_id)
                    ok, err = await pipeline.switch_engine(config.whisper.engine)
                    if ok:
                        save_runtime_settings()
                        await broadcast({
                            "type": "model_updated",
                            "model_id": config.whisper.hf_model_id,
                        })
                    else:
                        # Roll back so the dashboard reflects what's actually loaded.
                        config.whisper.apply_model_id(previous_model_id)
                        await broadcast({
                            "type": "model_update_failed",
                            "model_id": model_id,
                            "error": err or "Engine reload failed.",
                            "current_model_id": config.whisper.hf_model_id,
                        })

            elif msg_type == "set_engine":
                name = msg.get("engine")
                if name not in SUPPORTED_ENGINES:
                    await broadcast({
                        "type": "engine_update_failed",
                        "engine": name,
                        "error": f"Unknown engine '{name}'.",
                    })
                else:
                    await broadcast({
                        "type": "engine_loading",
                        "engine": name,
                    })
                    ok, err = await pipeline.switch_engine(name)
                    if ok:
                        save_runtime_settings()
                        await broadcast({
                            "type": "engine_updated",
                            "engine": config.whisper.engine,
                        })
                    else:
                        await broadcast({
                            "type": "engine_update_failed",
                            "engine": name,
                            "error": err or "Engine load failed.",
                            "current_engine": config.whisper.engine,
                        })

    finally:
        if websocket in clients:
            clients.remove(websocket)


@app.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    """Dedicated WebSocket for receiving raw audio from browser mic."""
    await websocket.accept()
    print("[Server] Remote audio client connected")

    try:
        import numpy as np
        while True:
            data = await websocket.receive_bytes()
            if pipeline and pipeline._audio_source == "remote":
                # Browser sends 16-bit PCM at 16kHz mono
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                pipeline.push_remote_audio(samples)
    except (WebSocketDisconnect, RuntimeError):
        print("[Server] Remote audio client disconnected")


@app.get("/api/status")
async def get_status():
    """Get current pipeline status."""
    if not pipeline:
        return {"running": False}
    return {
        "running": pipeline.running,
        "paused": pipeline.paused,
        "pipeline_state": pipeline.tracker.state.value,
        "current": pipeline.tracker.current.to_dict() if pipeline.tracker.current else None,
        "history_count": len(pipeline.tracker.history),
        "confidence_mode": pipeline.confidence_mode,
        "hypotheses": pipeline.tracker.get_hypotheses(),
    }


@app.get("/api/verses/{shabad_id}")
async def get_verses(shabad_id: int):
    """Fetch all verses for a shabad (fallback for when WebSocket broadcast misses)."""
    if not pipeline:
        return {"verses": []}

    # First try the cached verses from the tracker
    current = pipeline.tracker.current
    if current and current.shabad_id == shabad_id and current.verses:
        return {
            "shabad_id": shabad_id,
            "verses": [
                {"unicode": v.unicode, "english": v.english}
                for v in current.verses
            ],
        }

    # Otherwise fetch from the local DB (no external APIs).
    import asyncio
    verses = await asyncio.to_thread(pipeline.searcher.fetch_all_verses, shabad_id)
    return {
        "shabad_id": shabad_id,
        "verses": [
            {"unicode": v.unicode, "english": v.english}
            for v in verses
        ],
    }


@app.get("/api/devices")
async def get_audio_devices():
    """List available audio input devices."""
    from src.audio.capture import AudioCapture
    return {"devices": AudioCapture.list_devices()}


@app.get("/eval/audio/{video_id}")
async def eval_audio(video_id: str):
    """Serve cached yt-dlp audio for the eval player. Downloads if not cached."""
    import re
    from fastapi.responses import FileResponse
    from pathlib import Path
    # Sanitise video_id — only alphanumeric, dash, underscore allowed
    if not re.match(r'^[\w\-]{1,64}$', video_id):
        raise HTTPException(400, "Invalid video_id")
    cache_dir = Path(__file__).parent.parent.parent / "tests" / "eval" / "cache" / "audio"
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / f"{video_id}.opus"
    if not audio_path.exists():
        # Download via yt-dlp (same as headless eval does)
        try:
            from tests.eval.playback import YtDlpAudioFeeder
            feeder = YtDlpAudioFeeder(video_id=video_id, audio_t0=0, audio_t_end=None)
            await asyncio.to_thread(feeder._ensure_cached_sync)
        except Exception as exc:
            raise HTTPException(500, f"Download failed: {exc}")
    if not audio_path.exists():
        # Check for any cached file with a different extension
        candidates = list(cache_dir.glob(f"{video_id}.*"))
        if candidates:
            return FileResponse(str(candidates[0]), media_type="audio/ogg")
        raise HTTPException(404, "Audio not available")
    return FileResponse(str(audio_path), media_type="audio/ogg")


@app.get("/eval/session/{video_id}")
async def eval_session_info(video_id: str):
    """Return session metadata for a video (t0, t_end, duration, title)."""
    import re
    if not re.match(r'^[\w\-]{1,64}$', video_id):
        raise HTTPException(400, "Invalid video_id")
    try:
        from tests.eval.dataset import _DATASET, load_eval_sessions
        sessions = await asyncio.to_thread(load_eval_sessions, dataset_name=_DATASET, video_ids=[video_id])
        if not sessions:
            raise HTTPException(404, "Video not in eval dataset")
        s = sessions[0]
        return {
            "video_id": s.video_id,
            "session_id": s.session_id,
            "audio_t0": s.audio_t0,
            "audio_t_end": s.audio_t_end,
            "duration_s": s.duration_s,
            "title": getattr(s, "title", s.video_id),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Eval API ────────────────────────────────────────────────────────────────

class EvalRunRequest(BaseModel):
    mode: str = "headless"
    limit_videos: int | None = None
    min_match_score: float = 60.0
    video_ids: list[str] | None = None
    dataset: str | None = None
    audio_device: int | None = None


@app.post("/eval/run")
async def eval_run(req: EvalRunRequest):
    """Start a new eval run in the background."""
    if eval_manager is None:
        raise HTTPException(503, "Eval manager not ready")
    if eval_manager.is_running():
        raise HTTPException(409, "An eval run is already in progress")
    # For mic mode, pause the dashboard pipeline so it releases the audio device
    if req.mode == "mic" and pipeline is not None:
        pipeline.pause()
    run_id = await eval_manager.start_run(
        mode=req.mode,
        limit_videos=req.limit_videos,
        min_match_score=req.min_match_score,
        video_ids=req.video_ids,
        dataset=req.dataset,
        audio_device=req.audio_device,
    )
    # Resume the dashboard pipeline once the eval task completes
    if req.mode == "mic" and pipeline is not None:
        eval_manager._task.add_done_callback(lambda _: pipeline.resume() if pipeline else None)
    return {"run_id": run_id, "status": "started"}


@app.get("/eval/status")
async def eval_status():
    """Return current eval run state."""
    if eval_manager is None or eval_manager.state is None:
        return {"status": "idle"}
    s = eval_manager.state
    return {
        "run_id": s.run_id,
        "status": s.status,
        "mode": s.mode,
        "total_jobs": s.total_jobs,
        "completed": s.completed,
        "current_job_id": s.current_job_id,
        "error": s.error,
        "started_at": s.started_at,
        "finished_at": s.finished_at,
        "running": eval_manager.is_running(),
    }


@app.get("/eval/results")
async def eval_results():
    """Return completed eval results."""
    if eval_manager is None or eval_manager.state is None:
        return {"status": "idle", "report": None, "per_job": []}
    from dataclasses import asdict
    s = eval_manager.state
    return {
        "run_id": s.run_id,
        "status": s.status,
        "report": s.report,
        "per_job": [asdict(m) for m in s.per_job],
        "completed": s.completed,
        "total_jobs": s.total_jobs,
    }


@app.get("/eval/videos")
async def eval_videos():
    """List available videos in the eval dataset (lightweight, no audio loaded)."""
    from tests.eval.dataset import _DATASET, available_videos
    try:
        videos = await asyncio.to_thread(available_videos, _DATASET)
        return {"videos": videos}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/eval/runs")
async def eval_runs():
    """List completed eval runs saved to disk, newest first."""
    import json
    from pathlib import Path
    runs_dir = Path(__file__).parent.parent / "tests" / "eval" / "runs"
    if not runs_dir.exists():
        return {"runs": []}
    runs = []
    for report_path in sorted(runs_dir.glob("*/report.json"), reverse=True):
        try:
            data = json.loads(report_path.read_text())
            agg = data.get("aggregate", {})
            runs.append({
                "run_id": data.get("run_id", report_path.parent.name),
                "mode": data.get("mode", "headless"),
                "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"),
                "total_sessions": data.get("total_sessions", 0),
                "lock_accuracy_pct": agg.get("median_lock_accuracy_pct"),
                "line_accuracy_pm1_pct": agg.get("median_line_accuracy_pm1_pct"),
                "composite_pct_time_correct": agg.get("composite_pct_time_correct"),
                "overall_detection_rate_pct": agg.get("overall_detection_rate_pct"),
                "sessions": data.get("sessions", []),
            })
        except Exception:
            pass
    return {"runs": runs}
