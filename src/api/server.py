"""FastAPI application with WebSocket for real-time dashboard."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import config
from src.controller.sttm_http import STTMHttpController
from src.pipeline.orchestrator import PipelineOrchestrator

# Connected WebSocket clients
clients: list[WebSocket] = []

# Pipeline instance (initialized on startup)
pipeline: PipelineOrchestrator | None = None
runtime_settings_path = Path(__file__).parent.parent.parent / ".runtime_settings.json"
confidence_mode = "balanced"


def load_runtime_settings():
    """Load persisted runtime settings (if present)."""
    global confidence_mode
    if not runtime_settings_path.exists():
        return
    try:
        data = json.loads(runtime_settings_path.read_text(encoding="utf-8"))
        if "controller_pin" in data:
            value = data["controller_pin"]
            config.sttm.controller_pin = int(value) if value not in (None, "") else None
        mode = data.get("confidence_mode", "balanced")
        confidence_mode = mode if mode in ("conservative", "balanced", "fast") else "balanced"
        engine = data.get("transcription_engine", "whisper")
        config.transcription.engine = engine if engine in ("whisper", "google") else "whisper"
        if "google_credentials_path" in data and data["google_credentials_path"]:
            config.transcription.google_credentials_path = data["google_credentials_path"]
    except Exception as e:
        print(f"[Server] Could not load runtime settings: {e}")


def save_runtime_settings():
    """Persist runtime settings so they survive app restarts."""
    payload = {
        "controller_pin": config.sttm.controller_pin,
        "confidence_mode": confidence_mode,
        "transcription_engine": config.transcription.engine,
        "google_credentials_path": config.transcription.google_credentials_path,
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
    global pipeline
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
            "transcription_engine": pipeline.current_engine,
            "audio_source": pipeline._audio_source,
            "hypotheses": pipeline.tracker.get_hypotheses(),
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

            elif msg_type == "set_confidence_mode":
                mode = msg.get("mode", "balanced")
                if mode not in ("conservative", "balanced", "fast"):
                    mode = "balanced"
                confidence_mode = mode
                pipeline.set_confidence_mode(mode)
                save_runtime_settings()
                await broadcast({
                    "type": "confidence_mode_updated",
                    "mode": confidence_mode,
                })

            elif msg_type == "set_audio_source":
                source = msg.get("source", "local")
                if source in ("local", "remote"):
                    pipeline.set_audio_source(source)
                    await broadcast({
                        "type": "audio_source_updated",
                        "source": source,
                    })

            elif msg_type == "set_transcription_engine":
                engine = msg.get("engine", "whisper")
                if engine in ("whisper", "google"):
                    await broadcast({
                        "type": "engine_switching",
                        "engine": engine,
                    })
                    await pipeline.switch_engine(engine)
                    save_runtime_settings()
                    await broadcast({
                        "type": "transcription_engine_updated",
                        "engine": engine,
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

    # Otherwise fetch from BaniDB
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
