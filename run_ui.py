"""Launch the STTM Automate dashboard without requiring audio hardware.

Stubs out sounddevice so the FastAPI server can start on machines
without PortAudio (e.g. remote servers). The full pipeline won't
capture audio, but the UI and WebSocket connection will work.
"""

import sys
import types

# Stub sounddevice before anything imports it
sd_stub = types.ModuleType("sounddevice")
sd_stub.InputStream = type("InputStream", (), {"__init__": lambda *a, **kw: None})
sd_stub.query_devices = lambda *a, **kw: []
sd_stub.default = types.SimpleNamespace(device=[None, None])
sys.modules["sounddevice"] = sd_stub

# Also stub faster_whisper in case the model isn't downloaded
if "faster_whisper" not in sys.modules:
    fw_stub = types.ModuleType("faster_whisper")
    fw_stub.WhisperModel = type("WhisperModel", (), {
        "__init__": lambda *a, **kw: None,
        "transcribe": lambda *a, **kw: (iter([]), None),
    })
    sys.modules["faster_whisper"] = fw_stub

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=3002, log_level="info")
