"""Audio capture from microphone or line-in using sounddevice.

Maintains a real-time ring of the most recently captured audio (`window_duration`
seconds wide). The sounddevice callback writes directly into the ring; readers
can pull the freshest N seconds at any time via `latest_window()` — decode-side
code never has to drain a queue and so cannot fall behind real time.
"""

import numpy as np
from threading import Event, Lock

from src.config import config


def _get_sd():
    """Lazy import sounddevice (requires PortAudio)."""
    import sounddevice as sd
    return sd


class AudioCapture:
    """Captures audio from an input device and exposes the latest N seconds on demand."""

    def __init__(self, device: int | None = None):
        self.samplerate = config.audio.samplerate
        self.device = device or config.audio.device
        # Ring size == full rolling window. Reader asks for ≤ window_duration seconds.
        self._ring_samples = int(config.audio.window_duration * self.samplerate)
        self._ring = np.zeros(self._ring_samples, dtype=np.float32)
        self._ring_lock = Lock()
        self._samples_written = 0  # monotonic counter, used to detect ring warmup + gaps
        self._stream = None
        self._stop_event = Event()
        self.available = False

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"[AudioCapture] status: {status}")
        # Downmix to mono. Loopback devices (BlackHole) carry stereo; averaging
        # both channels preserves anything panned hard left/right (taking just
        # channel 0 would drop the right side of a stereo source).
        if indata.ndim > 1 and indata.shape[1] > 1:
            block = indata.mean(axis=1)
        else:
            block = indata[:, 0] if indata.ndim > 1 else indata
        n = block.shape[0]
        if n == 0:
            return
        if n >= self._ring_samples:
            # Oversize block — keep only the tail that fits.
            with self._ring_lock:
                self._ring[:] = block[-self._ring_samples:]
                self._samples_written += n
            return
        with self._ring_lock:
            # Shift the ring left by n, append new samples on the right.
            self._ring[:-n] = self._ring[n:]
            self._ring[-n:] = block
            self._samples_written += n

    def start(self):
        """Start capturing audio. Returns True if started, False if no audio hardware."""
        try:
            sd = _get_sd()
        except OSError:
            print("[AudioCapture] PortAudio not available. Local mic disabled — use remote mic.")
            self.available = False
            return False
        self._stop_event.clear()
        with self._ring_lock:
            self._ring.fill(0.0)
            self._samples_written = 0
        # Some devices (notably BlackHole loopback) only accept being opened at
        # their native channel count — CoreAudio rejects channels=1 with
        # PaErrorCode -9998. Query the device and request what it actually
        # supports; the callback downmixes to mono.
        channels = config.audio.channels
        if self.device is not None:
            try:
                info = sd.query_devices(self.device)
                max_in = int(info.get("max_input_channels", 0))
                if max_in >= 1:
                    channels = min(max(channels, 1), max_in) if max_in == 1 else max_in
            except Exception as e:
                print(f"[AudioCapture] Could not query device {self.device}: {e}")
        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=channels,
                dtype=config.audio.dtype,
                callback=self._callback,
                blocksize=int(self.samplerate * 0.5),  # 500ms blocks
                device=self.device,
            )
            self._stream.start()
            self.available = True
            return True
        except Exception as e:
            print(f"[AudioCapture] Could not start audio stream: {e}")
            self._stream = None
            self.available = False
            return False

    def push_external(self, block: np.ndarray) -> None:
        """Feed audio from an external source (e.g. browser mic) into the ring.

        Lets remote audio share the same "latest N seconds" surface as the local
        mic without the decode side having to branch on source.
        """
        if block.ndim > 1:
            block = block[:, 0]
        block = np.ascontiguousarray(block, dtype=np.float32)
        n = block.shape[0]
        if n == 0:
            return
        if n >= self._ring_samples:
            with self._ring_lock:
                self._ring[:] = block[-self._ring_samples:]
                self._samples_written += n
            return
        with self._ring_lock:
            self._ring[:-n] = self._ring[n:]
            self._ring[-n:] = block
            self._samples_written += n

    def reset_ring(self) -> None:
        """Zero the ring and samples counter (e.g. on source switch)."""
        with self._ring_lock:
            self._ring.fill(0.0)
            self._samples_written = 0

    def latest_window(self, seconds: float) -> np.ndarray:
        """Return a copy of the most recent `seconds` of captured audio (wall-clock).

        Older audio is silently discarded — callers can never fall behind real time.
        Before enough audio has been captured to fill the window, the leading portion
        of the returned buffer is zeros (matches the prior `AudioRingBuffer` warmup).
        """
        samples = max(1, int(seconds * self.samplerate))
        samples = min(samples, self._ring_samples)
        with self._ring_lock:
            return self._ring[-samples:].copy()

    def samples_written(self) -> int:
        """Monotonic count of audio samples the sounddevice callback has processed."""
        with self._ring_lock:
            return self._samples_written

    def stop(self):
        """Stop capturing audio."""
        self._stop_event.set()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.available = False
        with self._ring_lock:
            self._ring.fill(0.0)
            self._samples_written = 0

    # Virtual conferencing devices we never want to expose in the dashboard picker
    # (they capture nothing useful for kirtan recognition).
    _HIDDEN_DEVICE_HINTS = (
        "microsoft teams",
        "teams audio",
        "speaker audio recorder",
        "zoom",
        "aggregate",
        "multi-output",
    )

    # System-audio loopback devices — surfaced under a "System Audio" group so the
    # user can recognize whatever's playing through the Mac (e.g. a YouTube
    # gurdwara stream) instead of mic input. Requires the loopback driver to be
    # installed (e.g. `brew install blackhole-2ch`) and routed via a Multi-Output
    # Device — see scripts/setup_audio.py.
    _LOOPBACK_DEVICE_HINTS = (
        "blackhole",
        "loopback",
    )

    @staticmethod
    def list_devices(include_loopback: bool = True) -> list[dict]:
        """List input devices. Each entry carries `kind`: "mic" or "loopback".

        Loopback devices (BlackHole etc.) are returned when `include_loopback` is
        True so the dashboard can offer "System Audio" alongside physical mics.
        Conferencing virtual devices stay hidden regardless.
        """
        try:
            sd = _get_sd()
        except OSError:
            return []
        devices = sd.query_devices()
        inputs = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue
            name_lower = dev["name"].lower()
            if any(hint in name_lower for hint in AudioCapture._HIDDEN_DEVICE_HINTS):
                continue
            is_loopback = any(hint in name_lower for hint in AudioCapture._LOOPBACK_DEVICE_HINTS)
            if is_loopback and not include_loopback:
                continue
            inputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default": i == sd.default.device[0],
                "kind": "loopback" if is_loopback else "mic",
            })
        return inputs

    @staticmethod
    def find_blackhole_device() -> int | None:
        """Find BlackHole virtual audio device index, if installed."""
        try:
            sd = _get_sd()
        except OSError:
            return None
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and "blackhole" in dev["name"].lower():
                return i
        return None

    @staticmethod
    def find_best_device() -> int | None:
        """Auto-select the system default input (ignore virtual/loopback devices)."""
        try:
            sd = _get_sd()
        except OSError:
            print("[AudioCapture] PortAudio not available. Use remote mic mode.")
            return None
        default_idx = sd.default.device[0]
        try:
            info = sd.query_devices(default_idx)
            if info["max_input_channels"] > 0:
                print(f"[AudioCapture] Using default input: {info['name']} (device {default_idx})")
                return default_idx
        except Exception:
            pass
        return None
