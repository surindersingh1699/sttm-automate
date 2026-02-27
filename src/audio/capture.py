"""Audio capture from microphone or line-in using sounddevice."""

import numpy as np
import sounddevice as sd
from queue import Queue, Empty
from threading import Event

from src.config import config


class AudioCapture:
    """Captures audio from an input device and provides chunks for transcription."""

    def __init__(self, device: int | None = None):
        self.samplerate = config.audio.samplerate
        self.device = device or config.audio.device
        self._queue: Queue[np.ndarray] = Queue()
        self._stream: sd.InputStream | None = None
        self._stop_event = Event()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"[AudioCapture] status: {status}")
        self._queue.put(indata[:, 0].copy())  # mono: take first channel

    def start(self):
        """Start capturing audio."""
        self._stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=config.audio.channels,
            dtype=config.audio.dtype,
            callback=self._callback,
            blocksize=int(self.samplerate * 0.5),  # 500ms blocks
            device=self.device,
        )
        self._stream.start()

    def get_chunk(self, timeout: float = 10.0) -> np.ndarray | None:
        """
        Collect audio blocks until we have chunk_duration worth of audio.
        Returns a flat float32 numpy array, or None if stopped.
        """
        samples_needed = int(config.audio.chunk_duration * self.samplerate)
        collected: list[np.ndarray] = []
        collected_samples = 0

        while collected_samples < samples_needed:
            if self._stop_event.is_set():
                return None
            try:
                block = self._queue.get(timeout=timeout)
                collected.append(block)
                collected_samples += len(block)
            except Empty:
                if self._stop_event.is_set():
                    return None
                continue

        audio = np.concatenate(collected)[:samples_needed]
        return audio

    def stop(self):
        """Stop capturing audio."""
        self._stop_event.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices."""
        devices = sd.query_devices()
        inputs = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                inputs.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default": i == sd.default.device[0],
                })
        return inputs
