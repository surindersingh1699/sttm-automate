"""Sliding window buffer: short step, long context for Whisper."""

import numpy as np

from src.config import config


class AudioRingBuffer:
    """
    Maintains a sliding window of audio.
    Each call to process() appends new audio (step_duration)
    and returns the full window (window_duration) for Whisper.
    """

    def __init__(self):
        self.samplerate = config.audio.samplerate
        self.window_samples = int(config.audio.window_duration * self.samplerate)
        # Start with silence so Whisper always gets a full window
        self._buffer = np.zeros(self.window_samples, dtype=np.float32)

    def process(self, new_audio: np.ndarray) -> np.ndarray:
        """
        Append new audio, shift window, return full window_duration of audio.
        """
        n = len(new_audio)
        # Shift old audio left, append new audio at the end
        self._buffer = np.concatenate([self._buffer[n:], new_audio])
        return self._buffer.copy()

    def reset(self):
        """Clear the buffer."""
        self._buffer = np.zeros(self.window_samples, dtype=np.float32)
