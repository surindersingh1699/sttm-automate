"""Ring buffer for producing overlapping audio windows."""

import numpy as np

from src.config import config


class AudioRingBuffer:
    """
    Maintains a sliding window of audio data.
    Produces overlapping chunks for continuous transcription.
    """

    def __init__(self):
        self.samplerate = config.audio.samplerate
        self.chunk_samples = int(config.audio.chunk_duration * self.samplerate)
        self.overlap_samples = int(config.audio.overlap_duration * self.samplerate)
        self.advance_samples = self.chunk_samples - self.overlap_samples

        # Buffer holds current chunk + overlap from previous
        self._previous_tail: np.ndarray | None = None

    def process(self, new_audio: np.ndarray) -> np.ndarray:
        """
        Take new audio and prepend overlap from previous chunk.
        Returns a window of chunk_duration length.
        """
        if self._previous_tail is not None:
            window = np.concatenate([self._previous_tail, new_audio])
        else:
            window = new_audio

        # Save the tail for overlap with next chunk
        if len(new_audio) >= self.overlap_samples:
            self._previous_tail = new_audio[-self.overlap_samples:]
        else:
            self._previous_tail = new_audio.copy()

        # Trim to chunk size
        if len(window) > self.chunk_samples:
            window = window[-self.chunk_samples:]

        return window

    def reset(self):
        """Clear the overlap buffer."""
        self._previous_tail = None
