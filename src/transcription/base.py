"""Shared transcription engine interface + types.

All backends (faster-whisper, mlx-whisper, whisper.cpp) return the same
`TranscriptionSegment` list so the rest of the pipeline is engine-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


class BaseTranscriptionEngine(ABC):
    """Contract every Whisper backend must implement."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called once before transcribe()."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> list[TranscriptionSegment]:
        """Transcribe a 16kHz float32 mono chunk."""

    @staticmethod
    def _normalize(audio: np.ndarray, target_peak: float = 0.7) -> np.ndarray:
        if audio.size == 0:
            return audio
        peak = float(np.max(np.abs(audio)))
        if peak < 0.005:
            return audio
        gain = min(target_peak / peak, 20.0)
        if gain > 1.2:
            return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        return audio

    @staticmethod
    def has_vocal_content(audio: np.ndarray, samplerate: int = 16000) -> bool:
        if audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(audio**2)))
        return rms > 0.005
