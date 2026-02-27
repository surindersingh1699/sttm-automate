"""Faster-whisper transcription engine for Punjabi audio."""

from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from src.config import config


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


class TranscriptionEngine:
    """Loads faster-whisper and transcribes audio chunks to Punjabi text."""

    def __init__(self):
        self._model: WhisperModel | None = None

    def load(self):
        """Load the Whisper model. Call once at startup."""
        print(f"[Whisper] Loading '{config.whisper.model_size}' model "
              f"(device={config.whisper.device}, compute={config.whisper.compute_type})...")
        self._model = WhisperModel(
            config.whisper.model_size,
            device=config.whisper.device,
            compute_type=config.whisper.compute_type,
        )
        print("[Whisper] Model loaded.")

    def transcribe(self, audio: np.ndarray) -> list[TranscriptionSegment]:
        """
        Transcribe an audio chunk (16kHz float32 mono).
        Returns list of segments with timestamps and text.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        segments_iter, info = self._model.transcribe(
            audio,
            language=config.whisper.language,
            beam_size=config.whisper.beam_size,
            vad_filter=config.whisper.vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=config.whisper.vad_min_silence_ms,
                speech_pad_ms=config.whisper.vad_speech_pad_ms,
            ),
        )

        results = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                results.append(TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                ))

        return results
