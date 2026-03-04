"""Faster-whisper transcription engine for Punjabi kirtan audio."""

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
    """Transcribes audio using local faster-whisper models."""

    def __init__(self):
        self._model: WhisperModel | None = None
        self._language: str | None = config.whisper.language or None

    def load(self):
        """Load whisper model once at startup."""
        print(
            f"[Whisper] Loading '{config.whisper.model_size}' "
            f"(device={config.whisper.device}, compute={config.whisper.compute_type})..."
        )
        self._model = WhisperModel(
            config.whisper.model_size,
            device=config.whisper.device,
            compute_type=config.whisper.compute_type,
        )
        print("[Whisper] Ready.")

    def transcribe(self, audio: np.ndarray) -> list[TranscriptionSegment]:
        """
        Transcribe audio chunk (16kHz float32 mono) via faster-whisper.
        Returns list of segments with text.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not self.has_vocal_content(audio):
            return []

        # Auto-gain for quiet line-in captures.
        audio = self._normalize(audio)

        kwargs = {
            "beam_size": config.whisper.beam_size,
            "vad_filter": config.whisper.vad_filter,
            "vad_parameters": {
                "threshold": config.whisper.vad_threshold,
                "min_silence_duration_ms": config.whisper.vad_min_silence_ms,
                "speech_pad_ms": config.whisper.vad_speech_pad_ms,
            },
        }
        if self._language:
            kwargs["language"] = self._language

        try:
            segments_iter, info = self._transcribe_with_fallback(audio, kwargs)
        except Exception as e:
            print(f"[Whisper] Transcription error: {e}")
            return []

        if getattr(info, "language", None) and info.language != "pa":
            prob = getattr(info, "language_probability", 0.0)
            print(f"[Whisper] Detected language={info.language} (p={prob:.2f})")

        out: list[TranscriptionSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                out.append(
                    TranscriptionSegment(
                        start=seg.start,
                        end=seg.end,
                        text=text,
                    )
                )
        return out

    def _transcribe_with_fallback(self, audio: np.ndarray, kwargs: dict):
        """Fallback to language auto-detect if explicit language pinning fails."""
        try:
            return self._model.transcribe(audio, **kwargs)
        except ValueError as e:
            if self._language:
                print(
                    f"[Whisper] language='{self._language}' unavailable ({e}); "
                    "falling back to auto-detect."
                )
                self._language = None
                kwargs.pop("language", None)
                return self._model.transcribe(audio, **kwargs)
            raise

    @staticmethod
    def _normalize(audio: np.ndarray, target_peak: float = 0.7) -> np.ndarray:
        """Normalize quiet audio for more stable Whisper input."""
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
        """Check if audio has any content worth transcribing."""
        if audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(audio**2)))
        return rms > 0.001
