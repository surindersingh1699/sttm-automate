"""Vosk transcription engine for Punjabi kirtan audio (offline, small model).

Uses the vosk-model-small-hi (42MB Hindi model) for fast offline recognition.
Outputs Devanagari text which the pipeline converts to Gurmukhi via
devanagari_to_gurmukhi().

Key advantages over Whisper:
- 42MB model vs 244MB (Whisper small)
- RTF ~0.07 (real-time factor) — much faster than Whisper on CPU
- No GPU needed
"""

import json
from pathlib import Path

import numpy as np

from src.transcription.engine import TranscriptionSegment

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "vosk-model-small-hi-0.22"


class VoskTranscriptionEngine:
    """Transcribes audio using Vosk with the small Hindi model."""

    def __init__(self, model_path: str | Path | None = None):
        self._model_path = str(model_path or _DEFAULT_MODEL_PATH)
        self._model = None
        self._samplerate = 16000

    def load(self):
        """Load the Vosk model."""
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)  # suppress Vosk's verbose logging

        if not Path(self._model_path).exists():
            raise RuntimeError(
                f"Vosk model not found at {self._model_path}. "
                "Download from: https://alphacephei.com/vosk/models"
            )

        self._model = Model(self._model_path)
        print(f"[Vosk] Model loaded from {self._model_path}")

    def transcribe(self, audio: np.ndarray) -> list[TranscriptionSegment]:
        """Transcribe audio chunk (16kHz float32 mono) via Vosk."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not self.has_vocal_content(audio):
            return []

        audio = self._normalize(audio)

        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, self._samplerate)
        rec.SetWords(True)

        # Convert float32 [-1,1] to int16 PCM bytes
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Feed audio in chunks for Vosk's streaming interface
        chunk_size = 4000  # bytes (~125ms at 16kHz int16)
        for i in range(0, len(audio_bytes), chunk_size):
            rec.AcceptWaveform(audio_bytes[i:i + chunk_size])

        # Get final result
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        if not text:
            return []

        # Extract word-level timing if available
        words = result.get("result", [])
        start = words[0]["start"] if words else 0.0
        end = words[-1]["end"] if words else 0.0

        return [TranscriptionSegment(start=start, end=end, text=text)]

    @staticmethod
    def _normalize(audio: np.ndarray, target_peak: float = 0.7) -> np.ndarray:
        """Normalize quiet audio."""
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
        return rms > 0.005
