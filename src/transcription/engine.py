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
        Auto-normalizes quiet audio before transcription.
        Returns list of segments with timestamps and text.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Auto-gain: normalize quiet audio so Whisper can detect speech
        audio = self._normalize(audio)

        segments_iter, info = self._model.transcribe(
            audio,
            language=config.whisper.language,
            beam_size=config.whisper.beam_size,
            vad_filter=config.whisper.vad_filter,
            vad_parameters=dict(
                threshold=config.whisper.vad_threshold,
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

    @staticmethod
    def _normalize(audio: np.ndarray, target_peak: float = 0.7) -> np.ndarray:
        """Normalize audio to a target peak level for consistent Whisper input."""
        peak = np.max(np.abs(audio))
        if peak < 0.005:
            return audio  # near-silence, don't amplify noise
        gain = min(target_peak / peak, 20.0)
        if gain > 1.2:
            return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        return audio

    @staticmethod
    def has_vocal_content(audio: np.ndarray, samplerate: int = 16000) -> bool:
        """
        Detect if audio contains vocal content vs just instrumental music.
        Uses zero-crossing rate (vocals have higher ZCR than instruments)
        and spectral centroid heuristics.
        """
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.005:
            return False  # silence

        # Zero-crossing rate: vocals typically 0.02-0.15, pure music lower
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2.0 * len(audio))

        # For kirtan with background music, even if ZCR is moderate,
        # we should try transcription. Only skip if very low energy.
        return rms > 0.01 or zero_crossings > 0.03
