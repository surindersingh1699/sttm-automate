"""Google Cloud Speech-to-Text engine for Punjabi kirtan audio (online mode)."""

from dataclasses import dataclass

import numpy as np

from src.transcription.engine import TranscriptionSegment


class GoogleTranscriptionEngine:
    """Transcribes audio using Google Cloud Speech-to-Text API."""

    def __init__(self, credentials_path: str | None = None):
        self._client = None
        self._credentials_path = credentials_path

    def load(self):
        """Initialize the Google Speech client."""
        try:
            from google.cloud import speech
        except ImportError:
            raise RuntimeError(
                "google-cloud-speech not installed. "
                "Run: pip install google-cloud-speech"
            )

        if self._credentials_path:
            import os
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._credentials_path

        self._client = speech.SpeechClient()
        print("[Google STT] Client ready.")

    def transcribe(self, audio: np.ndarray) -> list[TranscriptionSegment]:
        """Transcribe audio chunk (16kHz float32 mono) via Google Speech API."""
        if self._client is None:
            raise RuntimeError("Client not loaded. Call load() first.")

        if not self.has_vocal_content(audio):
            return []

        audio = self._normalize(audio)

        from google.cloud import speech

        # Convert float32 to int16 PCM bytes for Google API
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        audio_content = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="pa-IN",
            alternative_language_codes=["pa-Guru-IN", "hi-IN"],
            enable_word_time_offsets=True,
            model="latest_long",
        )

        try:
            response = self._client.recognize(config=config, audio=audio_content)
        except Exception as e:
            print(f"[Google STT] API error: {e}")
            return []

        out: list[TranscriptionSegment] = []
        for result in response.results:
            if not result.alternatives:
                continue
            alt = result.alternatives[0]
            text = alt.transcript.strip()
            if not text:
                continue

            start = 0.0
            end = 0.0
            if alt.words:
                start = alt.words[0].start_time.total_seconds()
                end = alt.words[-1].end_time.total_seconds()

            out.append(TranscriptionSegment(start=start, end=end, text=text))

        return out

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
        return rms > 0.001
