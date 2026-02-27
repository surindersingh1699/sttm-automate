"""Transcription engine — Google Speech-to-Text for Punjabi kirtan audio."""

import io
import wave
from dataclasses import dataclass

import numpy as np
import speech_recognition as sr


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


class TranscriptionEngine:
    """Transcribes audio using Google Web Speech API (free, no key needed)."""

    def __init__(self):
        self._recognizer: sr.Recognizer | None = None

    def load(self):
        """Initialize the recognizer."""
        print("[Google STT] Initializing recognizer...")
        self._recognizer = sr.Recognizer()
        # Tune for kirtan: lower energy threshold since audio comes via BlackHole
        self._recognizer.energy_threshold = 50
        self._recognizer.dynamic_energy_threshold = False
        print("[Google STT] Ready.")

    def transcribe(self, audio: np.ndarray) -> list[TranscriptionSegment]:
        """
        Transcribe audio chunk (16kHz float32 mono) via Google Speech API.
        Returns list of segments with text.
        """
        if self._recognizer is None:
            raise RuntimeError("Recognizer not loaded. Call load() first.")

        # Skip near-silence
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 0.001:
            return []

        # Convert float32 audio to WAV bytes
        audio_data = self._to_speech_recognition(audio)

        try:
            text = self._recognizer.recognize_google(
                audio_data, language="pa-IN"
            )
            text = text.strip()
            if text:
                duration = len(audio) / 16000
                return [TranscriptionSegment(start=0.0, end=duration, text=text)]
        except sr.UnknownValueError:
            pass  # Google couldn't understand — normal for instrumental sections
        except sr.RequestError as e:
            print(f"[Google STT] API error: {e}")

        return []

    @staticmethod
    def _to_speech_recognition(audio: np.ndarray) -> sr.AudioData:
        """Convert numpy float32 audio to SpeechRecognition AudioData."""
        audio_int16 = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())
        buf.seek(0)
        with sr.AudioFile(buf) as source:
            recognizer = sr.Recognizer()
            return recognizer.record(source)

    @staticmethod
    def has_vocal_content(audio: np.ndarray, samplerate: int = 16000) -> bool:
        """Check if audio has any content worth transcribing."""
        rms = np.sqrt(np.mean(audio**2))
        return rms > 0.001
