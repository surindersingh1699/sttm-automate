"""whisper.cpp transcription engine via pywhispercpp bindings.

Cross-platform C++ backend (Windows/Linux/macOS + iOS when compiled natively).
Uses a GGML-format model file. Convert `surt-small-v3` with the whisper.cpp
`models/convert-h5-to-ggml.py` helper (or quantize with `./quantize`) and
point `config.whisper.whisper_cpp_model_path` at the resulting `.bin`.
"""

import re
from pathlib import Path

import numpy as np

from src.config import config
from src.transcription.base import BaseTranscriptionEngine, TranscriptionSegment


# Strip leading chars that can't legitimately start a Gurmukhi word:
#   - non-Gurmukhi script (whisper.cpp hallucinates "!" / "atenção…" prefixes)
#   - dependent vowel signs / halant / nasal marks (matras can only follow a base)
# A valid word-start is an independent vowel (ਅ–ਔ), a base consonant (ਕ–ਹ,
# ਖ਼–ਫ਼), ੳ/ੲ carriers, or whitespace. Anything else at the start is a decoder
# artifact from how whisper.cpp chunks multi-byte UTF-8 at BPE boundaries.
_LEADING_GARBAGE = re.compile(
    r"^[^\u0A05-\u0A39\u0A59-\u0A5E\u0A72-\u0A74\s]+\s*"
)


class WhisperCppEngine(BaseTranscriptionEngine):
    """Transcribes audio using whisper.cpp via pywhispercpp."""

    def __init__(self):
        self._model = None
        self._language: str | None = config.whisper.language or None

    def load(self):
        try:
            from pywhispercpp.model import Model
        except ImportError as e:
            raise RuntimeError(
                "pywhispercpp not installed. `pip install pywhispercpp`."
            ) from e

        model_path = self._resolve_model_path()
        print(
            f"[WhisperCpp] Loading {model_path} "
            f"(threads={config.whisper.whisper_cpp_threads}, "
            f"lang={self._language or 'auto'})..."
        )
        # Model() accepts either a shorthand ("tiny", "base") that auto-downloads
        # stock Whisper weights, or an absolute path to a local GGML file.
        self._model = Model(
            str(model_path),
            n_threads=config.whisper.whisper_cpp_threads,
            print_progress=False,
            print_realtime=False,
        )
        print("[WhisperCpp] Ready.")

    @staticmethod
    def _resolve_model_path() -> Path:
        """Return GGML file path; convert HF→GGML on first load if missing."""
        raw = config.whisper.whisper_cpp_model_path
        if not raw:
            raise RuntimeError(
                "whisper.cpp engine requires `config.whisper.whisper_cpp_model_path`."
            )
        p = Path(raw)
        project_root = Path(__file__).resolve().parents[2]
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if p.exists():
            return p

        from src.transcription._whisper_cpp_convert import convert_hf_to_ggml

        print(
            f"[WhisperCpp] Converting '{config.whisper.hf_model_id}' to GGML → {p}"
        )
        convert_hf_to_ggml(
            config.whisper.hf_model_id,
            p,
            cache_dir=project_root / "data" / "_whisper_cpp_assets",
            use_f16=True,
        )
        print("[WhisperCpp] Conversion complete.")
        return p

    def transcribe(self, audio: np.ndarray) -> list[TranscriptionSegment]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not self.has_vocal_content(audio):
            return []

        audio = self._normalize(audio).astype(np.float32)

        kwargs: dict = {}
        if self._language:
            kwargs["language"] = self._language
        if config.whisper.single_temperature:
            kwargs["temperature"] = 0.0
        if config.whisper.independent_windows:
            kwargs["no_context"] = True

        try:
            segments = self._model.transcribe(audio, **kwargs)
        except Exception as e:
            print(f"[WhisperCpp] Transcription error: {e}")
            return []

        out: list[TranscriptionSegment] = []
        for seg in segments:
            raw = (getattr(seg, "text", "") or "")
            # whisper.cpp emits U+FFFD at BPE boundaries when a multi-byte
            # UTF-8 char (common in Gurmukhi) is split across tokens. Strip it,
            # then drop any leading non-word-start artifacts.
            cleaned = raw.replace("\ufffd", "")
            cleaned = _LEADING_GARBAGE.sub("", cleaned, count=1)
            text = cleaned.strip()
            if not text:
                continue
            # pywhispercpp reports timestamps in centiseconds (1/100s).
            t0 = float(getattr(seg, "t0", 0)) / 100.0
            t1 = float(getattr(seg, "t1", 0)) / 100.0
            out.append(TranscriptionSegment(start=t0, end=t1, text=text))
        return out
