"""Post-processing for transcription output: cleanup, filtering, and deduplication.

Cross-window dedup is delegated to a strategy object (see ``src/pipeline/dedup``)
so the legitimate-repetition bug (rahau, ਵਾਹਿਗੁਰੂ jaaps, sangat echo) can be
fixed by switching to ``audio_time`` without otherwise changing the post-
processing pipeline. The strategy is read each call from
``config.streaming.dedup_strategy`` so the dashboard toggle takes effect live.
"""

import re

from src.config import config
from src.pipeline.dedup import (
    AudioTimeDedup,
    DedupStrategy,
    NoOpDedup,
    TextDedup,
)
from src.transcription.engine import TranscriptionSegment


# Patterns that indicate Whisper hallucination or garbage output
_GARBAGE_PATTERNS = [
    re.compile(r'^[.\s,،؟!]+$'),                    # only punctuation
    re.compile(r'^(.)\1{3,}'),                       # repeated same char (rrrrr)
    re.compile(r'[\u4e00-\u9fff]'),                  # Chinese characters
    re.compile(r'[\u3040-\u30ff]'),                  # Japanese
    re.compile(r'[\uac00-\ud7af]'),                  # Korean
    re.compile(r'(?i)subscribe|like|comment|video'),  # YouTube artifacts
    re.compile(r'(?i)thank you for watching'),
    re.compile(r'ॐ'),                                # Om hallucination from music/tabla
    re.compile(r'^[ॐ\s]+$'),                         # only Om symbols
    re.compile(r'(?i)^(music|tabla|harmonium|dhol)'), # instrument labels
    re.compile(r'^\W+$'),                            # only non-word characters
    re.compile(r'^(.{1,10})\1{2,}'),                 # short phrase repeated 3+ times
]

# Valid script ranges for Punjabi/Hindi transcription
_VALID_RANGES = [
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0900, 0x097F),  # Devanagari
    (0x0600, 0x06FF),  # Arabic/Shahmukhi
    (0x0041, 0x007A),  # ASCII letters (romanized)
]


def _is_valid_text(text: str) -> bool:
    """Check if text is valid Punjabi transcription (not garbage)."""
    text = text.strip()

    if len(text) < 2:
        return False

    for pattern in _GARBAGE_PATTERNS:
        if pattern.search(text):
            return False

    # Require at least some Gurmukhi or Devanagari characters —
    # pure English/ASCII output is not useful for shabad matching.
    punjabi_chars = 0
    for char in text:
        code = ord(char)
        if (0x0A00 <= code <= 0x0A7F) or (0x0900 <= code <= 0x097F):
            punjabi_chars += 1

    return punjabi_chars >= 2


class TranscriptionProcessor:
    """Cleans up, filters, and deduplicates transcription segments.

    Cross-window dedup is pluggable — strategies live in ``src/pipeline/dedup``
    and are selected by ``config.streaming.dedup_strategy``. The processor
    holds one instance per strategy and re-reads the config on every call so a
    dashboard toggle flip applies live.
    """

    def __init__(self) -> None:
        self._last_text: str = ""
        self._repeat_count: int = 0
        # Hold one stateful strategy per kind so flipping the toggle doesn't
        # blow away in-flight context. Switching from text → audio_time →
        # text re-uses the same TextDedup instance and resumes where it left
        # off, which keeps overlap-stripping continuous in the common case.
        self._strategies: dict[str, DedupStrategy] = {
            "text": TextDedup(),
            "audio_time": AudioTimeDedup(),
            "none": NoOpDedup(),
        }

    def _strategy(self) -> DedupStrategy:
        name = config.streaming.dedup_strategy
        return self._strategies.get(name, self._strategies["text"])

    def process(
        self,
        segments: list[TranscriptionSegment],
        audio_window_start: float | None = None,
    ) -> str:
        """Combine segments → single text. Filters garbage, dedups, detects repetition.

        ``audio_window_start`` is the absolute (monotonic) audio time at the
        start of this decode window. Required for ``audio_time`` dedup; ignored
        by ``text`` and ``none`` strategies. The orchestrator passes
        ``time.monotonic()`` (or equivalent) at the moment audio was snapshotted.
        """
        if not segments:
            return ""

        valid_segments = [seg for seg in segments if _is_valid_text(seg.text)]
        if not valid_segments:
            return ""

        combined = " ".join(seg.text for seg in valid_segments)

        # De-stutter within a single window — Whisper repeats words in singing.
        # This is intra-window cleanup, not cross-window dedup, so it always runs.
        combined = self._remove_repeated_words(combined)

        # Cross-window dedup — strategy chosen by config (live-readable).
        if audio_window_start is not None and valid_segments:
            seg_start = audio_window_start + min(s.start for s in valid_segments)
            seg_end = audio_window_start + max(s.end for s in valid_segments)
        else:
            seg_start = None
            seg_end = None
        combined = self._strategy().dedup(combined, seg_start, seg_end)

        # Hallucination guard — orthogonal to dedup. If the SAME text comes
        # through 3 consecutive windows, suppress it. This catches Whisper
        # locking onto a phrase and looping it, regardless of dedup mode.
        # Note: with audio_time dedup this stays correct because the strategy
        # has already preserved legit repetition; what reaches here as 3×
        # identical strings is still a hallucination signal.
        if combined and combined == self._last_text:
            self._repeat_count += 1
            if self._repeat_count >= 3:
                return ""
        else:
            self._repeat_count = 0

        if combined:
            self._last_text = combined
        return combined

    @staticmethod
    def _remove_repeated_words(text: str) -> str:
        """Remove consecutive duplicate words (e.g. 'ਕੋਇ ਕੋਇ ਕੋਇ' → 'ਕੋਇ')."""
        words = text.split()
        if not words:
            return text
        deduped = [words[0]]
        for word in words[1:]:
            if word != deduped[-1]:
                deduped.append(word)
        return " ".join(deduped)

    def reset(self) -> None:
        """Reset state for a new session."""
        self._last_text = ""
        self._repeat_count = 0
        for strategy in self._strategies.values():
            strategy.reset()
