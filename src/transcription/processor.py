"""Post-processing for transcription output: cleanup, filtering, and deduplication."""

import re

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

    # Check that at least some characters are from valid scripts
    valid_chars = 0
    for char in text:
        code = ord(char)
        for start, end in _VALID_RANGES:
            if start <= code <= end:
                valid_chars += 1
                break

    return valid_chars >= 2


class TranscriptionProcessor:
    """Cleans up, filters, and deduplicates transcription segments."""

    def __init__(self):
        self._last_text: str = ""
        self._repeat_count: int = 0

    def process(self, segments: list[TranscriptionSegment]) -> str:
        """
        Combine segments into a single text string.
        Filters garbage, deduplicates overlap, and detects repetition.
        """
        if not segments:
            return ""

        # Filter out garbage segments
        valid_segments = [seg for seg in segments if _is_valid_text(seg.text)]
        if not valid_segments:
            return ""

        combined = " ".join(seg.text for seg in valid_segments)

        # Remove repeated text from overlap with previous window
        if self._last_text and combined.startswith(self._last_text[:20]):
            overlap_len = min(len(self._last_text), len(combined) // 2)
            for i in range(overlap_len, 0, -1):
                if combined.startswith(self._last_text[-i:]):
                    combined = combined[i:].strip()
                    break

        # Detect exact repetition (Whisper hallucination)
        if combined == self._last_text:
            self._repeat_count += 1
            if self._repeat_count >= 3:
                return ""  # suppress repeated hallucinations
        else:
            self._repeat_count = 0

        self._last_text = combined
        return combined

    def reset(self):
        """Reset state for a new session."""
        self._last_text = ""
        self._repeat_count = 0
