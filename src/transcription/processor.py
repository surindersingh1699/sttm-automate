"""Post-processing for transcription output: cleanup and deduplication."""

from src.transcription.engine import TranscriptionSegment


class TranscriptionProcessor:
    """Cleans up and deduplicates transcription segments from overlapping windows."""

    def __init__(self):
        self._last_text: str = ""

    def process(self, segments: list[TranscriptionSegment]) -> str:
        """
        Combine segments into a single text string.
        Deduplicate overlap with previous window's output.
        """
        if not segments:
            return ""

        combined = " ".join(seg.text for seg in segments)

        # Remove repeated text from overlap with previous window
        if self._last_text and combined.startswith(self._last_text[:20]):
            # Find where the new content starts
            overlap_len = min(len(self._last_text), len(combined) // 2)
            for i in range(overlap_len, 0, -1):
                if combined.startswith(self._last_text[-i:]):
                    combined = combined[i:].strip()
                    break

        self._last_text = combined
        return combined

    def reset(self):
        """Reset state for a new session."""
        self._last_text = ""
