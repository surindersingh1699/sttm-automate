"""Confidence scoring for shabad candidates."""

from difflib import SequenceMatcher
import re

from src.config import config
from src.matcher.search import ShabadCandidate


_TOKEN_SPLIT = re.compile(r"\s+")


class ConfidenceScorer:
    """
    Scores how well a candidate shabad matches the transcription.

    Uses weighted combination of:
    - First-letter match ratio (how many letters match)
    - Consecutive match bonus (longer consecutive matches score higher)
    - Context match (same source/raag as current shabad)
    - Source priority (SGGS preferred over other sources)
    """

    def score(
        self,
        query_letters: str,
        candidate: ShabadCandidate,
        current_shabad_id: int | None = None,
    ) -> float:
        """
        Score a candidate against the query. Returns 0.0 to 1.0.
        """
        # Extract first Gurmukhi letter of each word from the unicode field
        candidate_letters = "".join(
            w[0] for w in candidate.unicode.split()
            if w and "\u0A00" <= w[0] <= "\u0A7F"
        )

        if not query_letters or not candidate_letters:
            return 0.0

        # 1. Overall sequence similarity (no .lower() needed — Gurmukhi has no case)
        letter_ratio = SequenceMatcher(
            None, query_letters, candidate_letters
        ).ratio()

        # 2. Consecutive match bonus
        consec = self._longest_consecutive_match(query_letters, candidate_letters)
        consec_ratio = consec / max(len(query_letters), 1)

        # 3. Context: boost if different from current (we're looking for matches)
        context_score = 0.5  # neutral by default
        if current_shabad_id is not None:
            if candidate.shabad_id == current_shabad_id:
                context_score = 1.0  # strong boost for current shabad

        # 4. Source priority (G = SGGS, highest priority)
        source_score = 1.0 if candidate.source_id == "G" else 0.5
        return (
            config.matcher.weight_letter_match * letter_ratio
            + config.matcher.weight_consecutive * consec_ratio
            + config.matcher.weight_context * context_score
            + config.matcher.weight_source * source_score
        )

    def score_line(self, query_letters: str, line_first_letters: str) -> float:
        """
        Score how well query matches a single verse line.

        Simpler than full score() — just letter similarity + consecutive bonus.
        Used in LOCKED state for line alignment within a known shabad.
        """
        if not query_letters or not line_first_letters:
            return 0.0

        letter_ratio = SequenceMatcher(
            None, query_letters, line_first_letters
        ).ratio()

        consec = self._longest_consecutive_match(query_letters, line_first_letters)
        consec_ratio = consec / max(len(query_letters), 1)

        # Equal weight: overall similarity + consecutive match bonus
        return 0.5 * letter_ratio + 0.5 * consec_ratio

    def classify(self, score: float) -> str:
        """Classify score into action: 'auto', 'suggest', or 'ignore'."""
        if score >= config.matcher.auto_threshold:
            return "auto"
        elif score >= config.matcher.suggest_threshold:
            return "suggest"
        return "ignore"

    def word_overlap_count(self, transcript_text: str, candidate_text: str) -> int:
        """Count overlapping normalized Punjabi words between transcript and candidate."""
        if not transcript_text or not candidate_text:
            return 0
        transcript_words = set(self._normalize_words(transcript_text))
        candidate_words = set(self._normalize_words(candidate_text))
        if not transcript_words or not candidate_words:
            return 0
        return len(transcript_words & candidate_words)

    def _longest_consecutive_match(self, a: str, b: str) -> int:
        """Find the longest consecutive matching substring length."""
        match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
        return match.size

    def _normalize_words(self, text: str) -> list[str]:
        """
        Normalize Devanagari/Gurmukhi mixed text into Gurmukhi-ish tokens and split words.
        Keeps only Punjabi script letters and spaces for robust overlap checks.
        """
        normalized_chars: list[str] = []
        for char in text:
            cp = ord(char)
            # Convert Devanagari block to Gurmukhi via Unicode offset.
            if 0x0900 <= cp <= 0x097F:
                mapped = cp + 0x0100
                if 0x0A00 <= mapped <= 0x0A7F:
                    normalized_chars.append(chr(mapped))
                else:
                    normalized_chars.append(" ")
                continue
            # Keep Gurmukhi chars.
            if 0x0A00 <= cp <= 0x0A7F:
                normalized_chars.append(char)
                continue
            # Treat everything else as separator.
            normalized_chars.append(" ")

        cleaned = "".join(normalized_chars)
        tokens = [token.strip() for token in _TOKEN_SPLIT.split(cleaned) if token.strip()]
        # Ignore tiny single-character tokens to reduce accidental overlap.
        return [token for token in tokens if len(token) >= 2]
