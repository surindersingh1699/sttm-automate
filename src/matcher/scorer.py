"""Confidence scoring for shabad candidates."""

from difflib import SequenceMatcher

from src.config import config
from src.matcher.search import ShabadCandidate


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

    def classify(self, score: float) -> str:
        """Classify score into action: 'auto', 'suggest', or 'ignore'."""
        if score >= config.matcher.auto_threshold:
            return "auto"
        elif score >= config.matcher.suggest_threshold:
            return "suggest"
        return "ignore"

    def _longest_consecutive_match(self, a: str, b: str) -> int:
        """Find the longest consecutive matching substring length."""
        match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
        return match.size
