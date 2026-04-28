"""Confidence scoring for shabad candidates."""

from difflib import SequenceMatcher
import re

from src.config import config
from src.matcher.search import ShabadCandidate
from src.transcription.transliterate import gurmukhi_to_ascii, normalize_first_letter


_TOKEN_SPLIT = re.compile(r"\s+")

# Gurmukhi matras (vowel signs) stripped before word-level comparison so that
# "ਨਾਮੁ" (u-matra) and "ਨਾਮਿ" (i-matra) both reduce to the same consonant stem.
_GURMUKHI_MATRAS = frozenset(
    "\u0A3E\u0A3F\u0A40\u0A41\u0A42\u0A47\u0A48\u0A4B\u0A4C"
    "\u0A70\u0A71\u0A02\u0A01\u0A4D\u0A3C"
)


def _strip_matras(text: str) -> str:
    return "".join(ch for ch in text if ch not in _GURMUKHI_MATRAS)


def _subsequence_coverage(query: str, target: str) -> float:
    """Fraction of target letters that appear as a subsequence of query.

    `len(LCS(query, target)) / len(target)`. Answers: "does the (clean) target
    show up inside the (possibly noisy) query in order?" Insensitive to how much
    filler sits between target letters in the query — exactly the property we
    need for scattered-word transcripts where the real shabad letters are there
    but surrounded by Whisper filler.
    """
    if not query or not target:
        return 0.0
    n = len(target)
    # Classic LCS length via two-row DP. Query on outer axis, target on inner —
    # O(|query| * |target|) time, O(|target|) space. Both strings are ≤ a few
    # hundred chars in this pipeline so this is negligible (~10µs per call).
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for qch in query:
        for j, tch in enumerate(target):
            if qch == tch:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev, curr = curr, prev
        for i in range(n + 1):
            curr[i] = 0
    return prev[n] / n


def _dense_substring_coverage(query: str, shabad_concat: str | None) -> float:
    """Longest-common-substring coverage against a shabad's full first-letters.

    When one audio window spans 2–3 consecutive DB lines (fast/dense recitation),
    the query's first-letters are a near-contiguous substring of the concatenation
    of every line in the shabad. `longest_common_substring(query, concat) / |query|`
    lands near 1.0 on correct multi-line windows and near 0 on unrelated ones.
    Returns 0.0 when the candidate doesn't carry a shabad concat (older callers
    or candidates that were never hydrated via the searcher).
    """
    if not query or not shabad_concat:
        return 0.0
    # Both inputs may be Gurmukhi-Unicode (from live transcription) OR ASCII
    # (the DB's `first_letters` column stores ASCII). Normalize both to ASCII so
    # comparison works regardless of source.
    q_ascii = gurmukhi_to_ascii(query) if query and query[0] > "\x7f" else query
    c_ascii = (
        gurmukhi_to_ascii(shabad_concat)
        if shabad_concat and shabad_concat[0] > "\x7f"
        else shabad_concat
    )
    if not q_ascii or not c_ascii:
        return 0.0
    match = SequenceMatcher(None, q_ascii, c_ascii).find_longest_match(
        0, len(q_ascii), 0, len(c_ascii)
    )
    return match.size / max(len(q_ascii), 1)


def _char_4grams(text: str) -> set[str]:
    s = (text or "").strip()
    if len(s) < 4:
        return {s} if s else set()
    return {s[i: i + 4] for i in range(len(s) - 3)}


def _ngram_overlap(a: set[str], b: set[str]) -> float:
    """Overlap coefficient |a∩b| / min(|a|,|b|). Robust when query >> verse."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


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
        """Score a candidate against the query. Returns 0.0 to 1.0."""
        return self.score_detailed(query_letters, candidate, current_shabad_id)["score"]

    def score_detailed(
        self,
        query_letters: str,
        candidate: ShabadCandidate,
        current_shabad_id: int | None = None,
    ) -> dict:
        """Full score breakdown so callers can gate decisions on which signal won.

        Returns a dict with keys: score, letter_ratio, coverage, dense_coverage,
        consec_ratio, dense_dominant. `dense_dominant=True` means the whole-shabad
        substring match beat both the single-line ratio AND the subsequence
        coverage — a sign that the hit may be spurious (dense substring inside
        an unrelated shabad) and should require stricter corroboration before
        triggering an instant switch or auto-lock.
        """
        # Extract first Gurmukhi letter of each word from the unicode field
        candidate_letters = "".join(
            normalize_first_letter(w[0]) for w in candidate.unicode.split()
            if w and "\u0A00" <= w[0] <= "\u0A7F"
        )

        if not query_letters or not candidate_letters:
            return {
                "score": 0.0,
                "letter_ratio": 0.0,
                "coverage": 0.0,
                "dense_coverage": 0.0,
                "consec_ratio": 0.0,
                "dense_dominant": False,
            }

        # 1. Overall sequence similarity (no .lower() needed — Gurmukhi has no case)
        letter_ratio = SequenceMatcher(
            None, query_letters, candidate_letters
        ).ratio()

        # 1b. Subsequence coverage: does the target appear as a subsequence inside
        # the (possibly noisy) query? Catches the scattered-words case where real
        # shabad letters are sprinkled among Whisper filler.
        coverage = _subsequence_coverage(query_letters, candidate_letters)

        # 1c. Dense-window coverage (Fix 3): query as substring of the shabad's
        # full concatenated first-letters. Powerful for multi-line windows but
        # prone to spurious matches against unrelated shabads that happen to
        # contain similar short sequences, so callers should treat a pure-dense
        # win skeptically.
        dense_coverage = _dense_substring_coverage(query_letters, candidate.full_first_letters)

        letter_term = max(letter_ratio, coverage, dense_coverage)
        dense_dominant = (
            dense_coverage > max(letter_ratio, coverage) + 1e-6
            and dense_coverage >= config.matcher.dense_dominant_margin
        )

        # 2. Consecutive match bonus
        consec = self._longest_consecutive_match(query_letters, candidate_letters)
        consec_ratio = consec / max(len(query_letters), 1)

        # 3. Context: boost if different from current (we're looking for matches)
        context_score = 0.5  # neutral by default
        if current_shabad_id is not None and candidate.shabad_id == current_shabad_id:
            context_score = 1.0

        # 4. Source priority (G = SGGS, highest priority)
        source_score = 1.0 if candidate.source_id == "G" else 0.5
        score = (
            config.matcher.weight_letter_match * letter_term
            + config.matcher.weight_consecutive * consec_ratio
            + config.matcher.weight_context * context_score
            + config.matcher.weight_source * source_score
        )
        return {
            "score": score,
            "letter_ratio": letter_ratio,
            "coverage": coverage,
            "dense_coverage": dense_coverage,
            "consec_ratio": consec_ratio,
            "dense_dominant": dense_dominant,
        }

    def score_line(self, query_letters: str, line_first_letters: str) -> float:
        """
        Score how well query matches a single verse line.

        Used in LOCKED state for line alignment within a known shabad. Two paths:
          (a) existing letter-ratio + consecutive-match bonus — dominates for clean,
              same-length queries.
          (b) subsequence-coverage + consecutive bonus — dominates when the query
              contains the line's letters as a subsequence buried in noise (Fix 2).
        Score is the max of the two — no regression on clean queries, big lift on
        scattered ones.
        """
        if not query_letters or not line_first_letters:
            return 0.0

        letter_ratio = SequenceMatcher(
            None, query_letters, line_first_letters
        ).ratio()

        consec = self._longest_consecutive_match(query_letters, line_first_letters)
        consec_ratio = consec / max(len(query_letters), 1)

        coverage = _subsequence_coverage(query_letters, line_first_letters)

        baseline = 0.5 * letter_ratio + 0.5 * consec_ratio
        coverage_path = config.matcher.dense_coverage_weight * coverage + (
            1.0 - config.matcher.dense_coverage_weight
        ) * consec_ratio
        return max(baseline, coverage_path)

    def score_line_ngram(self, transcript_text: str, verse_unicode: str) -> float:
        """Char-4-gram overlap coefficient between transcript and a verse's Unicode text.

        Normalizes the transcript to Gurmukhi (handles Devanagari / mixed script)
        then computes overlap-coefficient on 4-grams. Verbatim match → 1.0, partial
        or noisy → graceful decay. Complement to score_line: FL is sharp on correct
        first-letters; ngram is robust when matras or extra syllables differ.
        """
        if not transcript_text or not verse_unicode:
            return 0.0
        from src.transcription.transliterate import normalize_for_fullword_search
        normalized = normalize_for_fullword_search(transcript_text)
        if not normalized:
            return 0.0
        return _ngram_overlap(_char_4grams(normalized), _char_4grams(verse_unicode))

    def score_line_word_overlap(self, transcript_text: str, verse_unicode: str) -> float:
        """Word-set overlap: fraction of normalized transcript words found in the verse.

        Used by the word_match_line_scoring toggle — picks the line that contains
        the most transcript words regardless of order. Matras are stripped so
        "ਨਾਮੁ" and "ਨਾਮਿ" both reduce to the same consonant stem. Returns |q ∩ v| / |q|.
        """
        if not transcript_text or not verse_unicode:
            return 0.0
        from src.transcription.transliterate import normalize_for_fullword_search
        q_norm = _strip_matras(normalize_for_fullword_search(transcript_text))
        v_norm = _strip_matras(normalize_for_fullword_search(verse_unicode))
        if not q_norm or not v_norm:
            return 0.0
        q_words = set(q_norm.split())
        v_words = set(v_norm.split())
        if not q_words:
            return 0.0
        return len(q_words & v_words) / len(q_words)

    def score_line_word_match(self, transcript_text: str, verse_unicode: str) -> float:
        """Sequential normalized word match for short (2-word) queries within the locked shabad.

        Normalizes both sides then checks how many query words appear in order as a
        subsequence of the verse's word list. Returns matched / total so 2/2 → 1.0,
        1/2 → 0.5, 0/2 → 0.0. Used only for line pointer positioning, never for
        shabad retrieval or switching.
        """
        if not transcript_text or not verse_unicode:
            return 0.0
        from src.transcription.transliterate import normalize_for_fullword_search
        q_norm = _strip_matras(normalize_for_fullword_search(transcript_text))
        v_norm = _strip_matras(normalize_for_fullword_search(verse_unicode))
        if not q_norm or not v_norm:
            return 0.0
        q_words = q_norm.split()
        v_words = v_norm.split()
        if not q_words:
            return 0.0
        qi = 0
        for vw in v_words:
            if qi < len(q_words) and vw == q_words[qi]:
                qi += 1
        return qi / len(q_words)

    def score_line_sw(self, transcript_text: str, verse_unicode: str) -> float:
        """Word-level Smith-Waterman local alignment score.

        match=2, mismatch=-1, gap=-1. Normalized by 2 * len(query_words) and
        clamped to [0, 1]. Used in locked-state line scoring only (never retrieval).
        """
        if not transcript_text or not verse_unicode:
            return 0.0
        from src.transcription.transliterate import normalize_for_fullword_search
        q_norm = _strip_matras(normalize_for_fullword_search(transcript_text))
        v_norm = _strip_matras(normalize_for_fullword_search(verse_unicode))
        if not q_norm or not v_norm:
            return 0.0
        q_words = q_norm.split()
        v_words = v_norm.split()
        if not q_words:
            return 0.0
        MATCH, MISMATCH, GAP = 2, -1, -1
        m, n = len(q_words), len(v_words)
        H = [[0] * (n + 1) for _ in range(m + 1)]
        best = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                s = MATCH if q_words[i - 1] == v_words[j - 1] else MISMATCH
                H[i][j] = max(0, H[i - 1][j - 1] + s, H[i - 1][j] + GAP, H[i][j - 1] + GAP)
                if H[i][j] > best:
                    best = H[i][j]
        return min(1.0, max(0.0, best / (MATCH * m)))

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
