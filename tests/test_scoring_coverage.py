"""Unit tests for the subsequence-coverage and dense-substring-coverage scorer helpers.

These exercise Fix 2 (scattered-word scoring) and Fix 3 (dense multi-line windows)
without touching the SQLite DB — pure scoring logic on synthetic first-letters
strings so we can assert behaviour deterministically.
"""

from src.config import config
from src.matcher.scorer import (
    ConfidenceScorer,
    _dense_substring_coverage,
    _subsequence_coverage,
)
from src.matcher.search import ShabadCandidate


def _make_candidate(
    first_letter_target: str,
    full_first_letters: str | None = None,
    shabad_id: int = 1,
) -> ShabadCandidate:
    """Build a candidate whose `unicode` produces `first_letter_target` via score()."""
    # score() extracts one letter per word; join the target letters with spaces so
    # each char becomes its own "word". All letters are in the Gurmukhi block.
    unicode_text = " ".join(first_letter_target)
    return ShabadCandidate(
        shabad_id=shabad_id,
        gurmukhi=unicode_text,
        unicode=unicode_text,
        english="",
        source_id="G",
        page_no=0,
        full_first_letters=full_first_letters,
    )


# ---------- _subsequence_coverage ----------


def test_subsequence_coverage_full_subsequence():
    # Target letters all appear in order inside a noisy query.
    query = "ਬਅਅਬਲਤਗਬਅਸਦਛਬਦਜਗਤਮਬਬਬ"  # 20 chars, noisy
    target = "ਤਗਤਮ"  # 4 real letters, each present in order
    assert _subsequence_coverage(query, target) == 1.0


def test_subsequence_coverage_partial():
    query = "ਅਅਅਬਲ"
    target = "ਬਲਗ"
    # ਬ and ਲ appear in order; ਗ doesn't → 2/3.
    assert abs(_subsequence_coverage(query, target) - (2 / 3)) < 1e-9


def test_subsequence_coverage_empty_inputs_return_zero():
    assert _subsequence_coverage("", "abc") == 0.0
    assert _subsequence_coverage("abc", "") == 0.0


# ---------- _dense_substring_coverage ----------


def test_dense_substring_coverage_contiguous_match():
    query = "ssnhjslv"  # Japji Sahib mool line first-letters
    concat = "xyzsshnhjslvabc"  # target appears as contiguous substring
    # Longest common substring shared with query = "snhjslv" (7) or "sshnhjslv" overlap
    # Actual LCS: the substring "nhjslv" (6) is the longest exact match since concat
    # has "sshnhjslv" (the extra 'sh' breaks the full query). Score = 6/8 = 0.75.
    result = _dense_substring_coverage(query, concat)
    assert result >= 0.7


def test_dense_substring_coverage_no_concat_returns_zero():
    assert _dense_substring_coverage("abc", None) == 0.0
    assert _dense_substring_coverage("abc", "") == 0.0


# ---------- ConfidenceScorer.score_line — subsequence path ----------


def test_score_line_scattered_query_crosses_suggest_threshold():
    """The scattered-words symptom from the screenshot should score above suggest."""
    scorer = ConfidenceScorer()
    # Synthesized noisy query (20 chars) where the real target "ਤਗਤਮ" is present
    # scattered among filler. Baseline ratio would be low (~0.33) but coverage
    # path should dominate and push score above suggest_threshold.
    query = "ਬਅਅਬਲਤਗਬਅਸਦਛਬਦਜਗਤਮਬਬਬ"
    target = "ਤਗਤਮ"
    score = scorer.score_line(query, target)
    assert score >= config.matcher.suggest_threshold, (
        f"scattered-query score {score:.3f} < suggest_threshold "
        f"{config.matcher.suggest_threshold}"
    )


def test_score_line_clean_query_regression():
    """Clean Japji Sahib first letters must still score high on the right line."""
    scorer = ConfidenceScorer()
    # Japji mool first-letters: ਸ ਸ ਨ ਹ ਜ ਸ ਲ ਵ (from memory entry)
    query = "ਸਸਨਹਜਸਲਵ"
    target = "ਸਸਨਹਜਸਲਵ"
    score = scorer.score_line(query, target)
    # Identical query & target → both paths score 1.0.
    assert score >= 0.95


# ---------- ConfidenceScorer.score — dense_coverage path ----------


def test_score_uses_dense_coverage_when_query_spans_multiple_lines():
    """Fast-recitation query that spans 3 DB lines should score near auto via dense_coverage."""
    scorer = ConfidenceScorer()
    # Imagine 3 DB lines with first-letters "ਅਬ", "ਕਲ", "ਮਨ". The shabad's
    # full_first_letters concat is "ਅਬ ਕਲ ਮਨ". A fast window that captures all
    # three produces the query "ਅਬਕਲਮਨ".
    query = "ਅਬਕਲਮਨ"
    full_concat = "ਅਬ ਕਲ ਮਨ"
    # The single-line target is just one of those lines — short compared to query.
    # Without dense_coverage, score would be very low (query 6 vs target 2).
    candidate = _make_candidate("ਅਬ", full_first_letters=full_concat)
    score = scorer.score(query, candidate)
    # Dense-coverage path: LCS length between "ਅਬਕਲਮਨ" and "ਅਬ ਕਲ ਮਨ" (ignoring
    # spaces) / len(query) should be high because most of the query sits inside
    # the concat. Baseline letter_ratio would be ~0.5 at best.
    assert score >= config.matcher.suggest_threshold


def test_score_without_full_first_letters_falls_back_to_baseline():
    """Candidates without cached concat must still score sensibly on single-line matches."""
    scorer = ConfidenceScorer()
    # Same target as single-line first-letters → baseline letter_ratio dominates.
    candidate = _make_candidate("ਅਬਕਲ", full_first_letters=None)
    score = scorer.score("ਅਬਕਲ", candidate)
    assert score >= 0.7  # source bonus + perfect letter match
