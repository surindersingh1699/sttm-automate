"""Integration tests for word-vote retrieval (Fix 2) against the real DB.

Skips if `database.sqlite` isn't present locally (first-run environments).
"""

from pathlib import Path

import pytest

from src.matcher.offline_search import OfflineShabadSearcher


_DB = Path(__file__).resolve().parent.parent / "database.sqlite"
pytestmark = pytest.mark.skipif(not _DB.exists(), reason="ShabadOS DB not present")


@pytest.fixture(scope="module")
def searcher():
    return OfflineShabadSearcher()


def test_word_vote_search_finds_japji_from_real_words(searcher):
    """Real Japji words sprinkled among filler should retrieve Japji."""
    transcript = "ਹੈ ਹਉ ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ ਤਾਂ ਜੀ ਅੱਛਾ ਨ"
    # Ensure the index is built.
    searcher._ensure_word_index()
    candidates = searcher._word_vote_search(transcript, limit=5, signal="type3_words")
    assert candidates, "no word-vote candidates returned"
    # Japji Sahib shabad_id = 1 in the ShabadOS mapping.
    assert any(c.shabad_id == 1 for c in candidates[:5]), (
        f"Japji (shabad_id=1) not in top 5; got: "
        f"{[(c.shabad_id, c.word_vote_hits, c.word_vote_score) for c in candidates]}"
    )
    japji = next(c for c in candidates if c.shabad_id == 1)
    assert japji.word_vote_hits >= 2
    assert japji.word_vote_score is not None and japji.word_vote_score > 0


def test_word_vote_search_stop_words_produce_weak_candidates(searcher):
    """Pure stop-word transcripts may return some candidates but none strongly.

    The IDF cutoff trims truly-common words; residual near-threshold words can
    still accumulate a small score. The real safety net is the word-vote-only
    floor in `_score_candidates` which demotes such candidates out of auto-lock —
    here we just confirm that a stop-word-only transcript never produces a
    candidate whose raw word_vote_score dominates real-signal runs.
    """
    searcher._ensure_word_index()
    stop_only = searcher._word_vote_search(
        "ਹੈ ਕਉ ਨ ਜੀ ਮੈ ਹੋ", limit=5, signal="type3_words"
    )
    real_signal = searcher._word_vote_search(
        "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ",
        limit=5,
        signal="type3_words",
    )
    assert real_signal, "real-signal retrieval returned nothing (unexpected)"
    if stop_only:
        # Any stop-word-only candidate should score well below a real-signal candidate.
        top_stop = max(c.word_vote_score or 0.0 for c in stop_only)
        top_real = max(c.word_vote_score or 0.0 for c in real_signal)
        assert top_real > top_stop, (
            f"stop-word candidate (score={top_stop:.2f}) outranked real signal "
            f"(score={top_real:.2f})"
        )


def test_word_vote_search_skips_when_too_few_tokens(searcher):
    """A single-token transcript should not trigger retrieval."""
    searcher._ensure_word_index()
    candidates = searcher._word_vote_search("ਸੋਚੈ", limit=5, signal="type3_words")
    assert not candidates


def test_search_pipeline_populates_full_first_letters(searcher):
    """Candidates from search() must carry full_first_letters for dense_coverage scoring."""
    searcher._ensure_word_index()
    # "ssnhjslv" is Japji's mool first-letters per the memory entry.
    candidates = searcher.search("ਸਸਨਹਜਸਲਵ", max_results=5)
    assert candidates
    # After search() fills the cache, at least the first result should carry the concat.
    assert candidates[0].full_first_letters is not None
    assert len(candidates[0].full_first_letters) > 0
