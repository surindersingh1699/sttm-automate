"""Regression tests for unified span retrieval."""

from pathlib import Path

import pytest

from src.matcher.offline_search import OfflineShabadSearcher


_DB = Path(__file__).resolve().parent.parent / "database.sqlite"
pytestmark = pytest.mark.skipif(not _DB.exists(), reason="ShabadOS DB not present")


def test_span_search_returns_exact_pointer_span_for_dense_japji_window():
    searcher = OfflineShabadSearcher()

    candidates = searcher.search(
        "ਸਸਨਹਜਸਲਵ",
        max_results=5,
        transcript_text="ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ",
    )

    assert candidates
    top = candidates[0]
    assert top.shabad_id == 1
    assert "span" in top.retrieval_sources
    assert top.span_len > 1
    assert top.line_idx > 0
    assert top.span_score is not None and top.span_score >= 0.9
