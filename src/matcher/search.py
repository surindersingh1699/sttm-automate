"""Shabad candidate / verse dataclasses shared across the matcher.

This module used to also house an online BaniDB HTTP searcher — that path has
been removed. All search goes through `src.matcher.offline_search` against the
local ShabadOS SQLite DB. **No external APIs are called anywhere in this
codebase.**
"""

from dataclasses import dataclass, field


@dataclass
class ShabadVerse:
    """A single verse/line within a shabad, with pre-extracted first letters."""
    verse_id: int
    unicode: str
    gurmukhi: str
    english: str
    first_letters: str  # pre-extracted Gurmukhi first letters for scoring


@dataclass
class ShabadCandidate:
    shabad_id: int
    gurmukhi: str
    unicode: str
    english: str
    source_id: str
    page_no: int
    retrieval_sources: set[str] = field(default_factory=set)
    # Concatenated first-letters of every line in the shabad (space-separated per line).
    # Used by the scorer's dense_coverage term to handle fast/multi-line windows where
    # one query spans 2+ DB lines. None until populated by the searcher.
    full_first_letters: str | None = None
    # How strongly word-level retrieval voted for this candidate (IDF-weighted sum of
    # distinct transcript words that hit this shabad). None when type3_words didn't fire.
    word_vote_score: float | None = None
    # Count of distinct transcript words that hit this shabad via type3_words retrieval.
    word_vote_hits: int = 0
    # Best line/span hit inside the shabad. Open-corpus search and ordered bani
    # matching both operate on spans now; callers can land the pointer directly
    # instead of re-guessing the line from shabad-level evidence.
    line_idx: int = 0
    span_len: int = 1
    span_score: float | None = None
