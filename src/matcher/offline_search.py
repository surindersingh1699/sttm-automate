"""Offline Gurbani search against the raw ShabadOS SQLite database.

The upstream `database.sqlite` stores:
  - `lines.gurmukhi` in AnvaadLipi ASCII font (needs gurmukhiutils.unicode → Unicode Gurmukhi)
  - `lines.first_letters` in ASCII (matches `transliterate.gurmukhi_to_ascii` output)
  - `translations.translation` keyed by `translation_source_id` (1 = Dr. Sant Singh Khalsa English)

All queries are scoped to `shabads.source_id = 1` (Sri Guru Granth Sahib).
"""

import sqlite3
from functools import lru_cache
from pathlib import Path

from gurmukhiutils.unicode import unicode as _ascii_to_unicode_gurmukhi

from src.config import config
from src.matcher.search import ShabadCandidate, ShabadVerse
from src.transcription.transliterate import gurmukhi_to_ascii, normalize_first_letter

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SOURCE_SGGS = 1
_ENGLISH_TRANSLATION_SOURCE = 1  # Dr. Sant Singh Khalsa


def _resolve_db_path() -> Path:
    """Return the local ShabadOS DB path, auto-downloading from HF if missing."""
    local = _PROJECT_ROOT / config.database.local_filename
    if local.exists():
        return local

    print(
        f"[DB] '{local.name}' not found locally. "
        f"Downloading from HF dataset '{config.database.hf_dataset_id}'..."
    )
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=config.database.hf_dataset_id,
        filename=config.database.hf_filename,
        repo_type="dataset",
    )
    print(f"[DB] Cached at {cached}")
    return Path(cached)

# Template for the common line+shabad+english projection.
_LINE_SELECT = f"""
    SELECT
        l.gurmukhi       AS gurmukhi_ascii,
        l.first_letters  AS first_letters,
        l.order_id       AS order_id,
        l.source_page    AS source_page,
        s.sttm_id        AS sttm_id,
        (
            SELECT t.translation FROM translations t
            WHERE t.line_id = l.id AND t.translation_source_id = {_ENGLISH_TRANSLATION_SOURCE}
            LIMIT 1
        ) AS english
    FROM lines l
    JOIN shabads s ON l.shabad_id = s.id
    WHERE s.source_id = {_SOURCE_SGGS}
"""


@lru_cache(maxsize=131072)
def _to_unicode(ascii_gurmukhi: str) -> str:
    """AnvaadLipi ASCII → Unicode Gurmukhi (cached — 60k lines, stable)."""
    if not ascii_gurmukhi:
        return ""
    try:
        return _ascii_to_unicode_gurmukhi(ascii_gurmukhi)
    except Exception:
        return ascii_gurmukhi


def _extract_verse_first_letters(unicode_text: str) -> str:
    """Extract first Gurmukhi letter of each word from a verse's Unicode text."""
    letters = []
    for word in unicode_text.split():
        if word and "\u0A00" <= word[0] <= "\u0A7F":
            letters.append(normalize_first_letter(word[0]))
    return "".join(letters)


class OfflineShabadSearcher:
    """Searches the ShabadOS SQLite DB using ASCII first-letter indices."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = str(db_path) if db_path is not None else str(_resolve_db_path())
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA cache_size=-16000")  # 16MB cache

    def search(
        self,
        first_letters: str,
        max_results: int = 10,
        start_mode: bool = False,
        transcript_text: str = "",
    ) -> list[ShabadCandidate]:
        """Search using first-letter codes with multiple strategies."""
        if len(first_letters) < 3 and not transcript_text.strip():
            return []

        candidates: list[ShabadCandidate] = []
        seen_ids: set[int] = set()

        if len(first_letters) >= 3:
            ascii_fl = gurmukhi_to_ascii(first_letters)
            query = ascii_fl[: min(8, len(ascii_fl))] if start_mode else ascii_fl

            # Strategy 1: prefix match (BaniDB searchtype=0 equivalent)
            self._add_unique(self._prefix_search(query, max_results, "type0"), candidates, seen_ids)

            # Strategy 2: contains match (searchtype=1 equivalent)
            if len(candidates) < 3 and (not start_mode or len(candidates) == 0):
                self._add_unique(self._contains_search(query, max_results, "type1"), candidates, seen_ids)

            # Strategy 3: shorter substring fallback
            if len(candidates) < 2 and len(query) > 4:
                for sub in (query[:4], query[-4:]):
                    self._add_unique(self._prefix_search(sub, 5, "type0_sub"), candidates, seen_ids)

        # Strategy 4: full-word phrase search from transcript text
        if transcript_text.strip():
            self._add_unique(self._fullword_search(transcript_text, max_results, "type2"), candidates, seen_ids)

        return candidates

    def search_by_id(self, shabad_id: int) -> ShabadCandidate | None:
        """Fetch the first line of a specific shabad by its sttm_id."""
        row = self._conn.execute(
            _LINE_SELECT + " AND s.sttm_id = ? ORDER BY l.order_id LIMIT 1",
            (shabad_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_candidate(row, signal="id")

    def fetch_all_verses(self, shabad_id: int) -> list[ShabadVerse]:
        """Fetch all verses of a shabad for line-level tracking."""
        rows = self._conn.execute(
            _LINE_SELECT + " AND s.sttm_id = ? ORDER BY l.order_id",
            (shabad_id,),
        ).fetchall()

        verses: list[ShabadVerse] = []
        for row in rows:
            unicode_text = _to_unicode(row["gurmukhi_ascii"] or "")
            verses.append(
                ShabadVerse(
                    verse_id=0,
                    unicode=unicode_text,
                    gurmukhi=row["gurmukhi_ascii"] or "",
                    english=row["english"] or "",
                    first_letters=_extract_verse_first_letters(unicode_text),
                )
            )
        return verses

    def _prefix_search(self, ascii_query: str, limit: int, signal: str) -> list[ShabadCandidate]:
        """ASCII first-letter prefix match."""
        rows = self._conn.execute(
            _LINE_SELECT
            + """
            AND l.first_letters LIKE ? || '%'
            GROUP BY s.sttm_id
            ORDER BY l.order_id
            LIMIT ?
            """,
            (ascii_query, limit),
        ).fetchall()
        return [self._row_to_candidate(r, signal) for r in rows]

    def _contains_search(self, ascii_query: str, limit: int, signal: str) -> list[ShabadCandidate]:
        """ASCII first-letter contains match (broader)."""
        rows = self._conn.execute(
            _LINE_SELECT
            + """
            AND l.first_letters LIKE '%' || ? || '%'
            GROUP BY s.sttm_id
            ORDER BY l.order_id
            LIMIT ?
            """,
            (ascii_query, limit),
        ).fetchall()
        return [self._row_to_candidate(r, signal) for r in rows]

    def _fullword_search(self, transcript_text: str, limit: int, signal: str) -> list[ShabadCandidate]:
        """Phrase search: match transcript words against Unicode Gurmukhi lines.

        Lines are stored as AnvaadLipi ASCII, so we search in Python after converting
        a small candidate pool first — but a cheap pre-filter on `first_letters`
        would lose valid matches. Instead we push the match down into SQL by
        relying on the transliteration round-trip: not feasible directly, so
        fall back to a coarse filter on first letters of the normalized words.
        """
        from src.transcription.transliterate import (
            normalize_for_fullword_search,
            extract_first_letters,
        )

        normalized = normalize_for_fullword_search(transcript_text)
        words = [w for w in normalized.split() if len(w) >= 2]
        if len(words) < 2:
            return []

        phrase_fls = extract_first_letters(" ".join(words[:6]))
        if len(phrase_fls) < 3:
            return []
        ascii_fls = gurmukhi_to_ascii(phrase_fls)

        rows = self._conn.execute(
            _LINE_SELECT
            + """
            AND l.first_letters LIKE '%' || ? || '%'
            GROUP BY s.sttm_id
            ORDER BY l.order_id
            LIMIT ?
            """,
            (ascii_fls, limit),
        ).fetchall()
        return [self._row_to_candidate(r, signal) for r in rows]

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row, signal: str) -> ShabadCandidate:
        """Convert a SQLite row to a ShabadCandidate (ASCII → Unicode Gurmukhi)."""
        ascii_g = row["gurmukhi_ascii"] or ""
        return ShabadCandidate(
            shabad_id=row["sttm_id"],
            gurmukhi=ascii_g,
            unicode=_to_unicode(ascii_g),
            english=row["english"] or "",
            source_id="G",
            page_no=row["source_page"],
            retrieval_sources={signal} if signal else set(),
        )

    @staticmethod
    def _add_unique(
        new: list[ShabadCandidate],
        existing: list[ShabadCandidate],
        seen: set[int],
    ) -> None:
        """Add candidates that haven't been seen yet; merge retrieval signals on duplicates."""
        by_id = {c.shabad_id: c for c in existing}
        for c in new:
            if c.shabad_id not in seen:
                seen.add(c.shabad_id)
                existing.append(c)
                by_id[c.shabad_id] = c
            else:
                existing_c = by_id.get(c.shabad_id)
                if existing_c is not None:
                    existing_c.retrieval_sources.update(c.retrieval_sources)
