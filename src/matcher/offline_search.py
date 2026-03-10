"""Offline Gurbani search using local SQLite database (Shabad OS).

Replaces the BaniDB REST API with a local 20MB SQLite database containing
all 60,555 SGGS lines with pre-computed first-letter indices.

Search is ~100x faster (<25ms vs 1-2s) and requires no internet.
"""

import sqlite3
from pathlib import Path

from src.matcher.search import ShabadCandidate, ShabadVerse
from src.transcription.transliterate import gurmukhi_to_ascii, normalize_first_letter

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "gurbani.sqlite"


def _extract_verse_first_letters(unicode_text: str) -> str:
    """Extract first Gurmukhi letter of each word from a verse's unicode field."""
    letters = []
    for word in unicode_text.split():
        if word and "\u0A00" <= word[0] <= "\u0A7F":
            letters.append(normalize_first_letter(word[0]))
    return "".join(letters)


class OfflineShabadSearcher:
    """Searches local SQLite Gurbani database using first-letter indices."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = str(db_path or _DB_PATH)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA cache_size=-8000")  # 8MB cache

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
            query = ascii_fl
            if start_mode:
                query = ascii_fl[: min(8, len(ascii_fl))]

            # Strategy 1: Prefix match (equivalent to BaniDB searchtype=0)
            results = self._prefix_search(query, max_results, "type0")
            self._add_unique(results, candidates, seen_ids)

            # Strategy 2: Contains match (broader, like BaniDB searchtype=1)
            if len(candidates) < 3 and (not start_mode or len(candidates) == 0):
                results = self._contains_search(query, max_results, "type1")
                self._add_unique(results, candidates, seen_ids)

            # Strategy 3: Shorter substring fallback
            if len(candidates) < 2 and len(query) > 4:
                for sub in [query[:4], query[-4:]]:
                    results = self._prefix_search(sub, 5, "type0_sub")
                    self._add_unique(results, candidates, seen_ids)

        # Strategy 4: Full-word phrase search from transcript text
        if transcript_text.strip():
            results = self._fullword_search(transcript_text, max_results, "type2")
            self._add_unique(results, candidates, seen_ids)

        return candidates

    def search_by_id(self, shabad_id: int) -> ShabadCandidate | None:
        """Fetch a specific shabad by its STTM ID."""
        row = self._conn.execute("""
            SELECT l.gurmukhi, l.unicode, l.english, s.sttm_id, l.source_page
            FROM lines l JOIN shabads s ON l.shabad_id = s.id
            WHERE s.sttm_id = ?
            ORDER BY l.order_id LIMIT 1
        """, (shabad_id,)).fetchone()

        if not row:
            return None

        return ShabadCandidate(
            shabad_id=row["sttm_id"],
            gurmukhi=row["gurmukhi"] or "",
            unicode=row["unicode"] or "",
            english=row["english"] or "",
            source_id="G",
            page_no=row["source_page"],
            retrieval_sources={"id"},
        )

    def fetch_all_verses(self, shabad_id: int) -> list[ShabadVerse]:
        """Fetch all verses of a shabad for line-level tracking."""
        rows = self._conn.execute("""
            SELECT l.id, l.unicode, l.gurmukhi, l.english
            FROM lines l JOIN shabads s ON l.shabad_id = s.id
            WHERE s.sttm_id = ?
            ORDER BY l.order_id
        """, (shabad_id,)).fetchall()

        verses = []
        for row in rows:
            unicode_text = row["unicode"] or ""
            fl = _extract_verse_first_letters(unicode_text)
            verses.append(ShabadVerse(
                verse_id=0,
                unicode=unicode_text,
                gurmukhi=row["gurmukhi"] or "",
                english=row["english"] or "",
                first_letters=fl,
            ))
        return verses

    def _prefix_search(
        self, ascii_query: str, limit: int, signal: str
    ) -> list[ShabadCandidate]:
        """First-letter prefix search (equivalent to BaniDB searchtype=0)."""
        rows = self._conn.execute("""
            SELECT l.gurmukhi, l.unicode, l.english, s.sttm_id, l.source_page
            FROM lines l JOIN shabads s ON l.shabad_id = s.id
            WHERE l.first_letters LIKE ? || '%'
            GROUP BY s.sttm_id
            ORDER BY l.order_id
            LIMIT ?
        """, (ascii_query, limit)).fetchall()
        return self._rows_to_candidates(rows, signal)

    def _contains_search(
        self, ascii_query: str, limit: int, signal: str
    ) -> list[ShabadCandidate]:
        """First-letter contains search (equivalent to BaniDB searchtype=1)."""
        rows = self._conn.execute("""
            SELECT l.gurmukhi, l.unicode, l.english, s.sttm_id, l.source_page
            FROM lines l JOIN shabads s ON l.shabad_id = s.id
            WHERE l.first_letters LIKE '%' || ? || '%'
            GROUP BY s.sttm_id
            ORDER BY l.order_id
            LIMIT ?
        """, (ascii_query, limit)).fetchall()
        return self._rows_to_candidates(rows, signal)

    def _fullword_search(
        self, transcript_text: str, limit: int, signal: str
    ) -> list[ShabadCandidate]:
        """Search by matching words from transcript against Gurmukhi/Unicode text."""
        from src.transcription.transliterate import normalize_for_fullword_search

        normalized = normalize_for_fullword_search(transcript_text)
        words = [w for w in normalized.split() if len(w) >= 2]
        if len(words) < 2:
            return []

        # Use the longest phrase (up to 6 words) for LIKE search
        phrase = " ".join(words[:6])
        rows = self._conn.execute("""
            SELECT l.gurmukhi, l.unicode, l.english, s.sttm_id, l.source_page
            FROM lines l JOIN shabads s ON l.shabad_id = s.id
            WHERE l.unicode LIKE '%' || ? || '%'
            GROUP BY s.sttm_id
            ORDER BY l.order_id
            LIMIT ?
        """, (phrase, limit)).fetchall()
        return self._rows_to_candidates(rows, signal)

    @staticmethod
    def _rows_to_candidates(
        rows: list[sqlite3.Row], signal: str
    ) -> list[ShabadCandidate]:
        """Convert SQLite rows to ShabadCandidate objects."""
        return [
            ShabadCandidate(
                shabad_id=row["sttm_id"],
                gurmukhi=row["gurmukhi"] or "",
                unicode=row["unicode"] or "",
                english=row["english"] or "",
                source_id="G",
                page_no=row["source_page"],
                retrieval_sources={signal} if signal else set(),
            )
            for row in rows
        ]

    def _add_unique(
        self,
        new: list[ShabadCandidate],
        existing: list[ShabadCandidate],
        seen: set[int],
    ) -> None:
        """Add candidates that haven't been seen yet."""
        by_id = {c.shabad_id: c for c in existing}
        for c in new:
            if c.shabad_id not in seen:
                seen.add(c.shabad_id)
                existing.append(c)
                by_id[c.shabad_id] = c
            else:
                current = by_id.get(c.shabad_id)
                if current is not None:
                    current.retrieval_sources.update(c.retrieval_sources)
