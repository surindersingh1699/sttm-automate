"""Offline Gurbani search against the raw ShabadOS SQLite database.

The upstream `database.sqlite` stores:
  - `lines.gurmukhi` in AnvaadLipi ASCII font (needs gurmukhiutils.unicode → Unicode Gurmukhi)
  - `lines.first_letters` in ASCII (matches `transliterate.gurmukhi_to_ascii` output)
  - `translations.translation` keyed by `translation_source_id` (1 = Dr. Sant Singh Khalsa English)

Queries span all sources in the DB — Sri Guru Granth Sahib, Sri Dasam Granth,
Vaaran Bhai Gurdas, Bhai Nand Lal's banis, Sarabloh Granth, Rehitname, etc.
"""

import sqlite3
from functools import lru_cache
from pathlib import Path

from gurmukhiutils.unicode import unicode as _ascii_to_unicode_gurmukhi

from src.config import config
from src.matcher.search import ShabadCandidate, ShabadVerse
from src.transcription.transliterate import gurmukhi_to_ascii, normalize_first_letter

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENGLISH_TRANSLATION_SOURCE = 1  # Dr. Sant Singh Khalsa
_SOURCE_SGGS = 1  # shabads.source_id for Sri Guru Granth Sahib (used by sggs_only toggle)


def _scope_clause() -> str:
    """Return an ``AND`` clause restricting rows to SGGS when the toggle is on."""
    return f" AND s.source_id = {_SOURCE_SGGS}" if config.database.sggs_only else ""


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
# No source filter — matches across SGGS, Dasam Granth, Bhai Gurdas, Bhai Nand Lal, etc.
_LINE_SELECT = f"""
    SELECT
        l.gurmukhi       AS gurmukhi_ascii,
        l.first_letters  AS first_letters,
        l.order_id       AS order_id,
        l.source_page    AS source_page,
        s.sttm_id        AS sttm_id,
        s.source_id      AS source_id,
        (
            SELECT t.translation FROM translations t
            WHERE t.line_id = l.id AND t.translation_source_id = {_ENGLISH_TRANSLATION_SOURCE}
            LIMIT 1
        ) AS english
    FROM lines l
    JOIN shabads s ON l.shabad_id = s.id
    WHERE 1=1
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

        # Strategy 5: multi-line search — the window spans 2+ DB lines (nitnem / dense text).
        # Split the query in half and require consecutive line hits for both halves.
        if (
            config.matcher.multi_line_search
            and len(first_letters) >= config.matcher.multi_line_min_query_length
        ):
            ascii_fl_full = gurmukhi_to_ascii(first_letters)
            self._add_unique(
                self._multiline_search(ascii_fl_full, max_results, "multiline"),
                candidates,
                seen_ids,
            )

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
        """Fetch all verses of a shabad for line-level tracking.

        Skips the sggs_only scope — we always honor an explicit shabad_id lookup,
        so the user can still navigate a Dasam/Bhai Gurdas shabad that was manually
        selected even while SGGS-only is on for search.
        """
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
            + _scope_clause()
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
            + _scope_clause()
            + """
            AND l.first_letters LIKE '%' || ? || '%'
            GROUP BY s.sttm_id
            ORDER BY l.order_id
            LIMIT ?
            """,
            (ascii_query, limit),
        ).fetchall()
        return [self._row_to_candidate(r, signal) for r in rows]

    def _multiline_search(
        self, ascii_query: str, limit: int, signal: str
    ) -> list[ShabadCandidate]:
        """
        Split a long first-letter query in half and require BOTH halves to hit
        consecutive lines within the same shabad. Handles windows that span
        multiple DB lines (dense nitnem text, fast kirtan).
        """
        if len(ascii_query) < 8:
            return []

        # Try a few split positions since we don't know exact word boundaries
        # in the first-letter string.
        mid = len(ascii_query) // 2
        split_positions = {mid}
        if mid > 3:
            split_positions.add(mid - 1)
            split_positions.add(mid + 1)

        results: dict[int, sqlite3.Row] = {}
        for split_at in sorted(split_positions):
            head = ascii_query[:split_at]
            tail = ascii_query[split_at:]
            if len(head) < 3 or len(tail) < 3:
                continue

            # Self-join lines so each row pairs with its next-order sibling in
            # the same shabad. Match head against line-N, tail against line-N+1.
            scope = (
                f"AND s.source_id = {_SOURCE_SGGS}"
                if config.database.sggs_only
                else ""
            )
            rows = self._conn.execute(
                f"""
                SELECT
                    l1.gurmukhi       AS gurmukhi_ascii,
                    l1.first_letters  AS first_letters,
                    l1.order_id       AS order_id,
                    l1.source_page    AS source_page,
                    s.sttm_id         AS sttm_id,
                    s.source_id       AS source_id,
                    (
                        SELECT t.translation FROM translations t
                        WHERE t.line_id = l1.id AND t.translation_source_id = {_ENGLISH_TRANSLATION_SOURCE}
                        LIMIT 1
                    ) AS english
                FROM lines l1
                JOIN shabads s ON l1.shabad_id = s.id
                JOIN lines l2
                    ON l2.shabad_id = l1.shabad_id
                    AND l2.order_id = l1.order_id + 1
                WHERE l1.first_letters LIKE ? || '%'
                  AND l2.first_letters LIKE ? || '%'
                  {scope}
                GROUP BY s.sttm_id
                ORDER BY l1.order_id
                LIMIT ?
                """,
                (head, tail, limit),
            ).fetchall()

            for row in rows:
                # Keep the first hit per shabad (from the earliest split_at tried).
                results.setdefault(row["sttm_id"], row)
            if len(results) >= limit:
                break

        return [self._row_to_candidate(r, signal) for r in list(results.values())[:limit]]

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
            + _scope_clause()
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
        # Map DB source_id (1..12) → scorer code: SGGS="G", everything else="D" (treated
        # as secondary by the scorer). Older rows without source_id default to "G".
        try:
            db_source = row["source_id"]
        except (IndexError, KeyError):
            db_source = _SOURCE_SGGS
        source_code = "G" if db_source == _SOURCE_SGGS else "D"
        return ShabadCandidate(
            shabad_id=row["sttm_id"],
            gurmukhi=ascii_g,
            unicode=_to_unicode(ascii_g),
            english=row["english"] or "",
            source_id=source_code,
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
