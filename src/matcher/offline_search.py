"""Offline Gurbani search against the raw ShabadOS SQLite database.

The upstream `database.sqlite` stores:
  - `lines.gurmukhi` in AnvaadLipi ASCII font (needs gurmukhiutils.unicode → Unicode Gurmukhi)
  - `lines.first_letters` in ASCII (matches `transliterate.gurmukhi_to_ascii` output)
  - `translations.translation` keyed by `translation_source_id` (1 = Dr. Sant Singh Khalsa English)

Queries span all sources in the DB — Sri Guru Granth Sahib, Sri Dasam Granth,
Vaaran Bhai Gurdas, Bhai Nand Lal's banis, Sarabloh Granth, Rehitname, etc.
"""

import math
import re
import sqlite3
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from gurmukhiutils.unicode import unicode as _ascii_to_unicode_gurmukhi

from src.config import config
from src.matcher.search import ShabadCandidate, ShabadVerse
from src.transcription.transliterate import gurmukhi_to_ascii, normalize_first_letter

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENGLISH_TRANSLATION_SOURCE = 1  # Dr. Sant Singh Khalsa
_SOURCE_SGGS = 1  # shabads.source_id for Sri Guru Granth Sahib (used by sggs_only toggle)

# `shabads.sttm_id` is NULL for Dasam Granth (5470 rows) and Uggardanti (9 rows).
# Without a synthetic fallback every such candidate collapses to shabad_id=None,
# so _add_unique dedup's them all into a single entry. We synthesize an ID from
# `order_id` (unique across the whole table) offset well past the real sttm_id
# range (max ~30k). Anything above SYNTHETIC_ID_OFFSET is a non-SGGS fallback.
SYNTHETIC_ID_OFFSET = 100_000_000
_SHABAD_ID_EXPR = f"COALESCE(s.sttm_id, s.order_id + {SYNTHETIC_ID_OFFSET})"


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
        {_SHABAD_ID_EXPR} AS sttm_id,
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


_GURMUKHI_TOKEN = re.compile(r"[\u0A00-\u0A7F]+")


def _tokenize_gurmukhi_words(text: str) -> list[str]:
    """Extract Gurmukhi word tokens (≥2 chars) from Unicode text.

    Shared by the DB-side word index and the live-transcript word-vote retrieval
    so both sides use identical tokenization. Intentionally simple — strip
    diacritics/matras from short tokens by length filter, not by character
    class, to avoid dropping real short words like ਨ ਹੈ ਕਉ (kept at len=2).
    """
    return [token for token in _GURMUKHI_TOKEN.findall(text) if len(token) >= 2]


class OfflineShabadSearcher:
    """Searches the ShabadOS SQLite DB using ASCII first-letter indices."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = str(db_path) if db_path is not None else str(_resolve_db_path())
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA cache_size=-16000")  # 16MB cache
        # Word-level retrieval (Fix 2) + shabad-concat scoring (Fix 3).
        # Lazy because the Unicode conversion of 60k lines takes a few seconds —
        # pay the cost the first time a caller needs it, not on every construction
        # (e.g. test fixtures, search_by_id-only callers).
        self._word_to_shabads: dict[str, set[int]] | None = None
        self._doc_freq: dict[str, int] | None = None
        self._total_shabads: int = 0
        self._shabad_first_letters: dict[int, str] | None = None

    def _ensure_word_index(self) -> None:
        """Build word→shabad and shabad→first-letters indices on first use.

        One linear scan of `lines.gurmukhi` / `lines.first_letters`, converting
        ASCII-Gurmukhi to Unicode once per unique string (via `_to_unicode`'s
        LRU cache). Typical cost: 3–6 s cold, <200 ms warm (most shabads already
        hydrated by prior searches).
        """
        if self._word_to_shabads is not None:
            return

        word_to_shabads: dict[str, set[int]] = defaultdict(set)
        shabad_concat: dict[int, list[str]] = defaultdict(list)

        # Walk every line ordered within its shabad so concatenations read in
        # recitation order. Uses the same scoped `_SHABAD_ID_EXPR` synthetic ID
        # so Dasam / Bhai Gurdas etc. get indexed alongside SGGS.
        rows = self._conn.execute(
            f"""
            SELECT
                {_SHABAD_ID_EXPR} AS sttm_id,
                l.gurmukhi       AS gurmukhi_ascii,
                l.first_letters  AS first_letters
            FROM lines l
            JOIN shabads s ON l.shabad_id = s.id
            ORDER BY sttm_id, l.order_id
            """
        ).fetchall()

        for row in rows:
            sid = row["sttm_id"]
            if sid is None:
                continue
            fl = row["first_letters"] or ""
            if fl:
                shabad_concat[sid].append(fl)
            unicode_text = _to_unicode(row["gurmukhi_ascii"] or "")
            for token in _tokenize_gurmukhi_words(unicode_text):
                word_to_shabads[token].add(sid)

        doc_freq = {word: len(shabads) for word, shabads in word_to_shabads.items()}
        shabad_first_letters = {
            sid: " ".join(parts) for sid, parts in shabad_concat.items()
        }

        self._word_to_shabads = dict(word_to_shabads)
        self._doc_freq = doc_freq
        self._total_shabads = len(shabad_concat)
        self._shabad_first_letters = shabad_first_letters

    def shabad_first_letters(self, shabad_id: int) -> str | None:
        """Return concatenated line-first-letters for a shabad, or None if unknown.

        Exposed so scoring can compute `dense_coverage` for fast/multi-line windows
        without every caller having to poke the private cache.
        """
        self._ensure_word_index()
        assert self._shabad_first_letters is not None  # set by _ensure_word_index
        return self._shabad_first_letters.get(shabad_id)

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

            # Strategy 1: prefix match on first_letters index.
            self._add_unique(self._prefix_search(query, max_results, "type0"), candidates, seen_ids)

            # Strategy 2: substring match on first_letters index.
            if len(candidates) < 3 and (not start_mode or len(candidates) == 0):
                self._add_unique(self._contains_search(query, max_results, "type1"), candidates, seen_ids)

            # Strategy 3: shorter substring fallback — prefix + sliding contains.
            # Contains catches mid-line matches: kirtan often starts at a chorus
            # fragment rather than the shabad's first line, so the query sits
            # inside a line's first_letters rather than prefixing it.
            if len(candidates) < 2 and len(query) > 4:
                for sub in (query[:4], query[-4:]):
                    self._add_unique(self._prefix_search(sub, 5, "type0_sub"), candidates, seen_ids)
                seen_subs: set[str] = set()
                for start in (0, len(query) // 2 - 2, len(query) - 4):
                    start = max(0, min(start, len(query) - 4))
                    sub = query[start:start + 4]
                    if sub in seen_subs:
                        continue
                    seen_subs.add(sub)
                    self._add_unique(self._contains_search(sub, 5, "type1_sub"), candidates, seen_ids)

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
            # 3-way split fires *in addition* when the query is long enough to
            # plausibly span 3 consecutive DB lines (very fast recitation).
            if len(first_letters) >= config.matcher.multi_line_trinary_min_query_length:
                self._add_unique(
                    self._multiline3_search(ascii_fl_full, max_results, "multiline3"),
                    candidates,
                    seen_ids,
                )

        # Strategy 6 (Fix 2): word-level IDF-weighted voting. Catches shabads where
        # the transcript's real words survived among Whisper filler, even if the
        # resulting first-letters string is too noisy for prefix/contains search.
        if (
            config.matcher.word_vote_enabled
            and transcript_text.strip()
        ):
            self._add_unique(
                self._word_vote_search(transcript_text, max_results, "type3_words"),
                candidates,
                seen_ids,
            )

        # Populate the full-first-letters cache on every returned candidate so the
        # scorer's dense_coverage term (Fix 3) has the data it needs without a
        # second round-trip.
        if self._shabad_first_letters is not None:
            for candidate in candidates:
                if candidate.full_first_letters is None:
                    candidate.full_first_letters = self._shabad_first_letters.get(
                        candidate.shabad_id
                    )

        return candidates

    def search_by_id(self, shabad_id: int) -> ShabadCandidate | None:
        """Fetch the first line of a specific shabad by its (possibly synthetic) id."""
        row = self._conn.execute(
            _LINE_SELECT + f" AND {_SHABAD_ID_EXPR} = ? ORDER BY l.order_id LIMIT 1",
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
            _LINE_SELECT + f" AND {_SHABAD_ID_EXPR} = ? ORDER BY l.order_id",
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
                    l2.gurmukhi       AS gurmukhi_ascii_next,
                    l1.first_letters  AS first_letters,
                    l1.order_id       AS order_id,
                    l1.source_page    AS source_page,
                    {_SHABAD_ID_EXPR} AS sttm_id,
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

        return [
            self._row_to_multiline_candidate(r, signal)
            for r in list(results.values())[:limit]
        ]

    def _multiline3_search(
        self, ascii_query: str, limit: int, signal: str
    ) -> list[ShabadCandidate]:
        """3-way variant of `_multiline_search` for very fast / dense recitation.

        Splits the query into THREE chunks and requires three consecutive lines
        in the same shabad to each match one chunk. Fires alongside the 2-way
        split when the query is long enough to plausibly span 3 lines.
        """
        if len(ascii_query) < 9:
            return []

        # Evenly trisect with small jitter — first-letter strings don't carry
        # word boundaries so exact split positions are approximate.
        third = len(ascii_query) // 3
        split_pairs: set[tuple[int, int]] = set()
        for a_off in (-1, 0, 1):
            for b_off in (-1, 0, 1):
                a = third + a_off
                b = 2 * third + b_off
                if a >= 3 and (b - a) >= 3 and (len(ascii_query) - b) >= 3:
                    split_pairs.add((a, b))

        scope = (
            f"AND s.source_id = {_SOURCE_SGGS}"
            if config.database.sggs_only
            else ""
        )

        results: dict[int, sqlite3.Row] = {}
        for a, b in sorted(split_pairs):
            part1 = ascii_query[:a]
            part2 = ascii_query[a:b]
            part3 = ascii_query[b:]
            rows = self._conn.execute(
                f"""
                SELECT
                    l1.gurmukhi       AS gurmukhi_ascii,
                    l2.gurmukhi       AS gurmukhi_ascii_2,
                    l3.gurmukhi       AS gurmukhi_ascii_3,
                    l1.first_letters  AS first_letters,
                    l1.order_id       AS order_id,
                    l1.source_page    AS source_page,
                    {_SHABAD_ID_EXPR} AS sttm_id,
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
                JOIN lines l3
                    ON l3.shabad_id = l1.shabad_id
                    AND l3.order_id = l1.order_id + 2
                WHERE l1.first_letters LIKE ? || '%'
                  AND l2.first_letters LIKE ? || '%'
                  AND l3.first_letters LIKE ? || '%'
                  {scope}
                GROUP BY s.sttm_id
                ORDER BY l1.order_id
                LIMIT ?
                """,
                (part1, part2, part3, limit),
            ).fetchall()
            for row in rows:
                results.setdefault(row["sttm_id"], row)
            if len(results) >= limit:
                break

        return [
            self._row_to_multiline3_candidate(r, signal)
            for r in list(results.values())[:limit]
        ]

    @staticmethod
    def _row_to_multiline3_candidate(row: sqlite3.Row, signal: str) -> ShabadCandidate:
        """Stitch all 3 matched lines into a single candidate for downstream scoring."""
        ascii_parts = [
            row["gurmukhi_ascii"] or "",
            row["gurmukhi_ascii_2"] or "",
            row["gurmukhi_ascii_3"] or "",
        ]
        combined_ascii = " ".join(p for p in ascii_parts if p).strip()
        combined_unicode = " ".join(
            _to_unicode(p) for p in ascii_parts if p
        ).strip()
        try:
            db_source = row["source_id"]
        except (IndexError, KeyError):
            db_source = _SOURCE_SGGS
        source_code = "G" if db_source == _SOURCE_SGGS else "D"
        return ShabadCandidate(
            shabad_id=row["sttm_id"],
            gurmukhi=combined_ascii,
            unicode=combined_unicode,
            english=row["english"] or "",
            source_id=source_code,
            page_no=row["source_page"],
            retrieval_sources={signal} if signal else set(),
        )

    @staticmethod
    def _row_to_multiline_candidate(row: sqlite3.Row, signal: str) -> ShabadCandidate:
        """
        Build a candidate that carries BOTH matched lines' text, so the scorer's
        letter-ratio and word-overlap logic can see the full query's worth of
        content instead of only line N. Without this, a 12-letter query scored
        against ~6 letters of line N gets penalized even on a perfect 2-line hit.
        """
        ascii_line1 = row["gurmukhi_ascii"] or ""
        ascii_line2 = row["gurmukhi_ascii_next"] or ""
        combined_ascii = f"{ascii_line1} {ascii_line2}".strip()
        combined_unicode = f"{_to_unicode(ascii_line1)} {_to_unicode(ascii_line2)}".strip()
        return ShabadCandidate(
            shabad_id=row["sttm_id"],
            gurmukhi=combined_ascii,
            unicode=combined_unicode,
            english=row["english"] or "",
            source_id="G",
            page_no=row["source_page"],
            retrieval_sources={signal} if signal else set(),
        )

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

    def _word_vote_search(
        self, transcript_text: str, limit: int, signal: str
    ) -> list[ShabadCandidate]:
        """IDF-weighted word voting: score shabads by how many rare transcript words hit them.

        When a Whisper transcript contains real Gurbani words buried among filler
        (the scattered-match symptom), the first-letter string is too noisy for
        prefix / contains search but the underlying words are still there. This
        strategy ignores first-letters entirely and just asks: which shabads
        contain enough of the transcript's distinctive words?
        """
        self._ensure_word_index()
        if not self._word_to_shabads or not self._total_shabads:
            return []

        from src.transcription.transliterate import normalize_for_fullword_search

        normalized = normalize_for_fullword_search(transcript_text)
        tokens = _tokenize_gurmukhi_words(normalized)
        if len(tokens) < 2:
            return []

        # Deduplicate tokens so a word repeated in the transcript doesn't stack up votes.
        distinct_tokens = set(tokens)
        stop_cutoff = max(1, int(self._total_shabads * config.matcher.word_vote_stopword_df_ratio))

        shabad_scores: dict[int, float] = defaultdict(float)
        shabad_hits: dict[int, int] = defaultdict(int)
        contributing_words: dict[int, set[str]] = defaultdict(set)
        for token in distinct_tokens:
            df = self._doc_freq.get(token, 0) if self._doc_freq else 0
            if df == 0 or df >= stop_cutoff:
                continue
            weight = math.log(self._total_shabads / df)
            for sid in self._word_to_shabads.get(token, ()):
                shabad_scores[sid] += weight
                shabad_hits[sid] += 1
                contributing_words[sid].add(token)

        if not shabad_scores:
            return []

        # Require ≥ min_distinct_hits and ≥ min_score before a shabad is returned —
        # prevents a single rare word from inventing a candidate out of thin air.
        min_hits = config.matcher.word_vote_min_distinct_hits
        min_score = config.matcher.word_vote_min_score
        ranked = sorted(
            (
                (score, shabad_hits[sid], sid)
                for sid, score in shabad_scores.items()
                if shabad_hits[sid] >= min_hits and score >= min_score
            ),
            reverse=True,
        )
        if not ranked:
            return []

        # Hydrate top candidates via the existing line-select path so they carry
        # translation + source metadata identical to other retrieval strategies.
        sids = [sid for _, _, sid in ranked[:limit]]
        placeholders = ",".join("?" * len(sids))
        rows = self._conn.execute(
            _LINE_SELECT
            + _scope_clause()
            + f"""
            AND {_SHABAD_ID_EXPR} IN ({placeholders})
            GROUP BY s.sttm_id
            ORDER BY l.order_id
            """,
            sids,
        ).fetchall()
        by_id = {row["sttm_id"]: row for row in rows}

        candidates: list[ShabadCandidate] = []
        for score, hits, sid in ranked[:limit]:
            row = by_id.get(sid)
            if row is None:
                continue
            candidate = self._row_to_candidate(row, signal)
            candidate.word_vote_score = round(score, 3)
            candidate.word_vote_hits = hits
            candidates.append(candidate)
        return candidates

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
