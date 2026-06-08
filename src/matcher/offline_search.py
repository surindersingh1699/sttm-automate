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
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from gurmukhiutils.unicode import unicode as _ascii_to_unicode_gurmukhi

from src.config import config
from src.matcher.search import ShabadCandidate, ShabadVerse
from src.transcription.transliterate import gurmukhi_to_ascii, normalize_first_letter

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENGLISH_TRANSLATION_SOURCE = 1  # Dr. Sant Singh Khalsa
_SOURCE_SGGS = 1  # shabads.source_id for Sri Guru Granth Sahib (used for source labelling)

# `shabads.sttm_id` is NULL for Dasam Granth (5470 rows) and Uggardanti (9 rows).
# Without a synthetic fallback every such candidate collapses to shabad_id=None,
# so _add_unique dedup's them all into a single entry. We synthesize an ID from
# `order_id` (unique across the whole table) offset well past the real sttm_id
# range (max ~30k). Anything above SYNTHETIC_ID_OFFSET is a non-SGGS fallback.
SYNTHETIC_ID_OFFSET = 100_000_000
_SHABAD_ID_EXPR = f"COALESCE(s.sttm_id, s.order_id + {SYNTHETIC_ID_OFFSET})"


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

# Kirtan filler words that ragis commonly prefix/suffix to a tuk but are absent
# from the DB line.  Stripping them before FL extraction tightens the FL match
# and prevents the extra first-letter from degrading score margin.
_KIRTAN_FILLER = frozenset([
    "ਵਾਹਿਗੁਰੂ", "ਵਾਹਿਗੁਰ", "ਵਾਹੁਗੁਰੂ", "ਵਾਹਗੁਰੂ",
    "ਜੀਉ", "ਜੀਓ", "ਜੀ",
    "ਵਾਹੁ", "ਵਾਹ",
    "ਰਾਮ",
    "ਸਤਿਨਾਮੁ", "ਸਤਿਨਾਮ",
])


def _strip_filler_words(transcript: str) -> str:
    """Remove known kirtan filler words from the start and end of a transcript.

    Only strips from the edges — interior words are kept as they may be part of
    the actual tuk and removing them would corrupt the FL string.
    """
    if not transcript:
        return transcript
    words = transcript.split()
    while words and words[0] in _KIRTAN_FILLER:
        words = words[1:]
    while words and words[-1] in _KIRTAN_FILLER:
        words = words[:-1]
    return " ".join(words)


def _tokenize_gurmukhi_words(text: str) -> list[str]:
    """Extract Gurmukhi word tokens (≥2 chars) from Unicode text.

    Shared by the DB-side word index and the live-transcript word-vote retrieval
    so both sides use identical tokenization. Intentionally simple — strip
    diacritics/matras from short tokens by length filter, not by character
    class, to avoid dropping real short words like ਨ ਹੈ ਕਉ (kept at len=2).
    """
    return [token for token in _GURMUKHI_TOKEN.findall(text) if len(token) >= 2]


def _char_4grams(text: str) -> frozenset[str]:
    s = (text or "").strip()
    if len(s) < 4:
        return frozenset({s} if s else set())
    return frozenset(s[i: i + 4] for i in range(len(s) - 3))


def _lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for ach in a:
        for idx, bch in enumerate(b):
            if ach == bch:
                curr[idx + 1] = prev[idx] + 1
            else:
                curr[idx + 1] = max(prev[idx + 1], curr[idx])
        prev, curr = curr, prev
        for i in range(len(curr)):
            curr[i] = 0
    return prev[len(b)]


def _span_fl_score(query_ascii: str, span_ascii: str) -> float:
    """Shared first-letter score for one-line and multi-line spans."""
    if not query_ascii or not span_ascii:
        return 0.0
    ratio = SequenceMatcher(None, query_ascii, span_ascii).ratio()
    matched = _lcs_len(query_ascii, span_ascii)
    span_cov = matched / max(1, len(span_ascii))
    query_cov = matched / max(1, len(query_ascii))
    dense = 1.0 if span_ascii in query_ascii or query_ascii in span_ascii else 0.0
    return max(ratio, (0.65 * span_cov) + (0.35 * query_cov), dense)


def _overlap_coeff(left: set | frozenset, right: set | frozenset) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


@dataclass(frozen=True)
class _IndexedLine:
    shabad_id: int
    line_idx: int
    order_id: int
    gurmukhi_ascii: str
    unicode: str
    english: str
    first_letters_ascii: str
    source_code: str
    page_no: int
    words: frozenset[str]
    char4: frozenset[str]


@dataclass(frozen=True)
class _IndexedSpan:
    shabad_id: int
    line_idx: int
    span_len: int
    gurmukhi_ascii: str
    unicode: str
    english: str
    first_letters_ascii: str
    source_code: str
    page_no: int
    words: frozenset[str]
    char4: frozenset[str]


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
        # Char 4-gram Unicode index (Strategy 9): 4-gram → list[line_rowid]
        self._ngram4_index: dict[str, list[int]] | None = None
        # line_rowid → (sttm_id, unicode_text, first_letters_ascii, frozenset of 4-grams)
        self._ngram4_line_data: dict[int, tuple[int, str, str, frozenset]] | None = None
        # Unified span index: the same evidence model for Kirtan open search and
        # ordered bani/nitnem style matching.
        self._span_index: list[_IndexedSpan] | None = None
        self._span_fl_grams: dict[str, set[int]] | None = None
        self._span_words: dict[str, set[int]] | None = None
        self._span_char4: dict[str, set[int]] | None = None

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

    def _ensure_ngram_index(self) -> None:
        """Build char-4-gram Unicode index over all DB lines on first use.

        Catches end-fragment kirtan patterns (ragi sings only the 2nd half of a
        line) that FL prefix/contains strategies miss entirely, because the DB
        line's first letters don't match the query's first letters at all.

        Build cost: ~4-6 s cold; zero thereafter (checked via None sentinel).
        """
        if self._ngram4_index is not None:
            return

        from src.transcription.transliterate import normalize_for_fullword_search

        ngram4_index: dict[str, list[int]] = defaultdict(list)
        line_data: dict[int, tuple[int, str, str, frozenset]] = {}

        rows = self._conn.execute(
            f"""
            SELECT
                l.rowid          AS line_rowid,
                {_SHABAD_ID_EXPR} AS sttm_id,
                l.gurmukhi       AS gurmukhi_ascii,
                l.first_letters  AS first_letters
            FROM lines l
            JOIN shabads s ON l.shabad_id = s.id
            ORDER BY l.order_id
            """
        ).fetchall()

        for row in rows:
            sid = row["sttm_id"]
            if sid is None:
                continue
            line_rowid = row["line_rowid"]
            unicode_text = _to_unicode(row["gurmukhi_ascii"] or "")
            normalized = normalize_for_fullword_search(unicode_text)
            s = normalized.strip()
            if len(s) < 4:
                grams: frozenset = frozenset({s} if s else set())
            else:
                grams = frozenset(s[i: i + 4] for i in range(len(s) - 3))
            line_data[line_rowid] = (sid, unicode_text, row["first_letters"] or "", grams)
            for gram in grams:
                ngram4_index[gram].append(line_rowid)

        self._ngram4_index = dict(ngram4_index)
        self._ngram4_line_data = line_data

    def _ensure_span_index(self) -> None:
        """Build a shared 1/2/3-line span index for open and ordered matching.

        The previous retriever had separate SQL strategies for prefix, contains,
        full-word, 2-line, 3-line, word-vote and char-ngram matching. This index
        stores the common unit they were all trying to find: a specific line span.
        Search becomes "collect plausible spans, score them with the same three
        evidence channels, keep the best span per shabad."
        """
        if self._span_index is not None:
            return

        from src.transcription.transliterate import normalize_for_fullword_search

        rows = self._conn.execute(
            f"""
            SELECT
                {_SHABAD_ID_EXPR} AS sttm_id,
                l.gurmukhi       AS gurmukhi_ascii,
                l.first_letters  AS first_letters,
                l.order_id       AS order_id,
                l.source_page    AS source_page,
                s.source_id      AS source_id,
                (
                    SELECT t.translation FROM translations t
                    WHERE t.line_id = l.id AND t.translation_source_id = {_ENGLISH_TRANSLATION_SOURCE}
                    LIMIT 1
                ) AS english
            FROM lines l
            JOIN shabads s ON l.shabad_id = s.id
            ORDER BY sttm_id, l.order_id
            """
        ).fetchall()

        lines_by_shabad: dict[int, list[_IndexedLine]] = defaultdict(list)
        for row in rows:
            sid = row["sttm_id"]
            if sid is None:
                continue
            ascii_g = row["gurmukhi_ascii"] or ""
            unicode_text = _to_unicode(ascii_g)
            normalized = normalize_for_fullword_search(unicode_text)
            source_code = "G" if row["source_id"] == _SOURCE_SGGS else "D"
            lines_by_shabad[sid].append(
                _IndexedLine(
                    shabad_id=sid,
                    line_idx=len(lines_by_shabad[sid]),
                    order_id=row["order_id"],
                    gurmukhi_ascii=ascii_g,
                    unicode=unicode_text,
                    english=row["english"] or "",
                    first_letters_ascii=row["first_letters"] or "",
                    source_code=source_code,
                    page_no=row["source_page"] or 0,
                    words=frozenset(_tokenize_gurmukhi_words(normalized)),
                    char4=_char_4grams(normalized),
                )
            )

        span_index: list[_IndexedSpan] = []
        fl_grams: dict[str, set[int]] = defaultdict(set)
        word_index: dict[str, set[int]] = defaultdict(set)
        char4_index: dict[str, set[int]] = defaultdict(set)
        shabad_concat: dict[int, str] = {}

        for sid, lines in lines_by_shabad.items():
            shabad_concat[sid] = " ".join(line.first_letters_ascii for line in lines)
            for start_idx, line in enumerate(lines):
                ascii_parts: list[str] = []
                unicode_parts: list[str] = []
                fl_parts: list[str] = []
                words: set[str] = set()
                char4: set[str] = set()
                for end_idx in range(start_idx, min(len(lines), start_idx + 3)):
                    part = lines[end_idx]
                    ascii_parts.append(part.gurmukhi_ascii)
                    unicode_parts.append(part.unicode)
                    fl_parts.append(part.first_letters_ascii)
                    words.update(part.words)
                    char4.update(part.char4)
                    span_fl = "".join(fl_parts)
                    if not span_fl and not words and not char4:
                        continue
                    span_id = len(span_index)
                    span = _IndexedSpan(
                        shabad_id=sid,
                        line_idx=end_idx,
                        span_len=end_idx - start_idx + 1,
                        gurmukhi_ascii=" ".join(p for p in ascii_parts if p).strip(),
                        unicode=" ".join(p for p in unicode_parts if p).strip(),
                        english=line.english,
                        first_letters_ascii=span_fl,
                        source_code=line.source_code,
                        page_no=line.page_no,
                        words=frozenset(words),
                        char4=frozenset(char4),
                    )
                    span_index.append(span)
                    for gram in self._ascii_grams(span_fl):
                        fl_grams[gram].add(span_id)
                    for token in span.words:
                        word_index[token].add(span_id)
                    for gram in span.char4:
                        char4_index[gram].add(span_id)

        self._span_index = span_index
        self._span_fl_grams = dict(fl_grams)
        self._span_words = dict(word_index)
        self._span_char4 = dict(char4_index)
        if self._shabad_first_letters is None:
            self._shabad_first_letters = shabad_concat
            self._total_shabads = len(shabad_concat)

    @staticmethod
    def _ascii_grams(value: str) -> set[str]:
        value = value or ""
        if not value:
            return set()
        sizes = (3, 4) if len(value) >= 4 else (len(value),)
        grams: set[str] = set()
        for size in sizes:
            if size <= 0 or len(value) < size:
                continue
            grams.update(value[i: i + size] for i in range(len(value) - size + 1))
        return grams

    def _span_search(
        self,
        first_letters: str,
        transcript_text: str,
        limit: int,
        signal: str = "span",
        start_mode: bool = False,
    ) -> list[ShabadCandidate]:
        """Retrieve and rank exact line spans using FL, word, and char evidence."""
        if len(first_letters) < 3 and not transcript_text.strip():
            return []
        self._ensure_span_index()
        assert self._span_index is not None
        assert self._span_fl_grams is not None
        assert self._span_words is not None
        assert self._span_char4 is not None

        from src.transcription.transliterate import normalize_for_fullword_search

        query_ascii = gurmukhi_to_ascii(first_letters) if first_letters else ""
        if start_mode and query_ascii:
            query_ascii = query_ascii[: min(8, len(query_ascii))]
        normalized = normalize_for_fullword_search(transcript_text) if transcript_text else ""
        q_words = frozenset(_tokenize_gurmukhi_words(normalized))
        q_char4 = _char_4grams(normalized)

        seed_ids: set[int] = set()
        for gram in self._ascii_grams(query_ascii):
            seed_ids.update(self._span_fl_grams.get(gram, ()))
        for token in q_words:
            seed_ids.update(self._span_words.get(token, ()))
        for gram in q_char4:
            seed_ids.update(self._span_char4.get(gram, ()))

        if not seed_ids:
            return []

        best_per_shabad: dict[int, tuple[float, _IndexedSpan, int, float, float, float]] = {}
        for span_id in seed_ids:
            span = self._span_index[span_id]
            fl_score = _span_fl_score(query_ascii, span.first_letters_ascii)
            word_score = _overlap_coeff(q_words, span.words)
            char_score = _overlap_coeff(q_char4, span.char4)
            if fl_score < 0.20 and word_score < 0.34 and char_score < config.matcher.ngram4_min_overlap:
                continue

            weights: list[tuple[float, float]] = []
            if query_ascii:
                weights.append((0.58, fl_score))
            if q_words:
                weights.append((0.27, word_score))
            if q_char4:
                weights.append((0.15, char_score))
            if not weights:
                continue
            total_weight = sum(weight for weight, _ in weights)
            score = sum(weight * value for weight, value in weights) / total_weight

            # Longer spans are only a bonus when the query itself is long enough
            # to plausibly cover multiple panktis.
            if span.span_len > 1 and len(query_ascii) >= config.matcher.multi_line_min_query_length:
                score += 0.035 * (span.span_len - 1)
            if span.line_idx == 0 and config.matcher.penalize_heading_line:
                score -= 0.08
            score = max(0.0, min(1.0, score))

            current = best_per_shabad.get(span.shabad_id)
            if current is None or score > current[0]:
                best_per_shabad[span.shabad_id] = (
                    score,
                    span,
                    len(q_words & span.words),
                    word_score,
                    char_score,
                    fl_score,
                )

        ranked = sorted(best_per_shabad.values(), key=lambda item: item[0], reverse=True)
        candidates: list[ShabadCandidate] = []
        for score, span, word_hits, _word_score, _char_score, _fl_score in ranked[:limit]:
            candidate = ShabadCandidate(
                shabad_id=span.shabad_id,
                gurmukhi=span.gurmukhi_ascii,
                unicode=span.unicode,
                english=span.english,
                source_id=span.source_code,
                page_no=span.page_no,
                retrieval_sources={signal},
                full_first_letters=(
                    self._shabad_first_letters.get(span.shabad_id)
                    if self._shabad_first_letters
                    else None
                ),
                word_vote_hits=word_hits,
                line_idx=span.line_idx,
                span_len=span.span_len,
                span_score=round(score, 3),
            )
            candidates.append(candidate)
        return candidates

    def _ngram4_search(
        self,
        transcript_text: str,
        limit: int,
        signal: str,
    ) -> list["ShabadCandidate"]:
        """Char-4-gram overlap-coefficient retrieval over Unicode Gurmukhi lines.

        Overlap coefficient = |q4 ∩ l4| / min(|q4|, |l4|).  Verbatim match → 1.0,
        partial/noisy → graceful decay.  Ignores FL alignment entirely — designed
        for the kirtan pattern where the ragi sings only the 2nd half of a line,
        making FL search blind to the correct shabad.
        """
        self._ensure_ngram_index()
        assert self._ngram4_index is not None
        assert self._ngram4_line_data is not None

        from src.transcription.transliterate import normalize_for_fullword_search

        normalized = normalize_for_fullword_search(transcript_text)
        s = normalized.strip()
        if len(s) < 4:
            q4: frozenset = frozenset({s} if s else set())
        else:
            q4 = frozenset(s[i: i + 4] for i in range(len(s) - 3))
        if not q4:
            return []

        # Collect candidate line IDs via inverted index
        candidate_line_ids: set[int] = set()
        for gram in q4:
            for lid in self._ngram4_index.get(gram, ()):
                candidate_line_ids.add(lid)

        if not candidate_line_ids:
            return []

        min_overlap = config.matcher.ngram4_min_overlap
        q4_size = len(q4)

        # Score each candidate line; keep best per shabad
        best_per_shabad: dict[int, tuple[float, int]] = {}  # sttm_id → (score, line_rowid)
        for lid in candidate_line_ids:
            entry = self._ngram4_line_data.get(lid)
            if entry is None:
                continue
            sid, _, _, l4 = entry
            if not l4:
                continue
            intersection = len(q4 & l4)
            overlap = intersection / min(q4_size, len(l4))
            if overlap < min_overlap:
                continue
            prev = best_per_shabad.get(sid)
            if prev is None or overlap > prev[0]:
                best_per_shabad[sid] = (overlap, lid)

        if not best_per_shabad:
            return []

        ranked = sorted(best_per_shabad.items(), key=lambda kv: kv[1][0], reverse=True)

        # Hydrate top candidates using _row_to_candidate
        candidates: list[ShabadCandidate] = []
        for sttm_id, (score, line_rowid) in ranked[:limit]:
            entry = self._ngram4_line_data.get(line_rowid)
            if entry is None:
                continue
            _, unicode_text, fl_ascii, _ = entry
            # Fetch the full DB row to use _row_to_candidate
            row = self._conn.execute(
                _LINE_SELECT + " AND l.rowid = ?",
                (line_rowid,),
            ).fetchone()
            if row is None:
                continue
            candidate = self._row_to_candidate(row, signal)
            candidates.append(candidate)
        return candidates

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

        # Strip kirtan filler words (ਵਾਹਿਗੁਰੂ, ਜੀਉ, etc.) from the transcript
        # edges before deriving FL so the extra first-letter doesn't compress
        # the score margin for the real shabad.
        if transcript_text.strip():
            cleaned = _strip_filler_words(transcript_text)
            if cleaned != transcript_text:
                from src.transcription.transliterate import extract_first_letters as _efl
                first_letters = _efl(cleaned)
                transcript_text = cleaned

        candidates: list[ShabadCandidate] = []
        seen_ids: set[int] = set()

        # Unified span search first. This replaces the old "try many unrelated
        # strategies then reconcile later" shape with one candidate stream that
        # already knows the best line/span for the pointer.
        self._add_unique(
            self._span_search(
                first_letters,
                transcript_text,
                max(max_results * 2, 20),
                "span",
                start_mode,
            ),
            candidates,
            seen_ids,
        )

        # Legacy SQL/string strategies are now fallback only. The span index
        # already combines first-letter, word and char evidence and returns the
        # exact pointer line, so keep the older stack as a safety net for sparse
        # or unusual queries instead of letting it dominate normal ranking.
        legacy_fallback = len(candidates) < 3

        if legacy_fallback and len(first_letters) >= 3:
            ascii_fl = gurmukhi_to_ascii(first_letters)
            query = ascii_fl[: min(8, len(ascii_fl))] if start_mode else ascii_fl

            # Strategy 1: prefix match on first_letters index.
            self._add_unique(self._prefix_search(query, max_results, "type0"), candidates, seen_ids)

            # Strategy 2: substring match on first_letters index.
            if len(candidates) < 3 and (not start_mode or len(candidates) == 0):
                self._add_unique(self._contains_search(query, max_results, "type1"), candidates, seen_ids)

            # Strategy 2b: rotation fallback for short queries — ASR may capture words in
            # a different order than the DB line (e.g. query "tmm" → rotation "mmt" hits
            # "kkjmmthj"). Only fires for ≤4 chars to keep rotation count and false-positive
            # rate manageable.
            if len(candidates) < 2 and len(query) <= 4:
                self._add_unique(self._rotation_search(query, 5, "rotation"), candidates, seen_ids)

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
        if legacy_fallback and transcript_text.strip():
            self._add_unique(self._fullword_search(transcript_text, max_results, "type2"), candidates, seen_ids)

        # Strategy 6: rotation search — ragi sang words out of canonical line order.
        # Always run for short queries (≤4 chars): type1 may find other candidates
        # but the correct shabad requires a rotation (e.g. "tmm" misses "kkjmmthj"
        # but rotation "mmt" hits pos 3-5). Scorer ranks all candidates by confidence.
        if legacy_fallback and len(first_letters) >= 3 and len(first_letters) <= 4:
            ascii_rot = gurmukhi_to_ascii(first_letters)
            self._add_unique(
                self._rotation_search(ascii_rot, max_results, "type_rotation"),
                candidates,
                seen_ids,
            )

        # Strategy 7 (Fix 2): word-level IDF-weighted voting. Catches shabads where
        # the transcript's real words survived among Whisper filler, even if the
        # resulting first-letters string is too noisy for prefix/contains search.
        if (
            legacy_fallback
            and config.matcher.word_vote_enabled
            and transcript_text.strip()
        ):
            self._add_unique(
                self._word_vote_search(transcript_text, max_results, "type3_words", first_letters=first_letters),
                candidates,
                seen_ids,
            )

        # Strategy 8: phonetic substitution — retry with common ASR confusions.
        # ਬ(b)↔ਵ(v) are nearly identical in Punjabi speech; Whisper routinely
        # hears ਬਿਸਾਰਹੁ as ਵਿਸਾਰਿ, breaking every first-letter strategy above.
        # Also tries 4-char substrings of each phonetic variant to handle cases where
        # Whisper prepends an extra word (e.g. singing "ਤੇਰਾ ਮੋਹਿ..." before the tuk
        # starts at "ਮੋਹਿ" in the DB, so the full variant doesn't substring-match).
        # Only fires when the original query found few candidates.
        if legacy_fallback and len(first_letters) >= 3:
            ascii_fl_ph = gurmukhi_to_ascii(first_letters)
            for variant in self._phonetic_variants(ascii_fl_ph):
                self._add_unique(
                    self._prefix_search(variant, max_results, "phonetic"),
                    candidates,
                    seen_ids,
                )
                self._add_unique(
                    self._contains_search(variant, max_results, "phonetic"),
                    candidates,
                    seen_ids,
                )
                if len(variant) > 4:
                    seen_ph_subs: set[str] = set()
                    for ph_start in (0, len(variant) // 2 - 2, len(variant) - 4):
                        ph_start = max(0, min(ph_start, len(variant) - 4))
                        ph_sub = variant[ph_start:ph_start + 4]
                        if ph_sub in seen_ph_subs:
                            continue
                        seen_ph_subs.add(ph_sub)
                        self._add_unique(
                            self._contains_search(ph_sub, 5, "phonetic_sub"),
                            candidates,
                            seen_ids,
                        )

        # Strategy 9: char 4-gram Unicode retrieval — catches end-fragment kirtan
        # patterns (ragi repeats 2nd half of a line) that are invisible to FL search.
        # Gated by config so it can be disabled if the index build cost is a concern.
        if legacy_fallback and config.matcher.ngram4_search_enabled and transcript_text.strip():
            self._add_unique(
                self._ngram4_search(transcript_text, config.matcher.ngram4_max_results, "ngram4"),
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
        """Fetch all verses of a shabad for line-level tracking."""
        rows = self._conn.execute(
            _LINE_SELECT + f" AND {_SHABAD_ID_EXPR} = ? ORDER BY l.order_id",
            (shabad_id,),
        ).fetchall()

        verses: list[ShabadVerse] = []
        for row in rows:
            unicode_text = _to_unicode(row["gurmukhi_ascii"] or "")
            verses.append(
                ShabadVerse(
                    verse_id=int(row["order_id"]),
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
            +f"""
            AND l.first_letters LIKE ? || '%'
            GROUP BY {_SHABAD_ID_EXPR}
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
            +f"""
            AND l.first_letters LIKE '%' || ? || '%'
            GROUP BY {_SHABAD_ID_EXPR}
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
            +f"""
            AND l.first_letters LIKE '%' || ? || '%'
            GROUP BY {_SHABAD_ID_EXPR}
            ORDER BY l.order_id
            LIMIT ?
            """,
            (ascii_fls, limit),
        ).fetchall()
        return [self._row_to_candidate(r, signal) for r in rows]

    def _word_vote_search(
        self, transcript_text: str, limit: int, signal: str, first_letters: str = ""
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

        # Require ≥ min_distinct_hits and ≥ min_score before a shabad is returned.
        # Exception: a single word whose IDF weight alone clears single_hit_min_score
        # is strong enough evidence — covers kirtan repetition of one rare/distinctive word.
        min_hits = config.matcher.word_vote_min_distinct_hits
        min_score = config.matcher.word_vote_min_score
        single_hit_min = config.matcher.word_vote_single_hit_min_score
        ranked = sorted(
            (
                (score, shabad_hits[sid], sid)
                for sid, score in shabad_scores.items()
                if (shabad_hits[sid] >= min_hits and score >= min_score)
                or (shabad_hits[sid] >= 1 and score >= single_hit_min)
            ),
            reverse=True,
        )
        if not ranked:
            return []

        # Hydrate top candidates: fetch ALL lines for the winning shabads so we can
        # pick the line that best matches the audio query — not just the first line
        # (which is often a raag header like "ਗਉੜੀ ਮਹਲਾ ੫ ॥" that scores near-zero).
        sids = [sid for _, _, sid in ranked[:limit]]
        placeholders = ",".join("?" * len(sids))
        all_rows = self._conn.execute(
            _LINE_SELECT
            +f"AND {_SHABAD_ID_EXPR} IN ({placeholders}) ORDER BY l.order_id",
            sids,
        ).fetchall()

        from collections import defaultdict as _defaultdict
        lines_by_shabad: dict = _defaultdict(list)
        for row in all_rows:
            lines_by_shabad[row["sttm_id"]].append(row)

        ascii_fl = gurmukhi_to_ascii(first_letters) if first_letters else ""

        def _line_score(fl_row: str) -> float:
            if not ascii_fl or not fl_row:
                return 0.0
            # Fraction of the candidate line's letters that appear as a
            # subsequence of the query — identical to _subsequence_coverage in scorer.
            n = len(fl_row)
            j = 0
            for ch in ascii_fl:
                if j < n and ch == fl_row[j]:
                    j += 1
            return j / n

        candidates: list[ShabadCandidate] = []
        for score, hits, sid in ranked[:limit]:
            lines = lines_by_shabad.get(sid)
            if not lines:
                continue
            # Pick the line whose first_letters best matches the query; fall back to
            # the second line (skip typical single-line headers) when no query given.
            if ascii_fl:
                best_row = max(lines, key=lambda r: _line_score(r["first_letters"] or ""))
            else:
                best_row = lines[1] if len(lines) > 1 else lines[0]
            candidate = self._row_to_candidate(best_row, signal)
            candidate.word_vote_score = round(score, 3)
            candidate.word_vote_hits = hits
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _phonetic_variants(ascii_query: str) -> list[str]:
        """Generate ASCII first-letter variants with common Punjabi ASR confusions.

        Pairs: b↔v, s↔S (sa/sha), n↔N (dental/retroflex), t↔T, d↔D, plus h-drop.
        Generates single-substitution variants first, then second-level combinations
        of those, capped at phonetic_max_variants to bound search load.
        """
        _PAIRS: list[tuple[str, str]] = [
            ("b", "v"), ("v", "b"),
            ("s", "S"), ("S", "s"),
            ("n", "N"), ("N", "n"),
            ("t", "T"), ("T", "t"),
            ("d", "D"), ("D", "d"),
        ]
        max_v = config.matcher.phonetic_max_variants
        seen: set[str] = {ascii_query}
        variants: list[str] = []

        # Level 1: single-pair substitutions
        for src, dst in _PAIRS:
            if src in ascii_query:
                alt = ascii_query.replace(src, dst)
                if alt not in seen:
                    seen.add(alt)
                    variants.append(alt)
                    if len(variants) >= max_v:
                        return variants

        # H-drop: remove all 'h' characters
        if "h" in ascii_query:
            alt = ascii_query.replace("h", "")
            if alt and alt not in seen:
                seen.add(alt)
                variants.append(alt)
                if len(variants) >= max_v:
                    return variants

        # Level 2: apply pairs to each level-1 variant for combined confusions
        for base in list(variants):
            for src, dst in _PAIRS:
                if src in base:
                    alt = base.replace(src, dst)
                    if alt not in seen:
                        seen.add(alt)
                        variants.append(alt)
                        if len(variants) >= max_v:
                            return variants
            if "h" in base:
                alt = base.replace("h", "")
                if alt and alt not in seen:
                    seen.add(alt)
                    variants.append(alt)
                    if len(variants) >= max_v:
                        return variants

        return variants

    def _rotation_search(self, ascii_query: str, limit: int, signal: str) -> list[ShabadCandidate]:
        """Try all cyclic rotations of a short query as contains searches.

        Handles out-of-order singing: if the ragi starts mid-line, the first-letter
        query may be a rotation of what sits in the DB. Each unique rotation is tried
        as a contains search; the original query is skipped (already tried by type1).
        Uses a wider internal fetch (5× limit) because ORDER BY l.order_id in the
        grouped contains query would otherwise miss shabads whose matching line sits
        late in the shabad but is actually the best match.
        """
        n = len(ascii_query)
        seen_rotations: set[str] = {ascii_query}
        results: list[ShabadCandidate] = []
        seen_ids: set[int] = set()
        inner_limit = 250  # wide net — scorer handles ranking, not this method
        for i in range(1, n):
            rotation = ascii_query[i:] + ascii_query[:i]
            if rotation in seen_rotations:
                continue
            seen_rotations.add(rotation)
            hits = self._contains_search(rotation, inner_limit, signal)
            for c in hits:
                if c.shabad_id not in seen_ids:
                    seen_ids.add(c.shabad_id)
                    results.append(c)
        return results

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
                    # Propagate word-vote metadata — it's only set on candidates
                    # returned by _word_vote_search and is lost if another path
                    # already claimed the slot.
                    if c.word_vote_hits and not existing_c.word_vote_hits:
                        existing_c.word_vote_hits = c.word_vote_hits
                        existing_c.word_vote_score = c.word_vote_score
                    if c.span_score is not None and (
                        existing_c.span_score is None or c.span_score > existing_c.span_score
                    ):
                        existing_c.line_idx = c.line_idx
                        existing_c.span_len = c.span_len
                        existing_c.span_score = c.span_score
                        existing_c.gurmukhi = c.gurmukhi
                        existing_c.unicode = c.unicode
                        existing_c.english = c.english
