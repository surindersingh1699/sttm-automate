"""Load the ordered line sequence for a single bani (gutka follow-along mode).

Used by ``GutkaAligner`` to collapse the search space from "every shabad in
the canon" down to one bani's flat ``[(idx, first_letters_ascii)]`` list.
The 28 nitnem banis live in the local SQLite DB's ``banis`` table; their
ordered line sequence is in ``bani_lines`` joined to ``lines``.

``bani_lines`` row ordering: ``(line_group, lines.order_id)``. ``line_group``
segments banis that span multiple shabads (Rehras has 12 groups, Japji is 1
group); within a group ``lines.order_id`` is globally unique and monotonic
through the shabad.

STTM Desktop's Sundar Gutka controller does *not* highlight by normal
``Verse.ID``. In bani mode its ``verseId`` payload is matched against the
Realm ``Banis_Shabad.ID`` cross-platform id. When
``data/gutka_bani_pointer_map.json`` is present (built from STTM's Realm DB),
we overlay that exact pointer id onto each local line and sort by STTM's
``Seq``. Without the map, callers still get local lines with a conservative
fallback pointer so the dashboard can run, but STTM line sync may be imperfect.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from gurmukhiutils.unicode import unicode as _ascii_to_unicode_gurmukhi

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_GUTKA_POINTER_MAP_PATH = _PROJECT_ROOT / "data" / "gutka_bani_pointer_map.json"


@dataclass(frozen=True)
class GutkaLine:
    bani_line_idx: int            # 0-based position within the bani
    line_id: str                  # lines.id (4-char varchar)
    shabad_id: str                # lines.shabad_id (for display/debug only)
    order_id: int                 # local sqlite lines.order_id
    first_letters_ascii: str      # what we match ASR first-letters against
    gurmukhi_unicode: str         # display text (converted from AnvaadLipi ASCII)
    english: str | None
    # STTM-native bani pointer metadata. ``cross_platform_id`` is
    # Realm Banis_Shabad.ID and is the preferred bani-mode payload verseId.
    cross_platform_id: int | None = None
    realm_verse_id: int | None = None
    realm_bani_id: int | None = None
    sttm_seq: int | None = None
    pointer_source: str = "local_fallback"

    @property
    def sttm_pointer_id(self) -> int:
        """Best available id to send as STTM bani-mode ``verseId``."""
        return int(self.cross_platform_id or self.realm_verse_id or self.order_id)


# In-process cache: sqlite_bani_id → list[GutkaLine]. Banis are immutable in
# the DB; building the list is cheap (one query + one Unicode-conversion pass)
# but still wasteful to redo on every dropdown re-select.
_BANI_CACHE: dict[int, list[GutkaLine]] = {}
_POINTER_MAP_CACHE: dict | None = None

_LOAD_SQL = """
    SELECT
        bl.line_group   AS line_group,
        l.id            AS line_id,
        l.shabad_id     AS shabad_id,
        l.order_id      AS order_id,
        l.first_letters AS first_letters,
        l.gurmukhi      AS gurmukhi_ascii,
        (
            SELECT t.translation FROM translations t
            WHERE t.line_id = l.id AND t.translation_source_id = 1
            LIMIT 1
        ) AS english
    FROM bani_lines bl
    JOIN lines l ON bl.line_id = l.id
    WHERE bl.bani_id = ?
    ORDER BY bl.line_group, l.order_id
"""


def _to_unicode_safe(ascii_gurmukhi: str) -> str:
    if not ascii_gurmukhi:
        return ""
    try:
        return _ascii_to_unicode_gurmukhi(ascii_gurmukhi)
    except Exception:
        return ascii_gurmukhi


def _load_pointer_map() -> dict:
    """Load optional sqlite bani id -> line pointer map."""
    global _POINTER_MAP_CACHE
    if _POINTER_MAP_CACHE is not None:
        return _POINTER_MAP_CACHE
    try:
        _POINTER_MAP_CACHE = json.loads(_GUTKA_POINTER_MAP_PATH.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        _POINTER_MAP_CACHE = {}
    return _POINTER_MAP_CACHE


def _pointer_for_line(sqlite_bani_id: int, order_id: int) -> dict:
    bani_map = _load_pointer_map().get(str(sqlite_bani_id), {})
    lines = bani_map.get("lines", {}) if isinstance(bani_map, dict) else {}
    ptr = lines.get(str(order_id), {}) if isinstance(lines, dict) else {}
    return ptr if isinstance(ptr, dict) else {}


def load_bani(
    sqlite_bani_id: int,
    *,
    db_path: str | Path | None = None,
    use_cache: bool = True,
) -> list[GutkaLine]:
    """Return the ordered line list for a SQLite bani id (1-29).

    Raises ``KeyError`` if the bani id has no rows. Caches results in-process.
    """
    if use_cache and sqlite_bani_id in _BANI_CACHE:
        return _BANI_CACHE[sqlite_bani_id]

    if db_path is None:
        from src.matcher.offline_search import _resolve_db_path
        db_path = _resolve_db_path()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(_LOAD_SQL, (int(sqlite_bani_id),)).fetchall()
    finally:
        conn.close()

    if not rows:
        raise KeyError(f"No lines found for bani_id={sqlite_bani_id}")

    lines: list[GutkaLine] = []
    for idx, row in enumerate(rows):
        ptr = _pointer_for_line(int(sqlite_bani_id), int(row["order_id"]))
        lines.append(
            GutkaLine(
                bani_line_idx=idx,
                line_id=row["line_id"],
                shabad_id=row["shabad_id"],
                order_id=int(row["order_id"]),
                first_letters_ascii=row["first_letters"] or "",
                gurmukhi_unicode=_to_unicode_safe(row["gurmukhi_ascii"] or ""),
                english=row["english"],
                cross_platform_id=(
                    int(ptr["cross_platform_id"])
                    if ptr.get("cross_platform_id") is not None
                    else None
                ),
                realm_verse_id=(
                    int(ptr["realm_verse_id"])
                    if ptr.get("realm_verse_id") is not None
                    else None
                ),
                realm_bani_id=(
                    int(ptr["realm_bani_id"])
                    if ptr.get("realm_bani_id") is not None
                    else None
                ),
                sttm_seq=int(ptr["seq"]) if ptr.get("seq") is not None else None,
                pointer_source=ptr.get("source") or "local_fallback",
            )
        )

    if any(line.sttm_seq is not None for line in lines):
        lines.sort(
            key=lambda line: (
                line.sttm_seq if line.sttm_seq is not None else 10**9,
                line.order_id,
            )
        )
        lines = [
            GutkaLine(
                bani_line_idx=idx,
                line_id=line.line_id,
                shabad_id=line.shabad_id,
                order_id=line.order_id,
                first_letters_ascii=line.first_letters_ascii,
                gurmukhi_unicode=line.gurmukhi_unicode,
                english=line.english,
                cross_platform_id=line.cross_platform_id,
                realm_verse_id=line.realm_verse_id,
                realm_bani_id=line.realm_bani_id,
                sttm_seq=line.sttm_seq,
                pointer_source=line.pointer_source,
            )
            for idx, line in enumerate(lines)
        ]

    if use_cache:
        _BANI_CACHE[sqlite_bani_id] = lines
    return lines


def list_available_banis(
    sqlite_ids: list[int] | None = None,
    *,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Return ``[{sqlite_id, name_gurmukhi, name_english, line_count}]`` for the
    given sqlite bani ids (or all banis when ``sqlite_ids`` is None).

    Caller is expected to pass the keys of ``_SQLITE_TO_REALM_BANI`` so the
    dashboard picker only surfaces banis STTM Desktop can actually display.
    """
    if db_path is None:
        from src.matcher.offline_search import _resolve_db_path
        db_path = _resolve_db_path()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if sqlite_ids:
            placeholders = ",".join("?" * len(sqlite_ids))
            sql = (
                f"SELECT id, name_gurmukhi, name_english FROM banis "
                f"WHERE id IN ({placeholders}) ORDER BY id"
            )
            rows = conn.execute(sql, sqlite_ids).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name_gurmukhi, name_english FROM banis ORDER BY id"
            ).fetchall()
        counts = {
            int(row["bani_id"]): int(row["n"])
            for row in conn.execute(
                "SELECT bani_id, COUNT(*) AS n FROM bani_lines GROUP BY bani_id"
            )
        }
    finally:
        conn.close()

    out = []
    for row in rows:
        sid = int(row["id"])
        out.append({
            "sqlite_id": sid,
            "name_gurmukhi": _to_unicode_safe(row["name_gurmukhi"] or ""),
            "name_english": row["name_english"],
            "line_count": counts.get(sid, 0),
        })
    return out
