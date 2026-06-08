"""Build STTM-native Sundar Gutka line pointer map.

Inputs:
  * data/realm_gutka_lines.json — produced by scripts/dump_realm_gutka.js
  * database.sqlite             — local ShabadOS DB

Output:
  * data/gutka_bani_pointer_map.json

The output maps sqlite `banis.id` + local `lines.order_id` to STTM Desktop's
Realm `Banis_Shabad.ID` (`crossPlatformId`). That id is what STTM's
bani-controller path expects in the payload field named `verseId`.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_DUMP = _REPO_ROOT / "data" / "realm_gutka_lines.json"
_DEFAULT_DB = _REPO_ROOT / "database.sqlite"
_DEFAULT_VERSE_MAP = _REPO_ROOT / "data" / "order_id_to_verse_id.json"
_OUT = _REPO_ROOT / "data" / "gutka_bani_pointer_map.json"

_SQLITE_TO_REALM_BANI: dict[int, int] = {
    1: 2,
    2: 4,
    3: 6,
    4: 9,
    5: 10,
    6: 1000,
    7: 21,
    8: 21,
    9: 22,
    10: 22,
    11: 23,
    12: 31,
    14: 24,
    15: 30,
    16: 3,
    17: 5,
    18: 7,
    19: 29,
    20: 33,
    21: 34,
    22: 35,
    23: 27,
    24: 13,
    25: 11,
    27: 38,
    28: 46,
}

_STRIP_RE = re.compile(r"[;.,\[\]()\s]")


def _normalize(text: str) -> str:
    return _STRIP_RE.sub("", text or "")


def _load_local_bani(conn: sqlite3.Connection, sqlite_bani_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            bl.line_group AS line_group,
            l.id AS line_id,
            l.order_id AS order_id,
            l.gurmukhi AS gurmukhi,
            l.first_letters AS first_letters
        FROM bani_lines bl
        JOIN lines l ON bl.line_id = l.id
        WHERE bl.bani_id = ?
        ORDER BY bl.line_group, l.order_id
        """,
        (sqlite_bani_id,),
    ).fetchall()


def _choose_after(
    rows: list[dict],
    cursor: int,
    *,
    norm: str,
    realm_verse_id: int | None,
) -> tuple[dict | None, str]:
    after = [row for row in rows if int(row["seq"]) > cursor]
    if not after:
        if rows:
            return rows[-1], "last_row_fallback"
        return None, "missing"

    exact = [row for row in after if row.get("_norm") == norm]
    if exact:
        return min(exact, key=lambda row: int(row["seq"])), "exact_gurmukhi"

    if realm_verse_id is not None:
        verse_hits = [
            row for row in after
            if row.get("verse_id") is not None and int(row["verse_id"]) == realm_verse_id
        ]
        if verse_hits:
            return min(verse_hits, key=lambda row: int(row["seq"])), "realm_verse"

    if len(norm) >= 20:
        prefix = norm[:20]
        prefix_hits = [row for row in after if row.get("_norm", "").startswith(prefix)]
        if len(prefix_hits) == 1:
            return prefix_hits[0], "prefix"

    return after[0], "next_row_fallback"


def build(dump_path: Path, db_path: Path, verse_map_path: Path) -> None:
    if not dump_path.exists():
        sys.exit(
            f"Realm Gutka dump not found at {dump_path}.\n"
            "Run:\n"
            f"  node scripts/dump_realm_gutka.js > {dump_path}"
        )
    if not db_path.exists():
        sys.exit(f"sqlite DB not found at {db_path}")

    realm_rows = json.loads(dump_path.read_text())
    by_bani: dict[int, list[dict]] = {}
    for row in realm_rows:
        bid = row.get("bani_id")
        if bid is None:
            continue
        row["_norm"] = _normalize(row.get("gurmukhi") or "")
        by_bani.setdefault(int(bid), []).append(row)
    for rows in by_bani.values():
        rows.sort(key=lambda row: int(row["seq"]))

    try:
        verse_map = {
            int(k): int(v)
            for k, v in json.loads(verse_map_path.read_text()).items()
        }
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        verse_map = {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    out: dict[str, dict] = {}
    stats = {
        "exact_gurmukhi": 0,
        "realm_verse": 0,
        "prefix": 0,
        "next_row_fallback": 0,
        "last_row_fallback": 0,
        "missing": 0,
    }

    try:
        for sqlite_bani_id, realm_bani_id in sorted(_SQLITE_TO_REALM_BANI.items()):
            local_lines = _load_local_bani(conn, sqlite_bani_id)
            realm_bani_rows = by_bani.get(realm_bani_id, [])
            if not local_lines or not realm_bani_rows:
                continue
            cursor = -1
            mapped_lines: dict[str, dict] = {}
            for line in local_lines:
                order_id = int(line["order_id"])
                match, source = _choose_after(
                    realm_bani_rows,
                    cursor,
                    norm=_normalize(line["gurmukhi"] or ""),
                    realm_verse_id=verse_map.get(order_id),
                )
                stats[source] += 1
                if match is None:
                    continue
                seq = int(match["seq"])
                if not source.endswith("_fallback"):
                    cursor = seq
                mapped_lines[str(order_id)] = {
                    "cross_platform_id": int(match["cross_platform_id"]),
                    "realm_verse_id": (
                        int(match["verse_id"]) if match.get("verse_id") is not None else None
                    ),
                    "realm_bani_id": int(realm_bani_id),
                    "seq": seq,
                    "source": source,
                }
            out[str(sqlite_bani_id)] = {
                "realm_bani_id": int(realm_bani_id),
                "lines": mapped_lines,
            }
    finally:
        conn.close()

    _OUT.parent.mkdir(exist_ok=True)
    _OUT.write_text(json.dumps(out, separators=(",", ":")))
    total = sum(len(v["lines"]) for v in out.values())
    print(f"[build] wrote {total} Gutka line pointers -> {_OUT}", file=sys.stderr)
    print(f"[build] match stats: {stats}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=_DEFAULT_DUMP)
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--verse-map", type=Path, default=_DEFAULT_VERSE_MAP)
    args = parser.parse_args()
    build(args.dump, args.db, args.verse_map)


if __name__ == "__main__":
    main()
