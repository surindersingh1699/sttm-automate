"""Build runtime lookup tables that translate our local sqlite ids into the
ids STTM Desktop's Realm DB expects in `/api/bani-control` payloads.

Inputs:
  * `data/realm_verses.json` — produced by `scripts/dump_realm_verses.js`
  * `database.sqlite`         — local ShabadOS DB at repo root

Outputs (both under `data/`):
  * `order_id_to_verse_id.json`
        sqlite `lines.order_id` → Realm `Verse.ID`
        Lets the HTTP controller send a verseId STTM can actually highlight
        (fixes line-sync drift in bani-mode and long SGGS shabads).
  * `shabad_to_realm_shabad_id.json`
        sqlite synthetic shabad id (COALESCE(sttm_id, order_id + 1e8))
        → Realm `Shabads.ShabadID`
        Lets the controller send Dasam shabads via type:"shabad" even when
        there's no bani mapping (Gyan Prabodh, Charitropakhyan, etc.).

Matching strategy is exact-match on `gurmukhi` text. Our sqlite stores lines
in AnvaadLipi ASCII — same encoding STTM's Realm uses — so string equality
lines up the two datasets without any normalization gymnastics. Rows that
don't match (typically whitespace-only or empty lines) are skipped and
reported to stderr.

Ambiguous matches (multiple Realm verses share the same Gurmukhi text)
prefer the one whose Source disambiguates: for SGGS sqlite rows we pick the
Realm verse with Source == 'G', for Dasam we pick 'D', and so on. When that
still leaves multiple candidates we take the smallest Realm ID to keep the
output stable across rebuilds.

Usage:
    python scripts/build_sttm_mapping.py [--dump data/realm_verses.json]
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_DUMP = _REPO_ROOT / "data" / "realm_verses.json"
_DEFAULT_DB = _REPO_ROOT / "database.sqlite"
_DATA_DIR = _REPO_ROOT / "data"

# sqlite source_id → Realm SourceID single-char code
_SOURCE_ID_TO_CHAR = {
    1: "G",   # Sri Guru Granth Sahib
    2: "D",   # Sri Dasam Granth
    3: "B",   # Vaaran Bhai Gurdas
    4: "B",   # Kabit Savaiye Bhai Gurdas
    5: "N",   # Ghazals Bhai Nand Lal
    6: "N",   # Zindagi Nama
    7: "N",   # Ganj Nama
    8: "N",   # Jot Bigas
    9: "A",   # Ardaas
    10: "R",  # Rehitname
    11: "S",  # Sarabloh Granth
    12: "U",  # Uggardanti
}

_SYNTHETIC_ID_OFFSET = 100_000_000


# Visraam markers (; . ,) live in our sqlite's `lines.gurmukhi` but STTM's
# Realm stores the same lines without them. Realm also omits whitespace
# around punctuation that our sqlite preserves (e.g. "word [" vs "word["),
# so the matcher strips visraam + all whitespace before comparing.
_VISRAAM_RE = re.compile(r"[;.,]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    if not text:
        return ""
    return _WS_RE.sub("", _VISRAAM_RE.sub("", text))


def _disambiguate(
    candidates: list[dict], preferred_source: str | None
) -> dict | None:
    if not candidates:
        return None
    if preferred_source:
        filtered = [c for c in candidates if c.get("src") == preferred_source]
        if filtered:
            candidates = filtered
    # Stable pick: smallest Realm Verse.ID.
    return min(candidates, key=lambda c: c["i"])


def build(dump_path: Path, db_path: Path) -> None:
    if not dump_path.exists():
        sys.exit(
            f"Realm dump not found at {dump_path}.\n"
            f"Produce it first:\n"
            f"  node scripts/dump_realm_verses.js > {dump_path}"
        )
    if not db_path.exists():
        sys.exit(f"sqlite DB not found at {db_path}")

    print(f"[build] loading Realm dump from {dump_path}", file=sys.stderr)
    verses = json.loads(dump_path.read_text())
    print(f"[build]   {len(verses)} Realm verses", file=sys.stderr)

    # Index by normalized Gurmukhi text. Multiple verses may share identical
    # text — keep them all and disambiguate later by source.
    by_gurmukhi: dict[str, list[dict]] = defaultdict(list)
    for v in verses:
        g = _normalize(v.get("g") or "")
        if g:
            by_gurmukhi[g].append(v)

    _DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    verse_map: dict[str, int] = {}
    shabad_map: dict[str, int] = {}
    # shabad → multiple (first-line Gurmukhi, source_id) pairs: track all
    # first-lines so each synthetic shabad id gets the best Realm ShabadID.
    first_line_seen: set[int] = set()
    unmatched_lines = 0
    unmatched_shabads = 0

    line_query = """
        SELECT
            l.order_id   AS line_order_id,
            l.gurmukhi   AS gurmukhi,
            s.source_id  AS source_id,
            s.sttm_id    AS sttm_id,
            s.order_id   AS shabad_order_id,
            ROW_NUMBER() OVER (PARTITION BY s.id ORDER BY l.order_id) AS line_rank
        FROM lines l
        JOIN shabads s ON l.shabad_id = s.id
        ORDER BY l.order_id
    """

    for row in conn.execute(line_query):
        g = _normalize(row["gurmukhi"] or "")
        source_char = _SOURCE_ID_TO_CHAR.get(int(row["source_id"]))
        candidates = by_gurmukhi.get(g, [])
        match = _disambiguate(candidates, source_char)
        if match is None:
            unmatched_lines += 1
            continue

        verse_map[str(row["line_order_id"])] = int(match["i"])

        # For each shabad, pull the Realm ShabadID from its first matched line.
        # Realm Verse.Shabads is a link list but in practice single-element for
        # our purposes — take the first Shabad id.
        shabad_synthetic = (
            int(row["sttm_id"])
            if row["sttm_id"] is not None
            else int(row["shabad_order_id"]) + _SYNTHETIC_ID_OFFSET
        )
        if row["line_rank"] == 1 and shabad_synthetic not in first_line_seen:
            first_line_seen.add(shabad_synthetic)
            realm_shabad_ids = match.get("s") or []
            if realm_shabad_ids:
                shabad_map[str(shabad_synthetic)] = int(realm_shabad_ids[0])
            else:
                unmatched_shabads += 1

    conn.close()

    # Write both maps as compact JSON.
    verse_out = _DATA_DIR / "order_id_to_verse_id.json"
    shabad_out = _DATA_DIR / "shabad_to_realm_shabad_id.json"
    verse_out.write_text(json.dumps(verse_map, separators=(",", ":")))
    shabad_out.write_text(json.dumps(shabad_map, separators=(",", ":")))

    print(
        f"[build] verse map: {len(verse_map)} entries "
        f"(unmatched lines: {unmatched_lines}) -> {verse_out}",
        file=sys.stderr,
    )
    print(
        f"[build] shabad map: {len(shabad_map)} entries "
        f"(unmatched shabads: {unmatched_shabads}) -> {shabad_out}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", type=Path, default=_DEFAULT_DUMP,
        help="Path to realm_verses.json produced by dump_realm_verses.js",
    )
    parser.add_argument(
        "--db", type=Path, default=_DEFAULT_DB,
        help="Path to ShabadOS database.sqlite",
    )
    args = parser.parse_args()
    build(args.dump, args.db)


if __name__ == "__main__":
    main()
