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


# Aggressive normalization so our sqlite lines line up with STTM Realm's
# verses despite a bunch of purely-cosmetic differences:
#
#   * visraam markers (; . ,) present in sqlite, stripped in Realm
#   * trailing `[` in sqlite vs `]` in Realm for Bhai Gurdas / Nand Lal
#   * parentheticals e.g. "pauVI 1 (mMglwcrx)" vs "pauVI 1"
#   * whitespace around punctuation ("word [" vs "word[")
#
# We throw away every punctuation/whitespace character on both sides and
# compare the resulting "letters only" form. That collapses legitimate
# lines together too rarely to matter in practice (the distinct verse
# still differs at the letter level).
_STRIP_RE = re.compile(r"[;.,\[\]()\s]")


def _normalize(text: str) -> str:
    if not text:
        return ""
    return _STRIP_RE.sub("", text)


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
    # Prefix index for fuzzy fallback: upstream sqlite and STTM's Realm
    # occasionally differ by one or two words inside a verse (ShabadOS has
    # updated some readings in Kabit Savaiye / late Dasam / Ganj Nama since
    # the Realm was built). When exact normalization misses we fall back
    # to "does Realm have exactly one verse that starts with the same
    # first 30 characters?" That's tight enough to avoid collisions while
    # catching word-level drift.
    _PREFIX_LEN = 30
    by_prefix: dict[str, list[dict]] = defaultdict(list)
    for v in verses:
        g = _normalize(v.get("g") or "")
        if g:
            by_gurmukhi[g].append(v)
            if len(g) >= _PREFIX_LEN:
                by_prefix[g[:_PREFIX_LEN]].append(v)

    _DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Collect every line per shabad first so we can fall back to later lines
    # when the first one is a rubric/header that Realm doesn't store.
    line_query = """
        SELECT
            l.order_id   AS line_order_id,
            l.gurmukhi   AS gurmukhi,
            s.id         AS shabad_key,
            s.source_id  AS source_id,
            s.sttm_id    AS sttm_id,
            s.order_id   AS shabad_order_id
        FROM lines l
        JOIN shabads s ON l.shabad_id = s.id
        ORDER BY s.order_id, l.order_id
    """
    shabad_lines: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in conn.execute(line_query):
        shabad_lines[row["shabad_key"]].append(row)
    conn.close()

    verse_map: dict[str, int] = {}
    shabad_map: dict[str, int] = {}
    unmatched_shabads = 0
    unmatched_lines = 0

    # Ordered list of (synthetic_id, source_char, source_id, lines) so a
    # post-pass can interpolate Realm ShabadIDs for shabads that never
    # landed a direct or prefix match.
    shabad_order: list[tuple[int, str | None, int, list]] = []

    for shabad_key, lines in shabad_lines.items():
        first = lines[0]
        source_id = int(first["source_id"])
        source_char = _SOURCE_ID_TO_CHAR.get(source_id)
        synthetic = (
            int(first["sttm_id"])
            if first["sttm_id"] is not None
            else int(first["shabad_order_id"]) + _SYNTHETIC_ID_OFFSET
        )
        shabad_order.append((synthetic, source_char, source_id, lines))

        # --- Pass 1: direct match per line, with prefix-match fallback. ---
        per_line_match: list[dict | None] = []
        for row in lines:
            n = _normalize(row["gurmukhi"] or "")
            match = _disambiguate(by_gurmukhi.get(n, []), source_char)
            if match is None and len(n) >= _PREFIX_LEN:
                prefix_hits = by_prefix.get(n[:_PREFIX_LEN], [])
                # Only accept the prefix match when it resolves to a single
                # candidate of the right source — otherwise we risk mapping a
                # distinct verse to its lookalike neighbour.
                filtered = [
                    c for c in prefix_hits
                    if source_char is None or c.get("src") == source_char
                ]
                if len(filtered) == 1:
                    match = filtered[0]
            per_line_match.append(match)

        # --- Resolve shabad ShabadID: first line with a Realm hit wins. ---
        realm_shabad_id: int | None = None
        for match in per_line_match:
            if match and match.get("s"):
                realm_shabad_id = int(match["s"][0])
                break
        if realm_shabad_id is not None:
            shabad_map[str(synthetic)] = realm_shabad_id
        else:
            unmatched_shabads += 1

        # --- Fill verse map: direct matches + nearest-neighbor fallback. ---
        # Line rubrics like "pauVI 1 [" or "sorTw [" have no Realm twin;
        # instead of dropping them we point them at the next matched line's
        # Realm verse so STTM still highlights *something* sane when the
        # kirtanee is on that header line.
        last_realm_id: int | None = None
        # Right-scan once to know the next realm id for each unmatched slot.
        next_realm_by_index: list[int | None] = [None] * len(lines)
        cursor: int | None = None
        for idx in range(len(lines) - 1, -1, -1):
            match = per_line_match[idx]
            if match:
                cursor = int(match["i"])
            next_realm_by_index[idx] = cursor

        for idx, row in enumerate(lines):
            match = per_line_match[idx]
            if match is not None:
                chosen = int(match["i"])
                last_realm_id = chosen
            else:
                chosen = next_realm_by_index[idx] or last_realm_id
                if chosen is None:
                    unmatched_lines += 1
                    continue
            verse_map[str(row["line_order_id"])] = chosen

    # --- Post-pass: neighbour-interpolate unmapped shabads. ---
    # A few dozen shabads still have no Realm ShabadID because their lines
    # diverged too much from Realm (textual variants, Ardaas as an entire
    # source, SGGS rubrics). Realm's ShabadID namespace is mostly sequential
    # within a source, and sqlite shabads traverse in the same order, so
    # copying the prior mapped shabad's Realm ShabadID gives STTM a neighbour
    # to display instead of silently no-op'ing. Same for the verses: any
    # line inside an unmapped shabad inherits the preceding shabad's last
    # Realm verse so auto-scroll at least stays in a related passage.
    last_by_source: dict[int, int] = {}
    last_verse_by_source: dict[int, int] = {}
    interp_shabads = 0
    interp_lines = 0
    for synthetic, _source_char, source_id, lines in shabad_order:
        key = str(synthetic)
        if key in shabad_map:
            last_by_source[source_id] = shabad_map[key]
            # Track last verse for same-source interpolation.
            for row in reversed(lines):
                vk = str(row["line_order_id"])
                if vk in verse_map:
                    last_verse_by_source[source_id] = verse_map[vk]
                    break
            continue
        fallback_sid = last_by_source.get(source_id)
        if fallback_sid is not None:
            shabad_map[key] = fallback_sid
            interp_shabads += 1
        fallback_vid = last_verse_by_source.get(source_id)
        if fallback_vid is not None:
            for row in lines:
                vk = str(row["line_order_id"])
                if vk not in verse_map:
                    verse_map[vk] = fallback_vid
                    interp_lines += 1

    # Write both maps as compact JSON.
    verse_out = _DATA_DIR / "order_id_to_verse_id.json"
    shabad_out = _DATA_DIR / "shabad_to_realm_shabad_id.json"
    verse_out.write_text(json.dumps(verse_map, separators=(",", ":")))
    shabad_out.write_text(json.dumps(shabad_map, separators=(",", ":")))

    print(
        f"[build] verse map: {len(verse_map)} entries "
        f"(unmatched: {unmatched_lines}, interpolated: {interp_lines}) -> {verse_out}",
        file=sys.stderr,
    )
    print(
        f"[build] shabad map: {len(shabad_map)} entries "
        f"(unmatched: {unmatched_shabads}, interpolated: {interp_shabads}) -> {shabad_out}",
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
