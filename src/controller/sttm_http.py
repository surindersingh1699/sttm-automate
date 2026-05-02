"""Control STTM Desktop via HTTP POST to its Express server.

**No external APIs.** Verse IDs needed for the STTM controller payload are
resolved from the local ShabadOS SQLite DB (`database.sqlite`) — the same
schema STTM Desktop itself uses, so ``lines.order_id`` doubles as the
``verseId`` STTM expects. Previously this module called BaniDB over HTTP;
that path is removed.
"""

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx

from src.config import config
from src.controller.base import STTMController

_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Optional runtime mappings that translate our sqlite ids into the ids STTM
# Desktop's Realm DB actually uses. Produced by
# ``scripts/dump_realm_verses.js`` + ``scripts/build_sttm_mapping.py``.
#
# When these files are absent we fall back to sending raw sqlite ids, which
# works for SGGS + other sources with matching sttm_id but drifts for Dasam.
# When present we get exact line-highlight sync and can display Dasam shabads
# outside nitnem banis (Gyan Prabodh, Charitropakhyan, etc.) via type:"shabad".
_VERSE_MAP_PATH = _PROJECT_ROOT / "data" / "order_id_to_verse_id.json"
_SHABAD_MAP_PATH = _PROJECT_ROOT / "data" / "shabad_to_realm_shabad_id.json"


def _load_int_map(path: Path) -> dict[int, int]:
    """Load a JSON object whose keys+values are integer-strings/ints."""
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"[STTM HTTP] Failed to load mapping {path}: {e}")
        return {}
    return {int(k): int(v) for k, v in raw.items()}


_VERSE_MAP: dict[int, int] = _load_int_map(_VERSE_MAP_PATH)
_SHABAD_MAP: dict[int, int] = _load_int_map(_SHABAD_MAP_PATH)
if _VERSE_MAP:
    print(f"[STTM HTTP] Loaded verse map: {len(_VERSE_MAP)} entries")
if _SHABAD_MAP:
    print(f"[STTM HTTP] Loaded shabad map: {len(_SHABAD_MAP)} entries")

# Realm shabad_id → ordered list of Realm Verse.IDs, built from realm_verses.json.
# Used in navigate_line to get the correct verseId by index, bypassing cases where
# _VERSE_MAP maps multiple consecutive lines to the same Realm verse (bad match).
_REALM_SHABAD_VERSES_PATH = _PROJECT_ROOT / "data" / "realm_verses.json"


def _build_realm_shabad_verses(path: Path) -> dict[int, list[int]]:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    shabad_verses: dict[int, list[int]] = {}
    for v in data:
        for sid in v.get("s", []):
            shabad_verses.setdefault(sid, []).append(v["i"])
    return {sid: sorted(vs) for sid, vs in shabad_verses.items()}


_REALM_SHABAD_VERSES: dict[int, list[int]] = _build_realm_shabad_verses(
    _REALM_SHABAD_VERSES_PATH
)
if _REALM_SHABAD_VERSES:
    print(f"[STTM HTTP] Loaded realm shabad verses: {len(_REALM_SHABAD_VERSES)} shabads")

# shabados/database `banis.id` → STTM Desktop Realm `Banis.ID`.
# Built by matching Gurmukhi/Token across both DBs (scripts/dump_realm_banis.js).
# Entries missing from STTM's Realm (e.g. Asa Ki Var, Alahnia, Mundavnni) are
# intentionally absent — those banis can't be displayed via bani-mode.
_SQLITE_TO_REALM_BANI: dict[int, int] = {
    1: 2,     # Jap Ji Sahib → japji
    2: 4,     # Jaap Sahib → jaap
    3: 6,     # Tav Prasad Savaiye (Sravag Sudh) → svaiye
    4: 9,     # Benti Chaupai Sahib → chaupai
    5: 10,    # Anand Sahib → anand
    6: 1000,  # Anand Sahib (6 Pauris) → anand6
    7: 21,    # Rehras Sahib (S.) → rehras
    8: 21,    # Rehras Sahib (T.) → rehras
    9: 22,    # Aarti → aarti
    10: 22,   # Aarti (Longer) → aarti
    11: 23,   # Sohila Sahib → sohila
    12: 31,   # Sukhmani Sahib → sukhmani
    14: 24,   # Ardaas → ardas
    15: 30,   # Salok Mehla 9 → salokm9
    16: 3,    # Shabad Hazare → shabadhazare
    17: 5,    # Shabad Hazare Patshahi 10 → shabadhazare10
    18: 7,    # Tav Prasad Savaiye (Deenan Ki) → svaiyedeenan
    19: 29,   # Akal Ustat → akalustat
    20: 33,   # Bavan Akhri → bavanakhree
    21: 34,   # Sidh Gosht → sidhgosht
    22: 35,   # Oankaar → dhakhnioankar
    23: 27,   # Barah Maha → baarehmaha (Maajh)
    24: 13,   # Chandi Di Var → chandidivar
    25: 11,   # Lavan (Anand Karaj) → lavaa
    27: 38,   # GGS Paath Bhog (Ragmala) → raagmala
    28: 46,   # Raamkali Sad → sadd
}


class STTMHttpController(STTMController):
    """
    Controls STTM Desktop by sending HTTP requests to its local Express server.

    STTM Desktop runs Express.js on one of several known ports.
    It exposes POST /api/bani-control which sends data to the Electron main window.

    Verse lookups hit the local SQLite DB — no external API calls.
    """

    def __init__(self):
        self.base_url: str | None = None
        self._client = httpx.AsyncClient(timeout=config.sttm.connect_timeout)
        self._active_shabad_id: int | None = None
        self._active_bani_id: int | None = None
        # Realm ShabadID we actually sent to STTM (differs from self._active_shabad_id
        # for Dasam/Ardaas where our sqlite has no sttm_id). Needed so navigate_line
        # can keep re-sending the same id for subsequent lines of the same shabad.
        self._active_realm_shabad_id: int | None = None
        self._active_line_idx: int = 0
        self._first_verse_cache: dict[int, int] = {}
        self._verse_ids_cache: dict[int, list[int]] = {}
        self._bani_id_cache: dict[int, int | None] = {}
        self._db: sqlite3.Connection | None = None

    async def connect(self) -> bool:
        """Discover STTM's port and verify connectivity."""
        for port in config.sttm.ports:
            url = f"http://localhost:{port}"
            try:
                resp = await self._client.get(url, timeout=config.sttm.connect_timeout)
                if resp.status_code == 200:
                    self.base_url = url
                    print(f"[STTM HTTP] Connected on port {port}")
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                continue
        print("[STTM HTTP] Could not find STTM Desktop on any known port")
        return False

    async def search_shabad(self, query: str) -> bool:
        """Search is not used in the current STTM controller integration."""
        return False

    async def select_result(self, index: int = 0) -> bool:
        """Selecting a result is not supported via the current STTM controller payloads."""
        return False

    async def display_shabad(self, shabad_id: int) -> bool:
        """Display a shabad by its ID.

        Routing priority:

        1. **Realm shabad mapping present** — send ``type:"shabad"`` with the
           translated Realm ``ShabadID`` + translated ``Verse.ID``. Works for
           every shabad in STTM's Realm, including Dasam shabads with no
           ``sttm_id`` (Gyan Prabodh, Charitropakhyan, etc.).
        2. **Nitnem bani match** — when no shabad mapping exists but the line
           belongs to one of the 29 Sundar Gutka banis, fall back to
           ``type:"bani"``. Preserves the pre-mapping behaviour so the
           controller still works without running the build script.
        3. **Raw shabad id** — final fallback for SGGS (and anything else with
           a real ``sttm_id``). This was the only path before the mapping
           work; kept intact so removing the data files never breaks prod.
        """
        verse_ids = await self._get_verse_ids(shabad_id)
        first_verse_order_id = (
            verse_ids[0] if verse_ids else await self._get_first_verse_id(shabad_id)
        )
        realm_shabad_id = _SHABAD_MAP.get(shabad_id)
        realm_verses_for_shabad = (
            _REALM_SHABAD_VERSES.get(realm_shabad_id, []) if realm_shabad_id else []
        )
        if realm_verses_for_shabad:
            realm_verse_id = realm_verses_for_shabad[0]
        else:
            realm_verse_id = _VERSE_MAP.get(first_verse_order_id, first_verse_order_id)

        if realm_shabad_id is not None:
            ok = await self._send_control({
                "type": "shabad",
                "shabadId": realm_shabad_id,
                "id": realm_shabad_id,
                "verseId": realm_verse_id,
                "lineCount": 1,
                "highlight": realm_verse_id,
                "homeId": realm_verse_id,
            })
            if ok:
                self._active_shabad_id = shabad_id
                self._active_realm_shabad_id = realm_shabad_id
                self._active_bani_id = None
                self._active_line_idx = 0
            return ok

        bani_id = await self._get_bani_id(shabad_id)
        if bani_id is not None:
            ok = await self._send_control({
                "type": "bani",
                "baniId": bani_id,
                "verseId": realm_verse_id,
            })
            if ok:
                self._active_shabad_id = shabad_id
                self._active_realm_shabad_id = None
                self._active_bani_id = bani_id
                self._active_line_idx = 0
            return ok

        ok = await self._send_control({
            "type": "shabad",
            "shabadId": shabad_id,
            # Compatibility fields observed in STTM desktop internals.
            "id": shabad_id,
            "verseId": first_verse_order_id,
            "lineCount": 1,
            "highlight": first_verse_order_id,
            "homeId": first_verse_order_id,
        })
        if ok:
            self._active_shabad_id = shabad_id
            self._active_realm_shabad_id = None
            self._active_bani_id = None
            self._active_line_idx = 0
        return ok

    async def navigate_line(self, direction: str = "next") -> bool:
        """Move line by re-sending the active payload with the target verseId.

        Mirrors the routing in :meth:`display_shabad`: if the shabad was sent
        as a Realm-mapped shabad, keep re-sending shabad-mode with the Realm
        ``ShabadID`` and a translated ``Verse.ID``; if it was sent as a bani,
        keep bani-mode; otherwise fall back to raw sqlite ids.
        """
        if self._active_shabad_id is None:
            return False

        verse_ids = await self._get_verse_ids(self._active_shabad_id)
        if not verse_ids:
            return False

        if direction == "prev":
            next_idx = max(0, self._active_line_idx - 1)
        else:
            next_idx = min(len(verse_ids) - 1, self._active_line_idx + 1)

        raw_verse_id = verse_ids[next_idx]

        if self._active_realm_shabad_id is not None:
            # Prefer direct index lookup from realm_verses.json — avoids cases
            # where _VERSE_MAP collapses multiple lines onto the same Realm ID.
            realm_verses_for_shabad = _REALM_SHABAD_VERSES.get(
                self._active_realm_shabad_id, []
            )
            if next_idx < len(realm_verses_for_shabad):
                realm_verse_id = realm_verses_for_shabad[next_idx]
            else:
                realm_verse_id = _VERSE_MAP.get(raw_verse_id, raw_verse_id)
        else:
            realm_verse_id = _VERSE_MAP.get(raw_verse_id, raw_verse_id)

        if self._active_realm_shabad_id is not None:
            home_realm_verse_id = (
                realm_verses_for_shabad[0]
                if realm_verses_for_shabad
                else _VERSE_MAP.get(verse_ids[0], verse_ids[0])
            )
            payload = {
                "type": "shabad",
                "shabadId": self._active_realm_shabad_id,
                "id": self._active_realm_shabad_id,
                "verseId": realm_verse_id,
                "lineCount": next_idx + 1,
                "highlight": realm_verse_id,
                "homeId": home_realm_verse_id,
            }
        elif self._active_bani_id is not None:
            payload = {
                "type": "bani",
                "baniId": self._active_bani_id,
                "verseId": realm_verse_id,
            }
        else:
            payload = {
                "type": "shabad",
                "shabadId": self._active_shabad_id,
                "id": self._active_shabad_id,
                "verseId": raw_verse_id,
                "lineCount": next_idx + 1,
                "highlight": raw_verse_id,
                "homeId": verse_ids[0],
            }

        ok = await self._send_control(payload)
        if ok:
            self._active_line_idx = next_idx
        return ok

    async def navigate_to_line(self, target_idx: int) -> bool:
        """Jump directly to a specific line index (0-based) within the active shabad."""
        if self._active_shabad_id is None or target_idx <= 0:
            return False
        verse_ids = await self._get_verse_ids(self._active_shabad_id)
        if not verse_ids or target_idx >= len(verse_ids):
            return False
        next_idx = target_idx
        raw_verse_id = verse_ids[next_idx]
        if self._active_realm_shabad_id is not None:
            realm_verses_for_shabad = _REALM_SHABAD_VERSES.get(self._active_realm_shabad_id, [])
            if next_idx < len(realm_verses_for_shabad):
                realm_verse_id = realm_verses_for_shabad[next_idx]
            else:
                realm_verse_id = _VERSE_MAP.get(raw_verse_id, raw_verse_id)
            home_realm_verse_id = (
                realm_verses_for_shabad[0] if realm_verses_for_shabad
                else _VERSE_MAP.get(verse_ids[0], verse_ids[0])
            )
            payload = {
                "type": "shabad",
                "shabadId": self._active_realm_shabad_id,
                "id": self._active_realm_shabad_id,
                "verseId": realm_verse_id,
                "lineCount": next_idx + 1,
                "highlight": realm_verse_id,
                "homeId": home_realm_verse_id,
            }
        elif self._active_bani_id is not None:
            realm_verse_id = _VERSE_MAP.get(raw_verse_id, raw_verse_id)
            payload = {"type": "bani", "baniId": self._active_bani_id, "verseId": realm_verse_id}
        else:
            payload = {
                "type": "shabad",
                "shabadId": self._active_shabad_id,
                "id": self._active_shabad_id,
                "verseId": raw_verse_id,
                "lineCount": next_idx + 1,
                "highlight": raw_verse_id,
                "homeId": verse_ids[0],
            }
        ok = await self._send_control(payload)
        if ok:
            self._active_line_idx = next_idx
        return ok

    async def disconnect(self):
        """Close the HTTP client + local DB connection."""
        await self._client.aclose()
        if self._db is not None:
            self._db.close()
            self._db = None

    async def _get_first_verse_id(self, shabad_id: int) -> int:
        """Return the first line's `order_id` for a shabad (STTM controller verseId)."""
        cached = self._first_verse_cache.get(shabad_id)
        if cached is not None:
            return cached
        verse_ids = await self._get_verse_ids(shabad_id)
        return verse_ids[0] if verse_ids else 1

    async def _get_verse_ids(self, shabad_id: int) -> list[int]:
        """Return all line `order_id`s for a shabad in display order.

        `lines.order_id` is unique across the whole DB and is what STTM Desktop
        uses as `verseId` in its controller payload (same ShabadOS schema).
        For Dasam/other non-SGGS shabads, our `shabad_id` is synthesized as
        `shabads.order_id + SYNTHETIC_ID_OFFSET`; a single `COALESCE` in the
        query handles both real and synthetic ids.
        """
        cached = self._verse_ids_cache.get(shabad_id)
        if cached is not None:
            return cached

        verse_ids = await asyncio.to_thread(self._query_verse_ids, shabad_id)
        if verse_ids:
            self._verse_ids_cache[shabad_id] = verse_ids
            self._first_verse_cache[shabad_id] = verse_ids[0]
        return verse_ids

    def _query_verse_ids(self, shabad_id: int) -> list[int]:
        """Sync SQLite query — run via asyncio.to_thread from async callers."""
        self._ensure_db()
        from src.matcher.offline_search import SYNTHETIC_ID_OFFSET

        rows = self._db.execute(
            f"""
            SELECT l.order_id
            FROM lines l
            JOIN shabads s ON l.shabad_id = s.id
            WHERE COALESCE(s.sttm_id, s.order_id + {SYNTHETIC_ID_OFFSET}) = ?
            ORDER BY l.order_id
            """,
            (shabad_id,),
        ).fetchall()
        return [int(r[0]) for r in rows]

    async def _get_bani_id(self, shabad_id: int) -> int | None:
        """Return STTM Realm's ``Banis.ID`` this shabad belongs to, or None.

        Only synthesized shabad ids (Dasam Granth, Ardaas — no upstream
        ``sttm_id``) are resolved. SGGS shabads display via shabad-mode even
        when they're part of a bani — we don't want Japji / Sukhmani kirtan
        to silently switch STTM into Sundar Gutka mode.

        STTM Desktop's Realm DB uses different ``Banis.ID`` values than
        shabados/database, so we translate via ``_SQLITE_TO_REALM_BANI``.
        Banis absent from STTM's Realm (Asa Ki Var, Alahnia, Mundavnni)
        return None — caller will fall through to shabad-mode.
        """
        from src.matcher.offline_search import SYNTHETIC_ID_OFFSET
        if shabad_id < SYNTHETIC_ID_OFFSET:
            return None
        if shabad_id in self._bani_id_cache:
            return self._bani_id_cache[shabad_id]
        sqlite_bani_id = await asyncio.to_thread(self._query_bani_id, shabad_id)
        realm_bani_id = (
            _SQLITE_TO_REALM_BANI.get(sqlite_bani_id)
            if sqlite_bani_id is not None
            else None
        )
        self._bani_id_cache[shabad_id] = realm_bani_id
        return realm_bani_id

    def _query_bani_id(self, shabad_id: int) -> int | None:
        """Pick the Nitnem bani (shabados/database id) covering the most lines of this shabad."""
        self._ensure_db()
        from src.matcher.offline_search import SYNTHETIC_ID_OFFSET

        row = self._db.execute(
            f"""
            SELECT bl.bani_id
            FROM lines l
            JOIN shabads s ON l.shabad_id = s.id
            JOIN bani_lines bl ON bl.line_id = l.id
            WHERE COALESCE(s.sttm_id, s.order_id + {SYNTHETIC_ID_OFFSET}) = ?
            GROUP BY bl.bani_id
            ORDER BY COUNT(*) DESC, bl.bani_id
            LIMIT 1
            """,
            (shabad_id,),
        ).fetchone()
        return int(row[0]) if row else None

    def _ensure_db(self) -> None:
        """Open the shared SQLite connection lazily, once per controller."""
        from src.matcher.offline_search import _resolve_db_path
        if self._db is None:
            self._db = sqlite3.connect(str(_resolve_db_path()), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")

    async def _send_control(self, data: dict) -> bool:
        """
        Send a control command to STTM's /api/bani-control endpoint.

        NOTE: The exact payload format needs to be discovered by:
        1. Inspecting STTM source code for how 'bani-controller-data' is consumed
        2. Network inspection of the STTM mobile controller app
        3. Trial and error with test_sttm_connection.py

        The payloads here are our best guess and will need refinement.
        """
        if not self.base_url:
            connected = await self.connect()
            if not connected:
                print("[STTM HTTP] Not connected")
                return False

        payload = dict(data)
        if config.sttm.controller_pin is not None and "pin" not in payload:
            payload["pin"] = str(config.sttm.controller_pin)

        try:
            resp = await self._client.post(
                f"{self.base_url}/api/bani-control",
                json=payload,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"[STTM HTTP] Error on {self.base_url}: {e}. Rediscovering port...")
            self.base_url = None
            connected = await self.connect()
            if not connected:
                return False
            try:
                retry = await self._client.post(
                    f"{self.base_url}/api/bani-control",
                    json=payload,
                )
                return retry.status_code == 200
            except Exception as retry_error:
                print(f"[STTM HTTP] Retry failed: {retry_error}")
                return False
