"""Control STTM Desktop via HTTP POST to its Express server."""

import httpx

from src.config import config
from src.controller.base import STTMController

_BANIDB_API_BASE = "https://api.banidb.com/v2"


class STTMHttpController(STTMController):
    """
    Controls STTM Desktop by sending HTTP requests to its local Express server.

    STTM Desktop runs Express.js on one of several known ports.
    It exposes POST /api/bani-control which sends data to the Electron main window.
    """

    def __init__(self):
        self.base_url: str | None = None
        self._client = httpx.AsyncClient(timeout=config.sttm.connect_timeout)
        self._banidb = httpx.AsyncClient(base_url=_BANIDB_API_BASE, timeout=8.0)
        self._active_shabad_id: int | None = None
        self._first_verse_cache: dict[int, int] = {}

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
        """Display a shabad by its ID."""
        first_verse_id = await self._get_first_verse_id(shabad_id)
        ok = await self._send_control({
            "type": "shabad",
            "shabadId": shabad_id,
            # Compatibility fields observed in STTM desktop internals.
            "id": shabad_id,
            "verseId": first_verse_id,
            "lineCount": 1,
            "highlight": first_verse_id,
            "homeId": first_verse_id,
        })
        if ok:
            self._active_shabad_id = shabad_id
        return ok

    async def navigate_line(self, direction: str = "next") -> bool:
        """
        STTM's remote listener does not handle a dedicated 'navigate' message type.
        Sending unknown types can destabilize the remote controller UI, so this is a safe no-op.
        """
        return self._active_shabad_id is not None

    async def disconnect(self):
        """Close the HTTP client."""
        await self._client.aclose()
        await self._banidb.aclose()

    async def _get_first_verse_id(self, shabad_id: int) -> int:
        """Resolve and cache the first verseId for a shabad (needed by STTM controller payload)."""
        cached = self._first_verse_cache.get(shabad_id)
        if cached is not None:
            return cached
        try:
            resp = await self._banidb.get(f"/shabads/{shabad_id}")
            resp.raise_for_status()
            data = resp.json()
            verses = data.get("verses", [])
            if verses:
                verse_id = int(verses[0].get("verseId", 1))
                self._first_verse_cache[shabad_id] = verse_id
                return verse_id
        except Exception as e:
            print(f"[STTM HTTP] Could not resolve verseId for shabad {shabad_id}: {e}")
        return 1

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
