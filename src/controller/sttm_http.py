"""Control STTM Desktop via HTTP POST to its Express server."""

import httpx

from src.config import config
from src.controller.base import STTMController


class STTMHttpController(STTMController):
    """
    Controls STTM Desktop by sending HTTP requests to its local Express server.

    STTM Desktop runs Express.js on one of several known ports.
    It exposes POST /api/bani-control which sends data to the Electron main window.
    """

    def __init__(self):
        self.base_url: str | None = None
        self._client = httpx.AsyncClient(timeout=config.sttm.connect_timeout)

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
        """Send a search command to STTM."""
        return await self._send_control({
            "type": "search",
            "query": query,
        })

    async def select_result(self, index: int = 0) -> bool:
        """Select a search result."""
        return await self._send_control({
            "type": "select",
            "index": index,
        })

    async def display_shabad(self, shabad_id: int) -> bool:
        """Display a shabad by its ID."""
        return await self._send_control({
            "type": "shabad",
            "shabadId": shabad_id,
        })

    async def navigate_line(self, direction: str = "next") -> bool:
        """Navigate lines within current shabad."""
        return await self._send_control({
            "type": "navigate",
            "direction": direction,
        })

    async def disconnect(self):
        """Close the HTTP client."""
        await self._client.aclose()

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
            print("[STTM HTTP] Not connected")
            return False

        try:
            resp = await self._client.post(
                f"{self.base_url}/api/bani-control",
                json=data,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"[STTM HTTP] Error: {e}")
            return False
