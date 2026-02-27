"""Control STTM Desktop via Playwright browser automation (fallback)."""

from playwright.async_api import async_playwright, Browser, Page

from src.config import config
from src.controller.base import STTMController


class STTMPlaywrightController(STTMController):
    """
    Controls STTM Desktop by automating its Electron UI via Chrome DevTools Protocol.

    Prerequisites:
    - STTM Desktop must be launched with --remote-debugging-port=9222
    - On macOS: open -a "SikhiToTheMax" --args --remote-debugging-port=9222
    """

    def __init__(self):
        self._pw = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def connect(self) -> bool:
        """Connect to STTM Desktop's Electron renderer via CDP."""
        try:
            self._pw = await async_playwright().start()
            cdp_url = f"http://localhost:{config.sttm.cdp_port}"
            self._browser = await self._pw.chromium.connect_over_cdp(cdp_url)
            contexts = self._browser.contexts
            if contexts and contexts[0].pages:
                self._page = contexts[0].pages[0]
                print(f"[STTM Playwright] Connected via CDP on port {config.sttm.cdp_port}")
                return True
            print("[STTM Playwright] Connected but no pages found")
            return False
        except Exception as e:
            print(f"[STTM Playwright] Connection failed: {e}")
            print(f"  Make sure STTM is running with --remote-debugging-port={config.sttm.cdp_port}")
            return False

    async def search_shabad(self, query: str) -> bool:
        """Type the search query into STTM's search input."""
        if not self._page:
            return False
        try:
            # STTM's search input — selector may need adjustment based on STTM version
            search_input = self._page.locator("input.search-field, #search-field, input[type='search']").first
            await search_input.click()
            await search_input.fill("")
            await search_input.type(query, delay=50)
            # Wait briefly for results to load
            await self._page.wait_for_timeout(500)
            return True
        except Exception as e:
            print(f"[STTM Playwright] Search failed: {e}")
            return False

    async def select_result(self, index: int = 0) -> bool:
        """Click on a search result."""
        if not self._page:
            return False
        try:
            results = self._page.locator(".search-results .result-row, .search-result")
            count = await results.count()
            if count > index:
                await results.nth(index).click()
                return True
            print(f"[STTM Playwright] Only {count} results, can't select index {index}")
            return False
        except Exception as e:
            print(f"[STTM Playwright] Select failed: {e}")
            return False

    async def display_shabad(self, shabad_id: int) -> bool:
        """
        Display a shabad by ID.
        This searches for the shabad and selects the first result.
        For direct ID-based display, the HTTP controller may be more appropriate.
        """
        # Use search as a proxy — Playwright doesn't have direct ID access
        return await self.search_shabad(str(shabad_id))

    async def navigate_line(self, direction: str = "next") -> bool:
        """Navigate lines using keyboard arrows."""
        if not self._page:
            return False
        try:
            key = "ArrowDown" if direction == "next" else "ArrowUp"
            await self._page.keyboard.press(key)
            return True
        except Exception as e:
            print(f"[STTM Playwright] Navigate failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from STTM."""
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        self._page = None
        self._browser = None
        self._pw = None
