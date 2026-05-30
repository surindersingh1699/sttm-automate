"""Abstract base for STTM controllers."""

from abc import ABC, abstractmethod


class STTMController(ABC):
    """Interface for controlling SikhiToTheMax Desktop."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to STTM. Returns True on success."""
        ...

    @abstractmethod
    async def search_shabad(self, query: str) -> bool:
        """Search for a shabad using first-letter query. Returns True on success."""
        ...

    @abstractmethod
    async def select_result(self, index: int = 0) -> bool:
        """Select a search result by index. Returns True on success."""
        ...

    @abstractmethod
    async def display_shabad(self, shabad_id: int) -> bool:
        """Display a specific shabad by ID. Returns True on success."""
        ...

    @abstractmethod
    async def navigate_line(self, direction: str = "next") -> bool:
        """Navigate to next/previous line. Direction: 'next' or 'prev'. Returns True on success."""
        ...

    async def display_bani(self, sqlite_bani_id: int) -> bool:
        """Open a nitnem bani (gutka follow-along mode). Default no-op for
        controllers that don't support bani-level display (e.g. the
        Playwright fallback). Returns True on success.
        """
        return False

    async def navigate_to_bani_verse(self, verse_order_id: int) -> bool:
        """Highlight a specific line of the currently-displayed bani.
        ``verse_order_id`` is ``lines.order_id`` from the local SQLite DB —
        the same value STTM Desktop accepts as ``verseId``. Default no-op
        for controllers without bani support.
        """
        return False

    async def navigate_to_bani_line(
        self,
        pointer_id: int,
        *,
        line_count: int | None = None,
        fallback_verse_order_id: int | None = None,
    ) -> bool:
        """Highlight a Gutka line by STTM's bani-mode pointer id.

        ``pointer_id`` should be Realm ``Banis_Shabad.ID`` when available.
        ``fallback_verse_order_id`` lets older callers/controllers degrade to
        the legacy local verse id path.
        """
        if fallback_verse_order_id is None:
            fallback_verse_order_id = pointer_id
        return await self.navigate_to_bani_verse(fallback_verse_order_id)

    @abstractmethod
    async def disconnect(self):
        """Disconnect from STTM."""
        ...
