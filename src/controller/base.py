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

    @abstractmethod
    async def disconnect(self):
        """Disconnect from STTM."""
        ...
