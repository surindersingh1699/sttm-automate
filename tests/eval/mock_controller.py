"""Mock STTM controller that records all calls with virtual timestamps."""

from __future__ import annotations

import time
from dataclasses import dataclass
from src.controller.base import STTMController


@dataclass
class ControlEvent:
    method: str
    kwargs: dict
    t_ms: float  # virtual elapsed ms since sequence start


class MockSTTMController(STTMController):
    """Records every controller call so eval can verify what STTM would display."""

    def __init__(self):
        self.events: list[ControlEvent] = []
        self._t0: float = time.monotonic()

    def reset(self):
        self.events.clear()
        self._t0 = time.monotonic()

    def _record(self, method: str, **kwargs):
        self.events.append(ControlEvent(
            method=method,
            kwargs=kwargs,
            t_ms=(time.monotonic() - self._t0) * 1000,
        ))

    async def connect(self) -> bool:
        return True

    async def disconnect(self):
        pass

    async def search_shabad(self, query: str) -> bool:
        self._record("search_shabad", query=query)
        return True

    async def select_result(self, index: int = 0) -> bool:
        self._record("select_result", index=index)
        return True

    async def display_shabad(self, shabad_id: int) -> bool:
        self._record("display_shabad", shabad_id=shabad_id)
        return True

    async def navigate_line(self, direction: str = "next") -> bool:
        self._record("navigate_line", direction=direction)
        return True

    async def navigate_to_line(self, target_idx: int) -> bool:
        self._record("navigate_to_line", target_idx=target_idx)
        return True
