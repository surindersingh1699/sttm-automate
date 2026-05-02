"""EventLogger: subscribe to PipelineOrchestrator._broadcast and write JSONL.

Each line written is:
  {"t": <float seconds from eval start>, "type": <str>, ...payload fields}

t=0 corresponds to audio_t0 (first GT slide's start_time in the source video),
so t lines up with SessionDescriptor.gt_timeline[*].start_s.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Awaitable, Callable


class EventLogger:
    """Captures every _broadcast event into an in-memory list and optional JSONL file."""

    def __init__(self, out_path: Path | None = None):
        self._out_path = out_path
        self._events: list[dict] = []
        self._t0: float = 0.0
        self._file = None

    def open(self, t0: float | None = None):
        """Start logging. t0 is the wall-clock reference (time.monotonic()) for t=0."""
        self._t0 = t0 if t0 is not None else time.monotonic()
        self._events.clear()
        if self._out_path:
            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._out_path.open("w", encoding="utf-8")

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    @property
    def events(self) -> list[dict]:
        return self._events

    def make_broadcast(self) -> Callable[[dict], Awaitable[None]]:
        """Return an async broadcast callback suitable for PipelineOrchestrator."""

        async def _broadcast(msg: dict) -> None:
            t = time.monotonic() - self._t0
            record = {"t": round(t, 3), **msg}
            self._events.append(record)
            if self._file:
                self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._file.flush()

        return _broadcast

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in self._events) + "\n",
            encoding="utf-8",
        )


def load_event_log(path: Path) -> list[dict]:
    """Load a previously saved JSONL event log."""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events
