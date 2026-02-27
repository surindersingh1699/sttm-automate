"""Shabad state tracking: current shabad, line position, history, transitions."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ShabadState:
    shabad_id: int
    gurmukhi: str
    unicode: str
    english: str
    current_line: int = 0
    started_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "shabad_id": self.shabad_id,
            "gurmukhi": self.gurmukhi,
            "unicode": self.unicode,
            "english": self.english,
            "current_line": self.current_line,
            "started_at": self.started_at.isoformat(),
        }


class ShabadTracker:
    """
    Tracks the currently active shabad, detects transitions, and maintains history.

    Transition logic: Requires N consecutive matches to a different shabad
    before switching, to prevent flicker from noisy recognition.
    """

    def __init__(self, transition_threshold: int = 3):
        self.current: ShabadState | None = None
        self.history: list[ShabadState] = []
        self._transition_threshold = transition_threshold
        self._transition_buffer: deque[int] = deque(maxlen=transition_threshold)

    def update(self, matched_shabad_id: int, confidence: float) -> dict:
        """
        Update tracker with a new match result.

        Returns a dict describing what happened:
        - {"action": "same"} — still on the same shabad
        - {"action": "switched", "new_shabad_id": int} — transitioned to new shabad
        - {"action": "tracking", "count": int} — building confidence for a switch
        - {"action": "started", "shabad_id": int} — first shabad of session
        """
        if self.current is None:
            # First shabad of the session
            self.current = ShabadState(
                shabad_id=matched_shabad_id,
                gurmukhi="",
                unicode="",
                english="",
            )
            self._transition_buffer.clear()
            return {"action": "started", "shabad_id": matched_shabad_id}

        if matched_shabad_id == self.current.shabad_id:
            # Same shabad — reset transition counter
            self._transition_buffer.clear()
            return {"action": "same"}

        # Different shabad — track transition
        self._transition_buffer.append(matched_shabad_id)

        # Check if we have enough consecutive matches to the same new shabad
        if len(self._transition_buffer) >= self._transition_threshold:
            recent = list(self._transition_buffer)
            if len(set(recent)) == 1:
                # All recent matches agree on the new shabad — switch!
                new_id = recent[0]
                self._switch_to(new_id)
                return {"action": "switched", "new_shabad_id": new_id}

        return {
            "action": "tracking",
            "count": len(self._transition_buffer),
            "needed": self._transition_threshold,
        }

    def _switch_to(self, new_shabad_id: int):
        """Archive current shabad and switch to a new one."""
        if self.current:
            self.history.append(self.current)
        self.current = ShabadState(
            shabad_id=new_shabad_id,
            gurmukhi="",
            unicode="",
            english="",
        )
        self._transition_buffer.clear()

    def set_shabad_details(self, gurmukhi: str, unicode: str, english: str):
        """Update the current shabad's display text (after fetching from BaniDB)."""
        if self.current:
            self.current.gurmukhi = gurmukhi
            self.current.unicode = unicode
            self.current.english = english

    def advance_line(self):
        """Move to the next line in the current shabad."""
        if self.current:
            self.current.current_line += 1

    def set_line(self, line: int):
        """Jump to a specific line in the current shabad."""
        if self.current:
            self.current.current_line = line

    def recall_from_history(self, shabad_id: int) -> bool:
        """
        Bring a shabad from history back as the current shabad.
        Returns True if found and switched, False if not in history.
        """
        for i, state in enumerate(self.history):
            if state.shabad_id == shabad_id:
                # Archive current, restore from history
                if self.current:
                    self.history.append(self.current)
                self.current = self.history.pop(i)
                self._transition_buffer.clear()
                return True
        return False

    def get_history_list(self) -> list[dict]:
        """Get history as a list of dicts for the dashboard."""
        return [s.to_dict() for s in reversed(self.history)]

    def reset(self):
        """Clear all state for a new session."""
        self.current = None
        self.history.clear()
        self._transition_buffer.clear()
