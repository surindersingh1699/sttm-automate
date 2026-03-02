"""Shabad state tracking with SEARCHING/LOCKED state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.matcher.search import ShabadVerse


class PipelineState(Enum):
    SEARCHING = "searching"
    LOCKED = "locked"


@dataclass
class ShabadState:
    shabad_id: int
    gurmukhi: str
    unicode: str
    english: str
    current_line: int = 0
    verses: list[ShabadVerse] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "shabad_id": self.shabad_id,
            "gurmukhi": self.gurmukhi,
            "unicode": self.unicode,
            "english": self.english,
            "current_line": self.current_line,
            "total_lines": len(self.verses),
            "started_at": self.started_at.isoformat(),
        }


@dataclass
class ChallengerState:
    """Tracks a potential replacement shabad during LOCKED state."""
    shabad_id: int
    consecutive_wins: int = 0
    last_score: float = 0.0


class ShabadTracker:
    """
    State machine for shabad tracking.

    SEARCHING: No shabad locked. Look for strong match, confirm with second cycle.
    LOCKED: Shabad selected. Track line position. Switch only if challenger
            wins for K consecutive windows by a margin (anti-flap).
    """

    def __init__(self, challenger_windows: int = 3, challenger_margin: float = 0.10):
        self.state = PipelineState.SEARCHING
        self.current: ShabadState | None = None
        self.history: list[ShabadState] = []
        self._challenger_windows = challenger_windows
        self._challenger_margin = challenger_margin
        # SEARCHING: pending confirmation
        self._pending_id: int | None = None
        self._pending_confidence: float = 0.0
        # LOCKED: challenger tracking
        self._challenger: ChallengerState | None = None

    # --- SEARCHING state ---

    def try_lock(self, shabad_id: int, confidence: float, instant: bool = False) -> dict:
        """
        Called in SEARCHING state when a strong candidate is found.

        instant=True: lock immediately (high confidence, skip confirmation).
        instant=False: requires 2-cycle confirmation.
        """
        if instant:
            self._lock_shabad(shabad_id)
            self._pending_id = None
            self._pending_confidence = 0.0
            return {"action": "locked", "shabad_id": shabad_id}

        if self._pending_id is None:
            # First strong match — store as pending
            self._pending_id = shabad_id
            self._pending_confidence = confidence
            return {"action": "pending", "shabad_id": shabad_id}

        if self._pending_id == shabad_id:
            # Confirmed — same shabad matched twice, lock it
            self._lock_shabad(shabad_id)
            self._pending_id = None
            self._pending_confidence = 0.0
            return {"action": "locked", "shabad_id": shabad_id}

        # Different shabad than pending — reset to new pending
        self._pending_id = shabad_id
        self._pending_confidence = confidence
        return {"action": "pending", "shabad_id": shabad_id}

    # --- LOCKED state ---

    def update_line(self, line_index: int, line_score: float) -> dict:
        """Update the current line position within the locked shabad."""
        if not self.current:
            return {"action": "error"}
        self.current.current_line = line_index
        # Good line match — reset any challenger
        self._challenger = None
        return {
            "action": "aligned",
            "line_index": line_index,
            "line_score": round(line_score, 3),
        }

    def challenge(
        self, challenger_id: int, challenger_score: float, current_score: float
    ) -> dict:
        """
        Called in LOCKED state when a broad search finds a better match.

        Implements anti-flap: challenger must beat current by margin
        for K consecutive windows before triggering a switch.
        """
        margin = challenger_score - current_score
        if margin < self._challenger_margin:
            # Not enough margin — reset challenger
            self._challenger = None
            return {"action": "rejected", "margin": round(margin, 3)}

        # Track this challenger
        if self._challenger and self._challenger.shabad_id == challenger_id:
            self._challenger.consecutive_wins += 1
            self._challenger.last_score = challenger_score
        else:
            # New challenger replaces old one
            self._challenger = ChallengerState(
                shabad_id=challenger_id,
                consecutive_wins=1,
                last_score=challenger_score,
            )

        if self._challenger.consecutive_wins >= self._challenger_windows:
            # Challenger wins — switch
            new_id = self._challenger.shabad_id
            self._switch_to(new_id)
            return {"action": "switched", "new_shabad_id": new_id}

        return {
            "action": "challenging",
            "challenger_id": challenger_id,
            "wins": self._challenger.consecutive_wins,
            "needed": self._challenger_windows,
        }

    # --- Shared ---

    def _lock_shabad(self, shabad_id: int):
        """Lock a shabad (transition to LOCKED state)."""
        if self.current:
            self.history.append(self.current)
        self.current = ShabadState(
            shabad_id=shabad_id,
            gurmukhi="",
            unicode="",
            english="",
        )
        self.state = PipelineState.LOCKED
        self._challenger = None

    def _switch_to(self, new_shabad_id: int):
        """Archive current shabad and go back to SEARCHING for confirmation."""
        if self.current:
            self.history.append(self.current)
        self.current = None
        self.state = PipelineState.SEARCHING
        self._challenger = None
        # Pre-seed pending so the new shabad only needs one more confirmation
        self._pending_id = new_shabad_id
        self._pending_confidence = 0.0

    def set_shabad_details(
        self, gurmukhi: str, unicode: str, english: str, verses: list[ShabadVerse]
    ):
        """Update the current shabad's display text and cached verses."""
        if self.current:
            self.current.gurmukhi = gurmukhi
            self.current.unicode = unicode
            self.current.english = english
            self.current.verses = verses

    def advance_line(self):
        """Move to the next line in the current shabad."""
        if self.current:
            self.current.current_line += 1

    def set_line(self, line: int):
        """Jump to a specific line in the current shabad."""
        if self.current:
            self.current.current_line = line

    def manual_lock(self, shabad_id: int):
        """Force-lock a shabad from the dashboard (bypasses state machine)."""
        self._lock_shabad(shabad_id)

    def recall_from_history(self, shabad_id: int) -> bool:
        """Bring a shabad from history back as the current shabad."""
        for i, state in enumerate(self.history):
            if state.shabad_id == shabad_id:
                if self.current:
                    self.history.append(self.current)
                self.current = self.history.pop(i)
                self.state = PipelineState.LOCKED
                self._challenger = None
                self._pending_id = None
                return True
        return False

    def get_history_list(self) -> list[dict]:
        """Get history as a list of dicts for the dashboard."""
        return [s.to_dict() for s in reversed(self.history)]

    def reset(self):
        """Clear all state for a new session."""
        self.state = PipelineState.SEARCHING
        self.current = None
        self.history.clear()
        self._pending_id = None
        self._pending_confidence = 0.0
        self._challenger = None
