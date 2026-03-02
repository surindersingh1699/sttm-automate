"""Shabad state tracking with SEARCHING/CANDIDATE_LOCK/LOCKED/UNSTABLE_LOCK states."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.matcher.search import ShabadVerse


class PipelineState(Enum):
    SEARCHING = "searching"
    CANDIDATE_LOCK = "candidate_lock"
    LOCKED = "locked"
    UNSTABLE_LOCK = "unstable_lock"


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
    """Tracks a potential replacement shabad during LOCKED/UNSTABLE_LOCK."""

    shabad_id: int
    consecutive_wins: int = 0
    last_score: float = 0.0


@dataclass
class HypothesisState:
    """Decaying evidence for a shabad hypothesis."""

    shabad_id: int
    line_idx: int = 0
    score: float = 0.0
    cumulative_score: float = 0.0
    stability: int = 0
    last_seen_window: int = 0

    @property
    def evidence_score(self) -> float:
        # Favor cumulative evidence with a small stability bump.
        return min(1.0, self.cumulative_score + min(0.2, self.stability * 0.03))

    def to_dict(self) -> dict:
        return {
            "shabad_id": self.shabad_id,
            "line_idx": self.line_idx,
            "score": round(self.score, 3),
            "cumulative_score": round(self.cumulative_score, 3),
            "stability": self.stability,
            "evidence_score": round(self.evidence_score, 3),
            "last_seen_window": self.last_seen_window,
        }


class ShabadTracker:
    """
    State machine for shabad tracking.

    SEARCHING: No shabad locked.
    CANDIDATE_LOCK: Candidate confirmation windows.
    LOCKED: Stable shabad lock.
    UNSTABLE_LOCK: Locked but weak/noisy alignment; prefer recovery before switch.
    """

    def __init__(
        self,
        challenger_windows: int = 3,
        challenger_margin: float = 0.10,
        candidate_lock_windows: int = 2,
        hypothesis_top_k: int = 5,
        hypothesis_ttl_windows: int = 2,
        hypothesis_decay: float = 0.85,
    ):
        self.state = PipelineState.SEARCHING
        self.current: ShabadState | None = None
        self.history: list[ShabadState] = []
        self._challenger_windows = challenger_windows
        self._challenger_margin = challenger_margin
        self._candidate_lock_windows = max(1, candidate_lock_windows)
        self._hypothesis_top_k = max(1, hypothesis_top_k)
        self._hypothesis_ttl_windows = max(1, hypothesis_ttl_windows)
        self._hypothesis_decay = min(0.99, max(0.5, hypothesis_decay))
        self._pending_id: int | None = None
        self._pending_confidence: float = 0.0
        self._pending_wins: int = 0
        self._challenger: ChallengerState | None = None
        self._hypotheses: dict[int, HypothesisState] = {}

    def set_policy(
        self,
        challenger_windows: int,
        challenger_margin: float,
        candidate_lock_windows: int,
    ):
        """Update runtime policy (used by confidence mode changes)."""
        self._challenger_windows = max(1, challenger_windows)
        self._challenger_margin = max(0.01, challenger_margin)
        self._candidate_lock_windows = max(1, candidate_lock_windows)

    # --- Hypothesis layer ---

    def observe_candidates(self, candidates: list[dict], window_idx: int):
        """Update top-K decaying hypotheses from this window's candidates."""
        # Global decay each window.
        for hypothesis in self._hypotheses.values():
            hypothesis.cumulative_score *= self._hypothesis_decay

        for candidate in candidates[: self._hypothesis_top_k]:
            shabad_id = int(candidate.get("shabad_id", 0))
            if shabad_id <= 0:
                continue
            score = float(candidate.get("score", 0.0))
            line_idx = int(candidate.get("line_idx", 0))
            if shabad_id in self._hypotheses:
                hypothesis = self._hypotheses[shabad_id]
                hypothesis.score = score
                hypothesis.cumulative_score += score
                hypothesis.stability += 1
                hypothesis.line_idx = line_idx
                hypothesis.last_seen_window = window_idx
            else:
                self._hypotheses[shabad_id] = HypothesisState(
                    shabad_id=shabad_id,
                    line_idx=line_idx,
                    score=score,
                    cumulative_score=score,
                    stability=1,
                    last_seen_window=window_idx,
                )

        stale = [
            shabad_id
            for shabad_id, hypothesis in self._hypotheses.items()
            if window_idx - hypothesis.last_seen_window > self._hypothesis_ttl_windows
        ]
        for shabad_id in stale:
            self._hypotheses.pop(shabad_id, None)

        ranked = sorted(
            self._hypotheses.values(),
            key=lambda hypothesis: hypothesis.evidence_score,
            reverse=True,
        )
        self._hypotheses = {
            hypothesis.shabad_id: hypothesis
            for hypothesis in ranked[: self._hypothesis_top_k]
        }

    def get_hypotheses(self) -> list[dict]:
        """Expose ranked hypotheses for dashboard/debugging."""
        ranked = sorted(
            self._hypotheses.values(),
            key=lambda hypothesis: hypothesis.evidence_score,
            reverse=True,
        )
        return [hypothesis.to_dict() for hypothesis in ranked]

    def best_hypothesis(self) -> dict | None:
        """Get highest-evidence hypothesis."""
        hypotheses = self.get_hypotheses()
        return hypotheses[0] if hypotheses else None

    # --- SEARCHING / CANDIDATE_LOCK ---

    def try_lock(self, shabad_id: int, confidence: float, instant: bool = False) -> dict:
        """
        Called when a lockable candidate is found.

        instant=True: lock immediately.
        instant=False: use CANDIDATE_LOCK persistence windows.
        """
        if instant:
            self._lock_shabad(shabad_id)
            self._clear_pending()
            return {"action": "locked", "shabad_id": shabad_id}

        if self.state not in (PipelineState.SEARCHING, PipelineState.CANDIDATE_LOCK):
            # When already locked, searching lock flow should not run.
            return {"action": "ignored"}

        if self._pending_id == shabad_id:
            self._pending_wins += 1
            self._pending_confidence = confidence
        else:
            self._pending_id = shabad_id
            self._pending_confidence = confidence
            self._pending_wins = 1

        self.state = PipelineState.CANDIDATE_LOCK
        if self._pending_wins >= self._candidate_lock_windows:
            self._lock_shabad(shabad_id)
            self._clear_pending()
            return {"action": "locked", "shabad_id": shabad_id}

        return {
            "action": "pending",
            "shabad_id": shabad_id,
            "wins": self._pending_wins,
            "needed": self._candidate_lock_windows,
        }

    def clear_candidate_lock(self):
        """Drop pending candidate and return to SEARCHING if no current lock exists."""
        self._clear_pending()
        if not self.current:
            self.state = PipelineState.SEARCHING

    # --- LOCKED / UNSTABLE_LOCK ---

    def update_line(self, line_index: int, line_score: float) -> dict:
        """Update current line and mark lock as stable."""
        if not self.current:
            return {"action": "error"}
        self.current.current_line = line_index
        self.state = PipelineState.LOCKED
        self._challenger = None
        return {
            "action": "aligned",
            "line_index": line_index,
            "line_score": round(line_score, 3),
        }

    def mark_unstable(self):
        """Mark active lock as unstable while attempting in-shabad recovery."""
        if self.current:
            self.state = PipelineState.UNSTABLE_LOCK

    def mark_stable(self):
        """Return unstable lock to stable LOCKED state."""
        if self.current and self.state == PipelineState.UNSTABLE_LOCK:
            self.state = PipelineState.LOCKED

    def challenge(
        self, challenger_id: int, challenger_score: float, current_score: float
    ) -> dict:
        """
        Evaluate challenger with margin + persistence rule.
        """
        margin = challenger_score - current_score
        if margin < self._challenger_margin:
            self._challenger = None
            return {"action": "rejected", "margin": round(margin, 3)}

        if self._challenger and self._challenger.shabad_id == challenger_id:
            self._challenger.consecutive_wins += 1
            self._challenger.last_score = challenger_score
        else:
            self._challenger = ChallengerState(
                shabad_id=challenger_id,
                consecutive_wins=1,
                last_score=challenger_score,
            )

        if self._challenger.consecutive_wins >= self._challenger_windows:
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
        if self.current:
            self.history.append(self.current)
        self.current = None
        self.state = PipelineState.CANDIDATE_LOCK
        self._challenger = None
        self._pending_id = new_shabad_id
        self._pending_confidence = 0.0
        self._pending_wins = 1

    def _clear_pending(self):
        self._pending_id = None
        self._pending_confidence = 0.0
        self._pending_wins = 0

    def set_shabad_details(
        self, gurmukhi: str, unicode: str, english: str, verses: list[ShabadVerse]
    ):
        if self.current:
            self.current.gurmukhi = gurmukhi
            self.current.unicode = unicode
            self.current.english = english
            self.current.verses = verses

    def advance_line(self):
        if self.current:
            self.current.current_line += 1

    def set_line(self, line: int):
        if self.current:
            self.current.current_line = line

    def manual_lock(self, shabad_id: int):
        self._lock_shabad(shabad_id)
        self._clear_pending()

    def release_lock(self):
        if self.current:
            self.history.append(self.current)
        self.current = None
        self.state = PipelineState.SEARCHING
        self._challenger = None
        self._clear_pending()

    def recall_from_history(self, shabad_id: int) -> bool:
        for i, state in enumerate(self.history):
            if state.shabad_id == shabad_id:
                if self.current:
                    self.history.append(self.current)
                self.current = self.history.pop(i)
                self.state = PipelineState.LOCKED
                self._challenger = None
                self._clear_pending()
                return True
        return False

    def get_history_list(self) -> list[dict]:
        return [state.to_dict() for state in reversed(self.history)]

    def reset(self):
        self.state = PipelineState.SEARCHING
        self.current = None
        self.history.clear()
        self._challenger = None
        self._clear_pending()
        self._hypotheses.clear()
