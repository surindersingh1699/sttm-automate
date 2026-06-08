"""Monotonic streaming aligner for gutka follow-along mode.

Given a fixed bani's ordered line sequence (from :mod:`src.matcher.bani_loader`),
take ASR first-letter strings window-by-window and decide which line of the
bani the reciter is currently on. Replaces the open-ended 9-strategy
``OfflineShabadSearcher`` + lock state machine when the user has explicitly
selected a bani up front.

Two operating modes (runtime-toggleable):

- **Forward-only** (default): aligner can ADVANCE to a later line or REPEAT
  the current line (rahau-friendly), but never goes back. On low-confidence
  windows it HOLDs — the highlight stays where it is. Matches paper-gutka
  reading: you don't unread lines.
- **Bidirectional**: aligner can also BACK up a few lines if a stronger match
  shows up behind the current position. Useful if the reciter restarted from
  an earlier pauri.

Matching is cheap: for each candidate start line in a small window around the
current position, score one-line and multi-line spans. Fast paath often emits
partial current line + next line in one ASR chunk, so a single-line scorer
lags. The selected span's end line becomes the pointer target. The best score
has to clear ``min_score`` AND beat the runner-up by ``min_margin`` to commit
a move. Otherwise the aligner HOLDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.matcher.bani_loader import GutkaLine


class AlignAction(str, Enum):
    ADVANCE = "advance"  # moved forward to a later line
    REPEAT = "repeat"    # confident match on the same line (rahau / re-fired)
    BACK = "back"        # bidirectional only — moved to an earlier line
    HOLD = "hold"        # no confident match in window — position unchanged


@dataclass
class AlignmentDecision:
    action: AlignAction
    new_idx: int                # the line index after this decision
    score: float                # best candidate's match score (0..1)
    runner_up_score: float      # second-best score, for telemetry
    asr_first_letters: str      # what we scored against (echoed for logging)


def _lcs_len(a: str, b: str) -> int:
    """Length of the longest common subsequence of two short strings.

    O(len(a) * len(b)) DP. Bani lines are short (≤30 chars FL); ASR windows
    are short (≤60 chars FL). At ~6 candidates per window this is well under
    a millisecond — far cheaper than the existing 9-strategy SQLite search.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _coverage_score(asr_fl: str, line_fl: str) -> float:
    """Fraction of the line's first-letters present (in order) in the ASR FL.

    1.0 means every letter of the line shows up as a subsequence of the ASR
    output; 0.0 means none do. Normalising by ``len(line_fl)`` (rather than
    ``min(len, len)``) penalises ASR windows that are shorter than the line
    — those don't carry enough signal to confidently commit a move.
    """
    if not line_fl:
        return 0.0
    if not asr_fl:
        return 0.0
    return _lcs_len(asr_fl, line_fl) / len(line_fl)


def _span_score(asr_fl: str, span_fl: str) -> float:
    """Score ASR first letters against a line/span first-letter sequence.

    Combines two signals:
      * how much of the candidate span was covered
      * how much of the ASR output was explained

    This lets a short but clean chunk move promptly once it carries enough
    evidence, while still preferring full-span matches for multi-line chunks.
    """
    if not asr_fl or not span_fl:
        return 0.0
    matched = _lcs_len(asr_fl, span_fl)
    span_cov = matched / max(1, len(span_fl))
    asr_cov = matched / max(1, len(asr_fl))
    return (0.65 * span_cov) + (0.35 * asr_cov)


class GutkaAligner:
    """Streaming monotonic aligner over one bani's first-letter sequence."""

    def __init__(
        self,
        lines: list[GutkaLine],
        *,
        bidirectional: bool = False,
        forward_window: int = 6,
        backward_window: int = 2,
        min_score: float = 0.55,
        min_margin: float = 0.08,
        max_span_lines: int = 3,
        min_matched_letters: int = 2,
        rahau_repeat_enabled: bool = True,
    ):
        if not lines:
            raise ValueError("GutkaAligner needs at least one line")
        self.lines = lines
        self.bidirectional = bidirectional
        self.forward_window = max(1, forward_window)
        self.backward_window = max(0, backward_window)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.max_span_lines = max(1, int(max_span_lines))
        self.min_matched_letters = max(1, int(min_matched_letters))
        self.rahau_repeat_enabled = bool(rahau_repeat_enabled)
        self.current_idx = 0

    def reset(self, idx: int = 0) -> None:
        self.current_idx = max(0, min(idx, len(self.lines) - 1))

    def set_bidirectional(self, enabled: bool) -> None:
        self.bidirectional = bool(enabled)

    @property
    def current_line(self) -> GutkaLine:
        return self.lines[self.current_idx]

    def update(self, asr_first_letters_ascii: str) -> AlignmentDecision:
        """Consume one ASR first-letter window, return the alignment decision.

        Side effect: ``self.current_idx`` is updated for ADVANCE / BACK /
        REPEAT (REPEAT keeps idx but signals a confident re-hit).
        """
        asr_fl = (asr_first_letters_ascii or "").strip()
        if not asr_fl:
            return AlignmentDecision(
                AlignAction.HOLD, self.current_idx, 0.0, 0.0, asr_fl
            )

        lo = self.current_idx
        if self.bidirectional:
            lo = max(0, self.current_idx - self.backward_window)
        hi = min(len(self.lines) - 1, self.current_idx + self.forward_window)

        # Score one-line and multi-line spans. Candidate target is the span's
        # end line, so a chunk containing two panktis can catch the pointer up
        # immediately instead of landing on the first line and waiting.
        scored: list[tuple[int, float, int, int]] = []
        for start_idx in range(lo, hi + 1):
            span_parts: list[str] = []
            max_end = min(len(self.lines) - 1, start_idx + self.max_span_lines - 1)
            for end_idx in range(start_idx, max_end + 1):
                span_parts.append(self.lines[end_idx].first_letters_ascii)
                span_fl = "".join(span_parts)
                matched = _lcs_len(asr_fl, span_fl)
                score = _span_score(asr_fl, span_fl)
                if matched < self.min_matched_letters:
                    score = 0.0
                scored.append((end_idx, score, start_idx, end_idx - start_idx + 1))

        scored.sort(
            key=lambda kv: (
                -kv[1],
                abs(kv[0] - self.current_idx),
                kv[3],
                kv[2],
                kv[0],
            )
        )

        best_idx, best_score, _best_start, _best_span_len = scored[0]
        runner_up = scored[1][1] if len(scored) > 1 else 0.0

        confident = (
            best_score >= self.min_score
            and (best_score - runner_up) >= self.min_margin
        )
        if best_idx == self.current_idx:
            next_idx = self.current_idx + 1
            if next_idx < len(self.lines):
                next_fl = self.lines[next_idx].first_letters_ascii
                next_matched = _lcs_len(asr_fl, next_fl)
                next_coverage = (
                    next_matched / max(1, len(next_fl))
                    if next_fl
                    else 0.0
                )
                # Rolling Gutka windows often contain all of the current line
                # plus the first syllables of the next. The current line can
                # still win or tie the absolute score; the next-line evidence
                # rule keeps the pointer moving once enough of that next line
                # is already present.
                if (
                    next_matched >= self.min_matched_letters
                    and next_coverage >= (1.0 / 3.0)
                    and next_fl != self.lines[self.current_idx].first_letters_ascii
                    and best_score >= self.min_score
                ):
                    self.current_idx = next_idx
                    return AlignmentDecision(
                        AlignAction.ADVANCE,
                        next_idx,
                        max(best_score, next_coverage),
                        runner_up,
                        asr_fl,
                    )
        if not confident:
            return AlignmentDecision(
                AlignAction.HOLD, self.current_idx, best_score, runner_up, asr_fl
            )

        if best_idx > self.current_idx:
            self.current_idx = best_idx
            return AlignmentDecision(
                AlignAction.ADVANCE, best_idx, best_score, runner_up, asr_fl
            )
        if best_idx < self.current_idx:
            # Only reachable in bidirectional mode (lo = current_idx in
            # forward-only mode, so best_idx >= current_idx always).
            self.current_idx = best_idx
            return AlignmentDecision(
                AlignAction.BACK, best_idx, best_score, runner_up, asr_fl
            )
        # best_idx == current_idx — same line scored highest again.
        if self.rahau_repeat_enabled:
            return AlignmentDecision(
                AlignAction.REPEAT, best_idx, best_score, runner_up, asr_fl
            )
        return AlignmentDecision(
            AlignAction.HOLD, best_idx, best_score, runner_up, asr_fl
        )
