"""Prototype: soft-beam Bayesian filter over (shabad_id, line_idx).

Each beam entry is a hypothesis with a weight (unnormalized log-prob). On every
observation:

  1. Predict — spread each hypothesis via transition prior (stay/advance/jump).
  2. Inject — add fresh shabad candidates from corpus.search() results.
  3. Update — multiply by observation likelihood = exp(T · overlap_score).
  4. Normalise, prune to top-K.

Lock decisions are emergent:
  - LOCKED if top shabad has >= LOCK_MASS posterior for >= LOCK_PERSIST windows
  - SWITCH if a different shabad rises to >= LOCK_MASS for LOCK_PERSIST windows
  - Line pointer = highest-mass line within locked shabad
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from tests.eval.proto_corpus import Corpus, search


# ── tunables (kept tiny — no per-strategy weights) ──────────────────────────

BEAM_SIZE = 60
RETRIEVAL_TOP_K = 30

# Transition prior P(next_line | current_line). Indices are deltas.
TRANS_STAY = 0.50
TRANS_ADVANCE_1 = 0.32
TRANS_ADVANCE_2 = 0.06
TRANS_BACK = 0.04          # ragi returned to chorus
TRANS_TO_LINE_0 = 0.04     # jump to chorus / refrain
# Implicit: remainder mass becomes P(switch_shabad), allocated to fresh hits.

# Observation likelihood temperature: P(obs|line) ∝ exp(T · overlap)
OBS_TEMPERATURE = 8.0

# Floor for retrieval injection — anything below ignored
RETRIEVAL_MIN_OVERLAP = 0.18

# Posterior for fresh injections (un-normalised log weight)
INJECTION_LOG_PRIOR = -2.5

# Lock decision thresholds
LOCK_MASS = 0.45          # lock if posterior >= this for LOCK_PERSIST windows
LOCK_PERSIST = 2
SWITCH_MASS = 0.55

# Instant lock — fires on a single window when posterior is overwhelming
INSTANT_LOCK_MASS = 0.70
# And the top retrieved overlap must be strong (avoids fluke posteriors)
INSTANT_LOCK_MIN_OVERLAP = 0.55


@dataclass
class Hypothesis:
    shabad_id: int
    line_idx: int
    log_w: float


@dataclass
class FilterEvent:
    """Standard event emitted by replay — same shape as orchestrator output."""
    t: float
    type: str
    payload: dict = field(default_factory=dict)


LINE_PERSIST = 2          # require 2 consecutive windows agreeing on new line
LINE_MASS_FLOOR = 0.20    # don't update line if best line mass below this


@dataclass
class BeamFilter:
    corpus: Corpus
    beam: list[Hypothesis] = field(default_factory=list)
    locked_shabad: int | None = None
    locked_line: int | None = None
    persist_count: int = 0
    last_top_shabad: int | None = None
    history: list[int] = field(default_factory=list)   # locked shabads in order
    # Line-stability counters
    pending_line: int | None = None
    pending_line_count: int = 0

    # ---------- helpers ----------

    def _normalise(self):
        if not self.beam:
            return
        max_log = max(h.log_w for h in self.beam)
        s = sum(math.exp(h.log_w - max_log) for h in self.beam)
        if s <= 0:
            return
        log_z = max_log + math.log(s)
        for h in self.beam:
            h.log_w -= log_z

    def _shabad_mass(self) -> dict[int, float]:
        """Aggregate posterior mass per shabad (probability units)."""
        out: dict[int, float] = defaultdict(float)
        for h in self.beam:
            out[h.shabad_id] += math.exp(h.log_w)
        return dict(out)

    def _prune(self):
        if len(self.beam) <= BEAM_SIZE:
            return
        self.beam.sort(key=lambda h: h.log_w, reverse=True)
        self.beam = self.beam[:BEAM_SIZE]

    # ---------- predict / inject / update ----------

    def _predict(self):
        """Apply transition kernel to current beam."""
        new_beam: dict[tuple[int, int], float] = defaultdict(lambda: -math.inf)
        for h in self.beam:
            shabad_lines = self.corpus.by_shabad.get(h.shabad_id, [])
            n_lines = len(shabad_lines)
            if n_lines == 0:
                continue
            for delta, p in (
                (0, TRANS_STAY),
                (1, TRANS_ADVANCE_1),
                (2, TRANS_ADVANCE_2),
                (-1, TRANS_BACK),
            ):
                tgt = h.line_idx + delta
                if 0 <= tgt < n_lines:
                    log_p = math.log(p)
                    key = (h.shabad_id, tgt)
                    new_beam[key] = max(new_beam[key], h.log_w + log_p)
            # Jump to chorus (line 0) when not already there
            if h.line_idx > 0:
                key = (h.shabad_id, 0)
                log_p = math.log(TRANS_TO_LINE_0)
                new_beam[key] = max(new_beam[key], h.log_w + log_p)

        self.beam = [
            Hypothesis(shabad_id=sid, line_idx=lidx, log_w=lw)
            for (sid, lidx), lw in new_beam.items()
        ]

    def _inject(self, retrieved: list[tuple[int, float]]):
        """Add fresh shabad candidates from this window's retrieval."""
        existing: set[tuple[int, int]] = {(h.shabad_id, h.line_idx) for h in self.beam}
        for nidx, _score in retrieved:
            node = self.corpus.nodes[nidx]
            key = (node.shabad_id, node.line_idx)
            if key in existing:
                continue
            self.beam.append(
                Hypothesis(
                    shabad_id=node.shabad_id,
                    line_idx=node.line_idx,
                    log_w=INJECTION_LOG_PRIOR,
                )
            )
            existing.add(key)

    def _update_observation(self, retrieved: list[tuple[int, float]]):
        """Multiply each hypothesis weight by the observation likelihood."""
        # Build score map: (shabad_id, line_idx) → overlap. Default 0.
        score_map: dict[tuple[int, int], float] = {}
        for nidx, sc in retrieved:
            node = self.corpus.nodes[nidx]
            score_map[(node.shabad_id, node.line_idx)] = sc

        # If a hypothesis line wasn't returned, give it a small floor based on
        # the best shabad-mate score (so we don't crash legitimate locks just
        # because the exact line wasn't in top-K).
        best_per_shabad: dict[int, float] = {}
        for (sid, _), sc in score_map.items():
            best_per_shabad[sid] = max(best_per_shabad.get(sid, 0.0), sc)

        for h in self.beam:
            sc = score_map.get((h.shabad_id, h.line_idx))
            if sc is None:
                # Not observed for this exact line; degrade by half the
                # shabad-best, so a multi-line shabad still has stickiness.
                sc = 0.5 * best_per_shabad.get(h.shabad_id, 0.0)
            h.log_w += OBS_TEMPERATURE * sc

    # ---------- public step ----------

    def step(self, t: float, transcript: str) -> list[FilterEvent]:
        """Advance the filter one window. Returns emitted events."""
        events: list[FilterEvent] = []

        if not transcript or not transcript.strip():
            # No-op observation: still age the beam mildly.
            for h in self.beam:
                h.log_w *= 0.95   # gentle decay so stale hypotheses fade
            return events

        retrieved = search(
            self.corpus,
            transcript,
            top_k=RETRIEVAL_TOP_K,
            min_overlap=RETRIEVAL_MIN_OVERLAP,
        )
        if not retrieved:
            return events

        if not self.beam:
            # Cold start: seed beam from retrieval directly, weighted by score.
            for nidx, sc in retrieved:
                node = self.corpus.nodes[nidx]
                self.beam.append(
                    Hypothesis(
                        shabad_id=node.shabad_id,
                        line_idx=node.line_idx,
                        log_w=OBS_TEMPERATURE * sc,
                    )
                )
        else:
            self._predict()
            self._inject(retrieved)
            self._update_observation(retrieved)

        self._normalise()
        self._prune()

        # ----- decide lock / switch -----
        masses = self._shabad_mass()
        top_sid, top_mass = max(masses.items(), key=lambda kv: kv[1])

        if self.locked_shabad is None:
            # SEARCHING / CANDIDATE phase
            top_overlap = retrieved[0][1] if retrieved else 0.0
            instant_lock = (
                top_mass >= INSTANT_LOCK_MASS
                and top_overlap >= INSTANT_LOCK_MIN_OVERLAP
            )
            if top_mass >= LOCK_MASS:
                if self.last_top_shabad == top_sid:
                    self.persist_count += 1
                else:
                    self.persist_count = 1
                self.last_top_shabad = top_sid
                if instant_lock or self.persist_count >= LOCK_PERSIST:
                    self.locked_shabad = top_sid
                    self.persist_count = 0
                    # Pick best line within the shabad
                    best_h = max(
                        (h for h in self.beam if h.shabad_id == top_sid),
                        key=lambda h: h.log_w,
                    )
                    self.locked_line = best_h.line_idx
                    self.history.append(top_sid)
                    events.append(FilterEvent(
                        t=t, type="shabad_locked",
                        payload={"shabad_id": top_sid, "confidence": round(top_mass, 3)},
                    ))
                    events.append(FilterEvent(
                        t=t, type="line_aligned",
                        payload={
                            "line_index": self.locked_line,
                            "line_score": round(top_mass, 3),
                        },
                    ))
            else:
                self.persist_count = 0
                self.last_top_shabad = top_sid
        else:
            # LOCKED — check for switch or line change
            if top_sid != self.locked_shabad and top_mass >= SWITCH_MASS:
                if self.last_top_shabad == top_sid:
                    self.persist_count += 1
                else:
                    self.persist_count = 1
                self.last_top_shabad = top_sid
                if self.persist_count >= LOCK_PERSIST:
                    old_sid = self.locked_shabad
                    self.locked_shabad = top_sid
                    self.persist_count = 0
                    best_h = max(
                        (h for h in self.beam if h.shabad_id == top_sid),
                        key=lambda h: h.log_w,
                    )
                    self.locked_line = best_h.line_idx
                    self.history.append(top_sid)
                    events.append(FilterEvent(
                        t=t, type="shabad_switched",
                        payload={
                            "old_shabad_id": old_sid,
                            "new_shabad_id": top_sid,
                            "confidence": round(top_mass, 3),
                        },
                    ))
                    events.append(FilterEvent(
                        t=t, type="line_aligned",
                        payload={
                            "line_index": self.locked_line,
                            "line_score": round(top_mass, 3),
                        },
                    ))
            elif top_sid == self.locked_shabad:
                # Update line within shabad with persistence + mass floor
                self.persist_count = 0
                self.last_top_shabad = top_sid
                best_h = max(
                    (h for h in self.beam if h.shabad_id == self.locked_shabad),
                    key=lambda h: h.log_w,
                )
                best_line = best_h.line_idx
                best_line_mass = math.exp(best_h.log_w)
                if best_line == self.locked_line:
                    self.pending_line = None
                    self.pending_line_count = 0
                elif best_line_mass < LINE_MASS_FLOOR:
                    pass  # too weak — keep current line
                else:
                    if self.pending_line == best_line:
                        self.pending_line_count += 1
                    else:
                        self.pending_line = best_line
                        self.pending_line_count = 1
                    if self.pending_line_count >= LINE_PERSIST:
                        self.locked_line = best_line
                        self.pending_line = None
                        self.pending_line_count = 0
                        events.append(FilterEvent(
                            t=t, type="line_aligned",
                            payload={
                                "line_index": self.locked_line,
                                "line_score": round(best_line_mass, 3),
                            },
                        ))
            else:
                # Different top but not enough mass to switch
                self.persist_count = 0
                self.last_top_shabad = top_sid

        return events
