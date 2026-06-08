"""Pure 4-axis scorer: (event_log, SessionDescriptor) → SessionMetrics.

All functions are stateless — feed them the JSONL event list and the GT timeline,
get back typed dataclasses. No orchestrator coupling; re-run without re-running ASR.

Axis A – Lock the right shabad
Axis B – Re-lock on shabad transition
Axis C – Line tracking accuracy and latency
Axis D – Recovery and disruption

Virtual time convention
-----------------------
Event log times (field "t") and GT start_s/end_s are both in virtual seconds
(0 = audio_t0, the start of the session slice). They can be compared directly.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import Any

from tests.eval.dataset import GroundTruthEvent, SessionDescriptor, gt_at


# ── output types ──────────────────────────────────────────────────────────

@dataclass
class LockMetrics:
    """Axis A: did we lock the right shabad, and how fast?"""
    lock_accuracy_pct: float          # % of locked-time on correct shabad (excludes unlocked time)
    lock_coverage_pct: float          # % of vocal GT time where any lock held
    # ── real-world correctness view (user's 70% target) ───────────────
    # locked_correct_pct: vocal time on CORRECT shabad. Target: ≥ 70%.
    # locked_wrong_pct:   vocal time on a WRONG shabad — actively misleading
    #                     on the projector, worse than nothing.
    # unlocked_pct:       vocal time with no lock — inactive, not misleading.
    # net_lock_score_pct: locked_correct_pct − locked_wrong_pct. Positive =
    #                     net helpful; negative = net misleading. The single
    #                     number that captures the wrong-lock penalty.
    locked_correct_pct: float
    locked_wrong_pct: float
    unlocked_pct: float
    net_lock_score_pct: float
    # wrong_line_pct: vocal time where shabad is RIGHT but line is > ±1 off
    # GT — i.e. correct shabad, wrong pointer. The "wrong pointer move"
    # penalty term.
    wrong_line_pct: float
    ttfcl_s: float | None             # time-to-first-correct-lock from shabad start
    wrong_first_lock: bool            # did we lock a wrong shabad first?
    wrong_first_lock_duration_s: float | None
    never_locked: bool
    total_shabads: int                # distinct GT shabads in session


@dataclass
class TransitionMetrics:
    """Axis B: did we re-lock on shabad transitions?"""
    gt_transitions: int               # number of GT shabad changes (vocal→vocal)
    detected_within_10s: int          # transitions detected ≤ 10 s after GT change
    detection_rate_pct: float         # detected_within_10s / gt_transitions * 100
    transition_latencies_s: list[float]   # per-detected-transition lag (s)
    p50_transition_latency_s: float | None
    p90_transition_latency_s: float | None
    spurious_switches: int            # pipeline switches when GT did NOT change
    spurious_switch_rate_per_hr: float


@dataclass
class LineMetrics:
    """Axis C: line tracking accuracy and latency."""
    line_accuracy_exact_pct: float    # % GT-line events where system line == GT
    line_accuracy_pm1_pct: float      # same with ±1 tolerance
    line_lag_samples: list[float]     # seconds from GT slide change to system change
    p50_line_lag_s: float | None
    p90_line_lag_s: float | None
    line_skip_count: int              # GT lines never displayed
    line_backtrack_count: int         # navigate_line("prev") contradicting GT flow
    line_flicker_count: int           # A→B→A line changes within 2 s


@dataclass
class DisruptionMetrics:
    """Axis D: recovery from bad states."""
    ttr_samples: list[float]          # time-to-recovery from each wrong state
    p50_ttr_s: float | None
    p90_ttr_s: float | None
    disruption_events_per_hr: float   # weighted: wrong-lock=3, missed-transition=2, flicker=1
    pct_time_correct: float           # % total vocal time where locked_shabad == GT AND line ±1


@dataclass
class SessionMetrics:
    session_id: str
    video_id: str
    duration_s: float
    lock: LockMetrics
    transitions: TransitionMetrics
    line: LineMetrics
    disruption: DisruptionMetrics

    def to_dict(self) -> dict:
        return asdict(self)


# ── helpers ────────────────────────────────────────────────────────────────

def _med(vals: list[float]) -> float | None:
    return round(statistics.median(vals), 3) if vals else None

def _p90(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(int(len(s) * 0.9), len(s) - 1)], 3)

def _events_of_type(events: list[dict], *types: str) -> list[dict]:
    return [e for e in events if e.get("type") in types]

def _locked_shabad_at(events: list[dict], t: float) -> int | None:
    """Reconstruct which shabad the pipeline was locked on at virtual time t."""
    locked: int | None = None
    for ev in events:
        if ev["t"] > t:
            break
        tp = ev.get("type", "")
        if tp == "shabad_locked":
            locked = ev.get("shabad_id")
        elif tp == "shabad_switched":
            locked = ev.get("new_shabad_id")
        elif tp in ("force_unlock", "context_flushed"):
            locked = None
    return locked

def _current_line_at(events: list[dict], t: float) -> int | None:
    """Reconstruct pipeline's displayed line at virtual time t."""
    line: int | None = None
    for ev in events:
        if ev["t"] > t:
            break
        tp = ev.get("type", "")
        if tp in ("line_aligned", "line_update"):
            line = ev.get("line_index")
    return line


# ── axis A: lock ────────────────────────────────────────────────────────────

def score_lock(events: list[dict], session: SessionDescriptor) -> LockMetrics:
    vocal_gt = session.vocal_gt
    if not vocal_gt:
        return LockMetrics(
            lock_accuracy_pct=0, lock_coverage_pct=0,
            locked_correct_pct=0, locked_wrong_pct=0,
            unlocked_pct=0, net_lock_score_pct=0, wrong_line_pct=0,
            ttfcl_s=None,
            wrong_first_lock=False, wrong_first_lock_duration_s=None,
            never_locked=True, total_shabads=0,
        )

    # Distinct GT shabads (in session, after first/last exclusion)
    gt_shabads: list[int] = []
    for ev in vocal_gt:
        if not gt_shabads or ev.shabad_id != gt_shabads[-1]:
            gt_shabads.append(ev.shabad_id)
    total_shabads = len(set(gt_shabads))

    # Build a time-indexed view of lock state at each GT slide boundary
    lock_events = _events_of_type(events, "shabad_locked", "shabad_switched",
                                  "force_unlock", "context_flushed")

    # --- lock accuracy & coverage (sample at each GT vocal event midpoint) ---
    locked_correct = 0
    locked_any = 0
    total_vocal = len(vocal_gt)

    for gt_ev in vocal_gt:
        mid_t = (gt_ev.start_s + gt_ev.end_s) / 2
        locked_sid = _locked_shabad_at(lock_events, mid_t)
        if locked_sid is not None:
            locked_any += 1
            if locked_sid == gt_ev.shabad_id:
                locked_correct += 1

    lock_accuracy = round(locked_correct / locked_any * 100, 1) if locked_any else 0.0
    lock_coverage = round(locked_any / total_vocal * 100, 1) if total_vocal else 0.0
    # Locked-wrong: vocal events where a lock was held but on the wrong shabad.
    locked_wrong = locked_any - locked_correct
    locked_correct_pct = (
        round(locked_correct / total_vocal * 100, 1) if total_vocal else 0.0
    )
    locked_wrong_pct = (
        round(locked_wrong / total_vocal * 100, 1) if total_vocal else 0.0
    )
    unlocked_pct = round(100.0 - lock_coverage, 1) if total_vocal else 100.0
    # Net score: reward correct locks, penalise wrong locks. Unlocked is
    # neutral (inactive, not misleading). Range: −100 to +100.
    net_lock_score_pct = round(locked_correct_pct - locked_wrong_pct, 1)

    # Wrong-pointer-move penalty: vocal events where shabad lock IS correct
    # but the displayed line is more than ±1 off the GT line. Needs the
    # line_aligned / line_update event stream — keeps line metric and lock
    # metric in their own functions but the "right shabad, wrong line"
    # number belongs alongside the lock numbers because it's the projector's
    # other failure mode the user worries about.
    line_events_for_penalty = _events_of_type(events, "line_aligned", "line_update")
    wrong_line_count = 0
    for gt_ev in vocal_gt:
        mid_t = (gt_ev.start_s + gt_ev.end_s) / 2
        locked_sid = _locked_shabad_at(lock_events, mid_t)
        if locked_sid != gt_ev.shabad_id:
            continue  # only relevant when shabad is correct
        pipe_line = _current_line_at(line_events_for_penalty, mid_t)
        if pipe_line is None:
            continue
        if abs(pipe_line - gt_ev.line_idx_in_shabad) > 1:
            wrong_line_count += 1
    wrong_line_pct = (
        round(wrong_line_count / total_vocal * 100, 1) if total_vocal else 0.0
    )

    # --- time to first correct lock per primary shabad ---
    primary_shabad = gt_shabads[0] if gt_shabads else None
    primary_start = vocal_gt[0].start_s if vocal_gt else 0.0

    ttfcl: float | None = None
    first_correct_lock_t: float | None = None
    first_wrong_lock_t: float | None = None
    wrong_lock_end_t: float | None = None

    for ev in lock_events:
        t = ev["t"]
        tp = ev.get("type", "")
        if tp == "shabad_locked":
            sid = ev.get("shabad_id")
            if sid == primary_shabad and first_correct_lock_t is None:
                first_correct_lock_t = t
                if first_wrong_lock_t is not None and wrong_lock_end_t is None:
                    wrong_lock_end_t = t
            elif sid != primary_shabad and first_correct_lock_t is None:
                if first_wrong_lock_t is None:
                    first_wrong_lock_t = t
        elif tp == "shabad_switched":
            new_sid = ev.get("new_shabad_id")
            if new_sid == primary_shabad and first_correct_lock_t is None:
                first_correct_lock_t = t
                if first_wrong_lock_t is not None and wrong_lock_end_t is None:
                    wrong_lock_end_t = t

    if first_correct_lock_t is not None:
        ttfcl = round(first_correct_lock_t - primary_start, 2)

    never_locked = first_correct_lock_t is None
    wrong_first = first_wrong_lock_t is not None

    wrong_dur: float | None = None
    if wrong_first and wrong_lock_end_t is not None and first_wrong_lock_t is not None:
        wrong_dur = round(wrong_lock_end_t - first_wrong_lock_t, 2)
    elif wrong_first and first_wrong_lock_t is not None and lock_events:
        wrong_dur = round(lock_events[-1]["t"] - first_wrong_lock_t, 2)

    return LockMetrics(
        lock_accuracy_pct=lock_accuracy,
        lock_coverage_pct=lock_coverage,
        locked_correct_pct=locked_correct_pct,
        locked_wrong_pct=locked_wrong_pct,
        unlocked_pct=unlocked_pct,
        net_lock_score_pct=net_lock_score_pct,
        wrong_line_pct=wrong_line_pct,
        ttfcl_s=ttfcl,
        wrong_first_lock=wrong_first,
        wrong_first_lock_duration_s=wrong_dur,
        never_locked=never_locked,
        total_shabads=total_shabads,
    )


# ── axis B: transitions ─────────────────────────────────────────────────────

def score_transitions(events: list[dict], session: SessionDescriptor) -> TransitionMetrics:
    """Measure how well the pipeline detects shabad transitions."""
    DETECTION_WINDOW_S = 10.0

    # GT transitions: consecutive distinct shabad IDs in vocal GT events
    gt_trans: list[tuple[float, int | None, int | None]] = []
    prev_sid: int | None = None
    for ev in session.vocal_gt:
        if ev.shabad_id != prev_sid:
            if prev_sid is not None:  # skip the very first (no prior shabad to transition from)
                gt_trans.append((ev.start_s, prev_sid, ev.shabad_id))
            prev_sid = ev.shabad_id

    switch_events = _events_of_type(events, "shabad_switched")
    switch_times = [e["t"] for e in switch_events]
    switch_new = [e.get("new_shabad_id") for e in switch_events]

    detected = 0
    latencies: list[float] = []
    used_switch_idxs: set[int] = set()

    for (gt_t, _old_sid, new_sid) in gt_trans:
        deadline = gt_t + DETECTION_WINDOW_S
        for i, (st, sn) in enumerate(zip(switch_times, switch_new)):
            if i in used_switch_idxs:
                continue
            if st >= gt_t and st <= deadline and sn == new_sid:
                detected += 1
                latencies.append(round(st - gt_t, 2))
                used_switch_idxs.add(i)
                break

    detection_rate = round(detected / len(gt_trans) * 100, 1) if gt_trans else 100.0

    # Spurious switches: pipeline switched when GT had NOT changed
    spurious = 0
    for i, (st, sn) in enumerate(zip(switch_times, switch_new)):
        is_real = any(
            abs(st - gt_t) <= DETECTION_WINDOW_S and sn == new_sid
            for (gt_t, _old, new_sid) in gt_trans
        )
        if not is_real:
            spurious += 1

    session_hrs = session.duration_s / 3600
    spurious_per_hr = round(spurious / session_hrs, 2) if session_hrs > 0 else 0.0

    return TransitionMetrics(
        gt_transitions=len(gt_trans),
        detected_within_10s=detected,
        detection_rate_pct=detection_rate,
        transition_latencies_s=latencies,
        p50_transition_latency_s=_med(latencies),
        p90_transition_latency_s=_p90(latencies),
        spurious_switches=spurious,
        spurious_switch_rate_per_hr=spurious_per_hr,
    )


# ── axis C: line tracking ────────────────────────────────────────────────────

def score_line(events: list[dict], session: SessionDescriptor) -> LineMetrics:
    """Measure line-tracking accuracy and latency."""
    lock_events = _events_of_type(events, "shabad_locked", "shabad_switched",
                                  "force_unlock", "context_flushed")
    line_events = _events_of_type(events, "line_aligned", "line_update")

    vocal_gt = session.vocal_gt
    if not vocal_gt:
        return LineMetrics(
            line_accuracy_exact_pct=0, line_accuracy_pm1_pct=0,
            line_lag_samples=[], p50_line_lag_s=None, p90_line_lag_s=None,
            line_skip_count=0, line_backtrack_count=0, line_flicker_count=0,
        )

    exact_correct = 0
    pm1_correct = 0
    total_scored = 0
    lag_samples: list[float] = []
    gt_lines_seen: set[tuple[int, int]] = set()  # (shabad_id, line_idx)

    prev_gt_line_idx: int | None = None
    prev_gt_line_t: float | None = None

    for gt_ev in vocal_gt:
        mid_t = (gt_ev.start_s + gt_ev.end_s) / 2
        locked_sid = _locked_shabad_at(lock_events, mid_t)
        if locked_sid != gt_ev.shabad_id:
            continue  # only score when locked on correct shabad

        pipe_line = _current_line_at(line_events, mid_t)
        if pipe_line is None:
            continue

        gt_line = gt_ev.line_idx_in_shabad
        total_scored += 1
        gt_lines_seen.add((gt_ev.shabad_id, gt_line))

        if pipe_line == gt_line:
            exact_correct += 1
            pm1_correct += 1
        elif abs(pipe_line - gt_line) <= 1:
            pm1_correct += 1

        # Line lag: how long after GT slide change did pipeline follow?
        if gt_line != prev_gt_line_idx:
            prev_gt_line_t = gt_ev.start_s
            prev_gt_line_idx = gt_line
        elif prev_gt_line_t is not None:
            # Find the first line_event at or after prev_gt_line_t showing gt_line
            for le in line_events:
                if le["t"] >= prev_gt_line_t and le.get("line_index") == gt_line:
                    lag = le["t"] - prev_gt_line_t
                    if 0 <= lag <= 30:
                        lag_samples.append(round(lag, 2))
                    break

    exact_pct = round(exact_correct / total_scored * 100, 1) if total_scored else 0.0
    pm1_pct = round(pm1_correct / total_scored * 100, 1) if total_scored else 0.0

    # Line skip: GT lines never shown
    all_gt_lines = {(e.shabad_id, e.line_idx_in_shabad) for e in vocal_gt}
    skip_count = len(all_gt_lines - gt_lines_seen)

    # Backtrack: navigate_line("prev") calls
    nav_events = [e for e in events if e.get("type") == "navigate_line"
                  or (e.get("method") == "navigate_line")]
    backtrack = sum(1 for e in nav_events
                    if e.get("direction") == "prev" or e.get("kwargs", {}).get("direction") == "prev")

    # Line flicker: A→B→A within 2 s in line events
    flicker = 0
    for i in range(2, len(line_events)):
        a = line_events[i - 2].get("line_index")
        b = line_events[i - 1].get("line_index")
        c = line_events[i].get("line_index")
        dt = line_events[i]["t"] - line_events[i - 2]["t"]
        if a == c and a != b and dt <= 2.0:
            flicker += 1

    return LineMetrics(
        line_accuracy_exact_pct=exact_pct,
        line_accuracy_pm1_pct=pm1_pct,
        line_lag_samples=lag_samples,
        p50_line_lag_s=_med(lag_samples),
        p90_line_lag_s=_p90(lag_samples),
        line_skip_count=skip_count,
        line_backtrack_count=backtrack,
        line_flicker_count=flicker,
    )


# ── axis D: disruption & recovery ────────────────────────────────────────────

def score_disruption(
    events: list[dict],
    session: SessionDescriptor,
    lock_m: LockMetrics,
    trans_m: TransitionMetrics,
    line_m: LineMetrics,
) -> DisruptionMetrics:
    """Measure recovery time and disruption events per hour."""
    WRONG_LOCK_WEIGHT = 3
    MISSED_TRANSITION_WEIGHT = 2
    FLICKER_WEIGHT = 1

    # Time to recovery: from each wrong-lock start to correct lock
    ttr_samples: list[float] = []
    if lock_m.wrong_first_lock and lock_m.wrong_first_lock_duration_s is not None:
        ttr_samples.append(lock_m.wrong_first_lock_duration_s)

    # % time on correct shabad AND line ±1
    lock_events = _events_of_type(events, "shabad_locked", "shabad_switched",
                                  "force_unlock", "context_flushed")
    line_events = _events_of_type(events, "line_aligned", "line_update")
    vocal_gt = session.vocal_gt

    correct_vocal_windows = 0
    total_vocal_windows = 0
    for gt_ev in vocal_gt:
        mid_t = (gt_ev.start_s + gt_ev.end_s) / 2
        locked_sid = _locked_shabad_at(lock_events, mid_t)
        pipe_line = _current_line_at(line_events, mid_t)
        total_vocal_windows += 1
        if (locked_sid == gt_ev.shabad_id and pipe_line is not None
                and abs(pipe_line - gt_ev.line_idx_in_shabad) <= 1):
            correct_vocal_windows += 1

    pct_time_correct = (
        round(correct_vocal_windows / total_vocal_windows * 100, 1)
        if total_vocal_windows else 0.0
    )

    # Disruption events per hour
    wrong_lock_events = 1 if lock_m.wrong_first_lock else 0
    missed_transitions = trans_m.gt_transitions - trans_m.detected_within_10s
    flickers = line_m.line_flicker_count

    total_disruption = (
        wrong_lock_events * WRONG_LOCK_WEIGHT
        + missed_transitions * MISSED_TRANSITION_WEIGHT
        + flickers * FLICKER_WEIGHT
    )
    session_hrs = session.duration_s / 3600
    disruption_per_hr = round(total_disruption / session_hrs, 2) if session_hrs > 0 else 0.0

    return DisruptionMetrics(
        ttr_samples=ttr_samples,
        p50_ttr_s=_med(ttr_samples),
        p90_ttr_s=_p90(ttr_samples),
        disruption_events_per_hr=disruption_per_hr,
        pct_time_correct=pct_time_correct,
    )


# ── session scorer (entry point) ────────────────────────────────────────────

def score_session(events: list[dict], session: SessionDescriptor) -> SessionMetrics:
    """Score all four axes for one eval session."""
    lock_m = score_lock(events, session)
    trans_m = score_transitions(events, session)
    line_m = score_line(events, session)
    disrupt_m = score_disruption(events, session, lock_m, trans_m, line_m)
    return SessionMetrics(
        session_id=session.session_id,
        video_id=session.video_id,
        duration_s=session.duration_s,
        lock=lock_m,
        transitions=trans_m,
        line=line_m,
        disruption=disrupt_m,
    )


# ── aggregate across sessions ───────────────────────────────────────────────

@dataclass
class AggregateKPIs:
    """Headline KPIs across all sessions — four axes, separately reportable."""
    total_sessions: int
    mode: str
    total_duration_s: float

    # Axis A — Lock
    median_lock_accuracy_pct: float
    median_lock_coverage_pct: float
    # Real-world correctness view (user's 70% target). See LockMetrics docstring.
    median_locked_correct_pct: float
    median_locked_wrong_pct: float
    median_net_lock_score_pct: float
    median_wrong_line_pct: float
    p50_ttfcl_s: float | None
    p90_ttfcl_s: float | None
    wrong_first_lock_rate_pct: float
    never_locked_rate_pct: float

    # Axis B — Transitions
    total_gt_transitions: int
    overall_detection_rate_pct: float
    p50_transition_latency_s: float | None
    p90_transition_latency_s: float | None
    median_spurious_per_hr: float

    # Axis C — Line tracking
    median_line_accuracy_exact_pct: float
    median_line_accuracy_pm1_pct: float
    p50_line_lag_s: float | None
    p90_line_lag_s: float | None
    total_line_skips: int
    total_line_flickers: int

    # Axis D — Disruption
    p50_ttr_s: float | None
    p90_ttr_s: float | None
    median_disruption_per_hr: float
    median_pct_time_correct: float

    # Summary
    composite_pct_time_correct: float


def aggregate_sessions(metrics_list: list[SessionMetrics], mode: str = "headless") -> AggregateKPIs:
    if not metrics_list:
        return AggregateKPIs(
            total_sessions=0, mode=mode, total_duration_s=0,
            median_lock_accuracy_pct=0, median_lock_coverage_pct=0,
            median_locked_correct_pct=0, median_locked_wrong_pct=0,
            median_net_lock_score_pct=0, median_wrong_line_pct=0,
            p50_ttfcl_s=None, p90_ttfcl_s=None,
            wrong_first_lock_rate_pct=0, never_locked_rate_pct=0,
            total_gt_transitions=0, overall_detection_rate_pct=0,
            p50_transition_latency_s=None, p90_transition_latency_s=None,
            median_spurious_per_hr=0,
            median_line_accuracy_exact_pct=0, median_line_accuracy_pm1_pct=0,
            p50_line_lag_s=None, p90_line_lag_s=None,
            total_line_skips=0, total_line_flickers=0,
            p50_ttr_s=None, p90_ttr_s=None,
            median_disruption_per_hr=0, median_pct_time_correct=0,
            composite_pct_time_correct=0,
        )

    total = len(metrics_list)
    total_dur = sum(m.duration_s for m in metrics_list)

    # A
    ttfcl_all = [m.lock.ttfcl_s for m in metrics_list if m.lock.ttfcl_s is not None]
    wrong_first = sum(1 for m in metrics_list if m.lock.wrong_first_lock)
    never_locked = sum(1 for m in metrics_list if m.lock.never_locked)

    # B
    all_lats = [lat for m in metrics_list for lat in m.transitions.transition_latencies_s]
    total_gt_trans = sum(m.transitions.gt_transitions for m in metrics_list)
    total_detected = sum(m.transitions.detected_within_10s for m in metrics_list)
    detection_rate = round(total_detected / total_gt_trans * 100, 1) if total_gt_trans else 100.0

    # C
    all_lags = [lag for m in metrics_list for lag in m.line.line_lag_samples]
    total_skips = sum(m.line.line_skip_count for m in metrics_list)
    total_flickers = sum(m.line.line_flicker_count for m in metrics_list)

    # D
    all_ttrs = [t for m in metrics_list for t in m.disruption.ttr_samples]
    comp_correct = round(
        statistics.median([m.disruption.pct_time_correct for m in metrics_list]), 1
    )

    return AggregateKPIs(
        total_sessions=total,
        mode=mode,
        total_duration_s=total_dur,

        median_lock_accuracy_pct=round(statistics.median([m.lock.lock_accuracy_pct for m in metrics_list]), 1),
        median_lock_coverage_pct=round(statistics.median([m.lock.lock_coverage_pct for m in metrics_list]), 1),
        median_locked_correct_pct=round(statistics.median([m.lock.locked_correct_pct for m in metrics_list]), 1),
        median_locked_wrong_pct=round(statistics.median([m.lock.locked_wrong_pct for m in metrics_list]), 1),
        median_net_lock_score_pct=round(statistics.median([m.lock.net_lock_score_pct for m in metrics_list]), 1),
        median_wrong_line_pct=round(statistics.median([m.lock.wrong_line_pct for m in metrics_list]), 1),
        p50_ttfcl_s=_med(ttfcl_all),
        p90_ttfcl_s=_p90(ttfcl_all),
        wrong_first_lock_rate_pct=round(wrong_first / total * 100, 1),
        never_locked_rate_pct=round(never_locked / total * 100, 1),

        total_gt_transitions=total_gt_trans,
        overall_detection_rate_pct=detection_rate,
        p50_transition_latency_s=_med(all_lats),
        p90_transition_latency_s=_p90(all_lats),
        median_spurious_per_hr=round(
            statistics.median([m.transitions.spurious_switch_rate_per_hr for m in metrics_list]), 2
        ),

        median_line_accuracy_exact_pct=round(statistics.median([m.line.line_accuracy_exact_pct for m in metrics_list]), 1),
        median_line_accuracy_pm1_pct=round(statistics.median([m.line.line_accuracy_pm1_pct for m in metrics_list]), 1),
        p50_line_lag_s=_med(all_lags),
        p90_line_lag_s=_p90(all_lags),
        total_line_skips=total_skips,
        total_line_flickers=total_flickers,

        p50_ttr_s=_med(all_ttrs),
        p90_ttr_s=_p90(all_ttrs),
        median_disruption_per_hr=round(
            statistics.median([m.disruption.disruption_events_per_hr for m in metrics_list]), 2
        ),
        median_pct_time_correct=round(statistics.median([m.disruption.pct_time_correct for m in metrics_list]), 1),
        composite_pct_time_correct=comp_correct,
    )


def print_kpis(kpis: AggregateKPIs):
    bar = "=" * 72
    _f = lambda v: f"{v:.2f}s" if v is not None else "N/A"
    print(f"\n{bar}")
    print(f"  STTM-AUTOMATE — INTEGRATED PIPELINE EVAL   mode={kpis.mode.upper()}")
    print(f"  Sessions: {kpis.total_sessions}   Total audio: {kpis.total_duration_s/60:.1f} min")
    print(bar)

    print("\n  ── A. LOCK THE RIGHT SHABAD ────────────────────────────────")
    print(f"  ★ Locked-correct % (≥70 goal):  {kpis.median_locked_correct_pct}%")
    print(f"    Locked-wrong %  (penalised):  {kpis.median_locked_wrong_pct}%")
    print(f"    Wrong-line % (right shabad):  {kpis.median_wrong_line_pct}%")
    print(f"    Net lock score (correct−wrong): {kpis.median_net_lock_score_pct}%")
    print(f"  Lock accuracy (given locked):   {kpis.median_lock_accuracy_pct}%")
    print(f"  Lock coverage:                  {kpis.median_lock_coverage_pct}%")
    print(f"  Time-to-correct-lock P50:       {_f(kpis.p50_ttfcl_s)}")
    print(f"  Time-to-correct-lock P90:       {_f(kpis.p90_ttfcl_s)}")
    print(f"  Wrong-first-lock rate:          {kpis.wrong_first_lock_rate_pct}%")
    print(f"  Never-locked rate:              {kpis.never_locked_rate_pct}%")

    print("\n  ── B. RE-LOCK ON TRANSITION ────────────────────────────────")
    print(f"  GT transitions:           {kpis.total_gt_transitions}")
    print(f"  Detection rate (≤10s):    {kpis.overall_detection_rate_pct}%")
    print(f"  Transition latency P50:   {_f(kpis.p50_transition_latency_s)}")
    print(f"  Transition latency P90:   {_f(kpis.p90_transition_latency_s)}")
    print(f"  Spurious switches (med):  {kpis.median_spurious_per_hr}/hr")

    print("\n  ── C. SHOW CURRENT LINE FAST & ACCURATE ───────────────────")
    print(f"  Line accuracy (exact):    {kpis.median_line_accuracy_exact_pct}%")
    print(f"  Line accuracy (±1):       {kpis.median_line_accuracy_pm1_pct}%")
    print(f"  Line lag P50:             {_f(kpis.p50_line_lag_s)}")
    print(f"  Line lag P90:             {_f(kpis.p90_line_lag_s)}")
    print(f"  Lines skipped:            {kpis.total_line_skips}")
    print(f"  Line flickers:            {kpis.total_line_flickers}")

    print("\n  ── D. RECOVERY & DISRUPTION ───────────────────────────────")
    print(f"  Recovery time P50:        {_f(kpis.p50_ttr_s)}")
    print(f"  Recovery time P90:        {_f(kpis.p90_ttr_s)}")
    print(f"  Disruption events/hr:     {kpis.median_disruption_per_hr}")
    print(f"  % time on correct shabad+line: {kpis.composite_pct_time_correct}%")
    print(bar)
