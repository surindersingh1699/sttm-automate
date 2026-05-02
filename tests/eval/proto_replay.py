"""Replay harness: feed cached transcripts to the proto matcher and score it.

Usage:
    python -m tests.eval.proto_replay <run_dir> [<run_dir> ...]

For each run, reads <run>/<session>.jsonl, replays all `transcription` events
through BeamFilter, emits a synthetic event log shaped exactly like the
original (so the same scorer can run unchanged), and prints a per-session A/B
diff between the original (current matcher) and the replay (proto).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tests.eval.proto_corpus import build_corpus
from tests.eval.proto_filter import BeamFilter, FilterEvent
from tests.eval.scorer import score_session
from tests.eval.dataset import load_eval_sessions, SessionDescriptor


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _filter_transcripts(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("type") == "transcription"]


def _replay(transcripts: list[dict], filt: BeamFilter) -> list[dict]:
    """Run all transcripts through the filter, emit scorer-compatible events.

    The scorer cares about event shape: `t`, `type`, plus type-specific fields.
    We mirror those exactly so we don't have to touch scorer.py.
    """
    out: list[dict] = []
    for tr in transcripts:
        text = tr.get("text") or ""
        t = float(tr.get("t", 0.0))
        evs = filt.step(t, text)
        for ev in evs:
            out.append({"t": ev.t, "type": ev.type, **ev.payload})
    return out


def _summarise(metrics) -> dict:
    """Pull a flat headline subset from SessionMetrics for compact print."""
    m = asdict(metrics)
    return {
        "lock_acc%": m["lock"]["lock_accuracy_pct"],
        "lock_cov%": m["lock"]["lock_coverage_pct"],
        "ttfcl_s": m["lock"]["ttfcl_s"],
        "never_locked": m["lock"]["never_locked"],
        "wrong_first_lock": m["lock"]["wrong_first_lock"],
        "trans_detect%": m["transitions"]["detection_rate_pct"],
        "spurious/hr": m["transitions"]["spurious_switch_rate_per_hr"],
        "line_exact%": m["line"]["line_accuracy_exact_pct"],
        "line_pm1%": m["line"]["line_accuracy_pm1_pct"],
        "line_lag_p50": m["line"]["p50_line_lag_s"],
        "%correct": m["disruption"]["pct_time_correct"],
    }


_SESSION_CACHE: dict[str, SessionDescriptor] = {}


def _resolve_session(video_id: str) -> SessionDescriptor | None:
    if video_id in _SESSION_CACHE:
        return _SESSION_CACHE[video_id]
    sessions = load_eval_sessions(video_ids=[video_id])
    if sessions:
        _SESSION_CACHE[video_id] = sessions[0]
        return sessions[0]
    return None


def _row(label: str, vals: dict) -> str:
    parts = []
    for k, v in vals.items():
        if v is None:
            v = "—"
        elif isinstance(v, float):
            v = f"{v:.1f}"
        elif isinstance(v, bool):
            v = "Y" if v else "N"
        parts.append(f"{k}={v}")
    return f"  [{label}] " + "  ".join(parts)


def main(run_dirs: list[str]):
    print("[Build] Loading corpus …")
    corpus = build_corpus()

    overall_baseline = []
    overall_proto = []
    overall_meta = []

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        if not run_path.exists():
            print(f"[!] {run_dir} missing"); continue
        for session_log in sorted(run_path.glob("*.jsonl")):
            session_id = session_log.stem
            print(f"\n=== {run_path.name} / {session_id} ===")

            events = _read_jsonl(session_log)
            transcripts = _filter_transcripts(events)
            non_empty = sum(1 for e in transcripts if (e.get("text") or "").strip())
            print(f"  events={len(events)}  transcripts={len(transcripts)}  non_empty={non_empty}")
            if non_empty < 5:
                print(f"  -- skip (insufficient transcripts)")
                continue

            session = _resolve_session(session_id)
            if session is None:
                print(f"  -- skip (no GT for {session_id})")
                continue

            # --- baseline: original event log (what current matcher produced) ---
            baseline_metrics = score_session(events, session)

            # --- proto: replay transcripts through new filter ---
            t0 = time.time()
            filt = BeamFilter(corpus=corpus)
            proto_events = _replay(transcripts, filt)
            proto_dur = time.time() - t0
            print(f"  [proto] replayed in {proto_dur*1000:.0f} ms; emitted {len(proto_events)} events")

            proto_metrics = score_session(proto_events, session)

            base = _summarise(baseline_metrics)
            proto = _summarise(proto_metrics)
            print(_row("BASELINE", base))
            print(_row("PROTO   ", proto))

            overall_baseline.append(base)
            overall_proto.append(proto)
            overall_meta.append({
                "run": run_path.name,
                "session": session_id,
                "transcripts": non_empty,
            })

    if overall_baseline:
        print("\n──── AGGREGATE (median across sessions) ────")
        keys = list(overall_baseline[0].keys())
        import statistics
        def med(vals):
            vals = [v for v in vals if isinstance(v, (int, float))]
            return statistics.median(vals) if vals else None
        b_med = {k: med(v[k] for v in overall_baseline) for k in keys}
        p_med = {k: med(v[k] for v in overall_proto) for k in keys}
        print(_row("BASELINE-med", b_med))
        print(_row("PROTO-med   ", p_med))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
