"""Aggregate and report eval metrics from SessionResult objects.

SessionMetrics are computed by scorer.py (pure functions over event logs).
This module aggregates across sessions and handles JSON serialization.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tests.eval.scorer import (
    AggregateKPIs,
    SessionMetrics,
    aggregate_sessions,
    print_kpis,
)
from tests.eval.runner import SessionResult


def compute_aggregate(results: list[SessionResult], mode: str) -> AggregateKPIs:
    return aggregate_sessions([r.metrics for r in results], mode=mode)


def save_json(kpis: AggregateKPIs, sessions: list[SessionMetrics], path: str | Path):
    data = {
        "aggregate": asdict(kpis),
        "sessions": [s.to_dict() for s in sessions],
    }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[Metrics] Report saved → {path}")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def print_session_summary(result: SessionResult):
    m = result.metrics
    lk = m.lock
    status = "✓" if lk.lock_accuracy_pct >= 80 else "✗"
    ttfcl = f"{lk.ttfcl_s:.1f}s" if lk.ttfcl_s is not None else "N/A"
    trans_rate = f"{m.transitions.detection_rate_pct:.0f}%" if m.transitions.gt_transitions else "-"
    print(
        f"  {status} {m.session_id[:36]:<36} "
        f"lock={lk.lock_accuracy_pct:.0f}% "
        f"ttfcl={ttfcl}  "
        f"line±1={m.line.line_accuracy_pm1_pct:.0f}%  "
        f"trans={trans_rate}  "
        f"correct={m.disruption.pct_time_correct:.0f}%"
    )
