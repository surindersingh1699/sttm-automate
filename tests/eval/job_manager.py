"""Background eval job manager — runs eval sessions async and streams progress."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable, Literal


@dataclass
class EvalRunState:
    run_id: str
    status: Literal["loading", "running", "done", "error"]
    mode: str
    total_jobs: int
    completed: int
    current_job_id: str | None
    per_session: list         # list of SessionMetrics dicts
    report: dict | None
    error: str | None
    started_at: float
    finished_at: float | None


class EvalJobManager:
    def __init__(self, broadcast: Callable[[dict], Awaitable[None]]):
        self._broadcast = broadcast
        self._state: EvalRunState | None = None
        self._task: asyncio.Task | None = None

    @property
    def state(self) -> EvalRunState | None:
        return self._state

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start_run(
        self,
        mode: str = "headless",
        limit_videos: int | None = None,
        min_match_score: float = 60.0,
        video_ids: list[str] | None = None,
        dataset: str | None = None,
        audio_device: int | None = None,
    ) -> str:
        if self.is_running():
            raise RuntimeError("Eval run already in progress")

        run_id = str(uuid.uuid4())[:8]
        self._state = EvalRunState(
            run_id=run_id,
            status="loading",
            mode=mode,
            total_jobs=0,
            completed=0,
            current_job_id=None,
            per_session=[],
            report=None,
            error=None,
            started_at=time.time(),
            finished_at=None,
        )
        self._task = asyncio.create_task(
            self._run(run_id, mode, limit_videos, min_match_score, video_ids, dataset, audio_device)
        )
        return run_id

    async def _run(
        self,
        run_id: str,
        mode: str,
        limit_videos: int | None,
        min_match_score: float,
        video_ids: list[str] | None,
        dataset: str | None,
        audio_device: int | None = None,
    ):
        from tests.eval.dataset import _DATASET, load_eval_sessions
        from tests.eval.metrics import compute_aggregate, save_json
        from tests.eval.runner import HeadlessSessionDriver, MicSessionDriver, SessionResult
        from tests.eval.scorer import print_kpis

        state = self._state
        try:
            await self._broadcast({"type": "eval_status", "status": "loading", "run_id": run_id})

            sessions = await asyncio.to_thread(
                load_eval_sessions,
                dataset_name=dataset or _DATASET,
                limit_videos=limit_videos,
                min_match_score=min_match_score,
                video_ids=video_ids,
            )

            state.total_jobs = len(sessions)
            state.status = "running"
            await self._broadcast({
                "type": "eval_status",
                "status": "running",
                "run_id": run_id,
                "total": len(sessions),
                "mode": mode,
            })

            if mode == "headless":
                driver = HeadlessSessionDriver(run_id=run_id)
            else:
                driver = MicSessionDriver(run_id=run_id, audio_device=audio_device)
            results: list[SessionResult] = []

            for i, session in enumerate(sessions):
                state.current_job_id = session.session_id

                await self._broadcast({
                    "type": "eval_progress",
                    "run_id": run_id,
                    "job_idx": i,
                    "total": len(sessions),
                    "job_id": session.session_id,
                    "video_id": session.video_id,
                    "completed": state.completed,
                })

                async def _progress(elapsed, total, sid, _s=session):
                    pct = int(elapsed / total * 100) if total > 0 else 0
                    await self._broadcast({
                        "type": "eval_session_progress",
                        "run_id": run_id,
                        "session_id": _s.session_id,
                        "elapsed_s": round(elapsed, 1),
                        "total_s": round(total, 1),
                        "pct": pct,
                    })

                result = await driver.run_session(session, progress_cb=_progress)
                results.append(result)
                state.per_session.append(result.metrics.to_dict())
                state.completed = i + 1

                m = result.metrics
                await self._broadcast({
                    "type": "eval_session_done",
                    "run_id": run_id,
                    "session_id": m.session_id,
                    "video_id": m.video_id,
                    "duration_s": m.duration_s,
                    "completed": state.completed,
                    "total": state.total_jobs,
                    # Axis A
                    "lock_accuracy_pct": m.lock.lock_accuracy_pct,
                    "ttfcl_s": m.lock.ttfcl_s,
                    "wrong_first_lock": m.lock.wrong_first_lock,
                    "never_locked": m.lock.never_locked,
                    # Axis B
                    "gt_transitions": m.transitions.gt_transitions,
                    "detection_rate_pct": m.transitions.detection_rate_pct,
                    # Axis C
                    "line_accuracy_pm1_pct": m.line.line_accuracy_pm1_pct,
                    "p50_line_lag_s": m.line.p50_line_lag_s,
                    # Axis D
                    "pct_time_correct": m.disruption.pct_time_correct,
                    "disruption_per_hr": m.disruption.disruption_events_per_hr,
                })

            kpis = compute_aggregate(results, mode=mode)
            state.report = asdict(kpis)
            state.status = "done"
            state.finished_at = time.time()

            # Auto-save report to disk so the eval UI can list past runs
            try:
                import json
                from pathlib import Path
                report_dir = Path(__file__).parent / "runs" / run_id
                report_dir.mkdir(parents=True, exist_ok=True)
                report_data = {
                    "run_id": run_id,
                    "mode": mode,
                    "started_at": state.started_at,
                    "finished_at": state.finished_at,
                    "total_sessions": state.completed,
                    "aggregate": state.report,
                    "sessions": state.per_session,
                }
                (report_dir / "report.json").write_text(
                    json.dumps(report_data, indent=2, ensure_ascii=False)
                )
            except Exception:
                pass

            await self._broadcast({
                "type": "eval_complete",
                "run_id": run_id,
                "report": state.report,
                "completed": state.completed,
                "total": state.total_jobs,
            })

        except Exception as exc:
            import traceback
            traceback.print_exc()
            state.status = "error"
            state.error = str(exc)
            state.finished_at = time.time()
            await self._broadcast({
                "type": "eval_error",
                "run_id": run_id,
                "error": str(exc),
            })
