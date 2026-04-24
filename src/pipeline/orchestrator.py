"""Main pipeline: audio → transcription → matching → STTM control → dashboard."""

import asyncio
from datetime import datetime
import numpy as np
from typing import Callable, Awaitable

from src.config import config
from src.audio.capture import AudioCapture
from src.transcription.factory import create_engine
from src.transcription.processor import TranscriptionProcessor
from src.transcription.transliterate import extract_first_letters
from src.matcher.search import ShabadCandidate
from src.matcher.offline_search import OfflineShabadSearcher
from src.matcher.scorer import ConfidenceScorer
from src.matcher.tracker import ShabadTracker, PipelineState
from src.controller.base import STTMController


# Type for the dashboard broadcast callback
BroadcastFn = Callable[[dict], Awaitable[None]]


class PipelineOrchestrator:
    """
    Wires all components into a continuous processing loop.

    Uses a SEARCHING/LOCKED state machine:
    - SEARCHING: broad local-DB search, confirm strong match before locking
    - LOCKED: track line position within shabad, only switch on sustained challenger
    """

    def __init__(
        self,
        controller: STTMController,
        broadcast: BroadcastFn | None = None,
        audio_device: int | None = None,
    ):
        # Use config device if set, otherwise auto-detect (BlackHole > aggregate > default)
        if audio_device is None:
            audio_device = config.audio.device
        if audio_device is None:
            audio_device = AudioCapture.find_best_device()
        self.audio = AudioCapture(device=audio_device)
        self.transcriber = create_engine(config.whisper.engine)
        self.processor = TranscriptionProcessor()
        self.searcher = OfflineShabadSearcher()
        self.scorer = ConfidenceScorer()
        ttl_windows = max(
            1,
            int(
                round(
                    config.matcher.hypothesis_ttl_seconds
                    / max(config.audio.step_duration, 0.1)
                )
            ),
        )
        self.tracker = ShabadTracker(
            challenger_windows=config.matcher.challenger_windows,
            challenger_margin=config.matcher.challenger_margin,
            candidate_lock_windows=config.matcher.candidate_lock_windows,
            hypothesis_top_k=config.matcher.hypothesis_top_k,
            hypothesis_ttl_windows=ttl_windows,
            hypothesis_decay=config.matcher.hypothesis_decay,
        )
        self.controller = controller
        self._broadcast = broadcast or self._noop_broadcast
        self.running = False
        self.paused = False
        self._audio_source = "local"  # "local" or "remote"
        # Parallel capture ↔ decode handoff.
        # The capture task wakes every `step_duration` seconds, grabs the newest
        # `window_duration` seconds of audio from the AudioCapture ring, and stashes
        # it here. The decode task awaits `_window_ready`, reads whatever is most
        # recent, and runs Whisper on it. If the decoder is busy when new snapshots
        # arrive, they silently overwrite the slot — we always decode the freshest
        # audio, never a backlog. This is what keeps the UI live when Whisper RTF > 1.
        self._latest_window_data: dict | None = None
        self._window_ready = asyncio.Event()
        self._latest_window_lock = asyncio.Lock()
        self._weak_line_windows = 0
        self._silence_windows = 0
        self._after_break_windows = 0
        self._in_vocal_break = False
        self._silence_autolock_candidate: dict | None = None
        self._silence_autolock_ttl = 0
        self._window_index = 0
        self._confidence_mode = "balanced"
        self._prev_first_letters = ""
        self._candidate_lock_misses = 0
        self._speech_rate_lps = 0.0
        # Recent per-window line scores while LOCKED — used to decide if the lock is
        # "stable enough" to switch to the micro (3 s) listening window.
        self._recent_line_scores: list[float] = []
        # Alap / detour state: shabad_id → consecutive windows we've flagged this
        # history shabad as a detour. When it crosses alap_commit_windows we promote
        # to a real switch; otherwise we just display it and keep STTM on current.
        self._alap_detour_wins: dict[int, int] = {}
        # Fast-switch: count consecutive windows where the current shabad's line
        # alignment has been weak. Combined with a strong challenger this short-circuits
        # the usual 3-window challenger persistence.
        self._current_weak_windows = 0
        # Last listening mode actually used by the decoder — referenced by _handle_locked
        # to suppress fast-switch counting on micro (3 s) windows, which are inherently
        # noisier per-window and must not by themselves trigger a lock change.
        self._last_listening_mode: str = "search"

    async def start(self):
        """Initialize components and start the processing loop."""
        print("[Pipeline] Loading transcription engine...")
        await asyncio.to_thread(self.transcriber.load)

        print("[Pipeline] Connecting to STTM...")
        connected = await self.controller.connect()
        if not connected:
            print("[Pipeline] WARNING: STTM not connected. Running in monitor-only mode.")

        print("[Pipeline] Starting audio capture...")
        audio_ok = self.audio.start()
        if not audio_ok:
            print("[Pipeline] No local audio available. Defaulting to remote mic mode.")
            self._audio_source = "remote"
            await self._broadcast({"type": "audio_source_updated", "source": "remote"})
        self.running = True

        print("[Pipeline] Pipeline running. Listening for kirtan...")
        # Capture ticks on wall-clock time; decode runs at Whisper's pace and always
        # picks up the freshest snapshot. Running them concurrently is what keeps the
        # UI live even when the decoder is slower than realtime.
        capture_task = asyncio.create_task(self._capture_tick_task())
        decode_task = asyncio.create_task(self._decode_loop())
        try:
            done, pending = await asyncio.wait(
                {capture_task, decode_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done | pending:
                try:
                    await task
                except (asyncio.CancelledError, Exception) as exc:
                    if not isinstance(exc, asyncio.CancelledError):
                        print(f"[Pipeline] Task error: {exc}")
        finally:
            self.running = False

    async def stop(self):
        """Stop the pipeline."""
        self.running = False
        self.audio.stop()
        await self.controller.disconnect()
        print("[Pipeline] Stopped.")

    def pause(self):
        """Pause automatic processing (operator takes manual control)."""
        self.paused = True

    def resume(self):
        """Resume automatic processing."""
        self.paused = False

    def set_audio_source(self, source: str):
        """Switch between 'local' (server mic) and 'remote' (browser mic)."""
        if source not in ("local", "remote"):
            return
        if source == self._audio_source:
            return
        self._audio_source = source
        if source == "remote":
            self.audio.stop()
        elif source == "local":
            if not self.audio.start():
                print("[Pipeline] No local audio hardware. Staying on remote mic.")
                self._audio_source = "remote"
                return
        self.audio.reset_ring()
        print(f"[Pipeline] Audio source switched to: {source}")

    async def switch_engine(self, name: str) -> tuple[bool, str | None]:
        """Hot-swap the transcription backend. Returns (ok, error_message)."""
        from src.transcription.factory import SUPPORTED_ENGINES
        if name not in SUPPORTED_ENGINES:
            return False, f"Unknown engine '{name}'."
        if name == config.whisper.engine and self.transcriber is not None:
            return True, None
        prev = self.transcriber
        try:
            new_engine = create_engine(name)
            await asyncio.to_thread(new_engine.load)
        except Exception as e:
            print(f"[Pipeline] Engine switch to '{name}' failed: {e}")
            return False, str(e)
        # Only commit config/swap after the new engine has loaded successfully.
        self.transcriber = new_engine
        config.whisper.engine = name
        self.audio.reset_ring()
        # Release old model memory (helps mlx/ggml releases).
        del prev
        print(f"[Pipeline] Transcription engine switched to: {name}")
        return True, None

    def switch_audio_device(self, device_index: int | None):
        """Switch local audio input device (e.g. BlackHole vs MacBook Mic)."""
        self.audio.stop()
        self.audio = AudioCapture(device=device_index)
        config.audio.device = device_index
        if self._audio_source == "local":
            if not self.audio.start():
                print(f"[Pipeline] Could not start audio device {device_index}.")
                return False
        self.audio.reset_ring()
        device_name = self._get_device_name(device_index)
        print(f"[Pipeline] Audio device switched to: {device_name} (index={device_index})")
        return True

    @staticmethod
    def _get_device_name(device_index: int | None) -> str:
        """Get human-readable name for an audio device index."""
        if device_index is None:
            return "System Default"
        try:
            import sounddevice as sd
            info = sd.query_devices(device_index)
            return info["name"]
        except Exception:
            return f"Device {device_index}"

    def push_remote_audio(self, audio_data: np.ndarray):
        """Push audio from browser mic into the AudioCapture ring.

        Remote audio shares the same ring as the local mic — the decode path
        never has to branch on source, and old samples roll off naturally.
        """
        self.audio.push_external(audio_data)

    async def manual_select(self, shabad_id: int):
        """Manually select a shabad (override from dashboard)."""
        self.tracker.manual_lock(shabad_id)
        await self.controller.display_shabad(shabad_id)
        # Fetch verses for line tracking
        verses = await asyncio.to_thread(self.searcher.fetch_all_verses, shabad_id)
        candidate = await asyncio.to_thread(self.searcher.search_by_id, shabad_id)
        if candidate and verses:
            self.tracker.set_shabad_details(
                candidate.gurmukhi, candidate.unicode, candidate.english, verses
            )
        await self._broadcast({
            "type": "shabad_locked",
            "shabad_id": shabad_id,
            "state": self.tracker.state.value,
            "total_lines": len(verses) if verses else 0,
            "verses": [
                {"unicode": v.unicode, "english": v.english}
                for v in (verses or [])
            ],
        })

    async def manual_navigate(self, direction: str):
        """Manually navigate lines (override from dashboard)."""
        await self.controller.navigate_line(direction)
        if direction == "next":
            self.tracker.advance_line()
        elif direction == "prev" and self.tracker.current:
            self.tracker.set_line(max(0, self.tracker.current.current_line - 1))

    async def force_unlock(self):
        """Operator safety action: immediately release current lock."""
        current = self.tracker.current
        if not current:
            return
        old_id = current.shabad_id
        self.tracker.release_lock()
        self._weak_line_windows = 0
        self._recent_line_scores.clear()
        self._alap_detour_wins.clear()
        self._current_weak_windows = 0
        await self._broadcast({
            "type": "shabad_switched",
            "old_shabad_id": old_id,
            "new_shabad_id": None,
            "reason": "force_unlock",
        })

    async def flush_context(self):
        """
        Drop the rolling audio window and short-term pipeline memory so the
        next transcription starts from whatever is being recited right now.
        Keeps the current shabad lock (and its line position) intact.
        """
        self.audio.reset_ring()
        self._prev_first_letters = ""
        self._silence_autolock_candidate = None
        self._silence_autolock_ttl = 0
        self._weak_line_windows = 0
        self._candidate_lock_misses = 0
        self._after_break_windows = 0
        self._in_vocal_break = False
        self._silence_windows = 0
        self._speech_rate_lps = 0.0
        self._recent_line_scores.clear()
        self._alap_detour_wins.clear()
        self._current_weak_windows = 0
        self.tracker.clear_short_term_memory()
        print("[Pipeline] Context flushed — starting from fresh audio.")
        await self._broadcast({
            "type": "context_flushed",
            "shabad_id": self.tracker.current.shabad_id if self.tracker.current else None,
        })

    def set_confidence_mode(self, mode: str):
        """Apply runtime confidence profile."""
        profiles = {
            "conservative": {
                "auto_threshold": 0.82,
                "instant_lock_threshold": 0.92,
                "min_raw_lock_score": 0.74,
                "word_overlap_auto_min": 1,
                "word_overlap_evidence_min": 2,
                "word_overlap_instant_min": 1,
                "instant_challenger_switch_score": 0.94,
                "instant_challenger_switch_margin": 0.12,
                "word_overlap_instant_challenger_min": 2,
                "suggest_threshold": 0.66,
                "challenger_margin": 0.14,
                "challenger_windows": 4,
                "candidate_lock_windows": 3,
                "weak_line_recovery_windows": 4,
                "recovery_challenger_score": 0.72,
                "local_line_follow_threshold": 0.48,
                "silence_autolock_min_score": 0.90,
                "candidate_lock_miss_windows": 3,
            },
            "balanced": {
                "auto_threshold": 0.75,
                "instant_lock_threshold": 0.85,
                "min_raw_lock_score": 0.70,
                "word_overlap_auto_min": 1,
                "word_overlap_evidence_min": 2,
                "word_overlap_instant_min": 1,
                "instant_challenger_switch_score": 0.90,
                "instant_challenger_switch_margin": 0.08,
                "word_overlap_instant_challenger_min": 1,
                "suggest_threshold": 0.60,
                "challenger_margin": 0.10,
                "challenger_windows": 3,
                "candidate_lock_windows": 2,
                "weak_line_recovery_windows": 3,
                "recovery_challenger_score": 0.65,
                "local_line_follow_threshold": 0.42,
                "silence_autolock_min_score": 0.82,
                "candidate_lock_miss_windows": 4,
            },
            "fast": {
                "auto_threshold": 0.68,
                "instant_lock_threshold": 0.80,
                "min_raw_lock_score": 0.66,
                "word_overlap_auto_min": 1,
                "word_overlap_evidence_min": 1,
                "word_overlap_instant_min": 1,
                "instant_challenger_switch_score": 0.86,
                "instant_challenger_switch_margin": 0.05,
                "word_overlap_instant_challenger_min": 1,
                "suggest_threshold": 0.55,
                "challenger_margin": 0.08,
                "challenger_windows": 2,
                "candidate_lock_windows": 1,
                "weak_line_recovery_windows": 2,
                "recovery_challenger_score": 0.58,
                "local_line_follow_threshold": 0.38,
                "silence_autolock_min_score": 0.75,
                "candidate_lock_miss_windows": 5,
            },
        }
        selected = profiles.get(mode, profiles["balanced"])
        config.matcher.auto_threshold = selected["auto_threshold"]
        config.matcher.instant_lock_threshold = selected["instant_lock_threshold"]
        config.matcher.min_raw_lock_score = selected["min_raw_lock_score"]
        config.matcher.word_overlap_auto_min = selected["word_overlap_auto_min"]
        config.matcher.word_overlap_evidence_min = selected["word_overlap_evidence_min"]
        config.matcher.word_overlap_instant_min = selected["word_overlap_instant_min"]
        config.matcher.instant_challenger_switch_score = selected["instant_challenger_switch_score"]
        config.matcher.instant_challenger_switch_margin = selected["instant_challenger_switch_margin"]
        config.matcher.word_overlap_instant_challenger_min = selected["word_overlap_instant_challenger_min"]
        config.matcher.suggest_threshold = selected["suggest_threshold"]
        config.matcher.challenger_margin = selected["challenger_margin"]
        config.matcher.challenger_windows = selected["challenger_windows"]
        config.matcher.candidate_lock_windows = selected["candidate_lock_windows"]
        config.matcher.weak_line_recovery_windows = selected["weak_line_recovery_windows"]
        config.matcher.recovery_challenger_score = selected["recovery_challenger_score"]
        config.matcher.local_line_follow_threshold = selected["local_line_follow_threshold"]
        config.matcher.silence_autolock_min_score = selected["silence_autolock_min_score"]
        config.matcher.candidate_lock_miss_windows = selected["candidate_lock_miss_windows"]
        self.tracker.set_policy(
            challenger_windows=config.matcher.challenger_windows,
            challenger_margin=config.matcher.challenger_margin,
            candidate_lock_windows=config.matcher.candidate_lock_windows,
        )
        self._confidence_mode = mode if mode in profiles else "balanced"

    @property
    def confidence_mode(self) -> str:
        return self._confidence_mode

    async def recall_shabad(self, shabad_id: int):
        """Recall a shabad from history."""
        found = self.tracker.recall_from_history(shabad_id)
        if found:
            await self.controller.display_shabad(shabad_id)
            current = self.tracker.current
            verses = current.verses if current else []
            await self._broadcast({
                "type": "shabad_locked",
                "shabad_id": shabad_id,
                "total_lines": len(verses),
                "verses": [
                    {"unicode": v.unicode, "english": v.english}
                    for v in verses
                ],
            })

    async def _capture_tick_task(self):
        """Real-time capture ticker — independent of decoder speed.

        Every `step_duration` seconds of wall clock, snapshot the freshest audio
        from the AudioCapture ring, detect vocal breaks on the fresh slice, and
        publish a window for the decoder to pick up. Runs continuously; never
        waits on Whisper. This is what keeps the dashboard's audio level and
        vocal-break detection live even when decode is behind.
        """
        step = max(0.1, float(config.audio.step_duration))
        while self.running:
            try:
                fresh_chunk = self.audio.latest_window(step)
                window = self.audio.latest_window(config.audio.window_duration)

                chunk_rms = float(np.sqrt(np.mean(fresh_chunk**2))) if fresh_chunk.size else 0.0
                has_vocals = bool(self.transcriber.has_vocal_content(fresh_chunk))
                self._update_vocal_break_state(has_vocals)

                audio_for_stt, listening_mode, window_seconds = self._select_transcription_audio(window)

                await self._broadcast({
                    "type": "audio_level",
                    "rms": round(chunk_rms, 4),
                    "has_vocals": has_vocals,
                    "speech_rate_lps": round(self._speech_rate_lps, 2),
                    "listening_mode": listening_mode,
                    "window_seconds": round(window_seconds, 2),
                })

                async with self._latest_window_lock:
                    self._latest_window_data = {
                        "audio": audio_for_stt,
                        "listening_mode": listening_mode,
                        "window_seconds": window_seconds,
                        "chunk_rms": chunk_rms,
                        "has_vocals": has_vocals,
                    }
                    self._window_ready.set()
            except Exception as e:
                print(f"[Pipeline] Capture tick error: {e}")

            await asyncio.sleep(step)

    async def _decode_loop(self):
        """Consume the freshest captured window, transcribe, and match.

        Blocks on Whisper but never on capture. When decode takes longer than
        `step_duration`, the capture task has already overwritten the window slot
        with newer audio — we pick up that newer snapshot on the next iteration
        and skip the intermediate ones. This prevents backlog accumulation.
        """
        import time as _time

        while self.running:
            try:
                await self._window_ready.wait()
                async with self._latest_window_lock:
                    window_data = self._latest_window_data
                    self._latest_window_data = None
                    self._window_ready.clear()
                if window_data is None:
                    continue

                self._window_index += 1
                audio_for_stt = window_data["audio"]
                listening_mode = window_data["listening_mode"]
                self._last_listening_mode = listening_mode
                window_seconds = window_data["window_seconds"]
                chunk_rms = window_data["chunk_rms"]
                has_vocals = window_data["has_vocals"]

                if self.paused:
                    await self._broadcast({"type": "paused"})
                    await asyncio.sleep(0.1)
                    continue

                # Skip transcription if no vocal content (just music/silence)
                if not has_vocals:
                    self._prev_first_letters = ""
                    self._speech_rate_lps *= 0.9
                    await self._try_silence_autolock()
                    await self._broadcast({
                        "type": "transcription",
                        "text": "",
                        "first_letters": "",
                        "status": "music_only",
                    })
                    continue

                # 2. Transcribe
                _t0 = _time.monotonic()
                print(f"  [DEBUG] Starting Whisper transcription ({window_seconds:.1f}s audio)...")
                segments = await asyncio.to_thread(
                    self.transcriber.transcribe, audio_for_stt
                )
                _elapsed = _time.monotonic() - _t0
                text = self.processor.process(segments)
                print(f"  [DEBUG] Whisper done in {_elapsed:.1f}s → '{text[:80]}'" if text else f"  [DEBUG] Whisper done in {_elapsed:.1f}s → (empty)")

                # Drop windows where decode took too long (protects lock state from
                # junk output of runaway decodes). Transparent to UI via status flag.
                _rtf_value = _elapsed / max(window_seconds, 0.001)
                if (
                    config.whisper.skip_slow_windows
                    and _rtf_value > config.whisper.skip_slow_rtf_threshold
                ):
                    self._prev_first_letters = ""
                    print(
                        f"  [SLOW-DROP] RTF={_rtf_value:.2f} > "
                        f"{config.whisper.skip_slow_rtf_threshold:.1f} — "
                        f"dropping window to preserve lock"
                    )
                    await self._broadcast({
                        "type": "transcription",
                        "text": "",
                        "first_letters": "",
                        "status": "slow_window_dropped",
                        "transcribe_ms": int(_elapsed * 1000),
                        "rtf": round(_rtf_value, 3),
                        "window_seconds": round(window_seconds, 2),
                        "pipeline_state": self.tracker.state.value,
                    })
                    continue

                # 3. Extract first letters
                first_letters = extract_first_letters(text)
                instantaneous_lps = len(first_letters) / max(window_seconds, 0.1)
                alpha = max(0.01, min(0.99, config.matcher.speech_rate_ema_alpha))
                self._speech_rate_lps = (
                    (1.0 - alpha) * self._speech_rate_lps
                    + alpha * instantaneous_lps
                )

                # 4. Broadcast transcription (with realtime model speed).
                # Skip speed fields when inference short-circuited (<20ms = no real work),
                # otherwise the UI pill reads 0 ms / 0× on silence-suppressed windows.
                msg = {
                    "type": "transcription",
                    "text": text,
                    "first_letters": first_letters,
                    "speech_rate_lps": round(self._speech_rate_lps, 2),
                    "pipeline_state": self.tracker.state.value,
                    "listening_mode": listening_mode,
                    "window_seconds": round(window_seconds, 2),
                }
                if _elapsed >= 0.02:
                    msg["transcribe_ms"] = int(_elapsed * 1000)
                    msg["rtf"] = round(_elapsed / max(window_seconds, 0.001), 3)
                await self._broadcast(msg)

                # 6. Dispatch to state handler
                if self.tracker.state in (
                    PipelineState.SEARCHING,
                    PipelineState.CANDIDATE_LOCK,
                ):
                    if len(first_letters) < config.matcher.min_search_letters:
                        if self.tracker.state == PipelineState.CANDIDATE_LOCK:
                            self._candidate_lock_misses += 1
                            if self._candidate_lock_misses >= config.matcher.candidate_lock_miss_windows:
                                self.tracker.clear_candidate_lock()
                                self._candidate_lock_misses = 0
                        continue
                    await self._handle_searching(
                        first_letters,
                        start_mode=self._after_break_windows > 0,
                        transcript_text=text,
                    )
                    if self._after_break_windows > 0:
                        self._after_break_windows -= 1
                else:
                    stitched_letters = f"{self._prev_first_letters}{first_letters}"
                    if max(len(first_letters), len(stitched_letters)) < config.matcher.min_search_letters:
                        self.tracker.mark_unstable()
                        self._prev_first_letters = first_letters
                        continue
                    await self._handle_locked(
                        first_letters,
                        self._prev_first_letters,
                        transcript_text=text,
                    )
                    if self._after_break_windows > 0:
                        self._after_break_windows -= 1

                self._prev_first_letters = first_letters

                # 7. Broadcast current state (include verses when locked)
                current = self.tracker.current
                state_msg = {
                    "type": "state",
                    "pipeline_state": self.tracker.state.value,
                    "current": current.to_dict() if current else None,
                    "history": self.tracker.get_history_list(),
                    "confidence_mode": self._confidence_mode,
                    "hypotheses": self.tracker.get_hypotheses(),
                }
                if current and current.verses:
                    state_msg["verses"] = [
                        {"unicode": v.unicode, "english": v.english}
                        for v in current.verses
                    ]
                await self._broadcast(state_msg)

            except Exception as e:
                print(f"[Pipeline] Error in loop: {e}")
                await self._broadcast({"type": "error", "message": str(e)})
                await asyncio.sleep(1)

    async def _handle_searching(
        self,
        first_letters: str,
        start_mode: bool = False,
        transcript_text: str = "",
    ):
        """SEARCHING state: broad search, score candidates, try to lock."""
        # Broad local-DB search (offline SQLite).
        candidates = await asyncio.to_thread(
            self.searcher.search,
            first_letters,
            10,
            start_mode,
            transcript_text,
        )

        # Score candidates
        scored = self._score_candidates(first_letters, candidates, transcript_text)
        top_candidates = scored[:config.dashboard.max_candidates]
        self.tracker.observe_candidates(top_candidates, self._window_index)
        best_hypothesis = self.tracker.best_hypothesis()

        # Broadcast candidates
        await self._broadcast({
            "type": "candidates",
            "matches": top_candidates,
            "pipeline_state": "searching",
            "hypotheses": self.tracker.get_hypotheses(),
        })

        # Select winner from cumulative hypothesis evidence.
        top = None
        if best_hypothesis:
            top = next(
                (
                    candidate
                    for candidate in top_candidates
                    if candidate["shabad_id"] == best_hypothesis["shabad_id"]
                ),
                None,
            )
            if top:
                top = dict(top)
                top["raw_score"] = top["score"]
                top["evidence_score"] = round(best_hypothesis["evidence_score"], 3)
                top["stability"] = best_hypothesis["stability"]
        if top is None and top_candidates:
            top = dict(top_candidates[0])
            top["raw_score"] = top["score"]
            top["evidence_score"] = top["score"]
            top["stability"] = 1

        # Try to lock on cumulative strong match.
        if top:
            raw_score = float(top.get("raw_score", top.get("score", 0.0)))
            evidence_score = float(top.get("evidence_score", raw_score))
            word_overlap = int(top.get("word_overlap", 0))
            top["score"] = round(max(raw_score, evidence_score), 3)
            meets_raw_auto = (
                raw_score >= config.matcher.auto_threshold
                and word_overlap >= config.matcher.word_overlap_auto_min
            )
            meets_evidence = (
                evidence_score >= config.matcher.auto_threshold
                and int(top.get("stability", 1)) >= config.matcher.candidate_lock_windows
                and word_overlap >= config.matcher.word_overlap_evidence_min
            )
            lockable = (
                raw_score >= config.matcher.min_raw_lock_score
                and (meets_raw_auto or meets_evidence)
            )
        else:
            raw_score = 0.0
            evidence_score = 0.0
            word_overlap = 0
            lockable = False

        if top and lockable:
            self._silence_autolock_candidate = top
            self._silence_autolock_ttl = config.matcher.silence_autolock_windows
            instant = (
                raw_score >= config.matcher.instant_lock_threshold
                and word_overlap >= config.matcher.word_overlap_instant_min
            )
            result = self.tracker.try_lock(top["shabad_id"], raw_score, instant=instant)

            if result["action"] == "locked":
                self._candidate_lock_misses = 0
                await self._lock_shabad_from_top(top)
                self._after_break_windows = 0
                total = len(self.tracker.current.verses) if self.tracker.current else 0
                print(
                    f"  [LOCKED] Shabad {top['shabad_id']} — "
                    f"{top['unicode'][:60]} ({total} verses)"
                )

            elif result["action"] == "pending":
                self._candidate_lock_misses = 0
                await self._broadcast({
                    "type": "pending_lock",
                    "shabad": top,
                    "wins": result.get("wins", 1),
                    "needed": result.get("needed", config.matcher.candidate_lock_windows),
                })
                print(f"  [PENDING] Confirming shabad {top['shabad_id']}...")
        else:
            # No strong candidate in this cycle; don't keep stale silence-autolock state.
            self._silence_autolock_candidate = None
            self._silence_autolock_ttl = 0
            if self.tracker.state == PipelineState.CANDIDATE_LOCK:
                self._candidate_lock_misses += 1
                if self._candidate_lock_misses >= config.matcher.candidate_lock_miss_windows:
                    self.tracker.clear_candidate_lock()
                    self._candidate_lock_misses = 0

    async def _handle_locked(
        self,
        first_letters: str,
        prev_first_letters: str = "",
        transcript_text: str = "",
    ):
        """LOCKED state: align line within shabad, check for challenger."""
        current = self.tracker.current
        if not current or not current.verses:
            # No verses cached — fall back to searching
            self.tracker.state = PipelineState.SEARCHING
            return
        if current.current_line >= len(current.verses):
            current.current_line = len(current.verses) - 1
        elif current.current_line < 0:
            current.current_line = 0

        # First pass: fresh window only. This is the happy path — when the lock is
        # confident, scoring the fresh 3 s alone gives us the next line immediately
        # and avoids carrying the previous window's letters forward (which tends to
        # keep the pointer stuck on the previous line during fast recitation).
        current_scores: list[float] = []
        best_current_idx = 0
        best_current_score = 0.0
        for i, verse in enumerate(current.verses):
            raw_current = self.scorer.score_line(first_letters, verse.first_letters)
            current_score = self._apply_progression_bias(i, current.current_line, raw_current)
            current_scores.append(current_score)
            if current_score > best_current_score:
                best_current_score = current_score
                best_current_idx = i

        line_scores = current_scores
        best_line_idx = best_current_idx
        best_line_score = best_current_score
        best_line_variant = "current"

        # Fallback pass: only if the fresh window wasn't convincing on its own,
        # pay the cost of scoring stitched windows + consecutive-verse spans.
        need_fallback_scoring = best_current_score < config.matcher.suggest_threshold
        if need_fallback_scoring:
            combined_variants: list[tuple[str, str]] = []
            if prev_first_letters:
                candidate_combined = [
                    ("prev_current", f"{prev_first_letters}{first_letters}"),
                    ("current_prev", f"{first_letters}{prev_first_letters}"),
                ]
                seen_combined: set[str] = set()
                for label, letters in candidate_combined:
                    if letters and letters != first_letters and letters not in seen_combined:
                        combined_variants.append((label, letters))
                        seen_combined.add(letters)

            # Pair-scoring handles dense windows (one audio chunk holds 2+ verses).
            pair_align_enabled = (
                config.matcher.multi_line_locked_align
                and len(first_letters) >= config.matcher.multi_line_locked_min_query_length
            )
            # Triple-span alignment fires above the trinary query-length threshold
            # (Fix 3). Same shape as pair alignment but stitches verse[i]+[i+1]+[i+2].
            triple_align_enabled = (
                config.matcher.multi_line_locked_align
                and len(first_letters) >= config.matcher.multi_line_locked_trinary_min_query_length
            )

            combined_scores: list[float] = []
            combined_labels: list[str] = []
            best_combined_idx = 0
            best_combined_score = 0.0
            best_combined_label = "current"
            for i, verse in enumerate(current.verses):
                line_best_combined = current_scores[i]
                line_best_label = "current"
                for label, query_letters in combined_variants:
                    raw_score = self.scorer.score_line(query_letters, verse.first_letters)
                    candidate_score = self._apply_progression_bias(i, current.current_line, raw_score)
                    if candidate_score > line_best_combined:
                        line_best_combined = candidate_score
                        line_best_label = label

                if pair_align_enabled and i + 1 < len(current.verses):
                    paired_letters = f"{verse.first_letters}{current.verses[i + 1].first_letters}"
                    raw_pair = self.scorer.score_line(first_letters, paired_letters)
                    pair_score = self._apply_progression_bias(i, current.current_line, raw_pair)
                    if pair_score > line_best_combined:
                        line_best_combined = pair_score
                        line_best_label = "pair_i"
                    for label, query_letters in combined_variants:
                        raw_stitch = self.scorer.score_line(query_letters, paired_letters)
                        stitch_score = self._apply_progression_bias(i, current.current_line, raw_stitch)
                        if stitch_score > line_best_combined:
                            line_best_combined = stitch_score
                            line_best_label = f"pair_i+{label}"

                if triple_align_enabled and i + 2 < len(current.verses):
                    triple_letters = (
                        f"{verse.first_letters}"
                        f"{current.verses[i + 1].first_letters}"
                        f"{current.verses[i + 2].first_letters}"
                    )
                    raw_triple = self.scorer.score_line(first_letters, triple_letters)
                    triple_score = self._apply_progression_bias(i, current.current_line, raw_triple)
                    if triple_score > line_best_combined:
                        line_best_combined = triple_score
                        line_best_label = "triple_i"
                    for label, query_letters in combined_variants:
                        raw_stitch3 = self.scorer.score_line(query_letters, triple_letters)
                        stitch3_score = self._apply_progression_bias(i, current.current_line, raw_stitch3)
                        if stitch3_score > line_best_combined:
                            line_best_combined = stitch3_score
                            line_best_label = f"triple_i+{label}"

                combined_scores.append(line_best_combined)
                combined_labels.append(line_best_label)
                if line_best_combined > best_combined_score:
                    best_combined_score = line_best_combined
                    best_combined_idx = i
                    best_combined_label = line_best_label

            use_combined = best_combined_label != "current" and best_combined_score > best_current_score
            if use_combined:
                line_scores = combined_scores
                best_line_idx = best_combined_idx
                best_line_score = best_combined_score
                best_line_variant = best_combined_label

        # Fallback: follow nearby lines at lower confidence to avoid getting stuck.
        # This keeps movement local and avoids large random jumps.
        local_start = max(0, current.current_line - config.matcher.local_line_follow_window)
        local_end = min(
            len(current.verses),
            current.current_line + config.matcher.local_line_follow_window + 1,
        )
        local_best_idx = current.current_line
        local_best_score = line_scores[current.current_line]
        for idx in range(local_start, local_end):
            if line_scores[idx] > local_best_score:
                local_best_score = line_scores[idx]
                local_best_idx = idx

        # Shadow match against recently-sung shabads (sticky set) to detect alap —
        # a brief detour to another shabad the ragi already sang. We flag it on the
        # dashboard but do NOT move STTM unless the detour persists past the commit
        # threshold (at which point it's a real shabad switch).
        detour_match = self._score_sticky_set(first_letters)
        is_detour = False
        on_micro_window_flag = self._last_listening_mode == "locked_micro"
        if (
            detour_match
            and detour_match["score"] >= config.matcher.alap_detour_min_score
            and detour_match["score"] > best_line_score + 0.05
        ):
            is_detour = True
            shabad_id = detour_match["shabad_id"]
            # Show the detour on the dashboard regardless of window size, but only
            # accumulate toward the auto-commit on non-micro windows. 3 s of audio
            # is too little evidence on its own to take STTM off the current shabad.
            if not on_micro_window_flag:
                self._alap_detour_wins[shabad_id] = (
                    self._alap_detour_wins.get(shabad_id, 0) + 1
                )
                for sid in list(self._alap_detour_wins):
                    if sid != shabad_id:
                        self._alap_detour_wins[sid] = max(0, self._alap_detour_wins[sid] - 1)
                        if self._alap_detour_wins[sid] == 0:
                            self._alap_detour_wins.pop(sid, None)
            await self._broadcast({
                "type": "alap_detour",
                "shabad_id": shabad_id,
                "line_index": detour_match["line_idx"],
                "line_unicode": detour_match["unicode"],
                "line_english": detour_match["english"],
                "score": round(detour_match["score"], 3),
                "wins": self._alap_detour_wins.get(shabad_id, 0),
                "commit_at": config.matcher.alap_commit_windows,
                "current_shabad_id": current.shabad_id,
                "current_line_score": round(best_line_score, 3),
                "listening_mode": self._last_listening_mode,
            })
            if self._alap_detour_wins.get(shabad_id, 0) >= config.matcher.alap_commit_windows:
                print(
                    f"  [ALAP → SWITCH] Sustained detour to shabad {shabad_id} "
                    f"({self._alap_detour_wins[shabad_id]} windows) — committing switch"
                )
                self._alap_detour_wins.clear()
                await self._commit_sticky_switch(detour_match, current.shabad_id)
                return
        else:
            # Decay so a brief spurious detour window doesn't accumulate across alaps.
            for sid in list(self._alap_detour_wins):
                self._alap_detour_wins[sid] = max(0, self._alap_detour_wins[sid] - 1)
                if self._alap_detour_wins[sid] == 0:
                    self._alap_detour_wins.pop(sid, None)

        # Broadcast line alignment
        best_verse = current.verses[best_line_idx]
        await self._broadcast({
            "type": "line_aligned",
            "line_index": best_line_idx,
            "line_score": round(best_line_score, 3),
            "line_unicode": best_verse.unicode,
            "line_english": best_verse.english,
            "match_variant": best_line_variant,
            "pipeline_state": "locked",
            "is_detour": is_detour,
        })

        # Track recent line scores so _is_lock_stable can decide when the micro window
        # is safe to use. Detour windows don't represent current-shabad alignment, so
        # they don't contribute to stability.
        if not is_detour:
            self._recent_line_scores.append(best_line_score)
            if len(self._recent_line_scores) > 4:
                self._recent_line_scores.pop(0)

        # Fast-switch bookkeeping: count consecutive windows where current-shabad
        # alignment is weak. Detour windows don't count (expected misalignment), and
        # neither do micro-window cycles (3 s is noisier per-window — a single dip is
        # normal and must not on its own motivate a switch).
        on_micro_window = self._last_listening_mode == "locked_micro"
        if is_detour or on_micro_window:
            pass
        elif best_line_score < config.matcher.fast_switch_current_weak_score:
            self._current_weak_windows += 1
        else:
            self._current_weak_windows = 0

        # If current line matches well, or nearby line matches reasonably, update position.
        # During alap we intentionally freeze the line pointer — STTM stays on the current
        # shabad and we don't chase the detour audio.
        should_update_line = not is_detour and (
            best_line_score >= config.matcher.suggest_threshold
            or (
                local_best_score >= config.matcher.local_line_follow_threshold
                and local_best_idx != current.current_line
            )
        )
        if should_update_line:
            self._weak_line_windows = 0
            old_line = current.current_line
            target_idx = best_line_idx
            target_score = best_line_score
            if (
                best_line_score < config.matcher.suggest_threshold
                and local_best_score >= config.matcher.local_line_follow_threshold
            ):
                target_idx = local_best_idx
                target_score = local_best_score

            # Dwell gate: don't advance forward if we've been on the current line for
            # less than min_line_dwell_seconds, unless next-line evidence is
            # overwhelming. Moving backward (delta < 0) is always allowed — if we
            # accidentally jumped ahead we want to snap back immediately.
            if target_idx > old_line:
                dwell = (
                    datetime.now() - current.line_updated_at
                ).total_seconds()
                if (
                    dwell < config.matcher.min_line_dwell_seconds
                    and target_score < config.matcher.line_advance_override_score
                ):
                    print(
                        f"  [DWELL HOLD] line {old_line}→{target_idx} "
                        f"(dwell={dwell:.2f}s < {config.matcher.min_line_dwell_seconds:.1f}s, "
                        f"score={target_score:.2f} < {config.matcher.line_advance_override_score:.2f})"
                    )
                    target_idx = old_line
                    target_score = current_scores[old_line]

            self.tracker.update_line(target_idx, target_score)
            # Keep STTM line in sync in both directions to avoid drift.
            delta = target_idx - old_line
            if delta > 0:
                for _ in range(delta):
                    await self.controller.navigate_line("next")
            elif delta < 0:
                for _ in range(abs(delta)):
                    await self.controller.navigate_line("prev")
            target_verse = current.verses[target_idx]
            print(
                f"  [LINE {target_idx}/{len(current.verses)}] "
                f"score={target_score:.2f} — {target_verse.unicode[:50]}"
            )
        else:
            self.tracker.mark_unstable()

            if is_detour:
                # During alap we expect the current shabad to score weakly — don't
                # let that push us toward a recovery release.
                pass
            elif on_micro_window_flag:
                # A single-micro-window dip is normal (3 s = few letters); don't feed
                # it into the release counter either. If the micro window is persistently
                # bad _is_lock_stable() will switch us back off micro, after which the
                # regular counter can correctly detect a sustained weak alignment.
                pass
            elif best_line_score < config.matcher.weak_line_recovery_score:
                self._weak_line_windows += 1
            else:
                self._weak_line_windows = max(0, self._weak_line_windows - 1)

            if self._weak_line_windows >= config.matcher.weak_line_recovery_windows:
                print(
                    f"  [RECOVERY] weak line for {self._weak_line_windows} windows "
                    f"(score={best_line_score:.2f}) — releasing lock from shabad {current.shabad_id}"
                )
                self.tracker.release_lock()
                self._weak_line_windows = 0
                await self._broadcast({
                    "type": "shabad_switched",
                    "old_shabad_id": current.shabad_id,
                    "new_shabad_id": None,
                    "reason": "weak_locked_recovery",
                })
                return

        # Skip global challenger scan during an alap detour — the sticky-set handler
        # already tracks the would-be switch and will commit on its own timeline. Running
        # both races the counters and can cause premature switches on brief alaps.
        if is_detour:
            return

        # Run challenger scan every locked cycle (not only weak cycles) so wrong-lock
        # recovery can happen quickly.
        reason = "background_monitor" if should_update_line else "weak_line_match"
        await self._scan_challenger(
            first_letters=first_letters,
            transcript_text=transcript_text,
            current=current,
            best_line_idx=best_line_idx,
            best_line_score=best_line_score,
            reason=reason,
        )

    async def _scan_challenger(
        self,
        first_letters: str,
        transcript_text: str,
        current,
        best_line_idx: int,
        best_line_score: float,
        reason: str,
    ):
        """Search globally for challenger shabads while locked."""
        candidates = await asyncio.to_thread(
            self.searcher.search,
            first_letters,
            10,
            False,
            transcript_text,
        )
        scored = self._score_candidates(first_letters, candidates, transcript_text)
        for candidate in scored:
            if candidate["shabad_id"] == current.shabad_id:
                candidate["line_idx"] = best_line_idx
                break
        self.tracker.observe_candidates(scored, self._window_index)
        best_hypothesis = self.tracker.best_hypothesis()

        # Broadcast candidates (for dashboard visibility)
        await self._broadcast({
            "type": "candidates",
            "matches": scored[:config.dashboard.max_candidates],
            "pipeline_state": "locked",
            "reason": reason,
            "hypotheses": self.tracker.get_hypotheses(),
        })

        if not scored:
            return

        top = dict(scored[0])
        top["raw_score"] = top["score"]
        if best_hypothesis and best_hypothesis["shabad_id"] != current.shabad_id:
            from_hypothesis = next(
                (
                    candidate
                    for candidate in scored
                    if candidate["shabad_id"] == best_hypothesis["shabad_id"]
                ),
                None,
            )
            if from_hypothesis and best_hypothesis["stability"] >= 2:
                top = dict(from_hypothesis)
                top["raw_score"] = top["score"]
                top["score"] = round(
                    max(top["score"], best_hypothesis["evidence_score"]), 3
                )
        current_shabad_search_score = 0.0
        for candidate in scored:
            if candidate["shabad_id"] == current.shabad_id:
                current_shabad_search_score = candidate["score"]
                break

        # Only challenge if top result is a different shabad
        if top["shabad_id"] == current.shabad_id:
            return

        top_raw_score = float(top.get("raw_score", top["score"]))
        top_word_overlap = int(top.get("word_overlap", 0))
        top_dense_dominant = bool(top.get("dense_dominant", False))
        # When a challenger's score was driven by dense_coverage (substring match
        # against its whole-shabad first-letters), raise the overlap bar — such
        # hits are prone to spurious high scores against unrelated shabads that
        # happen to contain similar sequences somewhere in their text.
        required_overlap = (
            config.matcher.dense_dominant_instant_overlap_min
            if top_dense_dominant
            else config.matcher.word_overlap_instant_challenger_min
        )
        instant_switch = (
            top_raw_score >= config.matcher.instant_challenger_switch_score
            and top_word_overlap >= required_overlap
            and (
                top_raw_score - current_shabad_search_score
                >= config.matcher.instant_challenger_switch_margin
            )
        )
        if instant_switch:
            old_shabad_id = current.shabad_id
            print(
                f"  [INSTANT SWITCH] {old_shabad_id} -> {top['shabad_id']} "
                f"raw={top_raw_score:.2f} overlap={top_word_overlap}"
            )
            self.tracker.try_lock(top["shabad_id"], top_raw_score, instant=True)
            self._weak_line_windows = 0
            await self._broadcast({
                "type": "shabad_switched",
                "new_shabad_id": top["shabad_id"],
                "old_shabad_id": old_shabad_id,
                "reason": "instant_challenger",
            })
            await self._lock_shabad_from_top(top)
            return

        recovery_mode = (
            self._weak_line_windows >= max(1, config.matcher.weak_line_recovery_windows - 1)
            and best_line_score < config.matcher.weak_line_recovery_score
        )
        if top["action"] != "auto" and not (
            recovery_mode and top["score"] >= config.matcher.recovery_challenger_score
        ):
            # Not confident enough to challenge
            return

        if recovery_mode and top["action"] != "auto":
            print(
                f"  [RECOVERY] allowing challenger {top['shabad_id']} "
                f"with score={top['score']:.2f} (current line={best_line_score:.2f})"
            )

        # Fast-switch: if current-shabad alignment has been weak for several windows
        # in a row and the challenger is *clearly* stronger, shorten the persistence
        # requirement so the switch commits in ~6 s instead of ~9 s. All four guards
        # matter — without them transient noise + an opportunistic top candidate will
        # kick STTM off the correct shabad.
        top_word_overlap_for_fast = int(top.get("word_overlap", 0))
        fast_switch_active = (
            self._current_weak_windows >= config.matcher.fast_switch_current_weak_windows
            and top["action"] == "auto"
            and top["score"] >= config.matcher.auto_threshold
            and top_word_overlap_for_fast >= config.matcher.word_overlap_evidence_min
            and (top["score"] - current_shabad_search_score)
                >= max(config.matcher.challenger_margin, 0.15)
        )
        windows_override = (
            config.matcher.fast_switch_challenger_windows if fast_switch_active else None
        )
        if fast_switch_active:
            print(
                f"  [FAST-SWITCH] current weak for {self._current_weak_windows} windows, "
                f"challenger={top['shabad_id']} score={top['score']:.2f} — "
                f"requiring only {config.matcher.fast_switch_challenger_windows} wins"
            )
        result = self.tracker.challenge(
            top["shabad_id"],
            top["score"],
            current_shabad_search_score,
            windows_override=windows_override,
        )

        if result["action"] == "switched":
            self._weak_line_windows = 0
            new_id = result["new_shabad_id"]
            print(f"  [SWITCH] Challenger {new_id} wins! Transitioning...")
            await self._broadcast({
                "type": "shabad_switched",
                "new_shabad_id": new_id,
                "old_shabad_id": current.shabad_id,
            })
            # The tracker moved to CANDIDATE_LOCK with pending_id pre-seeded,
            # so next cycle's try_lock will confirm and lock the new shabad.
            return

        if result["action"] == "challenging":
            print(
                f"  [CHALLENGER] {top['shabad_id']} "
                f"wins {result['wins']}/{result['needed']}"
            )
            await self._broadcast({
                "type": "challenger_update",
                "challenger": top,
                "wins": result["wins"],
                "needed": result["needed"],
            })

    def _score_candidates(
        self,
        first_letters: str,
        candidates: list[ShabadCandidate],
        transcript_text: str = "",
    ) -> list[dict]:
        """Score and sort a list of candidates. Returns list of dicts."""
        current_id = self.tracker.current.shabad_id if self.tracker.current else None
        scored = []
        for candidate in candidates:
            detail = self.scorer.score_detailed(
                first_letters,
                candidate,
                current_id,
            )
            score = detail["score"]
            dense_dominant = detail["dense_dominant"]
            overlap_source = candidate.unicode
            if candidate.gurmukhi and candidate.gurmukhi not in overlap_source:
                overlap_source = f"{overlap_source} {candidate.gurmukhi}"
            word_overlap = self.scorer.word_overlap_count(transcript_text, overlap_source)
            retrieval_sources = sorted(candidate.retrieval_sources) if candidate.retrieval_sources else []
            if "type2" in retrieval_sources:
                # Type=2 phrase retrieval is high precision; add a modest bonus
                # when transcript words also overlap the verse.
                if word_overlap >= 2:
                    score += 0.10
                elif word_overlap >= 1:
                    score += 0.05
            if "multiline" in retrieval_sources:
                # Both halves of a long query hit consecutive lines — strong signal
                # for dense text (nitnem, fast kirtan) where one window = 2+ DB lines.
                score += config.matcher.multi_line_score_bonus
            if "multiline3" in retrieval_sources:
                # All three thirds of the query hit 3 consecutive lines — even
                # stronger evidence for very fast / dense recitation (Fix 3).
                score += config.matcher.multi_line_trinary_score_bonus
            if "type3_words" in retrieval_sources:
                # Word-level IDF vote (Fix 2). Bonus scaled by distinct-word-hits —
                # more distinct real words in the transcript hitting this shabad
                # means stronger corroboration on top of first-letter evidence.
                hits = candidate.word_vote_hits
                if hits >= 4:
                    score += config.matcher.word_vote_bonus_4plus
                elif hits == 3:
                    score += config.matcher.word_vote_bonus_3
                elif hits >= 2:
                    score += config.matcher.word_vote_bonus_2
            score = min(1.0, max(0.0, score))
            action = self.scorer.classify(score)
            # Safety floor: a candidate retrieved ONLY by word-vote (no first-letter
            # strategy backed it up) must clear a higher bar before it can auto-lock.
            # Prevents a buried-words false positive from hijacking the UI when the
            # first-letter evidence is weak.
            if action == "auto" and retrieval_sources == ["type3_words"]:
                if score < config.matcher.auto_threshold + 0.05:
                    action = "suggest"
            scored.append({
                "shabad_id": candidate.shabad_id,
                "gurmukhi": candidate.gurmukhi,
                "unicode": candidate.unicode,
                "english": candidate.english,
                "line_idx": 0,
                "score": round(score, 3),
                "word_overlap": word_overlap,
                "retrieval_sources": retrieval_sources,
                "action": action,
                "word_vote_hits": candidate.word_vote_hits,
                "word_vote_score": candidate.word_vote_score,
                "dense_dominant": dense_dominant,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _apply_progression_bias(
        self, index: int, current_line: int, raw_score: float
    ) -> float:
        """
        Prefer expected recitation flow: current, next, next+1.
        Penalize far jumps unless raw confidence is already very high.
        """
        if raw_score >= config.matcher.progression_high_confidence_bypass:
            return raw_score
        delta = index - current_line
        bonus = 0.0
        if delta == 0:
            # Prefer staying on the current line — kirtan lines last ~3 s and the
            # 3 s micro window catches the next line's first syllable near the
            # handover, which used to edge next ahead on tied scores.
            bonus = 0.16
        elif delta == 1:
            bonus = 0.12
        elif delta == 2:
            bonus = 0.08
        elif delta < 0:
            bonus = -0.04 * min(abs(delta), 3)
        elif delta >= 4:
            bonus = -0.08
        else:
            bonus = -0.02 * abs(delta)
        return min(1.0, max(0.0, raw_score + bonus))

    async def _try_silence_autolock(self):
        """
        During short no-lyrics gaps, lock a recently detected high-confidence
        shabad instead of waiting for new vocals.
        """
        if self.tracker.state not in (
            PipelineState.SEARCHING,
            PipelineState.CANDIDATE_LOCK,
        ):
            return
        if not self._silence_autolock_candidate or self._silence_autolock_ttl <= 0:
            return

        top = self._silence_autolock_candidate
        self._silence_autolock_ttl -= 1
        if self._silence_autolock_ttl <= 0:
            self._silence_autolock_candidate = None

        if top["score"] < config.matcher.silence_autolock_min_score:
            return
        if self._silence_windows > config.matcher.silence_autolock_windows:
            return

        result = self.tracker.try_lock(top["shabad_id"], top["score"], instant=True)
        if result.get("action") != "locked":
            return

        await self._lock_shabad_from_top(top)
        self._after_break_windows = 0
        print(
            f"  [SILENCE AUTO-LOCK] Shabad {top['shabad_id']} "
            f"score={top['score']:.2f}"
        )
        await self._broadcast({
            "type": "silence_autolock",
            "shabad_id": top["shabad_id"],
            "score": top["score"],
        })

    def _score_sticky_set(self, first_letters: str) -> dict | None:
        """
        Score the fresh window against each recently-sung shabad in the sticky set
        (tracker.history, TTL-bounded). Returns the best (shabad, line, score) across
        that set — caller decides whether it clears the alap detour threshold.
        """
        if not first_letters or len(first_letters) < config.matcher.min_search_letters:
            return None
        sticky = self.tracker.get_sticky_set(
            ttl_seconds=config.matcher.alap_sticky_ttl_seconds,
            max_size=config.matcher.alap_sticky_max_size,
        )
        if not sticky:
            return None
        best: dict | None = None
        for state in sticky:
            best_idx = 0
            best_score = 0.0
            for i, verse in enumerate(state.verses):
                score = self.scorer.score_line(first_letters, verse.first_letters)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_score <= 0:
                continue
            if best is None or best_score > best["score"]:
                verse = state.verses[best_idx]
                best = {
                    "shabad_id": state.shabad_id,
                    "line_idx": best_idx,
                    "score": best_score,
                    "unicode": verse.unicode,
                    "english": verse.english,
                }
        return best

    async def _commit_sticky_switch(self, detour_match: dict, previous_shabad_id: int):
        """
        Promote an alap detour to a real shabad switch: recall the history shabad
        (moves current → history, brings the detour back as current), drive STTM to
        the detour line, and broadcast the usual switch + lock events.
        """
        shabad_id = detour_match["shabad_id"]
        if not self.tracker.recall_from_history(shabad_id):
            return
        self._weak_line_windows = 0
        self._current_weak_windows = 0
        self._recent_line_scores.clear()
        current = self.tracker.current
        target_line = detour_match["line_idx"]
        await self.controller.display_shabad(shabad_id)
        if current and 0 <= target_line < len(current.verses):
            # display_shabad resets STTM to line 0 — step forward to the detour line.
            for _ in range(target_line):
                await self.controller.navigate_line("next")
            current.current_line = target_line
        verses = current.verses if current else []
        await self._broadcast({
            "type": "shabad_switched",
            "new_shabad_id": shabad_id,
            "old_shabad_id": previous_shabad_id,
            "reason": "alap_commit",
        })
        await self._broadcast({
            "type": "shabad_locked",
            "shabad_id": shabad_id,
            "total_lines": len(verses),
            "verses": [
                {"unicode": v.unicode, "english": v.english}
                for v in verses
            ],
        })

    async def _lock_shabad_from_top(self, top: dict):
        """Display a locked shabad and cache verses for line tracking."""
        await self.controller.display_shabad(top["shabad_id"])
        verses = await asyncio.to_thread(
            self.searcher.fetch_all_verses, top["shabad_id"]
        )
        self.tracker.set_shabad_details(
            top["gurmukhi"], top["unicode"], top["english"], verses
        )
        await self._broadcast({
            "type": "shabad_locked",
            "shabad_id": top["shabad_id"],
            "shabad": top,
            "total_lines": len(verses),
            "verses": [
                {"unicode": v.unicode, "english": v.english}
                for v in verses
            ],
        })
        self._silence_autolock_candidate = None
        self._silence_autolock_ttl = 0

    def _update_vocal_break_state(self, has_vocals: bool):
        """Track no-lyrics gaps and trigger a short post-break start mode."""
        if has_vocals:
            if self._in_vocal_break:
                self._after_break_windows = config.matcher.post_break_boost_windows
                self._in_vocal_break = False
                print(
                    f"  [BREAK END] Boosting start detection for "
                    f"{self._after_break_windows} windows"
                )
            self._silence_windows = 0
            return

        self._silence_windows += 1
        if (
            not self._in_vocal_break
            and self._silence_windows >= config.matcher.vocal_break_min_windows
        ):
            self._in_vocal_break = True
            print("  [BREAK START] Detected vocal pause")

    def _is_lock_stable(self) -> bool:
        """
        True when the last few windows have aligned strongly on the current shabad.
        Gates the shortest (micro) audio window — we only trust 3 s of audio when
        the line pointer has been moving confidently.
        """
        needed = max(1, config.matcher.locked_stable_min_windows)
        if len(self._recent_line_scores) < needed:
            return False
        threshold = config.matcher.locked_stable_score_threshold
        return all(score >= threshold for score in self._recent_line_scores[-needed:])

    def _select_transcription_audio(self, window):
        """Choose dynamic transcription window based on current tracking context."""
        samplerate = config.audio.samplerate
        if self._after_break_windows > 0:
            seconds = config.audio.start_window_duration
            mode = "start_boost"
        elif self.tracker.state in (PipelineState.LOCKED, PipelineState.UNSTABLE_LOCK):
            if self.tracker.state == PipelineState.UNSTABLE_LOCK or self._weak_line_windows > 0:
                seconds = config.audio.locked_recovery_window_duration
                mode = "locked_recover"
            elif self._is_lock_stable():
                # Ragi typically finishes a line in ~3 s. Once the lock is confirmed
                # stable we only need one line's worth of audio to move the pointer,
                # which lets the UI keep up with brisk recitation.
                seconds = config.audio.locked_micro_window_duration
                mode = "locked_micro"
            elif self._speech_rate_lps >= config.matcher.very_fast_speech_letters_per_second:
                # Very fast recitation — shrink the window further so one
                # window doesn't routinely span 3+ lines (Fix 3).
                seconds = config.audio.locked_very_fast_window_duration
                mode = "locked_very_fast"
            elif self._speech_rate_lps >= config.matcher.fast_speech_letters_per_second:
                seconds = config.audio.locked_fast_window_duration
                mode = "locked_fast"
            elif self._speech_rate_lps <= config.matcher.slow_speech_letters_per_second:
                seconds = config.audio.locked_recovery_window_duration
                mode = "locked_slow"
            else:
                seconds = config.audio.locked_window_duration
                mode = "locked_follow"
        else:
            if self._speech_rate_lps >= config.matcher.fast_speech_letters_per_second:
                seconds = config.audio.search_fast_window_duration
                mode = "search_fast"
            else:
                seconds = config.audio.window_duration
                mode = "search"

        samples = int(seconds * samplerate)
        if samples <= 0 or samples >= len(window):
            return window, mode, len(window) / samplerate
        return window[-samples:], mode, seconds

    @staticmethod
    async def _noop_broadcast(data: dict):
        """Default no-op broadcast (prints to console)."""
        msg_type = data.get("type", "")
        if msg_type == "transcription" and data.get("text"):
            state = data.get("pipeline_state", "?")
            print(f"  [{state.upper()}] Heard: {data['text']}")
            print(f"  [Letters] {data['first_letters']}")
        elif msg_type == "error":
            print(f"  [ERROR] {data['message']}")
