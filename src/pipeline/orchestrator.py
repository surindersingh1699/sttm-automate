"""Main pipeline: audio → transcription → matching → STTM control → dashboard."""

import asyncio
from typing import Callable, Awaitable

from src.config import config
from src.audio.capture import AudioCapture
from src.audio.buffer import AudioRingBuffer
from src.transcription.engine import TranscriptionEngine
from src.transcription.processor import TranscriptionProcessor
from src.transcription.transliterate import extract_first_letters
from src.matcher.search import ShabadSearcher, ShabadCandidate
from src.matcher.scorer import ConfidenceScorer
from src.matcher.tracker import ShabadTracker, PipelineState
from src.controller.base import STTMController


# Type for the dashboard broadcast callback
BroadcastFn = Callable[[dict], Awaitable[None]]


class PipelineOrchestrator:
    """
    Wires all components into a continuous processing loop.

    Uses a SEARCHING/LOCKED state machine:
    - SEARCHING: broad BaniDB search, confirm strong match before locking
    - LOCKED: track line position within shabad, only switch on sustained challenger
    """

    def __init__(
        self,
        controller: STTMController,
        broadcast: BroadcastFn | None = None,
        audio_device: int | None = None,
    ):
        # Auto-detect best audio device (BlackHole > aggregate > default)
        if audio_device is None:
            audio_device = AudioCapture.find_best_device()
        self.audio = AudioCapture(device=audio_device)
        self.buffer = AudioRingBuffer()
        self.transcriber = TranscriptionEngine()
        self.processor = TranscriptionProcessor()
        self.searcher = ShabadSearcher()
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

    async def start(self):
        """Initialize components and start the processing loop."""
        print("[Pipeline] Loading Whisper model...")
        await asyncio.to_thread(self.transcriber.load)

        print("[Pipeline] Connecting to STTM...")
        connected = await self.controller.connect()
        if not connected:
            print("[Pipeline] WARNING: STTM not connected. Running in monitor-only mode.")

        print("[Pipeline] Starting audio capture...")
        self.audio.start()
        self.running = True

        print("[Pipeline] Pipeline running. Listening for kirtan...")
        await self._run_loop()

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
        await self._broadcast({
            "type": "shabad_switched",
            "old_shabad_id": old_id,
            "new_shabad_id": None,
            "reason": "force_unlock",
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

    async def _run_loop(self):
        """Main processing loop with state machine dispatch."""
        import numpy as np

        while self.running:
            try:
                self._window_index += 1
                # 1. Get audio chunk
                chunk = await asyncio.to_thread(self.audio.get_chunk)
                if chunk is None:
                    break

                # Apply overlap buffer
                window = self.buffer.process(chunk)

                # Detect vocals from fresh chunk (not the long rolling window),
                # so real breaks are detected quickly.
                chunk_rms = float(np.sqrt(np.mean(chunk**2)))
                has_vocals = bool(self.transcriber.has_vocal_content(chunk))
                self._update_vocal_break_state(has_vocals)

                # Adaptive listening window: after a vocal break use short start window,
                # while locked use medium window, otherwise use full search window.
                audio_for_stt, listening_mode, window_seconds = self._select_transcription_audio(window)
                await self._broadcast({
                    "type": "audio_level",
                    "rms": round(chunk_rms, 4),
                    "has_vocals": has_vocals,
                    "speech_rate_lps": round(self._speech_rate_lps, 2),
                    "listening_mode": listening_mode,
                    "window_seconds": round(window_seconds, 2),
                })

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
                segments = await asyncio.to_thread(
                    self.transcriber.transcribe, audio_for_stt
                )
                text = self.processor.process(segments)

                # 3. Extract first letters
                first_letters = extract_first_letters(text)
                instantaneous_lps = len(first_letters) / max(window_seconds, 0.1)
                alpha = max(0.01, min(0.99, config.matcher.speech_rate_ema_alpha))
                self._speech_rate_lps = (
                    (1.0 - alpha) * self._speech_rate_lps
                    + alpha * instantaneous_lps
                )

                # 4. Broadcast transcription
                await self._broadcast({
                    "type": "transcription",
                    "text": text,
                    "first_letters": first_letters,
                    "speech_rate_lps": round(self._speech_rate_lps, 2),
                    "pipeline_state": self.tracker.state.value,
                    "listening_mode": listening_mode,
                    "window_seconds": round(window_seconds, 2),
                })

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
        # Broad BaniDB search
        candidates = await asyncio.to_thread(
            self.searcher.search,
            first_letters,
            10,
            start_mode,
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

        combined_variants: list[tuple[str, str]] = []
        if prev_first_letters:
            candidate_combined = [
                ("prev_current", f"{prev_first_letters}{first_letters}"),
                ("current_prev", f"{first_letters}{prev_first_letters}"),
            ]
            # De-duplicate equivalent stitched strings while preserving order.
            seen_combined: set[str] = set()
            for label, letters in candidate_combined:
                if letters and letters != first_letters and letters not in seen_combined:
                    combined_variants.append((label, letters))
                    seen_combined.add(letters)

        # Score against each line using current-only baseline and stitched alternatives.
        current_scores: list[float] = []
        combined_scores: list[float] = []
        combined_labels: list[str] = []
        best_current_idx = 0
        best_current_score = 0.0
        best_combined_idx = 0
        best_combined_score = 0.0
        best_combined_label = "current"
        for i, verse in enumerate(current.verses):
            raw_current = self.scorer.score_line(first_letters, verse.first_letters)
            current_score = self._apply_progression_bias(i, current.current_line, raw_current)
            current_scores.append(current_score)
            if current_score > best_current_score:
                best_current_score = current_score
                best_current_idx = i

            line_best_combined = current_score
            line_best_label = "current"
            for label, query_letters in combined_variants:
                raw_score = self.scorer.score_line(query_letters, verse.first_letters)
                candidate_score = self._apply_progression_bias(i, current.current_line, raw_score)
                if candidate_score > line_best_combined:
                    line_best_combined = candidate_score
                    line_best_label = label
            combined_scores.append(line_best_combined)
            combined_labels.append(line_best_label)
            if line_best_combined > best_combined_score:
                best_combined_score = line_best_combined
                best_combined_idx = i
                best_combined_label = line_best_label

        # Only pick stitched result when it beats the single-window baseline.
        use_combined = best_combined_label != "current" and best_combined_score > best_current_score
        if use_combined:
            line_scores = combined_scores
            best_line_idx = best_combined_idx
            best_line_score = best_combined_score
            best_line_variant = best_combined_label
        else:
            line_scores = current_scores
            best_line_idx = best_current_idx
            best_line_score = best_current_score
            best_line_variant = "current"

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
        })

        # If current line matches well, or nearby line matches reasonably, update position.
        should_update_line = (
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
            return
        self.tracker.mark_unstable()

        if best_line_score < config.matcher.weak_line_recovery_score:
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

        # Poor line match — do a broad search for potential challenger
        print(f"  [WEAK] line_score={best_line_score:.2f}, searching for challenger...")
        candidates = await asyncio.to_thread(self.searcher.search, first_letters)
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
            "reason": "weak_line_match",
            "hypotheses": self.tracker.get_hypotheses(),
        })

        if not scored:
            return

        top = scored[0]
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
            # Still matching current shabad (just a different line)
            self.tracker.update_line(best_line_idx, best_line_score)
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

        result = self.tracker.challenge(
            top["shabad_id"], top["score"], current_shabad_search_score
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
            # The tracker moved to SEARCHING with pending_id pre-seeded,
            # so next cycle's try_lock will confirm and lock the new shabad

        elif result["action"] == "challenging":
            print(f"  [CHALLENGER] {top['shabad_id']} wins {result['wins']}/{result['needed']}")
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
            score = self.scorer.score(
                first_letters,
                candidate,
                current_id,
            )
            action = self.scorer.classify(score)
            overlap_source = candidate.unicode
            if candidate.gurmukhi and candidate.gurmukhi not in overlap_source:
                overlap_source = f"{overlap_source} {candidate.gurmukhi}"
            word_overlap = self.scorer.word_overlap_count(transcript_text, overlap_source)
            scored.append({
                "shabad_id": candidate.shabad_id,
                "gurmukhi": candidate.gurmukhi,
                "unicode": candidate.unicode,
                "english": candidate.english,
                "line_idx": 0,
                "score": round(score, 3),
                "word_overlap": word_overlap,
                "action": action,
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
            bonus = 0.12
        elif delta == 1:
            bonus = 0.16
        elif delta == 2:
            bonus = 0.10
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
