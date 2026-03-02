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
        self.tracker = ShabadTracker(
            challenger_windows=config.matcher.challenger_windows,
            challenger_margin=config.matcher.challenger_margin,
        )
        self.controller = controller
        self._broadcast = broadcast or self._noop_broadcast
        self.running = False
        self.paused = False
        self._silence_s = 0.0

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
                # 1. Get audio chunk
                chunk = await asyncio.to_thread(self.audio.get_chunk)
                if chunk is None:
                    break

                # Apply overlap buffer
                window = self.buffer.process(chunk)

                # Log audio level
                rms = float(np.sqrt(np.mean(window**2)))
                has_vocals = bool(self.transcriber.has_vocal_content(window))
                await self._broadcast({
                    "type": "audio_level",
                    "rms": round(rms, 4),
                    "has_vocals": has_vocals,
                })

                if self.paused:
                    await self._broadcast({"type": "paused"})
                    await asyncio.sleep(0.1)
                    continue

                # Skip transcription if no vocal content (just music/silence)
                if not has_vocals:
                    self._silence_s += config.audio.step_duration
                    await self._broadcast({
                        "type": "transcription",
                        "text": "",
                        "first_letters": "",
                        "status": "music_only",
                    })
                    continue

                if self._silence_s >= config.matcher.long_vocal_break_s:
                    if self.tracker.state == PipelineState.LOCKED:
                        print(
                            f"  [BREAK] {self._silence_s:.1f}s vocal gap — releasing lock for fresh shabad detection"
                        )
                        self.tracker.release_lock()
                        await self._broadcast({
                            "type": "long_break_reset",
                            "silence_s": round(self._silence_s, 1),
                            "pipeline_state": self.tracker.state.value,
                        })
                self._silence_s = 0.0

                # 2. Transcribe
                segments = await asyncio.to_thread(self.transcriber.transcribe, window)
                text = self.processor.process(segments)

                # 3. Extract first letters
                first_letters = extract_first_letters(text)

                # 4. Broadcast transcription
                await self._broadcast({
                    "type": "transcription",
                    "text": text,
                    "first_letters": first_letters,
                    "pipeline_state": self.tracker.state.value,
                })

                # 5. Skip if too few letters
                if len(first_letters) < config.matcher.min_search_letters:
                    continue

                # 6. Dispatch to state handler
                if self.tracker.state == PipelineState.SEARCHING:
                    await self._handle_searching(first_letters)
                else:
                    await self._handle_locked(first_letters)

                # 7. Broadcast current state (include verses when locked)
                current = self.tracker.current
                state_msg = {
                    "type": "state",
                    "pipeline_state": self.tracker.state.value,
                    "current": current.to_dict() if current else None,
                    "history": self.tracker.get_history_list(),
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

    async def _handle_searching(self, first_letters: str):
        """SEARCHING state: low-latency primary search, then broaden only if needed."""
        search_limit = max(config.dashboard.max_candidates * 2, 10)
        # First pass: fastest/high-precision search (first-letter beginning only).
        primary = await asyncio.to_thread(
            self.searcher.search, first_letters, search_limit, "fast"
        )

        # Score candidates
        scored = self._score_candidates(first_letters, primary)

        # If primary is weak/sparse, broaden retrieval to improve recall.
        need_fallback = (
            not scored
            or scored[0]["action"] != "auto"
            or len(scored) < config.dashboard.max_candidates
        )
        if need_fallback:
            expanded = await asyncio.to_thread(
                self.searcher.search, first_letters, search_limit, "balanced"
            )
            merged = self._merge_candidates(primary, expanded)
            scored = self._score_candidates(first_letters, merged)

        top_candidates = scored[:config.dashboard.max_candidates]

        # Broadcast candidates
        await self._broadcast({
            "type": "candidates",
            "matches": top_candidates,
            "pipeline_state": "searching",
        })

        # Try to lock on strong match
        if top_candidates and top_candidates[0]["action"] == "auto":
            top = top_candidates[0]
            instant = top["score"] >= config.matcher.instant_lock_threshold
            result = self.tracker.try_lock(top["shabad_id"], top["score"], instant=instant)

            if result["action"] == "locked":
                verses = await self._lock_and_broadcast(top)
                print(f"  [LOCKED] Shabad {top['shabad_id']} — {top['unicode'][:60]} ({len(verses)} verses)")

            elif result["action"] == "pending":
                await self._broadcast({
                    "type": "pending_lock",
                    "shabad": top,
                })
                print(f"  [PENDING] Confirming shabad {top['shabad_id']}...")

    async def _handle_locked(self, first_letters: str):
        """LOCKED state: align line within shabad, check for challenger."""
        current = self.tracker.current
        if not current or not current.verses:
            # No verses cached — fall back to searching
            self.tracker.state = PipelineState.SEARCHING
            return

        # Score against each line of the locked shabad
        best_line_idx = 0
        best_line_score = 0.0
        for i, verse in enumerate(current.verses):
            score = self.scorer.score_line(first_letters, verse.first_letters)
            if score > best_line_score:
                best_line_score = score
                best_line_idx = i

        # Broadcast line alignment
        best_verse = current.verses[best_line_idx]
        await self._broadcast({
            "type": "line_aligned",
            "line_index": best_line_idx,
            "line_score": round(best_line_score, 3),
            "line_unicode": best_verse.unicode,
            "line_english": best_verse.english,
            "pipeline_state": "locked",
        })

        # If current line matches well, just update position
        if best_line_score >= config.matcher.suggest_threshold:
            old_line = current.current_line
            self.tracker.update_line(best_line_idx, best_line_score)
            # Navigate STTM forward if line advanced
            if best_line_idx > old_line:
                for _ in range(best_line_idx - old_line):
                    await self.controller.navigate_line("next")
            print(f"  [LINE {best_line_idx}/{len(current.verses)}] score={best_line_score:.2f} — {best_verse.unicode[:50]}")
            return

        # Poor line match — do a broad search for potential challenger
        print(f"  [WEAK] line_score={best_line_score:.2f}, searching for challenger...")
        candidates = await asyncio.to_thread(
            self.searcher.search, first_letters, 12, "broad"
        )
        scored = self._score_candidates(first_letters, candidates)

        # Broadcast candidates (for dashboard visibility)
        await self._broadcast({
            "type": "candidates",
            "matches": scored[:config.dashboard.max_candidates],
            "pipeline_state": "locked",
            "reason": "weak_line_match",
        })

        if not scored:
            return

        top = scored[0]
        # Only challenge if top result is a different shabad
        if top["shabad_id"] == current.shabad_id:
            # Still matching current shabad (just a different line)
            self.tracker.update_line(best_line_idx, best_line_score)
            return

        if top["action"] != "auto":
            # Not confident enough to challenge
            return

        # Very strong challenger: switch immediately to reduce lock-to-lock latency.
        instant_switch = (
            top["score"] >= config.matcher.instant_lock_threshold
            and (top["score"] - best_line_score) >= (config.matcher.challenger_margin + 0.10)
        )
        if instant_switch:
            print(f"  [FAST SWITCH] {current.shabad_id} -> {top['shabad_id']} "
                  f"(new={top['score']:.2f}, current={best_line_score:.2f})")
            self.tracker.manual_lock(top["shabad_id"])
            verses = await self._lock_and_broadcast(top)
            print(f"  [LOCKED] Shabad {top['shabad_id']} — {top['unicode'][:60]} ({len(verses)} verses)")
            return

        result = self.tracker.challenge(
            top["shabad_id"], top["score"], best_line_score
        )

        if result["action"] == "switched":
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
        self, first_letters: str, candidates: list[ShabadCandidate]
    ) -> list[dict]:
        """Score and sort a list of candidates. Returns list of dicts."""
        current_id = self.tracker.current.shabad_id if self.tracker.current else None
        scored = []
        for candidate in candidates:
            score = self.scorer.score(first_letters, candidate, current_id)
            action = self.scorer.classify(score)
            scored.append({
                "shabad_id": candidate.shabad_id,
                "gurmukhi": candidate.gurmukhi,
                "unicode": candidate.unicode,
                "english": candidate.english,
                "score": round(score, 3),
                "action": action,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _lock_and_broadcast(self, candidate: dict) -> list:
        """Display, cache verses, and broadcast lock event for a chosen shabad."""
        shabad_id = candidate["shabad_id"]
        await self.controller.display_shabad(shabad_id)
        verses = await asyncio.to_thread(self.searcher.fetch_all_verses, shabad_id)
        self.tracker.set_shabad_details(
            candidate.get("gurmukhi", ""),
            candidate.get("unicode", ""),
            candidate.get("english", ""),
            verses,
        )
        await self._broadcast({
            "type": "shabad_locked",
            "shabad_id": shabad_id,
            "shabad": candidate,
            "total_lines": len(verses),
            "verses": [
                {"unicode": v.unicode, "english": v.english}
                for v in verses
            ],
        })
        return verses

    @staticmethod
    def _merge_candidates(
        primary: list[ShabadCandidate], secondary: list[ShabadCandidate]
    ) -> list[ShabadCandidate]:
        """Merge candidate lists by shabad_id while preserving first-seen order."""
        merged: list[ShabadCandidate] = []
        seen: set[int] = set()
        for candidate in primary + secondary:
            if candidate.shabad_id in seen:
                continue
            seen.add(candidate.shabad_id)
            merged.append(candidate)
        return merged

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
