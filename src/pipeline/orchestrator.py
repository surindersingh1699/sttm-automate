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
from src.matcher.tracker import ShabadTracker
from src.controller.base import STTMController


# Type for the dashboard broadcast callback
BroadcastFn = Callable[[dict], Awaitable[None]]


class PipelineOrchestrator:
    """
    Wires all components into a continuous processing loop:
    audio capture → transcription → transliteration → search → score → control → broadcast
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
            transition_threshold=config.matcher.transition_count
        )
        self.controller = controller
        self._broadcast = broadcast or self._noop_broadcast
        self.running = False
        self.paused = False

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
        result = self.tracker.update(shabad_id, confidence=1.0)
        await self.controller.display_shabad(shabad_id)
        await self._broadcast({
            "type": "manual_selected",
            "shabad_id": shabad_id,
            "tracker": result,
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
            await self._broadcast({
                "type": "recalled",
                "shabad_id": shabad_id,
            })

    async def _run_loop(self):
        """Main processing loop."""
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
                    await self._broadcast({
                        "type": "transcription",
                        "text": "",
                        "first_letters": "",
                        "status": "music_only",
                    })
                    continue

                # 2. Transcribe
                segments = await asyncio.to_thread(self.transcriber.transcribe, window)
                text = self.processor.process(segments)

                # 3. Extract first letters
                first_letters = extract_first_letters(text)

                # 4. Broadcast transcription to dashboard
                await self._broadcast({
                    "type": "transcription",
                    "text": text,
                    "first_letters": first_letters,
                })

                # 5. Skip if too few letters
                if len(first_letters) < config.matcher.min_search_letters:
                    continue

                # 6. Search BaniDB
                candidates = await asyncio.to_thread(
                    self.searcher.search, first_letters
                )

                # 7. Score candidates
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
                top_candidates = scored[:config.dashboard.max_candidates]

                # 8. Broadcast candidates to dashboard
                await self._broadcast({
                    "type": "candidates",
                    "matches": top_candidates,
                })

                # 9. Act on top match
                if top_candidates and top_candidates[0]["action"] == "auto":
                    top = top_candidates[0]
                    tracker_result = self.tracker.update(
                        top["shabad_id"], top["score"]
                    )

                    if tracker_result["action"] in ("started", "switched"):
                        await self.controller.display_shabad(top["shabad_id"])
                        self.tracker.set_shabad_details(
                            top["gurmukhi"], top["unicode"], top["english"]
                        )
                        await self._broadcast({
                            "type": "auto_selected",
                            "shabad": top,
                            "tracker": tracker_result,
                        })

                # 10. Broadcast current state
                await self._broadcast({
                    "type": "state",
                    "current": self.tracker.current.to_dict() if self.tracker.current else None,
                    "history": self.tracker.get_history_list(),
                })

            except Exception as e:
                print(f"[Pipeline] Error in loop: {e}")
                await self._broadcast({"type": "error", "message": str(e)})
                await asyncio.sleep(1)

    @staticmethod
    async def _noop_broadcast(data: dict):
        """Default no-op broadcast (prints to console)."""
        msg_type = data.get("type", "")
        if msg_type == "transcription" and data.get("text"):
            print(f"  [Heard] {data['text']}")
            print(f"  [Letters] {data['first_letters']}")
        elif msg_type == "candidates" and data.get("matches"):
            top = data["matches"][0]
            print(f"  [Match] {top['unicode']} (score: {top['score']}, action: {top['action']})")
        elif msg_type == "auto_selected":
            print(f"  [AUTO] Selected shabad {data['shabad']['shabad_id']}")
        elif msg_type == "error":
            print(f"  [ERROR] {data['message']}")
