"""Main pipeline: audio → transcription → matching → STTM control → dashboard."""

import asyncio
from datetime import datetime
import numpy as np
from typing import Callable, Awaitable

from src.config import config
from src.audio.capture import AudioCapture
from src.audio.vad import StreamingVAD, Utterance
from src.pipeline.local_agreement import LocalAgreementBuffer
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

# Gurmukhi standalone vowels (U+0A05–U+0A14) and vowel matras (U+0A3E–U+0A4C).
# A word consisting only of these characters is a bare vowel sound — typical of
# melismatic/alaap singing where no consonant-rooted syllable is articulated.
_GURMUKHI_VOWELS = frozenset(
    "\u0A05\u0A06\u0A07\u0A08\u0A09\u0A0A\u0A0F\u0A10\u0A13\u0A14"  # standalone
    "\u0A3E\u0A3F\u0A40\u0A41\u0A42\u0A47\u0A48\u0A4B\u0A4C"        # matras
)


def _is_vowel_only(word: str) -> bool:
    """True when every Gurmukhi character in the word is a standalone vowel or matra."""
    gurmukhi_chars = [ch for ch in word if "\u0A00" <= ch <= "\u0A7F"]
    return bool(gurmukhi_chars) and all(ch in _GURMUKHI_VOWELS for ch in gurmukhi_chars)


def _text_to_segments(text: str):
    """Wrap a plain text string into a single-element ``TranscriptionSegment``
    list so ``LocalAgreementBuffer.commit_decode`` (which expects segments)
    can consume both raw Whisper output and post-processed strings."""
    from src.transcription.engine import TranscriptionSegment

    if not text:
        return []
    return [TranscriptionSegment(start=0.0, end=0.0, text=text)]


def _is_alaap_output(transcript_text: str) -> bool:
    """Detect non-lexical melismatic windows (alaap, vowel extension, melisma).

    Heuristics:
      - Empty / whitespace → silence/instrumental
      - >60% of words are bare vowel sounds
      - Only 1–2 distinct tokens repeated ≥3 times (raga syllable loop)
      - All tokens ≤2 chars (short melisma bursts with no real words)
    """
    if not transcript_text.strip():
        return True
    words = [w for w in transcript_text.split() if w]
    if not words:
        return True
    vowel_ratio = sum(1 for w in words if _is_vowel_only(w)) / len(words)
    if vowel_ratio > 0.6:
        return True
    if len(set(words)) <= 2 and len(words) >= 3:
        return True
    if all(len(w) <= 2 for w in words):
        return True
    return False


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
        self.mic_muted = False  # When True, local mic stream is stopped and remote audio is dropped.
        self._audio_source = "local"  # "local" or "remote"
        # Parallel capture ↔ decode handoff (naive streaming mode).
        # The capture task wakes every `step_duration` seconds, grabs the newest
        # `window_duration` seconds of audio from the AudioCapture ring, and stashes
        # it here. The decode task awaits `_window_ready`, reads whatever is most
        # recent, and runs Whisper on it. If the decoder is busy when new snapshots
        # arrive, they silently overwrite the slot — we always decode the freshest
        # audio, never a backlog. This is what keeps the UI live when Whisper RTF > 1.
        self._latest_window_data: dict | None = None
        self._window_ready = asyncio.Event()
        self._latest_window_lock = asyncio.Lock()
        # Live VAD instance (REA-10) — only constructed when streaming_mode is
        # vad_segmented or hybrid; ``flush_context`` resets it.
        self._streaming_vad: StreamingVAD | None = None
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
        # REA-11 zero-overlap mode: cache the most recent dynamic window
        # duration so _active_step_duration() can use it as the next hop.
        # 0.0 means "no decode yet" → falls through to step_duration default.
        self._last_window_seconds: float = 0.0
        # Manual navigation override — after the operator jumps to a line, suppress
        # automatic backward movement for this many windows so the pipeline doesn't
        # snap back to wherever the audio was.
        self._manual_nav_hold_windows: int = 0
        # Set to True immediately after a new shabad locks so the first _handle_locked
        # window can jump directly to the best-matching line (bypassing the 1-line cap
        # and dwell gate). Cleared as soon as the first successful line update runs.
        self._just_locked: bool = False
        # Alaap detection (Change 7): consecutive windows that scored as melismatic.
        self._alaap_window_count: int = 0
        # Transition mode (Change 8): entered when 2+ transition signals fire.
        self._in_transition_mode: bool = False
        self._transition_mode_start: float = 0.0
        self._transition_alaap_seconds: float = 0.0  # accumulated alaap/silence seconds
        # Tiered lock (Change 1): monotonic timestamp of the first window where each
        # shabad_id was the top suggest-level candidate.  Promoted to lockable once
        # (now - first_seen) >= suggest_confirmation_seconds.
        self._suggest_first_seen: dict[int, float] = {}
        # Challenger confirmation (Change 5): monotonic timestamp when each challenger
        # shabad_id first started outscoring the locked shabad.  Switch committed once
        # (now - first_seen) >= challenger_confirmation_seconds.
        # Reset whenever the current locked shabad wins a window.
        self._challenger_first_seen: dict[int, float] = {}
        # Last time a decode window was processed — used to detect ASR lag and
        # invalidate stale suggest/challenger memory.
        self._last_window_timestamp: float = 0.0

    async def start(self):
        """Initialize components and start the processing loop."""
        print("[Pipeline] Loading transcription engine...")
        await asyncio.to_thread(self.transcriber.load)

        print("[Pipeline] Connecting to STTM...")
        connected = await self.controller.connect()
        if not connected:
            print("[Pipeline] WARNING: STTM not connected. Running in monitor-only mode.")

        if self.mic_muted:
            print("[Pipeline] Mic muted on startup — skipping audio capture.")
        else:
            print("[Pipeline] Starting audio capture...")
            audio_ok = self.audio.start()
            if not audio_ok:
                print("[Pipeline] No local audio available. Defaulting to remote mic mode.")
                self._audio_source = "remote"
                await self._broadcast({"type": "audio_source_updated", "source": "remote"})
        self.running = True

        # REA-10: streaming-mode dispatch. All four modes wired:
        #   naive            — original capture-tick + decode-loop
        #   vad_segmented    — KirtanVAD utterance-bounded decode
        #   local_agreement  — re-decode growing buffer, commit on agreement
        #   hybrid           — VAD bounds + LocalAgreement-2 inside utterance
        mode = getattr(config.streaming, "streaming_mode", "naive")
        if mode == "vad_segmented":
            print(
                f"[Pipeline] streaming_mode=vad_segmented — driving decode from "
                f"VAD utterance boundaries (backend={config.streaming.vad_backend}, "
                f"threshold={config.streaming.vad_threshold}, "
                f"min_silence={config.streaming.vad_min_silence_ms}ms, "
                f"min_speech={config.streaming.vad_min_speech_ms}ms)"
            )
        elif mode == "local_agreement":
            print(
                f"[Pipeline] streaming_mode=local_agreement — re-decode "
                f"buffer every {config.streaming.local_agreement_decode_interval_ms}ms, "
                f"commit on {config.streaming.local_agreement_n}-way agreement"
            )
        elif mode == "hybrid":
            print(
                f"[Pipeline] streaming_mode=hybrid — VAD utterance bounds + "
                f"LocalAgreement-2 mid-utterance commits "
                f"(decode_interval={config.streaming.local_agreement_decode_interval_ms}ms)"
            )
        elif mode == "nemo_chunked":
            print(
                f"[Pipeline] streaming_mode=nemo_chunked — fixed "
                f"{config.whisper.nemo_chunk_len_s:.2f}s decodes, "
                f"context={config.whisper.nemo_chunk_context_s:.2f}s "
                f"(IndicConformer-only; bypasses VAD)"
            )

        print("[Pipeline] Pipeline running. Listening for kirtan...")
        # Streaming-mode-aware task layout. The naive path keeps capture and
        # decode tasks coupled via _latest_window_data. The vad_segmented and
        # hybrid paths run the VAD inline with audio capture and use a separate
        # meter task to keep the dashboard's audio-level indicator live (since
        # the VAD task only fires per utterance, not per tick).
        if mode == "vad_segmented":
            tasks = {
                asyncio.create_task(self._meter_tick_task()),
                asyncio.create_task(self._vad_streaming_loop()),
            }
        elif mode == "local_agreement":
            tasks = {
                asyncio.create_task(self._meter_tick_task()),
                asyncio.create_task(self._local_agreement_loop()),
            }
        elif mode == "hybrid":
            tasks = {
                asyncio.create_task(self._meter_tick_task()),
                asyncio.create_task(self._hybrid_streaming_loop()),
            }
        elif mode == "nemo_chunked":
            tasks = {
                asyncio.create_task(self._meter_tick_task()),
                asyncio.create_task(self._nemo_chunked_loop()),
            }
        else:
            tasks = {
                asyncio.create_task(self._capture_tick_task()),
                asyncio.create_task(self._decode_loop()),
            }
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
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

    def set_mic_muted(self, muted: bool) -> bool:
        """Mute or unmute the microphone.

        Mute fully releases the OS-level audio device (so the system mic-in-use
        indicator turns off and other apps can claim it) and drops any remote
        audio pushed while muted. Unmute restarts local capture if the source
        is "local". Returns True on success.
        """
        muted = bool(muted)
        if muted == self.mic_muted:
            return True
        if muted:
            self.mic_muted = True
            self.audio.stop()
            self.audio.reset_ring()
            print("[Pipeline] Mic muted.")
            return True
        # Unmuting
        self.mic_muted = False
        if self._audio_source == "local":
            if not self.audio.start():
                print("[Pipeline] Could not restart local audio on unmute.")
                return False
        self.audio.reset_ring()
        print("[Pipeline] Mic unmuted.")
        return True

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
            if self.mic_muted:
                # Honour mute — don't reopen the device; flip will take effect on unmute.
                pass
            elif not self.audio.start():
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
        # IndicConformer best-settings pin. Whisper-only knobs become no-ops
        # under NeMo; the dashboard hides them in lock-step (see
        # applyFamilyVisibility in app.js). Helper lives in factory so the
        # startup path can call it too — keeps the disk-loaded config
        # honest without re-running switch_engine.
        from src.transcription.factory import pin_indic_best_settings
        if pin_indic_best_settings():
            print(
                f"[Pipeline] Indic engine active — pinned streaming_mode="
                f"{config.streaming.streaming_mode}, dedup={config.streaming.dedup_strategy}, "
                f"locked_prompt_anchor=False, zero_overlap=False, hallucination_guards=False"
            )
        # Engine-family overlay: refresh VAD/capture defaults for whichever
        # family we just swapped to. "fast" speed if the user is already on
        # the indic_fast confidence mode, "normal" otherwise. set_confidence_mode
        # also calls apply_engine_profile when the user picks indic_fast, so
        # this is the symmetric path for "engine swap while keeping mode".
        from src.config import apply_engine_profile
        family = config.whisper.model_family()
        speed = "fast" if self._confidence_mode == "indic_fast" else "normal"
        if apply_engine_profile(family, speed):
            print(
                f"[Pipeline] Applied engine profile family={family} speed={speed} "
                f"(vad_min_silence={config.streaming.vad_min_silence_ms}ms, "
                f"vad_max_utt={config.streaming.vad_max_utterance_ms}ms)"
            )
        print(f"[Pipeline] Transcription engine switched to: {name}")
        return True, None

    def switch_audio_device(self, device_index: int | None):
        """Switch local audio input device (e.g. BlackHole vs MacBook Mic)."""
        self.audio.stop()
        self.audio = AudioCapture(device=device_index)
        config.audio.device = device_index
        if self._audio_source == "local" and not self.mic_muted:
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
        if self.mic_muted:
            return
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
        # Hold automatic line tracking for ~5 windows so the pipeline doesn't snap
        # back to the audio-derived line immediately after a manual jump.
        self._manual_nav_hold_windows = 5

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
        self._suggest_first_seen.clear()
        self._challenger_first_seen.clear()
        self._alaap_window_count = 0
        self._in_transition_mode = False
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
        self._suggest_first_seen.clear()
        self._challenger_first_seen.clear()
        self._alaap_window_count = 0
        self._in_transition_mode = False
        self._transition_alaap_seconds = 0.0
        # Reset VAD state so partial-utterance audio doesn't leak across the flush.
        if self._streaming_vad is not None:
            self._streaming_vad.reset()
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
            # IndicConformer + fast bani profile. Looser dwell, smaller
            # word-overlap floors, and lower thresholds so the matcher can
            # follow rapid Nitnem-style recitation where each VAD chunk is
            # ~1 pankti (≈ 1.5 s) instead of Whisper's 3 s window. Pairs with
            # apply_engine_profile("indicconformer", "fast") which tightens
            # the VAD knobs that produce these short chunks in the first place.
            "indic_fast": {
                "auto_threshold": 0.70,
                "instant_lock_threshold": 0.82,
                "min_raw_lock_score": 0.66,
                "word_overlap_auto_min": 1,
                "word_overlap_evidence_min": 1,
                "word_overlap_instant_min": 1,
                "instant_challenger_switch_score": 0.86,
                "instant_challenger_switch_margin": 0.06,
                "word_overlap_instant_challenger_min": 1,
                "suggest_threshold": 0.55,
                "challenger_margin": 0.08,
                "challenger_windows": 2,
                "candidate_lock_windows": 1,
                "weak_line_recovery_windows": 2,
                "recovery_challenger_score": 0.60,
                "local_line_follow_threshold": 0.38,
                "silence_autolock_min_score": 0.78,
                "candidate_lock_miss_windows": 5,
            },
            # Gurudwara — tighter than conservative on every threshold.
            # Live PA reverb, sangat noise, and the public visibility of false
            # locks all argue for "rather wait than lock wrong". Pair with the
            # streaming/VAD overrides documented in REA-10 ticket.
            "gurudwara": {
                "auto_threshold": 0.85,
                "instant_lock_threshold": 0.92,
                "min_raw_lock_score": 0.78,
                "word_overlap_auto_min": 2,
                "word_overlap_evidence_min": 2,
                "word_overlap_instant_min": 2,
                "instant_challenger_switch_score": 0.95,
                "instant_challenger_switch_margin": 0.16,
                "word_overlap_instant_challenger_min": 2,
                "suggest_threshold": 0.70,
                "challenger_margin": 0.16,
                "challenger_windows": 5,
                "candidate_lock_windows": 3,
                "weak_line_recovery_windows": 4,
                "recovery_challenger_score": 0.78,
                "local_line_follow_threshold": 0.50,
                "silence_autolock_min_score": 0.92,
                "candidate_lock_miss_windows": 3,
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
        # indic_fast also flips matcher knobs that aren't in the standard profile
        # shape — line-advance gating, alaap detection (false-positives on rapid
        # syllables), and fast-response — and wires up the matching engine-family
        # VAD overlay so the chunks coming in are actually short.
        if self._confidence_mode == "indic_fast":
            from src.config import apply_engine_profile
            config.matcher.min_line_dwell_seconds = 1.5
            config.matcher.min_words_for_line_advance = 1
            config.matcher.dense_dominant_margin = 0.50
            config.matcher.alaap_detection_enabled = False
            config.matcher.fast_response_enabled = True
            apply_engine_profile("indicconformer", "fast")

    @property
    def confidence_mode(self) -> str:
        return self._confidence_mode

    def _build_locked_prompt(self) -> str | None:
        """Return Gurmukhi prompt text for the current pankti, or None.

        Used when ``config.streaming.locked_prompt_anchor`` is enabled. We
        prefer the current line (most accurate "what's about to be sung")
        but fall back to the first line if the tracker is between lines.
        Returns None when the pipeline is not locked, so the engine reverts
        to its default no-prompt behavior.
        """
        current = self.tracker.current
        if current is None or self.tracker.state != PipelineState.LOCKED:
            return None
        if not current.verses:
            return None
        line_idx = max(0, min(current.current_line, len(current.verses) - 1))
        verse = current.verses[line_idx]
        text = (verse.unicode or "").strip()
        return text or None

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
        while self.running:
            # Re-read each iteration so the fast-response toggle takes effect live.
            step = max(0.1, float(self._active_step_duration()))
            try:
                fresh_chunk = self.audio.latest_window(step)
                window = self.audio.latest_window(config.audio.window_duration)

                chunk_rms = float(np.sqrt(np.mean(fresh_chunk**2))) if fresh_chunk.size else 0.0
                has_vocals = bool(self.transcriber.has_vocal_content(fresh_chunk))
                self._update_vocal_break_state(has_vocals)

                audio_for_stt, listening_mode, window_seconds = self._select_transcription_audio(window)
                # Cache for zero-overlap mode: next _active_step_duration() will
                # return this value so the next wake equals what we just decoded.
                self._last_window_seconds = float(window_seconds)

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

        Body delegated to ``_process_one_audio_window`` so the vad_segmented
        and (future) local_agreement modes share the same matcher pipeline
        without duplicating the state machine.
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
                await self._process_one_audio_window(
                    audio=window_data["audio"],
                    listening_mode=window_data["listening_mode"],
                    window_seconds=window_data["window_seconds"],
                    has_vocals=window_data["has_vocals"],
                    chunk_rms=window_data["chunk_rms"],
                    now_mono=_time.monotonic(),
                )
            except Exception as e:
                print(f"[Pipeline] Error in decode loop: {e}")
                await self._broadcast({"type": "error", "message": str(e)})
                await asyncio.sleep(1)

    # ──────────────────────────────────────────────────────────────────
    # vad_segmented streaming mode (REA-10 Phase 2)
    # ──────────────────────────────────────────────────────────────────

    def _make_streaming_vad(self) -> StreamingVAD:
        """Construct a fresh StreamingVAD wired to the current config knobs."""
        return StreamingVAD(
            backend=config.streaming.vad_backend,
            threshold=float(config.streaming.vad_threshold),
            min_silence_ms=int(config.streaming.vad_min_silence_ms),
            min_speech_ms=int(config.streaming.vad_min_speech_ms),
            speech_pad_ms=int(config.streaming.vad_speech_pad_ms),
            max_utterance_ms=int(config.streaming.vad_max_utterance_ms),
        )

    async def _meter_tick_task(self):
        """Lightweight ticker that keeps the dashboard's audio-level meter +
        vocal-break detection live in vad_segmented mode.

        Runs at a fixed cadence (independent of utterance boundaries) and
        broadcasts ``audio_level`` events. Cheap — only RMS + the existing
        ``has_vocal_content`` gate, no Whisper. Without this, the dashboard's
        audio bar would only update once per VAD utterance (1–8 s gaps).
        """
        meter_period_s = 0.5
        while self.running:
            try:
                fresh = self.audio.latest_window(meter_period_s)
                import numpy as _np
                chunk_rms = float(_np.sqrt(_np.mean(fresh ** 2))) if fresh.size else 0.0
                # has_vocal_content is the project's existing kirtan-friendly
                # presence detector — a simple RMS gate. We keep using it
                # alongside VAD to drive vocal-break state; VAD itself only
                # tells us about utterance boundaries.
                has_vocals = bool(self.transcriber.has_vocal_content(fresh))
                self._update_vocal_break_state(has_vocals)
                await self._broadcast({
                    "type": "audio_level",
                    "rms": round(chunk_rms, 4),
                    "has_vocals": has_vocals,
                    "speech_rate_lps": round(self._speech_rate_lps, 2),
                    "listening_mode": "utterance",
                    "window_seconds": 0.0,
                })
            except Exception as e:
                print(f"[Pipeline] Meter tick error: {e}")
            await asyncio.sleep(meter_period_s)

    async def _vad_streaming_loop(self):
        """Drive Whisper from VAD utterance boundaries, not a fixed clock.

        Polls the audio capture ring at a small step, feeds new samples into
        StreamingVAD, and runs Whisper exactly once per VAD-bounded utterance.
        Each second of audio is decoded ~once instead of ~3× — the headline
        CPU win in REA-10.

        Mic-mute and source-switch are honored via the audio ring (if the ring
        is empty/silent, VAD never fires). Pause is honored by checking the
        flag at utterance dispatch time. ``flush_context`` resets the VAD
        state so we don't carry stale partial-utterance buffers across an
        operator-initiated reset.
        """
        import time as _time
        import numpy as _np

        vad = self._make_streaming_vad()
        self._streaming_vad = vad
        # Poll cadence — tight enough that audio doesn't backlog in the ring,
        # loose enough that the asyncio loop has slack for the meter task and
        # for Whisper inference. 100 ms is a comfortable middle.
        poll_period_s = 0.1
        last_seen_samples = self.audio.samples_written()

        while self.running:
            try:
                # Pull only the audio that has arrived since our last poll.
                # We can't ask the ring for "samples since N" directly — it
                # only exposes the latest N seconds — so we read at our poll
                # cadence and let any small overlap be re-fed to VAD harmlessly.
                # Silero / KirtanVAD score frames independently; redundant
                # frames just produce the same prob and don't open a second
                # utterance.
                cur_samples = self.audio.samples_written()
                samples_delta = max(0, cur_samples - last_seen_samples)
                if samples_delta == 0:
                    await asyncio.sleep(poll_period_s)
                    continue
                # Cap the read at one poll-period of audio so we never try to
                # process more than ~poll_period_s + ring_size of samples.
                read_seconds = min(
                    samples_delta / float(config.audio.samplerate),
                    poll_period_s * 4,
                )
                fresh = self.audio.latest_window(read_seconds)
                last_seen_samples = cur_samples

                if self.paused or fresh.size == 0:
                    await asyncio.sleep(poll_period_s)
                    continue

                # Feed VAD; collect any utterances that closed during this poll.
                utterances = await asyncio.to_thread(vad.feed, fresh)
                for utt in utterances:
                    # Skip suspiciously-short utterances — VAD min_speech_ms
                    # already filters most, but a forced flush at utterance-
                    # boundary can produce a short tail. We don't want to
                    # spend a Whisper call on 200 ms of audio.
                    if utt.duration_s < 0.4:
                        continue
                    chunk_rms = float(_np.sqrt(_np.mean(utt.audio ** 2)))
                    await self._process_one_audio_window(
                        audio=utt.audio,
                        listening_mode="utterance",
                        window_seconds=float(utt.duration_s),
                        has_vocals=True,  # by construction — VAD said so
                        chunk_rms=chunk_rms,
                        now_mono=_time.monotonic(),
                    )
            except Exception as e:
                print(f"[Pipeline] VAD streaming loop error: {e}")
                await self._broadcast({"type": "error", "message": str(e)})
                await asyncio.sleep(1)
            await asyncio.sleep(poll_period_s)

    # ──────────────────────────────────────────────────────────────────
    # nemo_chunked streaming mode (IndicConformer-only)
    # ──────────────────────────────────────────────────────────────────

    async def _nemo_chunked_loop(self):
        """Decode IndicConformer at fixed audio intervals, no VAD gating.

        Why this exists: VAD-segmented mode cuts on silence, but very fast
        bani has almost no silence between panktis — multiple panktis fuse
        into a single 6-8 s utterance and the matcher can't track lines.
        This loop bypasses VAD entirely and feeds the model `nemo_chunk_len_s`
        seconds of fresh audio every cycle, giving deterministic ~chunk-len
        latency regardless of breath pattern.

        Whisper engines fall through to `naive` here — the chunked path
        depends on IndicConformer's tolerance for very short inputs.
        """
        import time as _time
        import numpy as _np

        if config.whisper.engine != "indicconformer":
            print(
                f"[Pipeline] nemo_chunked requested but active engine is "
                f"{config.whisper.engine!r} — falling back to naive."
            )
            await self._capture_tick_task()
            return

        chunk_len_s = max(0.3, float(config.whisper.nemo_chunk_len_s))
        context_s = max(0.0, float(config.whisper.nemo_chunk_context_s))
        samplerate = config.audio.samplerate
        chunk_samples = int(chunk_len_s * samplerate)
        last_seen = self.audio.samples_written()

        while self.running:
            try:
                cur = self.audio.samples_written()
                grew = cur - last_seen
                if grew < chunk_samples:
                    # Sleep just enough wall-clock time for the next chunk to
                    # land. Don't sub-poll — IndicConformer decodes are cheap
                    # but not free, and over-polling burns CPU on no-ops.
                    needed_samples = chunk_samples - grew
                    await asyncio.sleep(max(0.05, needed_samples / float(samplerate)))
                    continue
                last_seen = cur

                if self.paused:
                    await asyncio.sleep(chunk_len_s)
                    continue

                # Pull `chunk_len_s + context_s` of newest audio. The decoder
                # reads the whole thing — context exists only to warm up the
                # encoder. We attribute the produced text to chunk_len_s for
                # the speech-rate denominator (anything else would slowly
                # rotate LPS toward the context length).
                read_seconds = chunk_len_s + context_s
                audio = self.audio.latest_window(read_seconds)
                if audio.size == 0:
                    continue

                chunk_rms = float(_np.sqrt(_np.mean(audio ** 2)))
                has_vocals = bool(self.transcriber.has_vocal_content(audio))

                await self._broadcast({
                    "type": "audio_level",
                    "rms": round(chunk_rms, 4),
                    "has_vocals": has_vocals,
                    "speech_rate_lps": round(self._speech_rate_lps, 2),
                    "listening_mode": "nemo_chunked",
                    "window_seconds": round(chunk_len_s, 2),
                })

                if not has_vocals:
                    # No singing in this chunk — don't waste a decode.
                    continue

                await self._process_one_audio_window(
                    audio=audio,
                    listening_mode="nemo_chunked",
                    window_seconds=chunk_len_s,
                    has_vocals=True,
                    chunk_rms=chunk_rms,
                    now_mono=_time.monotonic(),
                )
            except Exception as e:
                print(f"[Pipeline] nemo_chunked loop error: {e}")
                await self._broadcast({"type": "error", "message": str(e)})
                await asyncio.sleep(1)

    # ──────────────────────────────────────────────────────────────────
    # local_agreement streaming mode (REA-10 Phase 3)
    # ──────────────────────────────────────────────────────────────────

    async def _local_agreement_loop(self):
        """LocalAgreement-2 (Macháček et al. 2023): re-decode a growing audio
        buffer every ``decode_interval_ms``, commit only the longest text
        prefix that two consecutive decodes agree on. The committed text
        runs through the matcher pipeline; the rest stays in the buffer
        until the next decode either confirms it (→ agreement → commit) or
        revises it.

        Trade vs. naive: similar steady-state CPU but **lower flicker** —
        nothing reaches the matcher until it has been confirmed by two
        decodes in a row. Trade vs. vad_segmented: higher CPU (re-decodes
        same audio multiple times), lower latency (commits land before any
        VAD offset would).
        """
        import time as _time
        import numpy as _np

        buf = LocalAgreementBuffer(
            agreement_n=int(config.streaming.local_agreement_n),
            max_buffer_ms=int(config.streaming.local_agreement_max_buffer_ms),
        )
        self._local_agreement_buf = buf
        decode_interval_s = max(
            0.5, float(config.streaming.local_agreement_decode_interval_ms) / 1000.0
        )
        # Minimum buffer size before we even try to decode — Whisper short
        # clips waste compute and produce nothing useful.
        min_decode_samples = int(0.6 * config.audio.samplerate)
        last_decode_t = _time.monotonic() - decode_interval_s
        last_seen_samples = self.audio.samples_written()

        print(
            f"[Pipeline] LocalAgreement-2 active: agreement_n="
            f"{buf.agreement_n}, decode_interval={decode_interval_s:.2f}s, "
            f"max_buffer={config.streaming.local_agreement_max_buffer_ms}ms"
        )

        while self.running:
            try:
                # Pull whatever audio has arrived since our last poll into the
                # LocalAgreement buffer. We don't dedupe at the byte level —
                # short overlaps (a few ms) just get re-fed and don't change
                # the agreement state.
                cur = self.audio.samples_written()
                samples_delta = max(0, cur - last_seen_samples)
                if samples_delta > 0:
                    read_seconds = min(
                        samples_delta / float(config.audio.samplerate),
                        decode_interval_s * 2,
                    )
                    fresh = self.audio.latest_window(read_seconds)
                    if fresh.size:
                        buf.append(fresh)
                    last_seen_samples = cur

                now = _time.monotonic()
                if now - last_decode_t < decode_interval_s:
                    await asyncio.sleep(0.05)
                    continue
                last_decode_t = now

                if self.paused:
                    await asyncio.sleep(0.1)
                    continue

                if buf.buffer_samples() < min_decode_samples:
                    await asyncio.sleep(0.05)
                    continue

                audio_for_stt = buf.working_audio()
                window_seconds = float(audio_for_stt.size) / float(config.audio.samplerate)

                initial_prompt = None
                if config.streaming.locked_prompt_anchor:
                    initial_prompt = self._build_locked_prompt()

                _t0 = _time.monotonic()
                segments = await asyncio.to_thread(
                    self.transcriber.transcribe, audio_for_stt, initial_prompt
                )
                elapsed = _time.monotonic() - _t0

                # Run the segments through the regular dedup processor — the
                # 'none' or 'audio_time' strategy works best here; 'text' would
                # double-strip what LocalAgreement already filters.
                window_audio_start = now - window_seconds
                processed_text = self.processor.process(
                    segments, audio_window_start=window_audio_start
                )

                # LocalAgreement-2 commit. Wrap the processed text into a
                # synthetic single-segment so commit_decode's text-LCP path
                # has stable input regardless of how many raw Whisper segments
                # came back.
                commit = buf.commit_decode(_text_to_segments(processed_text))

                # Anchor before the next iteration if buffer is now too big.
                buf.maybe_anchor()

                if not commit.new_text:
                    # Not yet two-way agreement, or no new text. Broadcast the
                    # in-flight (uncommitted) preview so the dashboard isn't dead.
                    await self._broadcast({
                        "type": "transcription",
                        "text": processed_text,
                        "first_letters": "",
                        "status": "local_agreement_pending",
                        "transcribe_ms": int(elapsed * 1000),
                        "listening_mode": "local_agreement",
                        "window_seconds": round(window_seconds, 2),
                        "pipeline_state": self.tracker.state.value,
                    })
                    continue

                # Stale-memory reset, mirrored from _process_one_audio_window
                # so the matcher's suggest/challenger memory doesn't outlive
                # an idle gap between commits.
                if (
                    self._last_window_timestamp > 0
                    and (now - self._last_window_timestamp)
                    > config.matcher.stale_memory_threshold_seconds
                ):
                    self._suggest_first_seen.clear()
                    self._challenger_first_seen.clear()
                self._last_window_timestamp = now
                self._window_index += 1
                self._last_listening_mode = "local_agreement"

                await self._process_decoded_text(
                    text=commit.new_text,
                    listening_mode="local_agreement",
                    window_seconds=decode_interval_s,
                    elapsed_decode_s=elapsed,
                )

            except Exception as e:
                print(f"[Pipeline] LocalAgreement loop error: {e}")
                await self._broadcast({"type": "error", "message": str(e)})
                await asyncio.sleep(1)

    # ──────────────────────────────────────────────────────────────────
    # hybrid streaming mode (REA-10 Phase 3) — VAD bounds + LocalAgreement-2
    # ──────────────────────────────────────────────────────────────────

    async def _hybrid_streaming_loop(self):
        """Best-of-both: VAD onset starts a fresh LocalAgreement-2 buffer,
        intra-utterance decodes commit agreed prefix to the matcher, VAD
        offset flushes anything still in flight.

        Properties:
        * Boundaries are clean (driven by VAD onset/offset)
        * Latency is low (LocalAgreement commits land before VAD offset)
        * Buffer is bounded (per-utterance, max_utterance_ms cap)

        This is the production target for live kirtan — naive and
        vad_segmented are easier to reason about, but hybrid pays back the
        extra complexity in dashboard responsiveness during long utterances.
        """
        import time as _time
        import numpy as _np

        vad = self._make_streaming_vad()
        self._streaming_vad = vad
        decode_interval_s = max(
            0.5, float(config.streaming.local_agreement_decode_interval_ms) / 1000.0
        )
        # Per-utterance LocalAgreement state. None when VAD says we're idle.
        utt_buf: LocalAgreementBuffer | None = None
        last_intra_decode_t = 0.0
        last_seen_samples = self.audio.samples_written()
        poll_period_s = 0.1

        print(
            f"[Pipeline] Hybrid (VAD + LocalAgreement-2) active: backend="
            f"{config.streaming.vad_backend}, threshold={config.streaming.vad_threshold}, "
            f"decode_interval={decode_interval_s:.2f}s"
        )

        while self.running:
            try:
                cur = self.audio.samples_written()
                samples_delta = max(0, cur - last_seen_samples)
                if samples_delta == 0:
                    await asyncio.sleep(poll_period_s)
                    continue
                read_seconds = min(
                    samples_delta / float(config.audio.samplerate),
                    poll_period_s * 4,
                )
                fresh = self.audio.latest_window(read_seconds)
                last_seen_samples = cur
                if self.paused or fresh.size == 0:
                    await asyncio.sleep(poll_period_s)
                    continue

                # Run VAD on the fresh chunk. Closed utterances (offsets) are
                # returned; in-flight utterance is accessible via peek_partial.
                closed = await asyncio.to_thread(vad.feed, fresh)

                # Mid-utterance LocalAgreement: while VAD says we're inside an
                # utterance, periodically decode the partial buffer and emit
                # any newly-agreed text.
                in_utt = vad.is_in_utterance()
                if in_utt:
                    if utt_buf is None:
                        utt_buf = LocalAgreementBuffer(
                            agreement_n=int(config.streaming.local_agreement_n),
                            max_buffer_ms=int(config.streaming.vad_max_utterance_ms),
                        )
                        last_intra_decode_t = _time.monotonic()

                    now = _time.monotonic()
                    if now - last_intra_decode_t >= decode_interval_s:
                        last_intra_decode_t = now
                        partial_audio = vad.peek_partial()
                        # Only re-decode if there's more audio than the last commit covered.
                        if partial_audio.size > utt_buf.buffer_samples():
                            # Re-base buffer to current peek_partial state.
                            utt_buf._audio = partial_audio  # noqa: SLF001 — module-internal
                            await self._maybe_emit_local_agreement(
                                utt_buf, listening_mode="hybrid_partial",
                                decode_interval_s=decode_interval_s,
                            )

                # Process VAD-closed utterances. For each, do one final
                # LocalAgreement decode (catches any text the partial decodes
                # missed) and clear the per-utterance state.
                for utt in closed:
                    if utt.duration_s < 0.4:
                        utt_buf = None
                        continue
                    if utt_buf is None:
                        utt_buf = LocalAgreementBuffer(
                            agreement_n=int(config.streaming.local_agreement_n),
                            max_buffer_ms=int(config.streaming.vad_max_utterance_ms),
                        )
                    utt_buf._audio = utt.audio  # noqa: SLF001
                    # Force enough history for agreement on the closing decode.
                    while len(utt_buf._history) < utt_buf.agreement_n - 1:  # noqa: SLF001
                        utt_buf._history.append("")  # noqa: SLF001
                    await self._maybe_emit_local_agreement(
                        utt_buf, listening_mode="hybrid_final",
                        decode_interval_s=float(utt.duration_s),
                    )
                    utt_buf = None

                if not in_utt and utt_buf is not None:
                    # Defensive: VAD says idle but we have a stale buffer —
                    # discard rather than carry over to the next utterance.
                    utt_buf = None

            except Exception as e:
                print(f"[Pipeline] Hybrid loop error: {e}")
                await self._broadcast({"type": "error", "message": str(e)})
                await asyncio.sleep(1)
            await asyncio.sleep(poll_period_s)

    async def _maybe_emit_local_agreement(
        self,
        buf: LocalAgreementBuffer,
        listening_mode: str,
        decode_interval_s: float,
    ):
        """Decode the buffer's working audio, run LocalAgreement, emit any
        newly-agreed text through the matcher pipeline. Shared between
        ``_local_agreement_loop`` and ``_hybrid_streaming_loop``.
        """
        import time as _time

        audio_for_stt = buf.working_audio()
        if audio_for_stt.size < int(0.5 * config.audio.samplerate):
            return
        window_seconds = float(audio_for_stt.size) / float(config.audio.samplerate)
        initial_prompt = (
            self._build_locked_prompt()
            if config.streaming.locked_prompt_anchor
            else None
        )
        _t0 = _time.monotonic()
        segments = await asyncio.to_thread(
            self.transcriber.transcribe, audio_for_stt, initial_prompt
        )
        elapsed = _time.monotonic() - _t0

        now_mono = _time.monotonic()
        window_audio_start = now_mono - window_seconds
        processed_text = self.processor.process(
            segments, audio_window_start=window_audio_start
        )

        commit = buf.commit_decode(_text_to_segments(processed_text))
        buf.maybe_anchor()
        if not commit.new_text:
            await self._broadcast({
                "type": "transcription",
                "text": processed_text,
                "first_letters": "",
                "status": f"{listening_mode}_pending",
                "transcribe_ms": int(elapsed * 1000),
                "listening_mode": listening_mode,
                "window_seconds": round(window_seconds, 2),
                "pipeline_state": self.tracker.state.value,
            })
            return

        if (
            self._last_window_timestamp > 0
            and (now_mono - self._last_window_timestamp)
            > config.matcher.stale_memory_threshold_seconds
        ):
            self._suggest_first_seen.clear()
            self._challenger_first_seen.clear()
        self._last_window_timestamp = now_mono
        self._window_index += 1
        self._last_listening_mode = listening_mode

        await self._process_decoded_text(
            text=commit.new_text,
            listening_mode=listening_mode,
            window_seconds=decode_interval_s,
            elapsed_decode_s=elapsed,
        )

    async def _process_one_audio_window(
        self,
        audio,
        listening_mode: str,
        window_seconds: float,
        has_vocals: bool,
        chunk_rms: float,
        now_mono: float,
    ):
        """Transcribe one chunk of audio and run it through the matcher.

        Shared between every streaming mode (naive sliding-window, VAD-segmented
        utterance, future LocalAgreement-2). The naive path's decode loop and
        the new ``_vad_streaming_loop`` both call this — keeping the lock state
        machine, alaap detection, transition mode, and broadcasts in one place.

        Parameters mirror the ``_latest_window_data`` shape so the naive caller
        can pass the dict through verbatim. ``listening_mode`` is a label for
        the dashboard: 'search' / 'locked' / 'micro' / 'recovery' for naive,
        'utterance' for VAD-segmented.
        """
        import time as _time

        self._window_index += 1
        # Stale-memory reset: if this window arrives too long after the last
        # one (ASR lag, paused pipeline, etc.) the suggest/challenger timestamps
        # are from a different audio context and must be discarded.
        if (
            self._last_window_timestamp > 0
            and (now_mono - self._last_window_timestamp)
            > config.matcher.stale_memory_threshold_seconds
        ):
            self._suggest_first_seen.clear()
            self._challenger_first_seen.clear()
            self._prev_first_letters = ""
            print(
                f"  [STALE RESET] gap={now_mono - self._last_window_timestamp:.1f}s "
                f"> {config.matcher.stale_memory_threshold_seconds:.0f}s — "
                "suggest/challenger memory cleared"
            )
        self._last_window_timestamp = now_mono
        self._last_listening_mode = listening_mode
        audio_for_stt = audio

        if self.paused:
            await self._broadcast({"type": "paused"})
            await asyncio.sleep(0.1)
            return

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
            return

        # 2. Transcribe
        # Optional locked-shabad prompt anchoring: when enabled and the
        # matcher is locked, pass the current pankti's Gurmukhi text to
        # Whisper as initial_prompt — biases the decoder toward the
        # actual shabad text. Off by default; enable per-session via
        # the dashboard.
        initial_prompt = None
        if config.streaming.locked_prompt_anchor:
            initial_prompt = self._build_locked_prompt()
        # Pass the absolute audio-window timestamp through the processor
        # so audio_time dedup can distinguish window-overlap from legit
        # repetition (REA-10).
        window_audio_start = now_mono - window_seconds
        _t0 = _time.monotonic()
        _engine_label = config.whisper.engine
        print(f"  [DEBUG] Starting {_engine_label} transcription ({window_seconds:.1f}s audio, mode={listening_mode})...")
        segments = await asyncio.to_thread(
            self.transcriber.transcribe, audio_for_stt, initial_prompt
        )
        _elapsed = _time.monotonic() - _t0
        text = self.processor.process(segments, audio_window_start=window_audio_start)
        print(f"  [DEBUG] {_engine_label} done in {_elapsed:.1f}s → '{text[:80]}'" if text else f"  [DEBUG] {_engine_label} done in {_elapsed:.1f}s → (empty)")

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
            return

        # Hand off to the post-decode helper. Splitting at this point lets
        # local_agreement / hybrid feed text-only commits (no fresh segments
        # of their own) through the same matcher pipeline.
        await self._process_decoded_text(
            text=text,
            listening_mode=listening_mode,
            window_seconds=window_seconds,
            elapsed_decode_s=_elapsed,
        )

    async def _process_decoded_text(
        self,
        text: str,
        listening_mode: str,
        window_seconds: float,
        elapsed_decode_s: float = 0.0,
    ):
        """Run already-transcribed text through the matcher pipeline.

        Shared by the audio-window path (called from
        ``_process_one_audio_window`` after Whisper) AND the LocalAgreement /
        hybrid streaming modes (which hand off newly-agreed text without a
        per-call audio window).

        ``window_seconds`` is the speech-rate / dwell-counter denominator —
        for naive windowed mode it's the actual window length; for streaming
        modes it's the approximate audio duration represented by the new
        text (decode interval, or VAD utterance length).
        """
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
        if elapsed_decode_s >= 0.02:
            msg["transcribe_ms"] = int(elapsed_decode_s * 1000)
            msg["rtf"] = round(elapsed_decode_s / max(window_seconds, 0.001), 3)
        await self._broadcast(msg)

        # 6a. Alaap detection (Change 7): detect melismatic/non-lexical windows.
        # When enabled, freeze line pointer and skip challenger logic for
        # windows where ASR output looks like alaap rather than lyrics.
        _is_alaap = (
            config.matcher.alaap_detection_enabled
            and _is_alaap_output(text)
        )
        if _is_alaap:
            self._alaap_window_count += 1
            self._transition_alaap_seconds += max(window_seconds, float(config.audio.step_duration))
        else:
            self._alaap_window_count = 0

        # 6b. Transition mode (Change 8): enter when 2+ signals suggest the
        # shabad is about to change, relaxing challenger/override thresholds.
        if config.matcher.transition_mode_enabled:
            self._update_transition_mode(window_seconds)

        # When alaap is confirmed (N consecutive windows) and we're LOCKED:
        # broadcast the freeze and skip matching for this window.
        _alaap_freeze = (
            _is_alaap
            and self._alaap_window_count >= config.matcher.alaap_consecutive_windows
            and self.tracker.state.value == "locked"
        )
        if _alaap_freeze:
            await self._broadcast({
                "type": "alaap_freeze",
                "windows": self._alaap_window_count,
                "transcript": text,
            })
            self._prev_first_letters = first_letters
            return

        # 6. Dispatch to state handler
        if self.tracker.state in (
            PipelineState.SEARCHING,
            PipelineState.CANDIDATE_LOCK,
        ):
            # When the current window alone is too short to search, try stitching
            # with the previous window's letters before giving up. Mirrors the
            # locked-state stitch path so a short-fragment window (e.g. ASR
            # returns 1-2 words mid-verse) can still contribute to retrieval.
            stitched_pc = f"{self._prev_first_letters}{first_letters}"
            stitched_cp = f"{first_letters}{self._prev_first_letters}"
            best_stitch_len = max(len(stitched_pc), len(stitched_cp))
            if (
                len(first_letters) < config.matcher.min_search_letters
                and best_stitch_len < config.matcher.min_search_letters
            ):
                if self.tracker.state == PipelineState.CANDIDATE_LOCK:
                    self._candidate_lock_misses += 1
                    if self._candidate_lock_misses >= config.matcher.candidate_lock_miss_windows:
                        self.tracker.clear_candidate_lock()
                        self._candidate_lock_misses = 0
                return
            await self._handle_searching(
                first_letters,
                start_mode=self._after_break_windows > 0,
                transcript_text=text,
                prev_first_letters=self._prev_first_letters,
            )
            if self._after_break_windows > 0:
                self._after_break_windows -= 1
        else:
            stitched_letters = f"{self._prev_first_letters}{first_letters}"
            if max(len(first_letters), len(stitched_letters)) < config.matcher.min_search_letters:
                self.tracker.mark_unstable()
                self._prev_first_letters = first_letters
                return
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

    async def _handle_searching(
        self,
        first_letters: str,
        start_mode: bool = False,
        transcript_text: str = "",
        prev_first_letters: str = "",
    ):
        """SEARCHING state: broad search, score candidates, try to lock."""
        # Pick the search query: when the fresh window alone is too short to
        # search reliably, fall back to a stitch with the previous window's
        # letters. Try both directions (prev+current and current+prev) — fast
        # kirtan can split a verse across the window boundary either way — and
        # use whichever yields the stronger top candidate. Mirrors the locked
        # path's stitch behaviour.
        scoring_letters = first_letters
        candidates: list = []
        if (
            prev_first_letters
            and len(first_letters) < config.matcher.min_search_letters
        ):
            best_top = -1.0
            for variant in (
                f"{prev_first_letters}{first_letters}",
                f"{first_letters}{prev_first_letters}",
            ):
                if len(variant) < config.matcher.min_search_letters:
                    continue
                trial = await asyncio.to_thread(
                    self.searcher.search,
                    variant,
                    10,
                    start_mode,
                    transcript_text,
                )
                trial_top = trial[0]["score"] if trial else 0.0
                if trial_top > best_top:
                    best_top = trial_top
                    scoring_letters = variant
                    candidates = trial
        else:
            candidates = await asyncio.to_thread(
                self.searcher.search,
                first_letters,
                10,
                start_mode,
                transcript_text,
            )

        # Score candidates against the same letters that retrieved them — when we
        # stitched, scoring with the original short query understates the match.
        scored = self._score_candidates(scoring_letters, candidates, transcript_text)
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
            hypothesis_candidate = next(
                (c for c in top_candidates if c["shabad_id"] == best_hypothesis["shabad_id"]),
                None,
            )
            if hypothesis_candidate:
                top_score_this_window = top_candidates[0]["score"] if top_candidates else 0.0
                # Only let accumulated evidence promote a hypothesis if it is
                # competitive this window. When the current audio strongly favours a
                # different shabad (gap > 0.12), trust the present over past windows —
                # otherwise a stale hypothesis can override the clearly-correct candidate.
                if hypothesis_candidate["score"] >= top_score_this_window - 0.12:
                    top = dict(hypothesis_candidate)
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
            top_dense_dominant = bool(top.get("dense_dominant", False))
            auto_overlap_min = (
                config.matcher.word_overlap_evidence_min
                if top_dense_dominant
                else config.matcher.word_overlap_auto_min
            )
            meets_raw_auto = (
                raw_score >= config.matcher.auto_threshold
                and word_overlap >= auto_overlap_min
            )
            meets_evidence = (
                evidence_score >= config.matcher.auto_threshold
                and int(top.get("stability", 1)) >= config.matcher.candidate_lock_windows
                and word_overlap >= config.matcher.word_overlap_evidence_min
            )

            # Change 4: confidence gap check.
            # High-confidence bypass (≥0.90): skip gap requirement entirely.
            # Otherwise require top-1 to lead top-2 by ≥ gap_threshold — prevents
            # coin-flip locks when two shabads score within noise of each other.
            second_best_score = top_candidates[1]["score"] if len(top_candidates) > 1 else 0.0
            gap = raw_score - second_best_score
            if raw_score < config.matcher.high_confidence_lock_threshold:
                if gap < config.matcher.gap_threshold:
                    meets_raw_auto = False  # kill per-window auto; evidence path still open

            lockable = (
                raw_score >= config.matcher.min_raw_lock_score
                and (meets_raw_auto or meets_evidence)
            )

            # Change 1: tiered lock — suggest-level candidate promoted to lockable
            # after suggest_confirmation_seconds of being the top candidate.
            import time as _t_now
            _now = _t_now.monotonic()
            top_id = top["shabad_id"]
            for sid in list(self._suggest_first_seen):
                if sid != top_id:
                    del self._suggest_first_seen[sid]
            if not lockable and raw_score >= config.matcher.suggest_threshold:
                if top_id not in self._suggest_first_seen:
                    self._suggest_first_seen[top_id] = _now
                elapsed = _now - self._suggest_first_seen[top_id]
                if elapsed >= self._active_suggest_confirmation_s():
                    lockable = True
            elif lockable:
                self._suggest_first_seen.clear()
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

        _word_count = len(transcript_text.split()) if transcript_text else 0

        # First pass: fresh window only. This is the happy path — when the lock is
        # confident, scoring the fresh 3 s alone gives us the next line immediately
        # and avoids carrying the previous window's letters forward (which tends to
        # keep the pointer stuck on the previous line during fast recitation).
        current_scores: list[float] = []
        raw_line_scores: list[float] = []  # Change 9: parallel raw track for jump bypass
        best_current_idx = 0
        best_current_score = 0.0
        best_raw_idx = 0
        best_raw_score_v = 0.0
        second_best_raw_score = 0.0
        for i, verse in enumerate(current.verses):
            # Line 0 is always the raag/mahala heading ("ਮਾਝ ਮਹਲਾ ੫ ॥") — never
            # sung.  Clamp its score to 0 so it never wins the line pointer race.
            if i == 0 and config.matcher.penalize_heading_line:
                current_scores.append(0.0)
                raw_line_scores.append(0.0)
                continue
            if transcript_text and _word_count == 2:
                raw_current = self.scorer.score_line_word_match(transcript_text, verse.unicode)
            elif transcript_text and len(first_letters) <= 3:
                raw_current = self.scorer.score_line_ngram(transcript_text, verse.unicode)
            else:
                raw_current = self.scorer.score_line(first_letters, verse.first_letters)
                if transcript_text:
                    raw_current = max(raw_current, self.scorer.score_line_ngram(transcript_text, verse.unicode))
            # Change 6: Smith-Waterman word alignment — takes max so it only helps.
            if config.matcher.sw_line_scoring_enabled and transcript_text and _word_count >= 2:
                raw_current = max(raw_current, self.scorer.score_line_sw(transcript_text, verse.unicode))
            raw_line_scores.append(raw_current)
            # Track top-2 raw scores for the gap-based bypass below
            if raw_current > best_raw_score_v:
                second_best_raw_score = best_raw_score_v
                best_raw_score_v = raw_current
                best_raw_idx = i
            elif raw_current > second_best_raw_score:
                second_best_raw_score = raw_current

            current_score = self._apply_progression_bias(i, current.current_line, raw_current)
            current_scores.append(current_score)
            if current_score > best_current_score:
                best_current_score = current_score
                best_current_idx = i

        # Change 9 (extended): two-phase confident-jump bypass.  When the raw best
        # line is clearly stronger — either by absolute threshold or by a wide gap
        # over second-best — skip progression bias entirely and use the raw winner.
        # This catches the case where every candidate scores below the absolute
        # threshold but the top-1 / top-2 spread makes the choice unambiguous.
        raw_gap = best_raw_score_v - second_best_raw_score
        confident_jump = best_raw_idx != current.current_line and (
            best_raw_score_v >= config.matcher.progression_confident_jump_threshold
            or raw_gap >= config.matcher.line_jump_gap_threshold
        )
        if confident_jump and best_raw_idx != best_current_idx:
            print(
                f"  [CONFIDENT JUMP] line {current.current_line} → {best_raw_idx} "
                f"raw={best_raw_score_v:.2f} gap={raw_gap:.2f}"
            )
            line_scores = raw_line_scores
            best_line_idx = best_raw_idx
            best_line_score = best_raw_score_v
            best_line_variant = "confident_jump"
        else:
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
                if i == 0 and config.matcher.penalize_heading_line:
                    combined_scores.append(0.0)
                    combined_labels.append("heading_skip")
                    continue
                line_best_combined = current_scores[i]
                line_best_label = "current"
                for label, query_letters in combined_variants:
                    if transcript_text and len(query_letters) <= 3:
                        raw_score = self.scorer.score_line_ngram(transcript_text, verse.unicode)
                    else:
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
        _enough_words = _word_count >= config.matcher.min_words_for_line_advance
        should_update_line = not is_detour and _enough_words and (
            best_line_score >= config.matcher.suggest_threshold
            or (
                local_best_score >= config.matcher.local_line_follow_threshold
                and local_best_idx != current.current_line
            )
        )
        if should_update_line:
            self._weak_line_windows = 0
            self._challenger_first_seen.clear()
            old_line = current.current_line
            target_idx = best_line_idx
            target_score = best_line_score
            if (
                best_line_score < config.matcher.suggest_threshold
                and local_best_score >= config.matcher.local_line_follow_threshold
            ):
                target_idx = local_best_idx
                target_score = local_best_score

            # Manual-nav hold: operator jumped to a specific line; don't snap back
            # for the next N windows. Allow forward advances (audio caught up) but
            # block any backward movement so the manual position is respected.
            if self._manual_nav_hold_windows > 0:
                self._manual_nav_hold_windows -= 1
                if target_idx < old_line:
                    target_idx = old_line
                    target_score = current_scores[old_line]

            if not self._just_locked:
                # Cap: never advance more than 1 line per window — kirtan never skips
                # 2+ lines in a single 3-9s chunk; larger jumps are always bad matches.
                # Change 9: a sufficiently confident match (≥ progression_confident_jump_threshold)
                # bypasses the cap so the pipeline can snap to the right line in one window.
                if (
                    target_idx > old_line + 1
                    and target_score < config.matcher.progression_confident_jump_threshold
                ):
                    target_idx = old_line + 1
                    target_score = line_scores[old_line + 1] if old_line + 1 < len(line_scores) else target_score

                # Dwell gate: don't advance forward if we've been on the current line for
                # less than min_line_dwell_seconds, unless next-line evidence is
                # overwhelming. Moving backward (delta < 0) is always allowed — if we
                # accidentally jumped ahead we want to snap back immediately.
                if target_idx > old_line:
                    dwell = (
                        datetime.now() - current.line_updated_at
                    ).total_seconds()
                    active_dwell = self._active_min_line_dwell_seconds()
                    if (
                        dwell < active_dwell
                        and target_score < config.matcher.line_advance_override_score
                    ):
                        print(
                            f"  [DWELL HOLD] line {old_line}→{target_idx} "
                            f"(dwell={dwell:.2f}s < {active_dwell:.1f}s, "
                            f"score={target_score:.2f} < {config.matcher.line_advance_override_score:.2f})"
                        )
                        target_idx = old_line
                        target_score = current_scores[old_line]

            self.tracker.update_line(target_idx, target_score)
            self._just_locked = False
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

        # Broadcast line alignment AFTER dwell-gate so the highlight reflects the
        # line we actually stayed on, not the scoring candidate before gating.
        _actual_idx = current.current_line
        if _actual_idx < len(current.verses):
            _actual_verse = current.verses[_actual_idx]
            _actual_score = line_scores[_actual_idx] if _actual_idx < len(line_scores) else 0.0
            await self._broadcast({
                "type": "line_aligned",
                "line_index": _actual_idx,
                "line_score": round(_actual_score, 3),
                "line_unicode": _actual_verse.unicode,
                "line_english": _actual_verse.english,
                "match_variant": best_line_variant,
                "pipeline_state": "locked",
                "is_detour": is_detour,
            })

        # Skip global challenger scan during an alap detour — the sticky-set handler
        # already tracks the would-be switch and will commit on its own timeline. Running
        # both races the counters and can cause premature switches on brief alaps.
        if is_detour:
            return

        # Micro windows (3 s) are too short to reliably identify a new shabad — they
        # exist only for fast line tracking. Skip the challenger scan unless the
        # current-line alignment has already gone weak, in which case recovery still
        # needs to fire.
        if on_micro_window_flag and self._weak_line_windows < max(
            1, config.matcher.weak_line_recovery_windows - 1
        ):
            return

        # Run challenger scan every locked cycle (not only weak cycles) so wrong-lock
        # recovery can happen quickly. Skip on short (≤2-word) windows — they only
        # drive line pointer positioning within the current shabad, never a switch.
        if _word_count <= 2:
            return
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
        required_overlap = (
            config.matcher.dense_dominant_instant_overlap_min
            if top_dense_dominant
            else config.matcher.word_overlap_instant_challenger_min
        )
        challenger_gap = top_raw_score - current_shabad_search_score
        # Change 5: strong override — immediate switch when challenger clearly leads.
        # Thresholds relax automatically in transition mode (Change 8).
        instant_switch = (
            top_raw_score >= self._active_override_threshold()
            and top_word_overlap >= required_overlap
            and challenger_gap >= self._active_override_min_gap()
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
            self._challenger_first_seen.clear()
            new_id = result["new_shabad_id"]
            print(f"  [SWITCH] Challenger {new_id} wins! Transitioning...")
            await self._broadcast({
                "type": "shabad_switched",
                "new_shabad_id": new_id,
                "old_shabad_id": current.shabad_id,
            })
            return

        if result["action"] == "challenging":
            # Change 5: time-based challenger confirmation.
            # Track the first time this challenger appeared; commit after
            # challenger_confirmation_seconds of sustained outscoring.
            import time as _t_ch
            _now_ch = _t_ch.monotonic()
            ch_id = top["shabad_id"]
            for sid in list(self._challenger_first_seen):
                if sid != ch_id:
                    del self._challenger_first_seen[sid]
            if ch_id not in self._challenger_first_seen:
                self._challenger_first_seen[ch_id] = _now_ch
            ch_elapsed = _now_ch - self._challenger_first_seen[ch_id]
            _req_s = self._active_challenger_confirmation_s()
            print(
                f"  [CHALLENGER] {ch_id} "
                f"wins {result['wins']}/{result['needed']} "
                f"({ch_elapsed:.1f}s / {_req_s:.1f}s"
                + (" TRANSITION" if self._in_transition_mode else "")
                + ")"
            )
            if ch_elapsed >= _req_s:
                # Time threshold met — force switch via tracker instant lock
                print(
                    f"  [TIMED SWITCH] {ch_id} confirmed after {ch_elapsed:.1f}s — committing"
                )
                self._challenger_first_seen.clear()
                self._weak_line_windows = 0
                self.tracker.try_lock(ch_id, top["score"], instant=True)
                await self._broadcast({
                    "type": "shabad_switched",
                    "new_shabad_id": ch_id,
                    "old_shabad_id": current.shabad_id,
                    "reason": "timed_challenger",
                })
                await self._lock_shabad_from_top(top)
                return
            await self._broadcast({
                "type": "challenger_update",
                "challenger": top,
                "wins": result["wins"],
                "needed": result["needed"],
                "elapsed_s": round(ch_elapsed, 1),
                "required_s": config.matcher.challenger_confirmation_seconds,
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
            # Change 3: dense_dominant + ngram4-only → downgrade to suggest.
            # When the only retrieval signal is the char-4-gram index AND the score
            # was driven by a dense substring match (not genuine line-level FL overlap),
            # the hit is prone to spurious substring collisions.  Let the tiered-lock
            # timer in Change 1 handle promotion after sustained evidence.
            if action == "auto" and dense_dominant and retrieval_sources == ["ngram4"]:
                action = "suggest"
            # Pick the best-matching line within this candidate so the snap in
            # _lock_shabad_from_top can land on the right verse instead of the
            # raag heading. We already have the per-line first-letters cached on
            # the candidate (space-separated, recitation order).
            line_idx = 0
            if first_letters and candidate.full_first_letters:
                line_fls = candidate.full_first_letters.split(" ")
                if len(line_fls) > 1:
                    # Score every line, then pick best — but apply a small penalty to
                    # line 0 (raag heading) so it only wins when it truly is the best
                    # match. Without this, ambiguous short queries tie across lines and
                    # line 0 wins the default tiebreak, leaving STTM stuck on the heading.
                    best_line_score = -1.0
                    for idx, line_fl in enumerate(line_fls):
                        if not line_fl:
                            continue
                        line_score = self.scorer.score_line(first_letters, line_fl)
                        if idx == 0 and config.matcher.penalize_heading_line:
                            line_score -= 0.15
                        if line_score > best_line_score:
                            best_line_score = line_score
                            line_idx = idx
            scored.append({
                "shabad_id": candidate.shabad_id,
                "gurmukhi": candidate.gurmukhi,
                "unicode": candidate.unicode,
                "english": candidate.english,
                "line_idx": line_idx,
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
        Pure raw-score comparison with a small current-line tiebreaker.

        Confident-jump bypass: a line clearing
        `progression_confident_jump_threshold` returns its raw score, so a
        clearly better match elsewhere in the shabad wins without competing
        against the current-line tiebreaker.
        """
        if raw_score >= config.matcher.progression_confident_jump_threshold:
            return raw_score

        delta = index - current_line
        if delta == 0:
            bonus = 0.05  # tiebreaker only — stay here when scores are equal
        elif raw_score >= config.matcher.progression_high_confidence_bypass:
            return raw_score
        elif delta < 0:
            bonus = -0.04 * min(abs(delta), 3)
        elif delta >= 4:
            bonus = -0.08
        else:
            bonus = -0.02 * abs(delta)  # delta=1 → -0.02, 2 → -0.04, 3 → -0.06
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
        # Snap the line pointer to the verse that actually matched during search,
        # so STTM doesn't start stuck on the raag heading (line 0).
        initial_line = top.get("line_idx", 0)
        if initial_line > 0 and initial_line < len(verses):
            await self.controller.navigate_to_line(initial_line)
            self.tracker.set_line(initial_line)
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
        self._suggest_first_seen.clear()
        self._challenger_first_seen.clear()
        self._in_transition_mode = False
        self._transition_alaap_seconds = 0.0
        self._alaap_window_count = 0
        # Drop carry-over text from the previous shabad so the stitched-window
        # scorer doesn't mix old lyrics into the first post-lock alignment pass.
        self._prev_first_letters = ""
        # Force 2 short (start-boost) windows after locking so the rolling audio
        # ring flushes old-shabad audio before we attempt line alignment.
        self._after_break_windows = max(self._after_break_windows, 2)
        self._just_locked = True

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

    def _update_transition_mode(self, window_seconds: float) -> None:
        """Change 8: enter/exit transition mode based on transition signal count."""
        import time as _tm
        _now = _tm.monotonic()

        # Exit if timed out
        if self._in_transition_mode:
            if _now - self._transition_mode_start >= config.matcher.transition_max_duration_seconds:
                self._in_transition_mode = False
                self._transition_alaap_seconds = 0.0
                print("  [TRANSITION] Timed out — exiting transition mode")
            return

        # Count signals
        if self.tracker.state.value != "locked":
            return
        signals = 0
        # Signal 1: locked shabad scoring low for ≥ transition_weak_seconds
        weak_s = self._weak_line_windows * float(config.audio.step_duration)
        if weak_s >= config.matcher.transition_weak_seconds:
            signals += 1
        # Signal 2: current line near end of shabad (last 2 lines)
        cur = self.tracker.current
        if cur and cur.verses and (len(cur.verses) - cur.current_line) <= 2:
            signals += 1
        # Signal 3: accumulated alaap/silence time
        if self._transition_alaap_seconds >= config.matcher.transition_silence_seconds:
            signals += 1
        # Signal 4: recent windows are all filler (alaap_window_count sustained)
        if self._alaap_window_count >= config.matcher.alaap_consecutive_windows:
            signals += 1

        if signals >= config.matcher.transition_min_signals:
            self._in_transition_mode = True
            self._transition_mode_start = _now
            print(f"  [TRANSITION] Entering transition mode ({signals} signals)")
            self._transition_alaap_seconds = 0.0

    def _active_challenger_confirmation_s(self) -> float:
        """Return the effective challenger confirmation seconds (relaxed in transition mode)."""
        if self._in_transition_mode:
            return config.matcher.transition_challenger_confirmation_s
        return config.matcher.challenger_confirmation_seconds

    def _active_step_duration(self) -> float:
        """Capture-tick cadence — controls how long we sleep between wakes.

        Priority order:
        1. Zero-overlap mode: hop = the most-recently-decoded dynamic window
           seconds, so each second of audio is transcribed exactly once.
           Falls back to step_duration on the very first tick (before any
           window has been decoded).
        2. Fast-response mode: shrinks the search-state cadence to 1.5 s
           without changing accuracy thresholds.
        3. Default: config.audio.step_duration (3 s).
        """
        if config.audio.zero_overlap_window and self._last_window_seconds > 0.0:
            return float(self._last_window_seconds)
        if config.matcher.fast_response_enabled:
            return config.audio.fast_step_duration
        return config.audio.step_duration

    def _active_min_line_dwell_seconds(self) -> float:
        """Minimum dwell on the current line before a forward pointer move (override score still bypasses)."""
        if config.matcher.fast_response_enabled:
            return config.matcher.fast_min_line_dwell_seconds
        return config.matcher.min_line_dwell_seconds

    def _active_suggest_confirmation_s(self) -> float:
        """Tiered-lock confirmation — how long a 0.60–0.74 candidate must stay top before promotion."""
        if config.matcher.fast_response_enabled:
            return config.matcher.fast_suggest_confirmation_seconds
        return config.matcher.suggest_confirmation_seconds

    def _active_override_threshold(self) -> float:
        if self._in_transition_mode:
            return config.matcher.transition_override_threshold
        return config.matcher.strong_override_threshold

    def _active_override_min_gap(self) -> float:
        if self._in_transition_mode:
            return config.matcher.transition_override_min_gap
        return config.matcher.override_min_gap

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
