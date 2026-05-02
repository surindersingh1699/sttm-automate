"""Voice activity detection for streaming Gurbani audio.

Provides two VAD backends, both with the same ``Utterance``-emitting interface:

KirtanVAD (default for this project)
    Spectral-envelope detector tuned for sung Gurbani. Voice formants live in
    300–3400 Hz; tabla transients are <300 Hz; harmonium drone is broadband but
    *spectrally stable*. The detector uses two signals:
      - **voice-band ratio** = energy in 300–3400 Hz ÷ total energy
      - **spectral flux** = how rapidly the spectrum is changing
    A frame is "voiced" when voice_ratio ≥ threshold AND rms > noise floor.
    Empirically, Silero VAD (a speech-only model) returns <1 % voiced on a
    representative kirtan recording — even at threshold 0.10. KirtanVAD finds
    the actual sung sections reliably because its features survive the
    speech↔singing transfer.

SileroVAD (kept for non-singing content, e.g. katha or spoken introductions)
    Standard Silero VAD via the ``silero-vad`` package. Use only when the input
    audio is closer to speech than song.

Both backends expose identical APIs:
    segment_utterances(audio, **knobs)  # batch — returns list[Utterance]
    StreamingVAD(**knobs).feed(samples)  # streaming — yields Utterance per call

Tunable knobs (all mirrored in ``src.config.streaming``):
    vad_threshold       : voice-band ratio cutoff (KirtanVAD) or speech-prob (Silero)
    vad_min_silence_ms  : silence duration before declaring offset
    vad_min_speech_ms   : speech duration before declaring onset
    vad_speech_pad_ms   : audio padding around utterance boundaries
    vad_max_utterance_ms: safety bound on runaway utterances
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

# All VADs in this module operate at fixed 16 kHz mono float32. The pipeline
# already runs at 16 kHz mono (see config.audio.samplerate).
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32 ms at 16 kHz
FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE  # 32.0


@dataclass(frozen=True)
class Utterance:
    """A VAD-bounded utterance with absolute audio timestamps.

    ``audio`` is a copy — safe to retain across the next streaming feed.
    Times are in seconds since the start of the streaming session
    (``StreamingVAD.reset()`` zeros them).
    """
    start_s: float
    end_s: float
    audio: np.ndarray  # 16 kHz mono float32

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ──────────────────────────────────────────────────────────────────────
# KirtanVAD — spectral-envelope detector tuned for sung Gurbani
# ──────────────────────────────────────────────────────────────────────

# Pre-compute frequency masks once at module load. Voice formants in
# 300–3400 Hz; bass/drone band <300 Hz (tabla impulses, harmonium drone).
_FREQ_BINS = np.fft.rfftfreq(FRAME_SAMPLES, 1.0 / SAMPLE_RATE)
_VOICE_BAND = (_FREQ_BINS >= 300) & (_FREQ_BINS <= 3400)
_HANN = np.hanning(FRAME_SAMPLES).astype(np.float32)
# RMS noise floor below which frames are unconditionally treated as silence,
# even if the voice-band ratio happens to be high (low-amp noise can have
# arbitrary spectral shape).
_NOISE_FLOOR = 0.005


def _frame_voice_score(frame: np.ndarray) -> tuple[float, float]:
    """Return (voice_band_ratio, rms) for a 512-sample frame.

    Voice-band ratio is the fraction of total spectral energy that falls in
    the voice formant range (300–3400 Hz). For sung kirtan this is typically
    0.5–0.85; for tabla/harmonium-only sections it drops below 0.4.
    """
    if frame.size != FRAME_SAMPLES:
        # Pad short tail to avoid silently truncating the spectrum.
        padded = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        padded[: frame.size] = frame
        frame = padded
    rms = float(np.sqrt(float(np.mean(frame * frame))))
    if rms < _NOISE_FLOOR:
        return 0.0, rms
    spec = np.abs(np.fft.rfft(frame * _HANN))
    total = float(np.sum(spec)) + 1e-9
    voice = float(np.sum(spec[_VOICE_BAND]))
    return voice / total, rms


def _kirtan_segment_batch(
    audio: np.ndarray,
    *,
    threshold: float,
    min_silence_ms: int,
    min_speech_ms: int,
    speech_pad_ms: int,
    max_utterance_ms: int,
) -> list[Utterance]:
    """Batch utterance segmentation using the spectral kirtan detector.

    Mirrors silero-vad's ``get_speech_timestamps`` semantics — same knobs,
    same output shape — so it can drop into the same downstream code.
    """
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        return []

    n_frames = audio.size // FRAME_SAMPLES
    if n_frames == 0:
        return []

    # Score every frame. Smoothing window of ~10 frames (320 ms) suppresses
    # single-frame flickers driven by transient tabla hits.
    voice_ratios = np.empty(n_frames, dtype=np.float32)
    rms_arr = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        voice_ratios[i], rms_arr[i] = _frame_voice_score(frame)

    if n_frames >= 10:
        kernel = np.ones(10, dtype=np.float32) / 10.0
        # 'same' to keep length, then re-clip to [0, 1] to ignore conv edge effects.
        voice_ratios = np.convolve(voice_ratios, kernel, mode="same").astype(np.float32)

    # Run hangover state machine.
    state = "idle"
    candidate_speech_ms = 0.0
    candidate_silence_ms = 0.0
    utterance_start_frame: int | None = None
    pad_frames = max(0, int(speech_pad_ms / FRAME_MS))

    utterances: list[Utterance] = []

    def _flush(end_frame: int) -> None:
        if utterance_start_frame is None:
            return
        s_frame = max(0, utterance_start_frame - pad_frames)
        e_frame = min(n_frames, end_frame + pad_frames)
        s_sample = s_frame * FRAME_SAMPLES
        e_sample = e_frame * FRAME_SAMPLES
        utterances.append(
            Utterance(
                start_s=s_sample / SAMPLE_RATE,
                end_s=e_sample / SAMPLE_RATE,
                audio=audio[s_sample:e_sample].copy(),
            )
        )

    for i in range(n_frames):
        is_voice = voice_ratios[i] >= threshold and rms_arr[i] > _NOISE_FLOOR

        if state == "idle":
            if is_voice:
                state = "maybe_speech"
                candidate_speech_ms = FRAME_MS
                utterance_start_frame = i

        elif state == "maybe_speech":
            if is_voice:
                candidate_speech_ms += FRAME_MS
                if candidate_speech_ms >= min_speech_ms:
                    state = "speech"
                    candidate_speech_ms = 0.0
            else:
                state = "idle"
                candidate_speech_ms = 0.0
                utterance_start_frame = None

        elif state == "speech":
            if not is_voice:
                state = "maybe_silence"
                candidate_silence_ms = FRAME_MS
            duration_ms = (i - (utterance_start_frame or i)) * FRAME_MS
            if duration_ms >= max_utterance_ms:
                _flush(i)
                state = "idle"
                utterance_start_frame = None

        elif state == "maybe_silence":
            if is_voice:
                state = "speech"
                candidate_silence_ms = 0.0
            else:
                candidate_silence_ms += FRAME_MS
                if candidate_silence_ms >= min_silence_ms:
                    _flush(i - int(candidate_silence_ms / FRAME_MS))
                    state = "idle"
                    utterance_start_frame = None

    if state in ("speech", "maybe_silence"):
        _flush(n_frames - 1)

    return utterances


def segment_utterances(
    audio: np.ndarray,
    *,
    threshold: float = 0.55,
    min_silence_ms: int = 400,
    min_speech_ms: int = 200,
    speech_pad_ms: int = 200,
    max_utterance_ms: int = 30000,
    backend: str = "kirtan",
) -> list[Utterance]:
    """Batch utterance segmentation. Used by the tuning script and unit tests.

    ``backend``:
      - ``"kirtan"`` (default) — spectral voice-band detector tuned for sung
        Gurbani audio. Threshold operates on voice_band_ratio ∈ [0, 1].
      - ``"silero"`` — silero-vad. Threshold operates on speech-probability.
        Use only for non-sung audio (katha, spoken introductions).
    """
    if backend == "kirtan":
        return _kirtan_segment_batch(
            audio,
            threshold=threshold,
            min_silence_ms=min_silence_ms,
            min_speech_ms=min_speech_ms,
            speech_pad_ms=speech_pad_ms,
            max_utterance_ms=max_utterance_ms,
        )
    if backend == "silero":
        return _silero_segment_batch(
            audio,
            threshold=threshold,
            min_silence_ms=min_silence_ms,
            min_speech_ms=min_speech_ms,
            speech_pad_ms=speech_pad_ms,
            max_utterance_ms=max_utterance_ms,
        )
    raise ValueError(f"Unknown backend: {backend}. Use 'kirtan' or 'silero'.")


# ──────────────────────────────────────────────────────────────────────
# Silero backend (kept for spoken-content fallback only)
# ──────────────────────────────────────────────────────────────────────

_silero_model_lock = threading.Lock()
_silero_model = None


def _get_silero_model():
    """Load the Silero model once per process."""
    global _silero_model
    if _silero_model is None:
        with _silero_model_lock:
            if _silero_model is None:
                from silero_vad import load_silero_vad
                _silero_model = load_silero_vad()
    return _silero_model


def _silero_segment_batch(
    audio: np.ndarray,
    *,
    threshold: float,
    min_silence_ms: int,
    min_speech_ms: int,
    speech_pad_ms: int,
    max_utterance_ms: int,
) -> list[Utterance]:
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        return []

    import torch
    from silero_vad import get_speech_timestamps

    model = _get_silero_model()
    tensor = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        speech_pad_ms=speech_pad_ms,
        max_speech_duration_s=max_utterance_ms / 1000.0,
        return_seconds=False,
    )
    out: list[Utterance] = []
    for ts in timestamps:
        s = int(ts["start"])
        e = int(ts["end"])
        out.append(
            Utterance(
                start_s=s / SAMPLE_RATE,
                end_s=e / SAMPLE_RATE,
                audio=audio[s:e].copy(),
            )
        )
    return out


# ──────────────────────────────────────────────────────────────────────
# Streaming wrapper (uses KirtanVAD by default, frame-by-frame)
# ──────────────────────────────────────────────────────────────────────


class StreamingVAD:
    """Frame-by-frame streaming VAD for the live pipeline.

    Feeds 32 ms (512-sample) frames into the configured backend, runs a
    hangover state machine on top, and emits ``Utterance`` objects when
    silence is confirmed (or when ``max_utterance_ms`` forces a flush).

    Default backend is ``"kirtan"`` (spectral voice-band detector). Use
    ``"silero"`` only if the input audio is dominated by speech (katha).

    The streaming surface mirrors a typical streaming gate:
      - ``feed(samples) -> list[Utterance]``
      - ``flush() -> Utterance | None``
      - ``reset()``

    Per-frame cost on a 4-core CPU is ~0.3 ms (kirtan backend, FFT-based) or
    ~1.5 ms (Silero backend). Both are negligible next to Whisper decoding.
    """

    _IDLE = "idle"
    _MAYBE_SPEECH = "maybe_speech"
    _SPEECH = "speech"
    _MAYBE_SILENCE = "maybe_silence"

    def __init__(
        self,
        *,
        threshold: float = 0.55,
        min_silence_ms: int = 400,
        min_speech_ms: int = 200,
        speech_pad_ms: int = 200,
        max_utterance_ms: int = 30000,
        backend: str = "kirtan",
    ) -> None:
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_utterance_ms = max_utterance_ms
        self.backend = backend

        if backend == "silero":
            self._silero_model = _get_silero_model()
        else:
            self._silero_model = None

        self._partial: np.ndarray = np.empty(0, dtype=np.float32)
        self._state = self._IDLE
        self._utterance_buffer: list[np.ndarray] = []
        self._pad_buffer: list[np.ndarray] = []
        self._pad_max_samples = max(0, int(speech_pad_ms / 1000.0 * SAMPLE_RATE))
        self._candidate_speech_ms = 0.0
        self._candidate_silence_ms = 0.0
        self._utterance_start_sample: int | None = None
        self._total_samples_seen = 0

    def reset(self) -> None:
        self._partial = np.empty(0, dtype=np.float32)
        self._state = self._IDLE
        self._utterance_buffer.clear()
        self._pad_buffer.clear()
        self._candidate_speech_ms = 0.0
        self._candidate_silence_ms = 0.0
        self._utterance_start_sample = None
        self._total_samples_seen = 0

    # ------------------------------------------------------------------
    # Per-frame scoring
    # ------------------------------------------------------------------

    def _is_speech_frame(self, frame: np.ndarray) -> bool:
        if self.backend == "silero":
            import torch
            with torch.inference_mode():
                tensor = torch.from_numpy(frame).float()
                prob = float(self._silero_model(tensor, SAMPLE_RATE).item())
            return prob >= self.threshold
        # kirtan backend
        voice_ratio, rms = _frame_voice_score(frame)
        return voice_ratio >= self.threshold and rms > _NOISE_FLOOR

    # ------------------------------------------------------------------
    # State-machine helpers
    # ------------------------------------------------------------------

    def _push_pre_roll(self, frame: np.ndarray) -> None:
        if self._pad_max_samples <= 0:
            return
        self._pad_buffer.append(frame)
        total = sum(b.shape[0] for b in self._pad_buffer)
        while total > self._pad_max_samples and self._pad_buffer:
            removed = self._pad_buffer.pop(0)
            total -= removed.shape[0]

    def _drain_pre_roll(self) -> np.ndarray:
        if not self._pad_buffer:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._pad_buffer)

    def _open_utterance(self, frame_start_sample: int) -> None:
        pre = self._drain_pre_roll()
        self._utterance_buffer = []
        if pre.size:
            self._utterance_buffer.append(pre)
            self._utterance_start_sample = frame_start_sample - pre.shape[0]
        else:
            self._utterance_start_sample = frame_start_sample
        self._pad_buffer.clear()

    def _close_utterance(self, frame_end_sample: int) -> Utterance | None:
        if not self._utterance_buffer or self._utterance_start_sample is None:
            return None
        audio = np.concatenate(self._utterance_buffer)
        start_s = self._utterance_start_sample / SAMPLE_RATE
        end_s = frame_end_sample / SAMPLE_RATE
        utt = Utterance(start_s=start_s, end_s=end_s, audio=audio)
        self._utterance_buffer = []
        self._utterance_start_sample = None
        self._candidate_speech_ms = 0.0
        self._candidate_silence_ms = 0.0
        return utt

    def _utterance_duration_ms(self) -> float:
        if self._utterance_start_sample is None:
            return 0.0
        cur_samples = sum(b.shape[0] for b in self._utterance_buffer)
        return cur_samples * 1000.0 / SAMPLE_RATE

    # ------------------------------------------------------------------
    # Public streaming surface
    # ------------------------------------------------------------------

    def feed(self, samples: np.ndarray) -> list[Utterance]:
        """Feed new audio samples, return zero-or-more closed utterances.

        Samples are float32 16 kHz mono. Any non-multiple-of-512 tail is
        kept in an internal buffer until the next call.
        """
        if samples.ndim > 1:
            samples = samples[:, 0]
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.size == 0:
            return []

        if self._partial.size:
            samples = np.concatenate([self._partial, samples])
        n = samples.size
        full_frames = n // FRAME_SAMPLES
        leftover = n - full_frames * FRAME_SAMPLES
        self._partial = samples[-leftover:].copy() if leftover else np.empty(0, dtype=np.float32)

        emitted: list[Utterance] = []
        for f in range(full_frames):
            frame = samples[f * FRAME_SAMPLES : (f + 1) * FRAME_SAMPLES]
            frame_start_sample = self._total_samples_seen
            self._total_samples_seen += FRAME_SAMPLES
            frame_end_sample = self._total_samples_seen

            is_speech = self._is_speech_frame(frame)

            if self._state == self._IDLE:
                self._push_pre_roll(frame)
                if is_speech:
                    self._state = self._MAYBE_SPEECH
                    self._candidate_speech_ms = FRAME_MS

            elif self._state == self._MAYBE_SPEECH:
                if is_speech:
                    self._candidate_speech_ms += FRAME_MS
                    if self._candidate_speech_ms >= self.min_speech_ms:
                        self._open_utterance(frame_start_sample)
                        self._utterance_buffer.append(frame)
                        self._state = self._SPEECH
                        self._candidate_speech_ms = 0.0
                else:
                    self._state = self._IDLE
                    self._candidate_speech_ms = 0.0
                    self._push_pre_roll(frame)

            elif self._state == self._SPEECH:
                self._utterance_buffer.append(frame)
                if not is_speech:
                    self._state = self._MAYBE_SILENCE
                    self._candidate_silence_ms = FRAME_MS
                if self._utterance_duration_ms() >= self.max_utterance_ms:
                    utt = self._close_utterance(frame_end_sample)
                    if utt is not None:
                        emitted.append(utt)
                    self._state = self._IDLE

            elif self._state == self._MAYBE_SILENCE:
                self._utterance_buffer.append(frame)
                if is_speech:
                    self._state = self._SPEECH
                    self._candidate_silence_ms = 0.0
                else:
                    self._candidate_silence_ms += FRAME_MS
                    if self._candidate_silence_ms >= self.min_silence_ms:
                        utt = self._close_utterance(frame_end_sample)
                        if utt is not None:
                            emitted.append(utt)
                        self._state = self._IDLE

        return emitted

    def flush(self) -> Utterance | None:
        """Force-close the in-flight utterance (e.g. on stream end)."""
        if self._state in (self._SPEECH, self._MAYBE_SILENCE):
            return self._close_utterance(self._total_samples_seen)
        return None

    def is_in_utterance(self) -> bool:
        """True when the VAD is currently inside (or just-closing) an utterance.

        Used by the hybrid streaming loop to decide whether to run
        LocalAgreement-2 mid-utterance commits. Returns False during
        IDLE / MAYBE_SPEECH so we don't waste decodes on confirmed-silence
        sections.
        """
        return self._state in (self._SPEECH, self._MAYBE_SILENCE)

    def peek_partial(self) -> np.ndarray:
        """Return the in-progress utterance audio without closing the utterance.

        Returns an empty array when not inside a speech run. Used by the
        hybrid streaming loop to do mid-utterance LocalAgreement-2 commits
        before VAD declares an offset. The array is a fresh concatenation —
        safe to retain across the next ``feed`` call.
        """
        if not self._utterance_buffer:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._utterance_buffer)
