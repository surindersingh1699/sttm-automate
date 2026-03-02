"""Central configuration for STTM Automate."""

from pydantic import BaseModel


class AudioConfig(BaseModel):
    samplerate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    step_duration: float = 3.0  # seconds of new audio per cycle (controls latency)
    window_duration: float = 10.0  # seconds of audio fed to Whisper (controls context)
    start_window_duration: float = 4.5  # shorter window right after a vocal break
    locked_window_duration: float = 7.0  # medium window while following a locked shabad
    device: int | None = None  # None = system default


class WhisperConfig(BaseModel):
    model_size: str = "small"  # tiny, base, small, medium, large-v3 (small = best accuracy/speed for Punjabi)
    device: str = "cpu"
    compute_type: str = "int8"  # int8 for CPU, float16 for GPU
    language: str = "pa"  # Punjabi
    beam_size: int = 5
    vad_filter: bool = True
    vad_threshold: float = 0.35  # lower = more sensitive to speech
    vad_min_silence_ms: int = 800  # kirtan has natural pauses between lines (~1s)
    vad_speech_pad_ms: int = 500  # pad more to catch singing onset/offset


class MatcherConfig(BaseModel):
    # Confidence thresholds
    auto_threshold: float = 0.75  # auto-select (2-cycle confirm at 75-84%, instant lock at 85%+)
    instant_lock_threshold: float = 0.85  # skip confirmation at this confidence
    suggest_threshold: float = 0.60
    # Scoring weights (must sum to 1.0)
    weight_letter_match: float = 0.4
    weight_consecutive: float = 0.3
    weight_context: float = 0.2
    weight_source: float = 0.1
    # State machine
    min_search_letters: int = 3  # minimum first letters before searching
    challenger_margin: float = 0.10  # how much better challenger must score vs current line
    challenger_windows: int = 3  # consecutive windows challenger must win before switching
    weak_line_recovery_score: float = 0.35  # treat locked line match below this as weak
    weak_line_recovery_windows: int = 3  # consecutive weak locked windows before releasing lock
    recovery_challenger_score: float = 0.65  # allow non-auto challenger in recovery mode
    local_line_follow_threshold: float = 0.42  # allow nearby line updates at lower confidence
    local_line_follow_window: int = 2  # consider +/- N lines around current line for fallback
    vocal_break_min_windows: int = 2  # consecutive non-vocal windows to mark a vocal break
    post_break_boost_windows: int = 3  # windows to stay in start-detection mode after break
    silence_autolock_min_score: float = 0.82  # minimum score to lock during no-lyrics gap
    silence_autolock_windows: int = 2  # how long a strong candidate stays eligible during silence
    hypothesis_top_k: int = 5  # keep top-K shabad hypotheses
    hypothesis_ttl_seconds: float = 5.0  # keep hypotheses alive for this long
    hypothesis_decay: float = 0.85  # cumulative evidence decay each window
    candidate_lock_windows: int = 2  # confirmations required in CANDIDATE_LOCK state
    progression_high_confidence_bypass: float = 0.88  # skip proximity penalty above this


class STTMConfig(BaseModel):
    ports: list[int] = [8001, 8000, 1397, 1469, 1539, 1552, 1574, 1581, 1606, 1644, 1661, 1665, 1675, 1708]
    connect_timeout: float = 1.0  # seconds per port attempt
    cdp_port: int = 9222  # Chrome DevTools Protocol port for Playwright
    controller_pin: int | None = 8945  # Optional Bani Controller PIN for authenticated control payloads


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    max_candidates: int = 5  # top N candidates to show


class AppConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    whisper: WhisperConfig = WhisperConfig()
    matcher: MatcherConfig = MatcherConfig()
    sttm: STTMConfig = STTMConfig()
    dashboard: DashboardConfig = DashboardConfig()


# Global config instance
config = AppConfig()
