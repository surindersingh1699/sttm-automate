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
    locked_fast_window_duration: float = 5.0  # short locked window for fast recitation
    locked_recovery_window_duration: float = 9.0  # longer locked window when recovering from weak match
    search_fast_window_duration: float = 8.0  # shorter search window for very fast speech
    device: int | None = None  # None = auto-detect: BlackHole > aggregate > default


class WhisperConfig(BaseModel):
    hf_model_id: str = "surindersinghssj/surt-small-v3"  # HuggingFace repo (auto-converted to CT2 int8 on first load)
    local_model_dir: str = "data/surt-small-v3-ct2"  # where the converted CT2 model lives
    device: str = "cpu"
    compute_type: str = "int8"  # int8 for CPU, float16 for GPU
    language: str = "pa"  # Punjabi
    beam_size: int = 5
    vad_filter: bool = False  # disabled — pipeline has its own vocal detection; Silero VAD rejects kirtan singing
    vad_threshold: float = 0.15  # (only used if vad_filter re-enabled)
    vad_min_silence_ms: int = 800  # kirtan has natural pauses between lines (~1s)
    vad_speech_pad_ms: int = 500  # pad more to catch singing onset/offset
    # Decoder toggles — exposed as UI checkboxes, read per transcribe() call (no reload).
    greedy_decode: bool = False         # True → beam_size=1 (3–5× faster, minor accuracy loss)
    single_temperature: bool = True     # True → temperature=[0.0] (no 6× fallback-retry loop)
    allow_repetition: bool = True       # True → compression_ratio_threshold=10.0 (kirtan is really repetitive)
    independent_windows: bool = False   # True → condition_on_previous_text=False (no carryover across windows)
    cap_decode_length: bool = True      # True → max_new_tokens=128 (safety cap; prevents runaway repetition loops)
    max_new_tokens_cap: int = 128       # tokens cap applied when cap_decode_length=True
    skip_slow_windows: bool = True      # True → drop transcription if RTF exceeds skip_slow_rtf_threshold
    skip_slow_rtf_threshold: float = 2.0  # windows slower than realtime × this are discarded before matching


class MatcherConfig(BaseModel):
    # Confidence thresholds
    auto_threshold: float = 0.75  # auto-select (2-cycle confirm at 75-84%, instant lock at 85%+)
    instant_lock_threshold: float = 0.85  # skip confirmation at this confidence
    min_raw_lock_score: float = 0.70  # require a minimum raw per-window score before any lock
    word_overlap_auto_min: int = 1  # min overlapping words for raw auto lock
    word_overlap_evidence_min: int = 2  # stricter overlap when relying on evidence+stability
    word_overlap_instant_min: int = 1  # min overlap for instant lock path
    instant_challenger_switch_score: float = 0.90  # immediate switch when challenger reaches this raw score
    instant_challenger_switch_margin: float = 0.08  # minimum lead over current shabad score for instant switch
    word_overlap_instant_challenger_min: int = 1  # min overlap words for instant challenger switch
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
    vocal_break_min_windows: int = 1  # consecutive non-vocal windows to mark a vocal break
    post_break_boost_windows: int = 3  # windows to stay in start-detection mode after break
    silence_autolock_min_score: float = 0.82  # minimum score to lock during no-lyrics gap
    silence_autolock_windows: int = 2  # how long a strong candidate stays eligible during silence
    hypothesis_top_k: int = 5  # keep top-K shabad hypotheses
    hypothesis_ttl_seconds: float = 5.0  # keep hypotheses alive for this long
    hypothesis_decay: float = 0.85  # cumulative evidence decay each window
    candidate_lock_windows: int = 2  # confirmations required in CANDIDATE_LOCK state
    candidate_lock_miss_windows: int = 4  # weak windows tolerated before dropping pending lock
    progression_high_confidence_bypass: float = 0.88  # skip proximity penalty above this
    fast_speech_letters_per_second: float = 1.50  # above this, favor shorter windows
    slow_speech_letters_per_second: float = 0.65  # below this, favor longer windows
    speech_rate_ema_alpha: float = 0.35  # smoothing factor for speech-rate estimate
    multi_line_search: bool = True  # True → also try 2-line split search for dense/fast text (nitnem)
    multi_line_min_query_length: int = 12  # query must be ≥ this many first-letters to trigger multi-line
    multi_line_score_bonus: float = 0.12  # score boost when both halves hit consecutive DB lines


class STTMConfig(BaseModel):
    ports: list[int] = [8001, 8000, 1397, 1469, 1539, 1552, 1574, 1581, 1606, 1644, 1661, 1665, 1675, 1708]
    connect_timeout: float = 1.0  # seconds per port attempt
    cdp_port: int = 9222  # Chrome DevTools Protocol port for Playwright
    controller_pin: int | None = 8945  # Optional Bani Controller PIN for authenticated control payloads


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    max_candidates: int = 5  # top N candidates to show


class DatabaseConfig(BaseModel):
    # ShabadOS SQLite DB — auto-downloaded from HF on first run if not present locally.
    local_filename: str = "database.sqlite"  # resolved relative to project root
    hf_dataset_id: str = "surindersinghssj/sttm-gurbani-db"
    hf_filename: str = "database.sqlite"
    # Search scope — False = all sources (SGGS + Dasam Granth + Vaaran Bhai Gurdas +
    # Bhai Nand Lal + Sarabloh + Rehitname + …). True = SGGS only.
    sggs_only: bool = False


class AppConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    whisper: WhisperConfig = WhisperConfig()
    matcher: MatcherConfig = MatcherConfig()
    sttm: STTMConfig = STTMConfig()
    dashboard: DashboardConfig = DashboardConfig()
    database: DatabaseConfig = DatabaseConfig()


# Global config instance
config = AppConfig()
