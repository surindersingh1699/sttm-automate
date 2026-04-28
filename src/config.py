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
    locked_micro_window_duration: float = 3.0  # minimal window while locked + stable (1 line ≈ 3s recitation)
    locked_very_fast_window_duration: float = 3.5  # locked window for *very* fast recitation (Fix 3 tier)
    search_fast_window_duration: float = 8.0  # shorter search window for very fast speech
    device: int | None = None  # None = auto-detect: BlackHole > aggregate > default


class WhisperConfig(BaseModel):
    # Which backend powers transcription. Swappable at runtime from the dashboard.
    # "faster-whisper" = CTranslate2 int8 (default, cross-platform CPU)
    # "mlx-whisper"    = Apple MLX (GPU/ANE, macOS Apple Silicon only)
    # "whisper-cpp"    = whisper.cpp via pywhispercpp (cross-platform incl. iOS)
    engine: str = "faster-whisper"
    hf_model_id: str = "surindersinghssj/surt-small-v3"  # HuggingFace repo (auto-converted to CT2 int8 on first load)
    local_model_dir: str = "data/surt-small-v3-ct2"  # where the converted CT2 model lives
    # MLX-specific:
    mlx_model_dir: str = "data/surt-small-v3-mlx"  # converted MLX weights cache
    mlx_quantize: bool = True  # 4-bit quantize on conversion to shrink + speed up
    mlx_quantize_bits: int = 4
    # whisper.cpp-specific: GGML file path. Auto-converted from `hf_model_id`
    # on first load via the vendored upstream converter.
    whisper_cpp_model_path: str = "data/surt-small-v3.ggml"
    whisper_cpp_threads: int = 4
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
    # Confidence gap (Change 4) — bypass above high_confidence, else require gap between top-1 and top-2
    high_confidence_lock_threshold: float = 0.90
    gap_threshold: float = 0.10
    # Tiered lock (Change 1) — suggest-level candidate promoted after this many seconds as top candidate
    suggest_confirmation_seconds: float = 4.0
    # Challenger switch (Change 5) — immediate override vs time-based confirmation
    strong_override_threshold: float = 0.90
    override_min_gap: float = 0.05
    challenger_confirmation_seconds: float = 4.0
    # Stale-memory reset — if gap between windows exceeds this, counters are too old to trust
    stale_memory_threshold_seconds: float = 10.0
    # Phonetic (Change 2) — cap on combinatorial variant count
    phonetic_max_variants: int = 32
    # Smith-Waterman word alignment in locked-state line scoring (Change 6)
    sw_line_scoring_enabled: bool = True
    # Alaap detection (Change 7): freeze line pointer on melismatic/non-lexical windows
    alaap_detection_enabled: bool = True
    alaap_consecutive_windows: int = 2  # N consecutive alaap windows before freezing
    # Transition mode (Change 8): relax thresholds when shabad is likely transitioning
    transition_mode_enabled: bool = True
    transition_min_signals: int = 2          # signals needed to enter transition mode
    transition_weak_seconds: float = 6.0    # seconds of low locked-line score = 1 signal
    transition_silence_seconds: float = 8.0  # recent alaap/silence seconds = 1 signal
    transition_max_duration_seconds: float = 30.0  # max time in transition mode
    transition_challenger_confirmation_s: float = 1.5   # relaxed challenger timer
    transition_override_threshold: float = 0.80         # relaxed override threshold
    transition_override_min_gap: float = 0.05           # relaxed gap
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
    # Change 9: confident-jump bypass — ANY line (including current) skips all bias
    # bonuses and the 1-line jump cap when its raw score clears this threshold.
    # Lower than progression_high_confidence_bypass so confident jumps win earlier.
    progression_confident_jump_threshold: float = 0.85
    # When True (default), the high-confidence bypass only fires for non-current lines so
    # the current-line +0.22 bonus always applies. Set False to restore the original
    # behaviour (bypass strips all bonuses including delta=0) for A/B comparison.
    progression_symmetric_bypass: bool = True
    # When False, the delta=+1 next-line bonus (0.12 + time-pressure ramp) is removed so
    # the next line must beat the current line on raw score alone. The current-line inertia
    # bonus is also reduced from 0.22 to 0.05 (pure tiebreaker). Enables pure evidence-
    # driven advancement — no time-based or positional nudges.
    next_line_bias_enabled: bool = False
    # Use char-4-gram overlap on full Unicode Gurmukhi verse text alongside first-letter
    # scoring inside locked state. Takes max(FL_score, ngram_score) per line so it only
    # helps, never hurts.
    ngram_line_scoring: bool = True
    # When True, line pointer positioning uses normalized word-set overlap instead of
    # FL/ngram scoring — picks the verse that contains the most transcript words.
    # Applies to all query lengths within the locked shabad. 2-word sequential match
    # and the ngram/FL paths are bypassed when this is on.
    word_match_line_scoring: bool = False
    # Minimum Gurmukhi word count in the transcript before the line pointer is updated.
    # 1-word fragments are too noisy to move the pointer. 2-word fragments are allowed
    # but use normalized full-word matching (not FL/ngram) — only within the current
    # locked shabad, never to challenge or switch to a different shabad.
    min_words_for_line_advance: int = 2
    # Line-advance gate — keeps the pointer on the current pangati until the ragi has
    # actually left it. Without this, the 3 s micro window advances mid-line as soon
    # as the next line's first syllable bleeds in. Matches the micro window duration
    # so the pointer cannot advance faster than one window tick.
    min_line_dwell_seconds: float = 4.5
    line_advance_override_score: float = 0.82  # next-line score that bypasses the dwell gate
    fast_speech_letters_per_second: float = 1.50  # above this, favor shorter windows
    slow_speech_letters_per_second: float = 0.65  # below this, favor longer windows
    speech_rate_ema_alpha: float = 0.35  # smoothing factor for speech-rate estimate
    multi_line_search: bool = True  # True → also try 2-line split search for dense/fast text (nitnem)
    multi_line_min_query_length: int = 12  # query must be ≥ this many first-letters to trigger multi-line
    multi_line_score_bonus: float = 0.12  # score boost when both halves hit consecutive DB lines
    multi_line_locked_align: bool = True  # True → in LOCKED state, also score against pairs of consecutive verses
    multi_line_locked_min_query_length: int = 10  # query must be ≥ this many letters to try pair alignment
    # 3-way multi-line split (Fix 3): fires alongside 2-way when the query is long enough
    # to plausibly span 3 consecutive DB lines (dense nitnem / very fast kirtan).
    multi_line_trinary_min_query_length: int = 18
    multi_line_trinary_score_bonus: float = 0.15  # slightly > 2-way bonus: 3 consecutive hits is stronger evidence
    multi_line_locked_trinary_min_query_length: int = 16
    # Dense-window scoring (Fix 3): score query against the shabad's full concatenated
    # first-letters so fast/multi-line windows don't get dragged down by line-level ratio.
    dense_coverage_weight: float = 0.7
    # When dense_coverage beats both letter_ratio and subsequence coverage by this margin,
    # flag it "dense_dominant" — callers require extra corroboration (higher word overlap)
    # before promoting such candidates to instant switches or auto-locks. Keeps spurious
    # substring matches against unrelated shabads from hijacking a good lock.
    dense_dominant_margin: float = 0.65
    # Extra word-overlap bar to use when the challenger signal is dense_dominant.
    dense_dominant_instant_overlap_min: int = 3
    # Very-fast speech window tier (Fix 3) — shrink locked window further above this LPS.
    very_fast_speech_letters_per_second: float = 2.2
    # Word-level retrieval (Fix 2 / type3_words). IDF-weighted voting on transcript words.
    word_vote_enabled: bool = True
    word_vote_stopword_df_ratio: float = 0.25   # words in >25% of shabads count as stop-words (weight ≈ 0)
    word_vote_min_distinct_hits: int = 2        # at least this many distinct transcript words must vote
    word_vote_min_score: float = 1.5            # summed IDF weight floor before a candidate is returned
    word_vote_single_hit_min_score: float = 3.5  # allow 1 hit if its IDF weight alone clears this (rare/distinctive word)
    # Char 4-gram Unicode retrieval (Strategy 9) — catches end-fragment kirtan patterns
    # where ragi sings only the 2nd half of a DB line (FL prefix/contains can't find it).
    ngram4_search_enabled: bool = True
    ngram4_min_overlap: float = 0.30  # overlap-coefficient floor before a line is considered
    ngram4_max_results: int = 8
    word_vote_bonus_2: float = 0.05             # _score_candidates bonus for 2 distinct word hits
    word_vote_bonus_3: float = 0.10             # … 3 hits
    word_vote_bonus_4plus: float = 0.15         # … 4+ hits
    word_vote_only_floor: float = 0.45          # candidates from word-vote alone must clear this first-letter score before auto-lock
    # Alap / detour handling — during alap ragi briefly sings a tuk from another shabad.
    # We detect it and flag it on the dashboard but don't move STTM off the current shabad.
    alap_detour_min_score: float = 0.70  # min sticky-set score to flag a detour
    alap_sticky_max_size: int = 3  # track up to N recently-sung shabads as alap candidates
    alap_sticky_ttl_seconds: float = 600.0  # drop history shabads from sticky set after this
    alap_commit_windows: int = 4  # sustained detour windows before promoting to a real shabad switch
    # Stable-lock fast tracking — when recent line alignments are strong, trust a very short window.
    # Penalize line 0 (raag heading) in locked-state line pointer so it never
    # wins the line race once we've locked.  Line 0 is always the raag/mahala
    # header ("ਮਾਝ ਮਹਲਾ ੫ ॥") and is never sung — a positive score for it is
    # always a false positive driven by short FL or tiebreaker arithmetic.
    penalize_heading_line: bool = True
    locked_stable_score_threshold: float = 0.55  # per-window score that counts as a "stable" line hit
    locked_stable_min_windows: int = 2  # that many stable windows in a row ⇒ use locked_micro_window
    # Fast real-switch path — if current shabad line is weak AND challenger is strong, switch fast.
    fast_switch_current_weak_score: float = 0.35  # treat current-line score below this as weak
    fast_switch_current_weak_windows: int = 2  # consecutive weak windows that unlock fast switch
    fast_switch_challenger_windows: int = 2  # challenger windows needed under fast-switch conditions
    # --- Predictive line tracking ---
    # Layer 1: time-aware progression bias (always active when LOCKED).
    # Scales the delta=+1 bonus linearly from 0 → max as the current line ages.
    predictive_time_bias_max: float = 0.03
    # Layer 2: per-session dwell-time estimator (toggle).
    predictive_dwell_enabled: bool = False
    predictive_dwell_ema_alpha: float = 0.30
    predictive_dwell_seed_seconds: float = 4.0
    predictive_dwell_min_seconds: float = 0.8   # ignore dwells outside this range
    predictive_dwell_max_seconds: float = 12.0
    # Layer 3: tentative predictive advance (toggle).
    predictive_advance_enabled: bool = False
    predictive_advance_threshold: float = 0.88  # time_pressure (0–1.5) that triggers advance
    predictive_advance_min_confirms: int = 3    # confirmed advances needed before predicting
    predictive_advance_repeat_score: float = 0.62  # current-line score above this → assume tuk repeat


class STTMConfig(BaseModel):
    ports: list[int] = [8001, 1397, 1469, 1539, 1552, 1574, 1581, 1606, 1644, 1661, 1665, 1675, 1708]
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
