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
    # Fast-response mode: when MatcherConfig.fast_response_enabled is True the capture
    # loop wakes every fast_step_duration seconds instead of step_duration.  The Whisper
    # window length is unchanged — we just look at the (already-buffered) audio twice as
    # often, halving the wake-up wait without changing decode cost or accuracy floors.
    fast_step_duration: float = 1.5
    # Zero-overlap mode (REA-11): when on, the capture loop sleeps exactly the
    # last-decoded *dynamic* window duration before the next wake-up. Each
    # second of audio gets transcribed once instead of overlapping under the
    # default 3 s wake / 10 s window. ~50 % encoder savings; trade-off is
    # potential word-boundary clipping at chunk edges (matcher's first-letter
    # retrieval is robust to one missing word in a multi-word query). Default
    # off to preserve existing capture behavior; user-togglable via dashboard.
    # Composes with fast_response: when both are on, zero-overlap wins because
    # its semantics ("hop = window") subsume the fast-response 1.5 s cadence.
    zero_overlap_window: bool = False
    device: int | None = None  # None = auto-detect: BlackHole > aggregate > default


class WhisperConfig(BaseModel):
    # Which backend powers transcription. Swappable at runtime from the dashboard.
    # "faster-whisper" = CTranslate2 int8 (default, cross-platform CPU)
    # "mlx-whisper"    = Apple MLX (GPU/ANE, macOS Apple Silicon only)
    # "whisper-cpp"    = whisper.cpp via pywhispercpp (cross-platform incl. iOS)
    # "indicconformer" = AI4Bharat IndicConformer hybrid CTC/RNN-T via NeMo
    engine: str = "faster-whisper"
    # Active model (HuggingFace repo id). User-selectable from the dashboard.
    # The engine-specific cache paths below derive from this — switching the
    # active model via apply_model_id() rewrites all four together.
    hf_model_id: str = "surindersinghssj/surt-small-v3"
    # Registry of selectable models. Add a new HF repo id here to expose it
    # in the dashboard's model dropdown. Each entry is auto-converted to the
    # active engine's cache format on first load (CT2 / MLX / GGML / .nemo).
    available_models: list[str] = [
        "surindersinghssj/surt-small-v3",
        "surindersinghssj/surt-small-turbo-baseline-v0",
        "surindersinghssj/indicconformer-pa-v3-kirtan",
    ]
    # Model family classifier — drives which engine set is valid for each
    # entry in available_models, and which Whisper-style cache paths get
    # rewritten by apply_model_id(). NeMo models bypass the GGML/CT2/MLX
    # conversion machinery entirely.
    model_families: dict[str, str] = {
        "surindersinghssj/surt-small-v3": "whisper",
        "surindersinghssj/surt-small-turbo-baseline-v0": "whisper",
        "surindersinghssj/indicconformer-pa-v3-kirtan": "indicconformer",
    }
    local_model_dir: str = "data/surt-small-v3-ct2"  # where the converted CT2 model lives
    # MLX-specific:
    mlx_model_dir: str = "data/surt-small-v3-mlx"  # converted MLX weights cache (Apple Metal)
    mlx_quantize: bool = True  # 4-bit quantize on conversion to shrink + speed up
    mlx_quantize_bits: int = 4
    # whisper.cpp-specific: GGML file paths. Auto-converted from `hf_model_id`
    # on first load via the vendored upstream converter.
    whisper_cpp_model_path: str = "data/surt-small-v3.ggml"
    whisper_cpp_q8_model_path: str = "data/surt-small-v3-q8_0.ggml"
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
    # whisper.cpp hallucination guards — bundles four rejection filters whisper.cpp
    # exposes but doesn't aggressively apply by default. When on, suspect segments
    # (low logprob, high entropy/repetition, high no-speech prob) get retried up
    # the temperature ladder and dropped if no temperature passes.
    hallucination_guards: bool = True
    # Kirtan-tuned thresholds: keep no-speech strict (Whisper's no-speech head is
    # the cleanest instrumental detector), but loosen logprob + entropy a lot —
    # sung vowels score low-confidence and kirtan refrains look repetitive, both
    # of which would trip the spoken-speech defaults (-1.0 and 2.4).
    hg_no_speech_thold: float = 0.9      # very lenient — sustained sung vowels can fire Whisper's no-speech head; only drop if it's REALLY sure
    hg_logprob_thold: float = -2.5       # very lenient — sung speech has much lower per-token confidence
    hg_entropy_thold: float = 8.0        # loosened from 2.4 (kirtan refrains are legitimately repetitive)
    hg_temperature_inc: float = 0.2      # fallback ladder step (0 → no fallback)
    # ── IndicConformer (ONNX, CTC) ──────────────────────────────────────
    # CTC-only ONNX export of our fine-tuned IndicConformer-pa, served via
    # sherpa-onnx. Three precisions live side-by-side under
    # ``{onnx_model_dir}/{onnx_precision}/indicconformer-pa-ctc.onnx`` with a
    # shared ``tokens.txt`` at the root; the engine downloads the relevant
    # files from the HF repo on first load if they aren't already there.
    onnx_model_dir: str = "~/models/exports-pa"
    # Active precision — selectable from the dashboard.
    #   fp32 — accuracy reference (~470 MB)
    #   fp16 — currently BROKEN: NeMo→ONNX fp16 export left mixed float/half
    #          tensors at /pre_encode/Add and onnxruntime refuses to load it.
    #          Re-export with the cast fixed before re-enabling.
    #   int8 — fastest + smallest (~134 MB). Default until fp16 is re-exported.
    onnx_precision: str = "int8"
    available_precisions: tuple[str, ...] = ("fp32", "int8")
    # CPU thread count for the ONNX session.
    onnx_threads: int = 4
    # Chunked streaming for IndicConformer (StreamingConfig.streaming_mode="nemo_chunked").
    # Decode every `nemo_chunk_len_s` seconds of audio standalone (no VAD gating,
    # no rolling window). Latency ≈ chunk_len_s; CPU cost ≈ one decode per chunk.
    # Smaller chunk = lower latency + more boundary errors; matcher's first-letter
    # retrieval is robust to a missing word at the edges, so 1.0–1.5 s works well
    # for fast bani. Set to 0 to disable (fall back to vad_segmented behavior).
    nemo_chunk_len_s: float = 1.5
    # Optional left-context audio fed to the model alongside each chunk to give
    # the encoder a warm-up region before the new audio. We only KEEP the new
    # text — context exists purely to stabilize the encoder's first frames.
    # 0 disables; 0.5 s is a safe default (~half a pankti syllable).
    nemo_chunk_context_s: float = 0.5

    def model_family(self, model_id: str | None = None) -> str:
        """Return ``"whisper"`` or ``"indicconformer"`` for the given model id.

        Defaults to the active model. Unknown ids fall back to ``"whisper"``
        — the legacy assumption — so config files written before the family
        registry shipped continue to work.
        """
        mid = model_id or self.hf_model_id
        return self.model_families.get(mid, "whisper")

    def apply_model_id(self, model_id: str) -> None:
        """Switch the active model; rewrite engine-specific cache paths.

        For Whisper models we update all four cache paths (CT2/MLX/GGML/GGML-q8)
        together so any of the four engines can pick up the swap. For
        IndicConformer the active artifacts live under ``onnx_model_dir`` and
        track the HF repo id directly, so there's no per-model cache path to
        rewrite here.

        Caller is responsible for triggering an engine reload
        (``pipeline.switch_engine(current_engine)``) afterward so the new
        weights are actually loaded.
        """
        short = model_id.rsplit("/", 1)[-1]
        self.hf_model_id = model_id
        if self.model_family(model_id) == "indicconformer":
            return
        self.local_model_dir = f"data/{short}-ct2"
        self.mlx_model_dir = f"data/{short}-mlx"
        self.whisper_cpp_model_path = f"data/{short}.ggml"
        self.whisper_cpp_q8_model_path = f"data/{short}-q8_0.ggml"


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
    # Two-phase line selection: bypass progression bias entirely when the *raw*
    # best line clears progression_confident_jump_threshold OR when raw top-1
    # leads raw top-2 by this gap.  Catches confident jumps that neither line
    # individually meets the absolute threshold.
    line_jump_gap_threshold: float = 0.20
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
    # Fast-response toggle — single switch for "react faster without lowering accuracy floors".
    # When True the orchestrator uses:
    #   * audio.fast_step_duration instead of audio.step_duration (capture cadence)
    #   * fast_min_line_dwell_seconds instead of min_line_dwell_seconds (line move gate)
    #   * fast_suggest_confirmation_seconds instead of suggest_confirmation_seconds
    # All accuracy thresholds (auto/instant lock, line override score, etc.) stay the same.
    fast_response_enabled: bool = False
    fast_min_line_dwell_seconds: float = 3.0
    fast_suggest_confirmation_seconds: float = 2.5


class StreamingConfig(BaseModel):
    """Streaming pipeline behavior — how audio is gated, decoded, and deduplicated.

    All fields default to legacy behavior. Flip toggles via the dashboard or
    edit `.runtime_settings.json["streaming"]` to A/B against the original
    naive-sliding-window pipeline. See REA-10 for the full design.
    """
    # ── Streaming mode ──────────────────────────────────────────────────
    # naive            — original: wake every step_duration, decode latest window
    #                     (current default; preserves all behavior pre-REA-10).
    # vad_segmented    — Silero VAD bounds utterances; Whisper runs once per utterance.
    #                     Pankti boundaries are natural breath pauses, so each utterance
    #                     ≈ one pankti. Eliminates window overlap and the dedup it requires.
    # local_agreement  — accumulating audio buffer; commit only the longest token
    #                     prefix two consecutive decodes agree on (Macháček 2023).
    # hybrid           — VAD-bounded utterance buffers + LocalAgreement-2 inside.
    #                     Production target.
    # nemo_chunked     — IndicConformer-only: decode every `nemo_chunk_len_s` seconds
    #                     of audio standalone. Bypasses VAD entirely so fast bani
    #                     (no breath gaps) doesn't merge multiple panktis into one
    #                     decode. Whisper-engine guard rails make this a no-op for
    #                     non-NeMo engines — they fall back to naive automatically.
    streaming_mode: str = "naive"

    # ── VAD ─────────────────────────────────────────────────────────────
    # Backend selects the underlying detector:
    #   "kirtan" — spectral voice-band detector (DEFAULT). Tuned for sung
    #             Gurbani audio. Threshold operates on voice-band energy ratio
    #             (300–3400 Hz / total). Empirically good on kirtan; Silero
    #             was confirmed to reject sung Gurbani as non-speech.
    #   "silero" — silero-vad. Use only for spoken katha-style content.
    vad_backend: str = "kirtan"
    # Detection threshold. Semantics depend on backend:
    #   kirtan: voice-band energy ratio cutoff (0.0–1.0). Tuned on cached
    #           kirtan eval — 0.65 produced p50=2.85 s utterances and 88 %
    #           voice ratio (kirtan-shaped). See scripts/tune_vad.py.
    #   silero: speech-probability cutoff (0.0–1.0).
    vad_threshold: float = 0.65
    # Minimum silence run after speech before declaring utterance offset.
    # 200 ms tolerates mid-pankti breaths without splitting; 400 ms = more
    # conservative. Tuned default keeps median utterance ≈ pankti length.
    vad_min_silence_ms: int = 200
    # Minimum confirmed speech run before declaring utterance onset.
    # 300 ms suppresses spurious tabla transients while still catching short
    # tuks and jaaps.
    vad_min_speech_ms: int = 300
    # Pad audio around utterance boundaries to avoid cutting off vowel tails.
    vad_speech_pad_ms: int = 200
    # Maximum utterance length before forced flush (safety bound for runaway sustains).
    vad_max_utterance_ms: int = 30000

    # ── LocalAgreement-2 ────────────────────────────────────────────────
    # Number of consecutive decodes whose prefixes must agree before committing.
    # 2 = standard whisper-streaming setting; 3+ = more conservative, slower commit.
    local_agreement_n: int = 2
    # How often to re-decode the streaming buffer.
    local_agreement_decode_interval_ms: int = 1500
    # Drop the streaming buffer once it exceeds this duration (forces a re-anchor).
    local_agreement_max_buffer_ms: int = 20000

    # ── Dedup strategy ──────────────────────────────────────────────────
    # text       — current behavior: strip overlapping prefixes from the new
    #              transcript when they match the tail of the last commit.
    #              Side effect: drops legitimate repetition (rahau, jaap repetition).
    # audio_time — only dedup when AUDIO time ranges overlap. Fixes legit repetition.
    # none       — no dedup at all (raw segments through). Useful for streaming modes
    #              that already commit at audio boundaries (vad_segmented, local_agreement).
    dedup_strategy: str = "text"

    # ── Locked-shabad prompt anchoring (REA-10) ─────────────────────────
    # When the matcher is LOCKED on a shabad, pass the current pankti's
    # Gurmukhi text to Whisper as ``initial_prompt``. Biases decode toward
    # the actual shabad text — typically +5–10 WER points. Off by default
    # because a wrong lock + prompt anchor double-down on the wrong text;
    # enable once the matcher's lock confidence is trusted.
    locked_prompt_anchor: bool = False


class STTMConfig(BaseModel):
    ports: list[int] = [8001, 8000, 1397, 1469, 1539, 1552, 1574, 1581, 1606, 1644, 1661, 1665, 1675, 1708]
    connect_timeout: float = 1.0  # seconds per port attempt
    cdp_port: int = 9222  # Chrome DevTools Protocol port for Playwright
    controller_pin: int | None = 8945  # Optional Bani Controller PIN for authenticated control payloads


class DashboardConfig(BaseModel):
    # Default to localhost-only. Token auth is meant as a polite gate, not a
    # MITM defence — keeping the listener bound to 127.0.0.1 means an
    # attacker has to already be on this Mac to even *try* a token guess.
    # Set STTM_LAN_MODE=1 (or `lan_mode = true` in .runtime_settings.json)
    # to expose the controller on the local network for sangat-mode clients.
    host: str = "127.0.0.1"
    port: int = 8080
    lan_mode: bool = False  # when True, host flips to 0.0.0.0
    max_candidates: int = 5  # top N candidates to show


class DatabaseConfig(BaseModel):
    # ShabadOS SQLite DB — auto-downloaded from HF on first run if not present locally.
    local_filename: str = "database.sqlite"  # resolved relative to project root
    hf_dataset_id: str = "surindersinghssj/sttm-gurbani-db"
    hf_filename: str = "database.sqlite"


class AppConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    whisper: WhisperConfig = WhisperConfig()
    matcher: MatcherConfig = MatcherConfig()
    streaming: StreamingConfig = StreamingConfig()
    sttm: STTMConfig = STTMConfig()
    dashboard: DashboardConfig = DashboardConfig()
    database: DatabaseConfig = DatabaseConfig()


# Global config instance
config = AppConfig()


# ──────────────────────────────────────────────────────────────────────
# Engine-family tuning profiles
# ──────────────────────────────────────────────────────────────────────
#
# These profiles flip a small whitelist of audio + streaming knobs so the
# capture pipeline matches what each engine family actually wants. They do
# NOT touch matcher knobs — that's `set_confidence_mode`'s job. Call
# `apply_engine_profile()` after a model/engine swap or at startup once
# `.runtime_settings.json` has been merged in.
#
# Why a whitelist instead of full presets? The user-tunable knobs in
# `.runtime_settings.json` should keep winning when they don't conflict
# with engine-family invariants. We only override the keys whose "right
# default" depends on the engine.

# Per (family, speed) overlay. Speed is a hint — "normal" (default) is the
# safe baseline; "fast" tightens VAD + capture cadence for fast recitation.
_ENGINE_PROFILES: dict[tuple[str, str], dict[str, dict[str, object]]] = {
    # Whisper baseline — restores legacy values so swapping back from
    # IndicConformer doesn't strand the operator with indic-tuned VAD.
    ("whisper", "normal"): {
        "streaming": {
            "vad_min_silence_ms": 200,
            "vad_min_speech_ms": 300,
            "vad_max_utterance_ms": 30000,
        },
        "audio": {
            "fast_step_duration": 1.5,
        },
    },
    # IndicConformer baseline — loosely matches what the existing
    # `pin_indic_best_settings` already enforces, plus VAD knobs.
    ("indicconformer", "normal"): {
        "streaming": {
            "vad_min_silence_ms": 180,
            "vad_min_speech_ms": 250,
            "vad_max_utterance_ms": 8000,
        },
        "audio": {
            "fast_step_duration": 1.2,
        },
    },
    # IndicConformer FAST — tuned for very rapid Nitnem-style recitation.
    # Cuts utterances on shorter breaths, force-flushes utterances at 2.5 s
    # so one decode ≈ one pankti even when the ragi barely pauses.
    ("indicconformer", "fast"): {
        "streaming": {
            "vad_min_silence_ms": 100,
            "vad_min_speech_ms": 150,
            "vad_max_utterance_ms": 2500,
            "vad_speech_pad_ms": 100,
        },
        "audio": {
            "fast_step_duration": 0.8,
        },
    },
}


def apply_engine_profile(family: str | None = None, speed: str = "normal") -> bool:
    """Apply audio/streaming defaults for the given engine family.

    Returns True if a profile was applied. Unknown (family, speed) pairs are
    a no-op (return False) so callers can pass through user input safely.
    Idempotent — calling twice with the same args is identical to calling once.
    """
    if family is None:
        family = config.whisper.model_family()
    overlay = _ENGINE_PROFILES.get((family, speed))
    if overlay is None:
        return False
    for section_name, section_overrides in overlay.items():
        section = getattr(config, section_name, None)
        if section is None:
            continue
        for key, value in section_overrides.items():
            if hasattr(section, key):
                setattr(section, key, value)
    return True
