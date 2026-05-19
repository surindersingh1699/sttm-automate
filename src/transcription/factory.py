"""Engine factory — picks a Whisper backend by name."""

from src.config import config
from src.transcription.base import BaseTranscriptionEngine


# Engine names tagged by family. The dashboard reads this split to drive its
# Whisper/Indic family toggle — only Whisper engines apply to Whisper models,
# only Indic engines to IndicConformer models.
WHISPER_ENGINES = ("faster-whisper", "mlx-whisper", "whisper-cpp", "whisper-cpp-q8")
INDIC_ENGINES = ("indicconformer",)
SUPPORTED_ENGINES = WHISPER_ENGINES + INDIC_ENGINES


def pin_indic_best_settings() -> bool:
    """Force the streaming/decoder settings that match IndicConformer's needs.

    IndicConformer is decoded once per VAD-bounded utterance — there is no
    rolling Whisper window, no inter-window dedup, and the engine ignores
    ``initial_prompt``. Whisper-only knobs are dead weight under Indic, but
    a stale "true" loaded from ``.runtime_settings.json`` would still show
    in the UI and confuse the operator. Call this both at startup (after
    runtime settings load) and from ``switch_engine`` so the in-memory
    config never disagrees with what the pipeline is actually doing.

    Returns True if the active engine is Indic (settings pinned), else False.
    """
    if config.whisper.engine not in INDIC_ENGINES:
        return False
    # Preserve a deliberate `nemo_chunked` choice — that mode is also
    # IndicConformer-native and exists specifically as an escape valve when
    # VAD-segmented can't close utterances on quiet/noisy audio. Pin to
    # vad_segmented only when the current mode is one that would fight with
    # IndicConformer (e.g. naive's rolling Whisper window).
    if config.streaming.streaming_mode != "nemo_chunked":
        config.streaming.streaming_mode = "vad_segmented"
    config.streaming.dedup_strategy = "none"
    config.streaming.locked_prompt_anchor = False
    config.audio.zero_overlap_window = False
    config.whisper.hallucination_guards = False
    # Tighten VAD bounds for IndicConformer. The default 30 s utterance cap is
    # a Whisper-era safety bound; IndicConformer was trained on ≤15 s clips and
    # silently returns empty text on very long inputs. Keep the bound below
    # the model's training distribution by default. apply_engine_profile is
    # idempotent and only touches a small whitelist, so it's safe to chain.
    from src.config import apply_engine_profile
    apply_engine_profile("indicconformer", "normal")
    return True


def create_engine(name: str) -> BaseTranscriptionEngine:
    """Instantiate a transcription engine by its config name.

    All engines read their model location from ``config.whisper`` — switching
    the active model via ``config.whisper.apply_model_id()`` and then calling
    ``pipeline.switch_engine(current_engine)`` reloads with the new weights.
    """
    family = config.whisper.model_family()
    if name in INDIC_ENGINES and family != "indicconformer":
        raise ValueError(
            f"Engine '{name}' requires an IndicConformer model, but active "
            f"model '{config.whisper.hf_model_id}' is a {family} checkpoint. "
            f"Switch the model first via config.whisper.apply_model_id()."
        )
    if name in WHISPER_ENGINES and family != "whisper":
        raise ValueError(
            f"Engine '{name}' requires a Whisper model, but active model "
            f"'{config.whisper.hf_model_id}' is a {family} checkpoint. "
            f"Switch the model first via config.whisper.apply_model_id()."
        )
    if name == "faster-whisper":
        from src.transcription.engine import FasterWhisperEngine
        return FasterWhisperEngine()
    if name == "mlx-whisper":
        from src.transcription.mlx_whisper_engine import MlxWhisperEngine
        return MlxWhisperEngine()
    if name == "whisper-cpp":
        from src.transcription.whisper_cpp_engine import WhisperCppEngine
        return WhisperCppEngine()
    if name == "whisper-cpp-q8":
        # Quantized q8_0 — ~265 MB, ~88% word fidelity to f16, ships cross-platform.
        # Path derives from `config.whisper.hf_model_id` so it tracks the
        # active model when the user switches via the dashboard.
        from src.transcription.whisper_cpp_engine import WhisperCppEngine
        return WhisperCppEngine(model_path=config.whisper.whisper_cpp_q8_model_path)
    if name == "indicconformer":
        # CTC-only ONNX export served via sherpa-onnx. Active precision lives
        # in config.whisper.onnx_precision and is editable from the dashboard;
        # the engine reloads itself on the next transcribe() when that changes.
        from src.transcription.onnx_engine import IndicConformerEngine
        return IndicConformerEngine()
    raise ValueError(
        f"Unknown transcription engine '{name}'. "
        f"Supported: {', '.join(SUPPORTED_ENGINES)}."
    )
