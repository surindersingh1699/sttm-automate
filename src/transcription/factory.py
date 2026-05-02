"""Engine factory — picks a Whisper backend by name."""

from src.config import config
from src.transcription.base import BaseTranscriptionEngine


SUPPORTED_ENGINES = ("faster-whisper", "mlx-whisper", "whisper-cpp", "whisper-cpp-q8")


def create_engine(name: str) -> BaseTranscriptionEngine:
    """Instantiate a transcription engine by its config name.

    All engines read their model location from ``config.whisper`` — switching
    the active model via ``config.whisper.apply_model_id()`` and then calling
    ``pipeline.switch_engine(current_engine)`` reloads with the new weights.
    """
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
    raise ValueError(
        f"Unknown transcription engine '{name}'. "
        f"Supported: {', '.join(SUPPORTED_ENGINES)}."
    )
