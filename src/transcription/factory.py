"""Engine factory — picks a Whisper backend by name."""

from src.transcription.base import BaseTranscriptionEngine


SUPPORTED_ENGINES = ("faster-whisper", "mlx-whisper", "whisper-cpp")


def create_engine(name: str) -> BaseTranscriptionEngine:
    """Instantiate a transcription engine by its config name."""
    if name == "faster-whisper":
        from src.transcription.engine import FasterWhisperEngine
        return FasterWhisperEngine()
    if name == "mlx-whisper":
        from src.transcription.mlx_whisper_engine import MlxWhisperEngine
        return MlxWhisperEngine()
    if name == "whisper-cpp":
        from src.transcription.whisper_cpp_engine import WhisperCppEngine
        return WhisperCppEngine()
    raise ValueError(
        f"Unknown transcription engine '{name}'. "
        f"Supported: {', '.join(SUPPORTED_ENGINES)}."
    )
