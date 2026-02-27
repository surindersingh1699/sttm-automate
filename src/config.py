"""Central configuration for STTM Automate."""

from pydantic import BaseModel


class AudioConfig(BaseModel):
    samplerate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    chunk_duration: float = 5.0  # seconds per transcription window
    overlap_duration: float = 1.0  # overlap between windows
    device: int | None = None  # None = system default


class WhisperConfig(BaseModel):
    model_size: str = "base"  # tiny, base, small, medium, large-v3
    device: str = "cpu"
    compute_type: str = "int8"  # int8 for CPU, float16 for GPU
    language: str = "pa"  # Punjabi
    beam_size: int = 5
    vad_filter: bool = True
    vad_threshold: float = 0.3  # lower = more sensitive to speech
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 300


class MatcherConfig(BaseModel):
    # Confidence thresholds
    auto_threshold: float = 0.85
    suggest_threshold: float = 0.60
    # Scoring weights (must sum to 1.0)
    weight_letter_match: float = 0.4
    weight_consecutive: float = 0.3
    weight_context: float = 0.2
    weight_source: float = 0.1
    # Transition detection
    transition_count: int = 3  # consecutive matches to new shabad before switching
    min_search_letters: int = 3  # minimum first letters before searching


class STTMConfig(BaseModel):
    ports: list[int] = [8000, 1397, 1469, 1539, 1552, 1574, 1581, 1606, 1644, 1661, 1665, 1675, 1708]
    connect_timeout: float = 1.0  # seconds per port attempt
    cdp_port: int = 9222  # Chrome DevTools Protocol port for Playwright


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
