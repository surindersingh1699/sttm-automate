"""Faster-whisper transcription engine for Punjabi kirtan audio.

Uses `surindersinghssj/surt-small-v3` (Whisper-small fine-tuned for Punjabi Gurbani).
The HF repo ships HF Transformers weights; on first load we convert to CTranslate2
int8 format and cache it under `data/surt-small-v3-ct2/`.
"""

from pathlib import Path

import numpy as np

from src.config import config
from src.transcription.base import BaseTranscriptionEngine, TranscriptionSegment


class FasterWhisperEngine(BaseTranscriptionEngine):
    """Transcribes audio using local faster-whisper (CTranslate2) models."""

    def __init__(self):
        self._model = None
        self._language: str | None = config.whisper.language or None

    def load(self):
        """Load (and lazily convert) the fine-tuned Whisper model."""
        from faster_whisper import WhisperModel
        model_dir = self._ensure_ct2_model()
        print(
            f"[FasterWhisper] Loading '{config.whisper.hf_model_id}' from {model_dir} "
            f"(device={config.whisper.device}, compute={config.whisper.compute_type})..."
        )
        self._model = WhisperModel(
            str(model_dir),
            device=config.whisper.device,
            compute_type=config.whisper.compute_type,
        )
        print("[FasterWhisper] Ready.")

    @staticmethod
    def _ensure_ct2_model() -> Path:
        """Convert HF model to CTranslate2 int8 format on first load; cache on disk."""
        project_root = Path(__file__).resolve().parents[2]
        model_dir = (project_root / config.whisper.local_model_dir).resolve()

        if (model_dir / "model.bin").exists() and (model_dir / "config.json").exists():
            return model_dir

        print(
            f"[Whisper] Converting '{config.whisper.hf_model_id}' to CTranslate2 int8 → {model_dir}"
        )
        model_dir.parent.mkdir(parents=True, exist_ok=True)

        # Snapshot the HF repo locally and patch the legacy tokenizer config
        # (`extra_special_tokens` ships as a list but new transformers expects a dict).
        from huggingface_hub import snapshot_download
        import json as _json

        snapshot = Path(
            snapshot_download(
                repo_id=config.whisper.hf_model_id,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.safetensors",
                    "*.bin",
                    "*.model",
                ],
            )
        )
        tok_cfg = snapshot / "tokenizer_config.json"
        if tok_cfg.exists():
            cfg = _json.loads(tok_cfg.read_text(encoding="utf-8"))
            if isinstance(cfg.get("extra_special_tokens"), list):
                cfg.pop("extra_special_tokens", None)
                tok_cfg.write_text(_json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

        from ctranslate2.converters import TransformersConverter

        converter = TransformersConverter(
            str(snapshot),
            copy_files=[
                "tokenizer.json",
                "preprocessor_config.json",
                "generation_config.json",
            ],
            load_as_float16=False,
        )
        converter.convert(
            str(model_dir),
            quantization="int8",
            force=False,
        )
        print("[Whisper] Conversion complete.")
        return model_dir

    def transcribe(self, audio: np.ndarray) -> list[TranscriptionSegment]:
        """
        Transcribe audio chunk (16kHz float32 mono) via faster-whisper.
        Returns list of segments with text.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not self.has_vocal_content(audio):
            return []

        audio = self._normalize(audio)

        # Decoder toggles are read per-call so UI checkboxes take effect without model reload.
        beam_size = 1 if config.whisper.greedy_decode else config.whisper.beam_size
        kwargs = {
            "beam_size": beam_size,
            "vad_filter": config.whisper.vad_filter,
            "vad_parameters": {
                "threshold": config.whisper.vad_threshold,
                "min_silence_duration_ms": config.whisper.vad_min_silence_ms,
                "speech_pad_ms": config.whisper.vad_speech_pad_ms,
            },
        }
        if config.whisper.single_temperature:
            kwargs["temperature"] = [0.0]
        if config.whisper.allow_repetition:
            kwargs["compression_ratio_threshold"] = 10.0
        if config.whisper.independent_windows:
            kwargs["condition_on_previous_text"] = False
        if config.whisper.cap_decode_length:
            # Hard cap prevents runaway decodes when repetition-rejection is relaxed.
            kwargs["max_new_tokens"] = max(16, int(config.whisper.max_new_tokens_cap))
        if self._language:
            kwargs["language"] = self._language

        try:
            segments_iter, info = self._transcribe_with_fallback(audio, kwargs)
        except Exception as e:
            print(f"[Whisper] Transcription error: {e}")
            return []

        if getattr(info, "language", None) and info.language != "pa":
            prob = getattr(info, "language_probability", 0.0)
            print(f"[Whisper] Detected language={info.language} (p={prob:.2f})")

        out: list[TranscriptionSegment] = []
        # Per-segment diagnostics to pinpoint RTF spikes.
        # temperature > 0 ⇒ fallback loop fired (previous T failed quality gates).
        # compression_ratio > 2.4 ⇒ repetition trigger (default retry threshold).
        # avg_logprob  < -1.0  ⇒ low-confidence trigger (default retry threshold).
        # no_speech_prob > 0.6 ⇒ Whisper thinks it's silence.
        for seg in segments_iter:
            text = seg.text.strip()
            marker = ""
            if seg.temperature > 0:
                marker += f" FALLBACK@T={seg.temperature:.1f}"
            if seg.compression_ratio > 2.4:
                marker += f" COMP_RATIO={seg.compression_ratio:.2f}"
            if seg.avg_logprob < -1.0:
                marker += f" LOW_LOGPROB={seg.avg_logprob:.2f}"
            if seg.no_speech_prob > 0.6:
                marker += f" NOSPEECH={seg.no_speech_prob:.2f}"
            if marker:
                print(
                    f"[Whisper-diag]{marker}  T={seg.temperature:.1f} "
                    f"comp={seg.compression_ratio:.2f} "
                    f"logp={seg.avg_logprob:.2f} "
                    f"nosp={seg.no_speech_prob:.2f}  → '{text[:60]}'"
                )
            if text:
                out.append(
                    TranscriptionSegment(
                        start=seg.start,
                        end=seg.end,
                        text=text,
                    )
                )
        return out

    def _transcribe_with_fallback(self, audio: np.ndarray, kwargs: dict):
        """Fallback to language auto-detect if explicit language pinning fails."""
        try:
            return self._model.transcribe(audio, **kwargs)
        except ValueError as e:
            if self._language:
                print(
                    f"[Whisper] language='{self._language}' unavailable ({e}); "
                    "falling back to auto-detect."
                )
                self._language = None
                kwargs.pop("language", None)
                return self._model.transcribe(audio, **kwargs)
            raise

# Back-compat alias for older imports (`TranscriptionEngine`).
TranscriptionEngine = FasterWhisperEngine
