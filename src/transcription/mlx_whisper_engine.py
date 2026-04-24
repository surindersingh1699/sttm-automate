"""MLX-Whisper transcription engine for Apple Silicon.

Uses Apple's MLX framework (GPU + ANE) for ~2-3x speedup vs faster-whisper CPU
on M-series Macs. mlx-whisper accepts HF repos directly — it auto-downloads
the MLX weights on first use. For the fine-tuned `surt-small-v3` model we do
a one-shot HF→MLX conversion and cache it on disk.
"""

from pathlib import Path

import numpy as np

from src.config import config
from src.transcription.base import BaseTranscriptionEngine, TranscriptionSegment


class MlxWhisperEngine(BaseTranscriptionEngine):
    """Transcribes audio using mlx-whisper (Apple Silicon GPU/ANE)."""

    def __init__(self):
        self._model_path: str | None = None
        self._language: str | None = config.whisper.language or None

    def load(self):
        """Resolve (and lazily convert) the MLX model directory."""
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "mlx-whisper not installed. Apple Silicon only — "
                "`pip install mlx-whisper`."
            ) from e

        model_path = self._ensure_mlx_model()
        self._model_path = str(model_path)
        print(
            f"[MlxWhisper] Ready (model={self._model_path}, "
            f"lang={self._language or 'auto'})."
        )

    @staticmethod
    def _ensure_mlx_model() -> Path:
        """Convert HF model to MLX format on first load; cache on disk.

        The PyPI `mlx-whisper` package ships no conversion API, so we call into
        a vendored copy of `mlx-examples/whisper/convert.py` — same algorithm
        the upstream CLI uses. Output matches what `mlx_whisper.load_models`
        expects: `model.safetensors` + `config.json`.
        """
        project_root = Path(__file__).resolve().parents[2]
        model_dir = (project_root / config.whisper.mlx_model_dir).resolve()

        # `load_models.load_model` reads `weights.safetensors` (falls back to
        # `weights.npz`); the upstream CLI writes `model.safetensors` — a real
        # mismatch in mlx-whisper 0.4.3. We use the loader's expected name.
        if (model_dir / "weights.safetensors").exists() and (model_dir / "config.json").exists():
            return model_dir

        print(
            f"[MlxWhisper] Converting '{config.whisper.hf_model_id}' to MLX → {model_dir}"
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        import json
        from dataclasses import asdict

        import mlx.core as mx
        from mlx.utils import tree_flatten

        from src.transcription import _mlx_convert

        dtype = mx.float16  # Apple Silicon GPU path prefers fp16.
        model = _mlx_convert.convert(config.whisper.hf_model_id, dtype)
        cfg = asdict(model.dims)
        weights = dict(tree_flatten(model.parameters()))

        if bool(config.whisper.mlx_quantize):
            class _QArgs:
                q_group_size = 64
                q_bits = int(config.whisper.mlx_quantize_bits)
            print(f"[MlxWhisper] Quantizing ({_QArgs.q_bits}-bit)")
            weights, cfg = _mlx_convert.quantize(weights, cfg, _QArgs)

        mx.save_safetensors(str(model_dir / "weights.safetensors"), weights)
        cfg["model_type"] = "whisper"
        (model_dir / "config.json").write_text(json.dumps(cfg, indent=4))
        print("[MlxWhisper] Conversion complete.")
        return model_dir

    def transcribe(self, audio: np.ndarray) -> list[TranscriptionSegment]:
        if self._model_path is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not self.has_vocal_content(audio):
            return []

        audio = self._normalize(audio)

        import mlx_whisper

        kwargs: dict = {
            "path_or_hf_repo": self._model_path,
            "verbose": None,
        }
        if self._language:
            kwargs["language"] = self._language
        if config.whisper.single_temperature:
            kwargs["temperature"] = 0.0
        if config.whisper.allow_repetition:
            kwargs["compression_ratio_threshold"] = 10.0
        if config.whisper.independent_windows:
            kwargs["condition_on_previous_text"] = False
        # mlx-whisper respects word_timestamps/no_speech_threshold defaults;
        # decoder toggles that don't map are silently ignored (by design).

        try:
            result = mlx_whisper.transcribe(audio.astype(np.float32), **kwargs)
        except Exception as e:
            print(f"[MlxWhisper] Transcription error: {e}")
            return []

        detected = result.get("language")
        if detected and detected != "pa":
            print(f"[MlxWhisper] Detected language={detected}")

        out: list[TranscriptionSegment] = []
        for seg in result.get("segments", []):
            text = (seg.get("text") or "").strip()
            if text:
                out.append(
                    TranscriptionSegment(
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        text=text,
                    )
                )
        return out
