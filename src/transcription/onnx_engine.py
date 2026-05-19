"""IndicConformer (ONNX) transcription engine.

Loads a CTC-only ONNX export of our fine-tuned IndicConformer-pa with
``onnxruntime``, computes log-mel features locally with torchaudio (matching
NeMo's ``AudioToMelSpectrogramPreprocessor`` defaults), and greedy-decodes the
CTC log-probs against ``tokens.txt``. One ONNX bundle per precision lives at
``~/models/exports-pa/{fp32,fp16,int8}/indicconformer-pa-ctc.onnx`` with a
shared ``tokens.txt`` at the root; switching precision triggers a reload so
the dashboard can flip between size/quality tradeoffs at runtime.

Why not sherpa-onnx: ``OfflineRecognizer.from_nemo_ctc`` requires
sherpa-specific metadata fields (``vocab_size``, ``model_type``, etc.) that
NeMo's vanilla ``model.export()`` doesn't write into the ONNX. Hand-rolling
the front-end + CTC decode is small, dependency-light, and gives us identical
outputs without re-exporting the model.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.config import config
from src.transcription import lm_scorer
from src.transcription.base import BaseTranscriptionEngine, TranscriptionSegment


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOKENS_FILENAME = "tokens.txt"
_MODEL_FILENAME = "indicconformer-pa-ctc.onnx"

# NeMo AudioToMelSpectrogramPreprocessor defaults for IndicConformer.
_SAMPLE_RATE = 16000
_N_FFT = 512
_WIN_LENGTH = 400      # 25 ms
_HOP_LENGTH = 160      # 10 ms
_N_MELS = 80
_LOG_EPS = 2 ** -24    # NeMo's log_zero_guard_value default


class IndicConformerEngine(BaseTranscriptionEngine):
    """Transcribes audio using the IndicConformer CTC ONNX export."""

    def __init__(self) -> None:
        self._sess = None
        self._mel = None
        self._tokens: list[str] = []
        self._blank_id: int = -1
        self._loaded_precision: str | None = None
        # ORT sessions are thread-safe for run() but reload swaps the session
        # object; the lock guards reload vs. transcribe.
        self._lock = threading.Lock()
        # KenLM-fused beam decoder. None when the LM toggle is off or any of
        # the required artifacts / deps are missing — caller still gets a
        # transcript via the existing greedy path.
        self._decoder: Any = None
        # Snapshot of the LM settings that produced ``self._decoder``. When
        # the user flips the toggle from the dashboard, we compare against
        # this and rebuild lazily on the next ``transcribe()``.
        self._lm_snapshot: tuple[bool, str, str, float, float, int] | None = None

    def load(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime not installed. `pip install onnxruntime`."
            ) from e
        try:
            import torchaudio  # noqa: F401  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "torchaudio not installed. `pip install torchaudio`."
            ) from e

        precision = self._resolved_precision()
        model_path, tokens_path = self._ensure_assets(precision)
        threads = max(1, int(getattr(config.whisper, "onnx_threads", 4)))
        print(
            f"[IndicConformer-ONNX] Loading {model_path.parent.name}/{model_path.name} "
            f"(precision={precision}, threads={threads})..."
        )
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self._tokens = self._load_tokens(tokens_path)
        # tokens.txt convention: <blank> sits at the last line, index = vocab-1.
        # Look it up explicitly so an out-of-order tokens file still works.
        self._blank_id = next(
            (i for i, t in enumerate(self._tokens) if t == "<blank>"),
            len(self._tokens) - 1,
        )
        if self._mel is None:
            self._mel = self._build_mel_extractor()
        self._sess = sess
        self._loaded_precision = precision
        print(
            f"[IndicConformer-ONNX] Ready. vocab={len(self._tokens)} "
            f"blank_id={self._blank_id}"
        )
        # Build the KenLM-fused beam decoder if the toggle is on. Failures here
        # are non-fatal — we just log and fall back to greedy CTC.
        self._refresh_lm_decoder()

    @staticmethod
    def _build_mel_extractor():
        import torchaudio.transforms as T  # type: ignore
        # Match NeMo: hann window, slaney mel scale + slaney norm, power=2.
        return T.MelSpectrogram(
            sample_rate=_SAMPLE_RATE,
            n_fft=_N_FFT,
            win_length=_WIN_LENGTH,
            hop_length=_HOP_LENGTH,
            n_mels=_N_MELS,
            window_fn=__import__("torch").hann_window,
            power=2.0,
            mel_scale="slaney",
            norm="slaney",
            center=True,
        )

    @staticmethod
    def _load_tokens(path: Path) -> list[str]:
        tokens: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            # Lines look like "▁ਸ 1" — split on the LAST whitespace so tokens
            # containing spaces survive intact.
            parts = line.rsplit(maxsplit=1)
            tokens.append(parts[0] if len(parts) == 2 else line)
        return tokens

    @staticmethod
    def _resolved_precision() -> str:
        precision = (getattr(config.whisper, "onnx_precision", "fp16") or "fp16").lower()
        if precision not in ("fp32", "fp16", "int8"):
            print(f"[IndicConformer-ONNX] Unknown precision {precision!r} — using int8.")
            return "int8"
        return precision

    @classmethod
    def _ensure_assets(cls, precision: str) -> tuple[Path, Path]:
        """Resolve absolute paths to the ONNX file and tokens.txt.

        Falls back to downloading from HuggingFace on first use if the local
        ``~/models/exports-pa`` tree isn't populated.
        """
        raw_dir = getattr(config.whisper, "onnx_model_dir", None)
        if not raw_dir:
            raise RuntimeError(
                "IndicConformer ONNX engine requires `config.whisper.onnx_model_dir`."
            )
        base = Path(raw_dir).expanduser()
        if not base.is_absolute():
            base = (_PROJECT_ROOT / base).resolve()
        model_path = base / precision / _MODEL_FILENAME
        tokens_path = base / _TOKENS_FILENAME
        if model_path.exists() and tokens_path.exists():
            return model_path, tokens_path

        from huggingface_hub import hf_hub_download

        repo_id = config.whisper.hf_model_id
        cache_dir = str(_PROJECT_ROOT / "data" / "_onnx_cache")
        if not model_path.exists():
            print(f"[IndicConformer-ONNX] Downloading {repo_id}/onnx-pa-only/{precision}/{_MODEL_FILENAME}")
            cached_model = hf_hub_download(
                repo_id=repo_id,
                filename=f"onnx-pa-only/{precision}/{_MODEL_FILENAME}",
                cache_dir=cache_dir,
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            cls._link_or_copy(Path(cached_model), model_path)
        if not tokens_path.exists():
            print(f"[IndicConformer-ONNX] Downloading {repo_id}/onnx-pa-only/{_TOKENS_FILENAME}")
            cached_tokens = hf_hub_download(
                repo_id=repo_id,
                filename=f"onnx-pa-only/{_TOKENS_FILENAME}",
                cache_dir=cache_dir,
            )
            tokens_path.parent.mkdir(parents=True, exist_ok=True)
            cls._link_or_copy(Path(cached_tokens), tokens_path)
        return model_path, tokens_path

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        try:
            dst.symlink_to(src)
        except (OSError, FileExistsError):
            import shutil
            shutil.copy2(src, dst)

    def reload_if_precision_changed(self) -> bool:
        if self._sess is None:
            return False
        target = self._resolved_precision()
        if target == self._loaded_precision:
            return False
        with self._lock:
            self.load()
        return True

    # ── KenLM language-model fusion (in-beam BPE + post-hoc char gate) ──

    @staticmethod
    def _current_lm_snapshot() -> tuple[bool, str, str, float, float, int]:
        """Snapshot of the LM settings used to rebuild the pyctcdecode decoder."""
        return (
            bool(getattr(config.whisper, "lm_enabled", False)),
            str(getattr(config.whisper, "lm_bpe_path", "")),
            str(getattr(config.whisper, "lm_char_path", "")),
            float(getattr(config.whisper, "lm_alpha", 0.5)),
            float(getattr(config.whisper, "lm_beta", 1.5)),
            int(getattr(config.whisper, "lm_beam_width", 100)),
        )

    def _refresh_lm_decoder(self) -> None:
        """Build or tear down the pyctcdecode beam decoder to match config.

        Idempotent. Called from ``load()`` and from ``reload_if_lm_changed()``.
        Any failure (missing dep, missing .bin) leaves ``self._decoder`` at
        ``None`` — callers fall back to greedy decode.
        """
        snapshot = self._current_lm_snapshot()
        if snapshot == self._lm_snapshot and self._decoder is not None:
            return
        self._lm_snapshot = snapshot

        enabled, bpe_path_str, char_path_str, alpha, beta, _beam = snapshot
        # Always refresh the char LM scorer so its singleton tracks the
        # configured path even when in-beam fusion is off.
        char_path = Path(char_path_str).expanduser() if char_path_str else None
        if char_path is not None and not char_path.is_absolute():
            char_path = (_PROJECT_ROOT / char_path).resolve()
        lm_scorer.configure(char_path)

        if not enabled:
            if self._decoder is not None:
                print("[IndicConformer-ONNX] LM disabled — reverting to greedy decode.")
            self._decoder = None
            return
        if not self._tokens:
            return  # model not loaded yet; ``load()`` will call us again

        bpe_path = Path(bpe_path_str).expanduser() if bpe_path_str else None
        if bpe_path is not None and not bpe_path.is_absolute():
            bpe_path = (_PROJECT_ROOT / bpe_path).resolve()
        if bpe_path is None or not bpe_path.exists():
            print(
                f"[IndicConformer-ONNX] LM enabled but BPE LM not found at "
                f"{bpe_path} — using greedy decode."
            )
            self._decoder = None
            return
        try:
            from pyctcdecode import build_ctcdecoder  # type: ignore  # noqa: PLC0415
        except ImportError:
            print(
                "[IndicConformer-ONNX] LM enabled but `pyctcdecode` is not "
                "installed (pip install pyctcdecode kenlm) — using greedy."
            )
            self._decoder = None
            return

        # pyctcdecode adds its own blank internally; strip <blank> from the
        # labels we hand it so the implicit-blank slot lines up with vocab-1.
        labels = [t for t in self._tokens if t != "<blank>"]
        try:
            self._decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=str(bpe_path),
                alpha=alpha,
                beta=beta,
            )
            print(
                f"[IndicConformer-ONNX] LM on: BPE fusion via {bpe_path.name} "
                f"(α={alpha}, β={beta}); char gate via "
                f"{Path(char_path).name if char_path else '—'}."
            )
        except Exception as e:  # noqa: BLE001
            print(f"[IndicConformer-ONNX] Failed to build LM decoder: {e}")
            self._decoder = None

    def reload_if_lm_changed(self) -> bool:
        """Rebuild the decoder if the LM toggle / paths / α / β changed."""
        if self._sess is None:
            return False
        if self._current_lm_snapshot() == self._lm_snapshot:
            return False
        with self._lock:
            self._refresh_lm_decoder()
        return True

    # IndicConformer was fine-tuned on clips up to ~15 s. Inputs much longer
    # than that silently degrade; cap and split on the rough midpoint silence.
    _MAX_AUDIO_SECONDS = 12.0

    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: str | None = None,
    ) -> list[TranscriptionSegment]:
        if self._sess is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        del initial_prompt  # IndicConformer doesn't support prompt anchoring.

        if not self.has_vocal_content(audio):
            return []

        audio = self._normalize(np.asarray(audio, dtype=np.float32))
        if audio.size == 0:
            return []

        self.reload_if_precision_changed()
        self.reload_if_lm_changed()

        max_samples = int(self._MAX_AUDIO_SECONDS * _SAMPLE_RATE)
        if audio.size > max_samples:
            print(
                f"[IndicConformer-ONNX] Audio {audio.size / _SAMPLE_RATE:.1f}s exceeds "
                f"{self._MAX_AUDIO_SECONDS:.1f}s cap — splitting into chunks."
            )
            return self._transcribe_chunked(audio, max_samples)

        text = self._decode_one(audio)
        if not text:
            return []
        duration = float(audio.size) / _SAMPLE_RATE
        return [TranscriptionSegment(start=0.0, end=duration, text=text)]

    def _transcribe_chunked(
        self, audio: np.ndarray, max_samples: int
    ) -> list[TranscriptionSegment]:
        segments: list[TranscriptionSegment] = []
        offset_s = 0.0
        for start in range(0, audio.size, max_samples):
            chunk = audio[start : start + max_samples]
            if chunk.size == 0:
                continue
            text = self._decode_one(chunk)
            chunk_dur = float(chunk.size) / _SAMPLE_RATE
            if text:
                segments.append(
                    TranscriptionSegment(
                        start=offset_s, end=offset_s + chunk_dur, text=text
                    )
                )
            offset_s += chunk_dur
        return segments

    def _decode_one(self, audio: np.ndarray) -> str:
        feat, length = self._features(audio)
        with self._lock:
            outputs = self._sess.run(
                ["logprobs"],
                {"audio_signal": feat, "length": length},
            )
        log_probs = outputs[0][0]  # [T, V]
        if self._decoder is not None:
            text = self._lm_beam_decode(log_probs)
        else:
            text = self._ctc_decode(log_probs)
        return self._apply_hallucination_gate(text)

    def _lm_beam_decode(self, log_probs: np.ndarray) -> str:
        """In-beam BPE-LM fusion via pyctcdecode. Falls back to greedy on error."""
        beam_width = int(getattr(config.whisper, "lm_beam_width", 100))
        try:
            text = self._decoder.decode(log_probs, beam_width=beam_width)
        except Exception as e:  # noqa: BLE001
            print(f"[IndicConformer-ONNX] LM beam decode failed ({e}) — using greedy.")
            return self._ctc_decode(log_probs)
        return text.strip()

    def _apply_hallucination_gate(self, text: str) -> str:
        """Post-hoc char-LM PPL gate. Drops outputs that look like hallucinations.

        Skipped entirely when the LM toggle is off or the char .bin can't be
        loaded — fail-open is intentional so the engine stays useful in
        environments without kenlm.
        """
        if not text:
            return text
        if not bool(getattr(config.whisper, "lm_enabled", False)):
            return text
        min_chars = int(getattr(config.whisper, "lm_gate_min_chars", 6))
        if len(text.strip()) < min_chars:
            return text
        hall = float(getattr(config.whisper, "lm_hallucination_ppl_threshold", 25.0))
        low = float(getattr(config.whisper, "lm_low_confidence_ppl_threshold", 12.0))
        s = lm_scorer.score(
            text,
            hallucination_threshold=hall,
            low_confidence_threshold=low,
        )
        if s is None:
            return text
        if s.is_hallucination:
            print(
                f"[IndicConformer-ONNX] LM gate dropped output "
                f"(per_char_PPL={s.per_char_ppl:.1f} > {hall:.0f}): {text!r}"
            )
            return ""
        return text

    def _features(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (audio_signal[1,80,T], length[1]) — NeMo per-feature CMVN."""
        import torch  # type: ignore
        wav = torch.from_numpy(audio.astype(np.float32))
        mel = self._mel(wav)                     # [80, T]
        log_mel = torch.log(mel + _LOG_EPS)
        # Per-feature normalization (NeMo "normalize: per_feature"): subtract
        # mean and divide by std across time, per mel band, per utterance.
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True).clamp_min(1e-5)
        feat = (log_mel - mean) / std
        feat_np = feat.unsqueeze(0).numpy().astype(np.float32)  # [1, 80, T]
        length = np.array([feat_np.shape[-1]], dtype=np.int64)
        return feat_np, length

    def _ctc_decode(self, log_probs: np.ndarray) -> str:
        """Greedy CTC decode → tokens → text. SentencePiece ▁ marks word starts."""
        ids = log_probs.argmax(axis=-1)  # [T]
        prev = -1
        decoded: list[int] = []
        for i in ids.tolist():
            if i != prev and i != self._blank_id:
                decoded.append(i)
            prev = i
        if not decoded:
            return ""
        pieces = [self._tokens[i] for i in decoded if 0 <= i < len(self._tokens)]
        # SentencePiece: ▁ (U+2581) is the word-boundary marker.
        return "".join(pieces).replace("\u2581", " ").strip()
