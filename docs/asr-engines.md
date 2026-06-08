# ASR engines in `sttm-automate`

The dashboard exposes a Whisper / Indic radio toggle, plus a precision dropdown
that applies to whichever engine family is selected. This doc explains what
those toggles map to under the hood, and how the optional KenLM language model
pair fits in.

---

## Engines at a glance

| Family | Engine name (factory) | Default? | Architecture | Decoder | Backend |
| --- | --- | --- | --- | --- | --- |
| IndicConformer | `indicconformer` | **yes** | CTC encoder | Greedy CTC by default; **pyctcdecode beam + KenLM** when LM toggle is on | `onnxruntime` (CPU) |
| Whisper | `mlx-whisper` | no | Encoder-decoder (transformer) | Beam search built into the decoder | `mlx-whisper` (Apple Silicon Metal) |
| Whisper | `whisper-cpp` | no | Encoder-decoder (transformer) | Beam search built into the decoder | `whisper.cpp` (cross-platform) |
| Whisper | `faster-whisper` | no | Encoder-decoder (transformer) | Beam search built into the decoder | `CTranslate2` (CPU only on Mac) |

Engine selection happens at two levels:

1. **Engine family**, driven by the active HF model id and the
   `WhisperConfig.model_families` registry — set via
   `config.whisper.apply_model_id(model_id)` from the dashboard's model
   dropdown.
2. **Engine implementation within a family** — for Whisper, picked by the
   `engine` field in `.runtime_settings.json` (`mlx-whisper` / `whisper-cpp` /
   `faster-whisper`). The IndicConformer family has one implementation today.

Code lives at:

- [src/transcription/factory.py](../src/transcription/factory.py) — registers
  engines, enforces "this engine requires this model family" coherence, and
  pins streaming-mode defaults that fit each family.
- [src/transcription/onnx_engine.py](../src/transcription/onnx_engine.py) —
  `IndicConformerEngine`: log-mel features (torchaudio matches NeMo's
  `AudioToMelSpectrogramPreprocessor`), ONNX session, CTC decode.
- [src/transcription/mlx_whisper_engine.py](../src/transcription/mlx_whisper_engine.py),
  [src/transcription/whisper_cpp_engine.py](../src/transcription/whisper_cpp_engine.py) —
  Whisper variants.

---

## IndicConformer (default)

Fine-tuned IndicConformer-pa CTC model, served as an ONNX export. Three
precisions live side-by-side under `~/models/exports-pa/{fp32,fp16,int8}/` and
are downloaded lazily from the HF repo at `config.whisper.hf_model_id` on
first use:

| Precision | File size | Notes |
| --- | --- | --- |
| `fp32` | ~470 MB | Accuracy reference. |
| `fp16` | — | Currently **broken**: NeMo→ONNX export left mixed float/half tensors at `/pre_encode/Add` and `onnxruntime` refuses to load it. Re-export with the cast fixed before re-enabling. |
| `int8` | ~134 MB | Default. Fastest and smallest with no measurable accuracy loss on kirtan eval. |

Switching precision from the dashboard triggers an in-place reload — the
ONNX session is swapped under a lock without tearing down the audio pipeline.

### Streaming

IndicConformer is decoded once per VAD-bounded utterance, not on a rolling
Whisper-style window. The factory pins these streaming defaults whenever
IndicConformer is the active engine:

- `streaming_mode = "nemo_chunked"`
- Chunk length: `config.whisper.nemo_chunk_len_s` (1.5 s default).
  Smaller = lower latency + more boundary errors; the matcher's first-letter
  retrieval is robust to single-word edge losses, so 1.0–1.5 s works for
  fast bani.
- Optional left-context audio: `nemo_chunk_context_s` (0.5 s default) — fed
  to the model as warm-up frames but its text output is discarded. Stabilises
  the encoder's first frames.

The engine ignores `initial_prompt` — there's no transformer decoder to
condition on it. Whisper-only knobs are dead weight under IndicConformer.

---

## Whisper (fallback)

Selected when the active model id maps to a Whisper checkpoint (any of the
`surt-small-*` repos). Within the Whisper family the runtime picks one of
three implementations:

- **MLX Whisper** — Apple Silicon Metal kernel. Fastest on the user's M-series Mac. Default on macOS arm64.
- **whisper.cpp** — cross-platform, GGML quantizations (q4_0, q5_1, q8_0, f16) under `data/`. Used on non-Mac or for benchmarks.
- **faster-whisper** — CTranslate2 backend, CPU-only on Mac (no Metal). Kept around for the `ct2-transformers-converter` integration but slower than the alternatives on Apple Silicon.

Whisper engines have no LM fusion hook — the decoder has its own implicit
LM via cross-attention. The KenLM toggle described below is **disabled** in
the UI when a Whisper engine is selected.

---

## KenLM LM pair (IndicConformer only)

Optional. Off by default. Enable from the dashboard via the **"Use Gurbani
LM"** checkbox that appears when an IndicConformer model is active.

### What it does

Two complementary KenLM models, each addressing a different failure mode:

```
audio → ONNX encoder → logprobs [T, V]
                          │
                          ▼
              ┌── pyctcdecode beam search ──────────────────┐
              │  (BPE 4-gram KenLM, α=0.5, β=1.5, beam=100) │   ← in-beam fusion
              └─────────────────────────────────────────────┘
                          │
                          ▼
                  best Gurmukhi hypothesis
                          │
                          ▼
              ┌── lm_scorer.score(text) ────────┐
              │  (char 6-gram KenLM, per-char PPL) │             ← post-hoc gate
              └────────────────────────────────────┘
                          │
              per_char_ppl > 25   →  drop (hallucination)
              per_char_ppl > 12   →  flag low-confidence
              otherwise           →  pass to matcher
```

### Files

| Path | Size | Purpose |
| --- | --- | --- |
| `models/lm/gurbani_canon_bpe_4gram.bin` | 2.94 MB | In-beam fusion LM. Built with the SentencePiece tokenizer the IndicConformer uses. |
| `models/lm/gurbani_canon_char_6gram.bin` | 4.81 MB | Hallucination gate LM. Character-level, `\|`-separated words. |
| `models/onnx/pa_tokenizer.model` | small | SentencePiece model required to keep BPE LM labels aligned with the AM. |

All three are gitignored. The build script lives in the sister project
[`Gurbani_ASR_v4/scripts/build_kenlm.py`](https://github.com/surindersingh1699/Gurbani_ASR_v4)
— see it for retraining if the canon corpus or tokenizer changes.

### Calibration (char 6-gram, per-char PPL)

| Input | Per-char PPL |
| --- | --- |
| Canon Gurbani (Mool Mantar, kirtan refrains) | 3–4 |
| Modern Punjabi (legal Gurmukhi, not Gurbani) | 6–8 |
| Random Gurmukhi garbage | 200+ |
| **Hallucination cutoff** | **25** (~8× below garbage, ~10× headroom over canon) |
| **Low-confidence flag** | **12** |

Thresholds tunable via `config.whisper.lm_hallucination_ppl_threshold` and
`config.whisper.lm_low_confidence_ppl_threshold`.

### Tuning knobs

| Config field | Default | Effect |
| --- | --- | --- |
| `lm_enabled` | `False` | Master switch — when off, IndicConformer uses plain greedy CTC decode. |
| `lm_alpha` | `0.5` | LM weight in the beam score (`AM_logp + α·LM_logp + β·words`). Sweep 0.3–0.8 on the eval set. |
| `lm_beta` | `1.5` | Word-insertion bonus. Higher = bias toward more words per second. |
| `lm_beam_width` | `100` | pyctcdecode beam width. Cost is roughly linear; 50 is fine for low-resource CPU. |

### Required Python deps

Added to `requirements.txt` when this feature lands:

- `kenlm` — Python bindings for the KenLM C++ library. Needs a C++ compiler at install time on macOS (Xcode CLT). Pip wheels usually available for Linux.
- `pyctcdecode` — CTC beam search + scorer plug-in interface.
- `sentencepiece` — already needed by IndicConformer for tokenizer inspection.

### Failure modes worth knowing

1. **pyctcdecode strips `<blank>` from labels** — it adds its own internally. Our `tokens.txt` has `<blank>` as the last entry; the decoder must skip it when building the label list. Getting this wrong = silently bad transcripts.
2. **BPE LM and AM tokenizer must match.** If the IndicConformer model is ever re-trained with a different SentencePiece tokenizer, the BPE LM must be rebuilt — they aren't interchangeable.
3. **Per-char PPL is length-dependent at very short inputs.** Outputs under ~6 chars can spike past the threshold spuriously. The scorer skips the gate when `n_chars == 0` and we treat anything under a configurable `min_chars_for_gate` as ungated to avoid false drops on single-syllable callouts.

---

## Picking a configuration

| Scenario | Recommended setting |
| --- | --- |
| Production projection on Apple Silicon, want best accuracy | IndicConformer int8 + LM on |
| Production projection, fastest possible | IndicConformer int8 + LM off |
| Diagnosing whether bad output is engine or matcher | Switch to Whisper to compare |
| Eval / benchmarking | `scripts/bench_engines.py` for cross-engine RTF and first-letter overlap; `scripts/compare_streaming_modes.py` for the streaming side. |
