# Project: sttm-automate

## What this is

Automation layer for SikhiToTheMax (STTM) Desktop that listens to live kirtan audio, recognizes which shabad is being sung, and automatically controls STTM to display the correct shabad.

## Tech Stack

- Python 3.11+
- **IndicConformer-pa (default)** — fine-tuned CTC model via `onnxruntime`; fp32 / fp16 / int8 precisions, switchable at runtime from the dashboard
- **Whisper (fallback)** — MLX Whisper on Apple Silicon (Metal) or whisper.cpp (cross-platform); selectable per session via the Whisper/Indic radio in the dashboard
- KenLM Gurbani language models — paired with the IndicConformer engine and toggleable per-session from the dashboard:
  - `models/lm/gurbani_canon_bpe_4gram.bin` (2.94 MB) — BPE-level n-gram, fused **in-beam** via `pyctcdecode.build_ctcdecoder(..., alpha, beta)`. Improves recognition by anchoring beams to canon Gurbani.
  - `models/lm/gurbani_canon_char_6gram.bin` (4.81 MB) — character-level n-gram, run **post-hoc** as a hallucination gate (drop outputs where `per_char_PPL > 25`; canon scores ~3-4, garbage 200+).
  - `models/onnx/pa_tokenizer.model` — the SentencePiece model the BPE LM was trained against. Required for in-beam fusion to align labels with LM vocab.
  - See [docs/asr-engines.md](docs/asr-engines.md) for the full integration story.
- Local ShabadOS SQLite DB (`database.sqlite`) — auto-downloaded from HF on first run
- FastAPI + WebSocket (dashboard server) — loopback-only by default, no auth
- sounddevice (audio capture)
- Playwright (STTM browser automation fallback)
- httpx (STTM HTTP control only — localhost)

## No external APIs

All shabad search, verse lookup, and transliteration happens against the local
SQLite DB. **BaniDB's REST API is not used anywhere** (it was removed in favor
of the offline DB — ~100× faster and works without internet). If you find
yourself reaching for `api.banidb.com`, don't — extend `OfflineShabadSearcher`
or query `database.sqlite` directly instead.

The only HTTP calls this project makes are to STTM Desktop's own Express server
on `localhost` (for controlling display) and HuggingFace Hub on first startup
(to download the IndicConformer ONNX bundle, the Whisper checkpoints used by
the fallback engine, and the SQLite DB).

## Search scope

By default the search covers every source in the DB: SGGS, Sri Dasam Granth,
Vaaran Bhai Gurdas, Bhai Nand Lal's banis, Sarabloh Granth, Rehitname, Uggardanti.
A dashboard toggle ("SGGS only") restricts to `source_id = 1` when you want to
narrow it down. Non-SGGS rows (Dasam, Uggardanti) have `sttm_id = NULL` in the
upstream DB, so we synthesize IDs via `COALESCE(s.sttm_id, s.order_id + 100_000_000)`
to keep candidate dedup working.

## Architecture

Audio (sounddevice) → VAD (KirtanVAD or Silero) → Transcription (IndicConformer CTC by default; Whisper fallback) → LocalAgreement-2 streaming buffer + cross-window dedup → Transliteration (Gurmukhi→first-letter codes) → Local SQLite Search (9-strategy retrieval) → Confidence Scoring → STTM Control → Web Dashboard

## Running the app

Project-local virtualenv lives at `.venv/` (Python 3.11). Start the dashboard with:

```bash
source .venv/bin/activate
python -m src.main dashboard
```

For a detached background launch (survives shell exit):

```bash
nohup .venv/bin/python -m src.main dashboard > /tmp/sttm-automate.log 2>&1 &
disown
```

Open `http://localhost:8080` once the process is running. Logs go to
`/tmp/sttm-automate.log`.

The dashboard binds to **loopback only by default**. To expose it on the LAN
(e.g. for a phone running on the same Wi-Fi), set `STTM_LAN_MODE=1` before
starting, or flip `config.dashboard.lan_mode = True` in `src/config.py`. There
is no auth on the dashboard — when running in LAN mode, anyone on the same
network can drive STTM, so only enable it on networks you trust.

If `.venv/` is missing, recreate it:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Audio input sources

The dashboard's audio-device picker offers two groups:

- **Microphones** — physical input devices (laptop mic, USB mic, audio interface).
- **System Audio (loopback)** — virtual inputs that capture whatever is playing
  through the Mac (e.g. a YouTube gurdwara stream, Spotify, an STTM session
  audio playback). Requires a loopback driver: `brew install blackhole-2ch`,
  then a Multi-Output Device routing speakers + BlackHole. The interactive
  setup helper at `scripts/setup_audio.py` walks through this and verifies
  capture works. Once routed, BlackHole appears in the picker under the
  "System Audio" group — selecting it streams system audio through the same
  VAD → ASR → search pipeline as a real mic.

Conferencing virtual devices (Teams, Zoom, aggregate/multi-output sources) are
always hidden — they would capture nothing useful for kirtan recognition.

## Model artifacts (gitignored — regenerate locally)

Large binaries under `data/`, `index/`, `~/models/exports-pa/`, and `models/lm/`
are excluded from git. None ship in the repo; recreate them on a fresh
checkout as needed:

| Path | What it is | How to (re)build |
| --- | --- | --- |
| `~/models/exports-pa/{fp32,fp16,int8}/indicconformer-pa-ctc.onnx` + `tokens.txt` | The IndicConformer CTC ONNX bundle in three precisions, sharing one tokens vocab at the root. Path is `config.whisper.onnx_model_dir`. | Auto-downloaded on first run by `IndicConformerEngine._ensure_assets` from the HF repo at `config.whisper.hf_model_id` (`onnx-pa-only/{precision}/indicconformer-pa-ctc.onnx`). |
| `data/_onnx_cache/` | HF download cache for the IndicConformer ONNX assets. | Repopulated on demand. Safe to delete. |
| `data/indicconformer-pa-v3-kirtan.nemo`, `data/_nemo_cache/` | Raw NeMo checkpoint + cache, only needed if re-exporting the ONNX bundle from source. | NeMo's `model.export()` from the fine-tuned checkpoint (see [docs/fine-tuning-gurbani-whisper.md](docs/fine-tuning-gurbani-whisper.md) for the Whisper-side fine-tune; IndicConformer export pipeline lives in `Gurbani_ASR_v4`). |
| `models/lm/gurbani_canon_bpe_4gram.bin` (2.94 MB) | KenLM BPE 4-gram trained on the Gurbani canon with the same SentencePiece tokenizer the IndicConformer model uses. Fed into `pyctcdecode` for **in-beam shallow fusion** during CTC beam search. | Reproducible build script at `Gurbani_ASR_v4/scripts/build_kenlm.py`; we copy the prebuilt `.bin` here, or fetch from HF. |
| `models/lm/gurbani_canon_char_6gram.bin` (4.81 MB) | KenLM character 6-gram trained on the Gurbani canon. Used **post-hoc** as a hallucination gate — drops outputs with `per_char_PPL > 25` (calibrated: canon ~3-4, modern Punjabi ~6-8, garbage 200+). | Same `build_kenlm.py`, char-tokenized input. Copied / HF-fetched. |
| `models/onnx/pa_tokenizer.model` | SentencePiece model the IndicConformer was trained with; needed to keep BPE LM labels and AM labels aligned during pyctcdecode fusion. | Extracted from the source `.nemo` checkpoint; ships with the IndicConformer HF repo. |
| `data/surt-small-turbo-baseline-v0-{ct2,mlx}/`, `data/surt-small-turbo-baseline-v0{,-q8_0}.ggml`, `data/_baseline-v0-source/` | All four engine formats of the `surindersinghssj/surt-small-turbo-baseline-v0` Whisper fine-tune (used by the Whisper fallback). | `.venv/bin/python scripts/preconvert_baseline_v0.py` (idempotent — skips formats that already exist). |
| `data/surt-small-v3.ggml` (f16 reference) | f16 ggml of `surt-small-v3` for whisper.cpp. | Convert from the HF model with whisper.cpp's `models/convert-h5-to-ggml.py`. |
| `data/surt-small-v3-q{4_0,5_1,8_0}.ggml` | Quantized Whisper variants used by `scripts/bench_engines.py`. | Build whisper.cpp's `quantize` binary, then `quantize data/surt-small-v3.ggml data/surt-small-v3-q8_0.ggml q8_0` (and `q5_1`, `q4_0`). |
| `data/surt-small-v3-{ct2,mlx}/` | faster-whisper (CTranslate2) and mlx-whisper conversions of `surt-small-v3`. | `ct2-transformers-converter --model surindersinghssj/surt-small-v3 --output_dir data/surt-small-v3-ct2 --quantization int8` for ct2; `python -m mlx_whisper.convert --hf-repo surindersinghssj/surt-small-v3 --mlx-path data/surt-small-v3-mlx -q` for mlx. |
| `index/*.faiss`, `index/*.pkl` | FAISS indexes for prototype semantic matching (`tests/eval/proto_*.py`). | Built from `database.sqlite` by the proto experiments — not on the production path. |
| `tests/eval/cache/`, `tests/eval/runs/` | Cached YouTube audio + scorer outputs from past eval runs. | Repopulated automatically by `tests/eval/runner.py`. |

`database.sqlite`, `data/gurbani.sqlite`, and `data/realm_verses.json` are
auto-downloaded from HF on first run by the app itself (already noted above).

## Conventions

- Use `async/await` throughout the pipeline
- All config in `src/config.py` (no magic numbers in other files)
- Type hints on all function signatures
- Keep modules focused — one responsibility per file

## Git commits

- Use conventional commit style: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Please commit after implementing feat, fix, refactor, test or doc
