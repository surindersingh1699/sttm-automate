# STTM Automate

Listens to live kirtan, recognizes the shabad being sung, and drives
[SikhiToTheMax (STTM) Desktop](https://www.sikhitothemax.org) to display the
right lines in sync — no cloud APIs on the hot path.

## How it works

```text
Audio (sounddevice)
  → VAD (KirtanVAD or Silero)
  → ASR  (IndicConformer-pa CTC ONNX — default; Whisper fallback)
  → LocalAgreement-2 streaming buffer + cross-window dedup
  → Gurmukhi → first-letter codes
  → Offline SQLite search (9-strategy retrieval)
  → Confidence scoring + line tracking
  → STTM control (HTTP on localhost)
  → Web dashboard (FastAPI + WebSocket)
```

The search corpus covers every source in the local ShabadOS DB: SGGS, Sri Dasam
Granth, Vaaran Bhai Gurdas, Bhai Nand Lal, Sarabloh Granth, Rehitname,
Uggardanti. A dashboard toggle restricts to SGGS-only when wanted.

## Prerequisites

- Python 3.11+
- [STTM Desktop](https://www.sikhitothemax.org) running locally
- macOS: [BlackHole](https://existential.audio/blackhole/) recommended for
  capturing the kirtan stream (loopback audio routing). On Linux you can use
  PulseAudio monitor sources.
- Xcode Command Line Tools on macOS (needed by `kenlm` at install time)

## Install

```bash
git clone https://github.com/surindersingh1699/sttm-automate.git
cd sttm-automate
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run will auto-download from HuggingFace:

- IndicConformer-pa ONNX bundle (fp32 / fp16 / int8) + tokenizer
- The Whisper checkpoints used by the fallback engine
- ShabadOS SQLite DB (`database.sqlite`)
- KenLM Gurbani LM pair (`models/lm/*.bin`)

No model files ship in the repo — see "Model artifacts" in
[CLAUDE.md](CLAUDE.md) for the full list and how to rebuild any of them.

## Run

```bash
source .venv/bin/activate
python -m src.main dashboard
```

At startup the process prints a one-time login URL:

```text
http://127.0.0.1:8080/auth?token=…
```

Open it once to set the auth cookie, then use `http://localhost:8080` normally.

Detached background launch (survives shell exit):

```bash
nohup .venv/bin/python -m src.main dashboard > /tmp/sttm-automate.log 2>&1 &
disown
```

Logs go to `/tmp/sttm-automate.log`.

## Dashboard authentication

The dashboard and its WebSocket are gated by a per-install secret token,
generated on first run and persisted to `.controller_token` at the project
root (or `$STTM_TOKEN_PATH` if set). The token is accepted via:

- `?token=…` query param (the `/auth?token=…` URL printed at startup)
- `X-STTM-Token` header (for programmatic clients)
- `sttm_token` cookie (set after the first `/auth?token=…` hit)

Without a valid token, HTTP returns 401 and WebSocket handshake is refused
with 403. **This is not a TLS replacement** — it just prevents a curious
LAN-mate from hijacking the dashboard or flipping the engine mid-kirtan.

`.controller_token` is gitignored — never commit it.

## Loopback vs. LAN

The dashboard binds to **loopback only by default**. To expose it on the LAN
(for a phone running on the same Wi-Fi):

```bash
STTM_LAN_MODE=1 python -m src.main dashboard
```

…or flip `config.dashboard.lan_mode = True` in `src/config.py`. Pair LAN mode
with the token gate; never expose this directly to the public internet.

## ASR engines

The active engine is selectable from the dashboard at runtime.

| Engine | Use case | Notes |
| --- | --- | --- |
| **IndicConformer-pa (default)** | Production kirtan | Fine-tuned CTC; ONNX in fp32 / fp16 / int8; toggle KenLM fusion |
| MLX Whisper | Apple Silicon fallback | Metal-accelerated |
| whisper.cpp | Cross-platform fallback | Quantized ggml |
| faster-whisper (CT2) | CPU fallback | Slowest on Apple Silicon |

KenLM Gurbani LM pair (optional, dashboard toggle):

- `gurbani_canon_bpe_4gram.bin` — BPE 4-gram fused **in-beam** via
  `pyctcdecode` during CTC beam search.
- `gurbani_canon_char_6gram.bin` — char 6-gram run **post-hoc** as a
  hallucination gate (drops outputs with `per_char_PPL > 25`).

Full integration story: [docs/asr-engines.md](docs/asr-engines.md).

## Retrieval

Local SQLite search with 9 strategies (first-letter exact / prefix / contains,
rotations, semantic / phonetic, multi-line, alaap detour, locked-state pair
alignment). Documented in [docs/retrieval_system.md](docs/retrieval_system.md).

The previously-used BaniDB REST API is **not** called anywhere — the offline
DB is ~100× faster and works without internet.

## Eval

Two modes in `tests/eval/`:

- **Headless** — yt-dlp pulls audio from a HF kirtan dataset, runs end-to-end,
  scores against ground truth. Used in CI.
- **Mic** — plays cached opus through your speakers into real STTM to validate
  the full stack including line tracking.

## Troubleshooting

If audio capture is not working:

```bash
python scripts/setup_audio.py
```

If `.venv/` is missing, recreate it:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Project layout

- `src/main.py` — entry point (`dashboard` subcommand)
- `src/config.py` — all tunables (no magic numbers elsewhere)
- `src/transcription/` — ASR engines + transliteration
- `src/matcher/` — offline SQLite search, scoring, line tracking
- `src/controller/` — STTM control (HTTP + Playwright fallback)
- `src/dashboard/` — FastAPI server + static UI
- `scripts/` — calibration, benchmarks, conversion utilities
- `tests/eval/` — integrated behaviour eval

See [CLAUDE.md](CLAUDE.md) for deeper architecture notes and contributor
conventions.
