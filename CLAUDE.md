# Project: sttm-automate

## What this is

Automation layer for SikhiToTheMax (STTM) Desktop that listens to live kirtan audio, recognizes which shabad is being sung, and automatically controls STTM to display the correct shabad.

## Tech Stack

- Python 3.11+
- faster-whisper (local Punjabi speech recognition)
- Local ShabadOS SQLite DB (`database.sqlite`) — auto-downloaded from HF on first run
- FastAPI + WebSocket (dashboard server)
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
(to download the Whisper model and the SQLite DB).

## Search scope

By default the search covers every source in the DB: SGGS, Sri Dasam Granth,
Vaaran Bhai Gurdas, Bhai Nand Lal's banis, Sarabloh Granth, Rehitname, Uggardanti.
A dashboard toggle ("SGGS only") restricts to `source_id = 1` when you want to
narrow it down. Non-SGGS rows (Dasam, Uggardanti) have `sttm_id = NULL` in the
upstream DB, so we synthesize IDs via `COALESCE(s.sttm_id, s.order_id + 100_000_000)`
to keep candidate dedup working.

## Architecture

Audio (sounddevice) → Transcription (faster-whisper) → Transliteration (Gurmukhi→first-letter codes) → Local SQLite Search → Confidence Scoring → STTM Control → Web Dashboard

## Running the app

Project-local virtualenv lives at `.venv/` (Python 3.11). Start the dashboard with:

```bash
source .venv/bin/activate
uvicorn src.api.server:app --host 0.0.0.0 --port 8080 --reload
```

For a detached background launch (survives shell exit):

```bash
nohup .venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8080 --reload > /tmp/sttm-automate.log 2>&1 &
disown
```

Dashboard is at <http://localhost:8080>. Logs go to `/tmp/sttm-automate.log`.

If `.venv/` is missing, recreate it:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Conventions

- Use `async/await` throughout the pipeline
- All config in `src/config.py` (no magic numbers in other files)
- Type hints on all function signatures
- Keep modules focused — one responsibility per file

## Git commits

- Use conventional commit style: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Please commit after implementing feat, fix, refactor, test or doc
