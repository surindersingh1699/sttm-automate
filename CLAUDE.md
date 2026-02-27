# Project: sttm-automate

## What this is
Automation layer for SikhiToTheMax (STTM) Desktop that listens to live kirtan audio, recognizes which shabad is being sung, and automatically controls STTM to display the correct shabad.

## Tech Stack
- Python 3.11+
- faster-whisper (local Punjabi speech recognition)
- BaniDB Python API (shabad database search)
- FastAPI + WebSocket (dashboard server)
- sounddevice (audio capture)
- Playwright (STTM browser automation fallback)
- httpx (STTM HTTP control)

## Architecture
Audio (sounddevice) → Transcription (faster-whisper) → Transliteration (Gurmukhi→first-letter codes) → BaniDB Search → Confidence Scoring → STTM Control → Web Dashboard

## Conventions
- Use `async/await` throughout the pipeline
- All config in `src/config.py` (no magic numbers in other files)
- Type hints on all function signatures
- Keep modules focused — one responsibility per file

## Git commits
- Use conventional commit style: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Please commit after implementing feat, fix, refactor, test or doc