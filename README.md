# STTM Automate

Automated Gurbani projection for live kirtan with SikhiToTheMax (STTM).

## Prerequisites

- Python 3.11+
- STTM Desktop installed
- BlackHole (recommended for reliable kirtan audio capture on macOS)

## Install

```bash
git clone https://github.com/surindersingh1699/sttm-automate.git
cd sttm-automate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.main dashboard
```

Then open:

```text
http://localhost:8080
```

## Audio Setup

If capture is not working:

```bash
python scripts/setup_audio.py
```
