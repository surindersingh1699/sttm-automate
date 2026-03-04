# STTM Automate

Automated Gurbani projection for live kirtan with SikhiToTheMax (STTM).

## Phase 1 Distribution (Technical Users)

Phase 1 is local-first installation from GitHub:
- App runs on each laptop (audio + STTM control stay local).
- GitHub is used for code delivery and updates.

Landing page:
- https://landing-mu-sandy.vercel.app

## Prerequisites

- macOS (current setup targets STTM Desktop on macOS)
- Python 3.11+
- `git`
- STTM Desktop installed
- BlackHole (recommended for reliable kirtan audio capture)

## Install

Run:

```bash
curl -fsSL https://raw.githubusercontent.com/surindersingh1699/sttm-automate/master/scripts/install.sh | bash
```

The installer will:
- clone/update to `~/.sttm-automate`
- create `~/.sttm-automate/.venv`
- install dependencies
- install Playwright Chromium runtime
- create launcher: `~/.local/bin/sttm-automate`

## Run

```bash
sttm-automate dashboard
```

Then open:

```text
http://localhost:8080
```

Or run as a macOS desktop app window:

```bash
sttm-automate mac-app
```

## Build macOS .app bundle

To generate a clickable app bundle locally:

```bash
./scripts/build_mac_app.sh
```

Output:

```text
dist/STTM Automate.app
```

## Build Windows .exe (shareable)

Windows `.exe` builds are automated via GitHub Actions:

- Workflow: `.github/workflows/build-windows-exe.yml`
- Job artifact: `STTM-Automate-windows-exe`
- Output binary inside artifact: `dist/STTM-Automate.exe`

You can trigger it from GitHub Actions (`Build Windows EXE`), then download the artifact and share the `.exe`.

## Host Downloads Online (recommended)

Use GitHub Releases as your download host:

1. Create and push a tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

2. Workflow `Release Binaries` builds and publishes:
- macOS zip (`STTM-Automate-mac-*.zip`)
- Windows exe (`STTM-Automate-windows-*.exe`)

3. Landing page auto-fetches latest release assets from:

```text
https://api.github.com/repos/surindersingh1699/sttm-automate/releases/latest
```

See full details:
- `docs/download-hosting.md`

## Update

Re-run the same install command:

```bash
curl -fsSL https://raw.githubusercontent.com/surindersingh1699/sttm-automate/master/scripts/install.sh | bash
```

## Uninstall

```bash
rm -rf ~/.sttm-automate ~/.local/bin/sttm-automate
```

## Audio Setup

If capture is not working:

```bash
~/.sttm-automate/.venv/bin/python ~/.sttm-automate/scripts/setup_audio.py
```
