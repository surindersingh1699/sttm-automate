#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/surindersingh1699/sttm-automate.git}"
REF="${REF:-master}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.sttm-automate}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_BIN_DIR="${LOCAL_BIN_DIR:-$HOME/.local/bin}"

log() {
  printf '[install] %s\n' "$1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

log "Checking prerequisites..."
require_cmd git
require_cmd "$PYTHON_BIN"

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  printf 'Python 3.11+ is required.\n' >&2
  exit 1
fi

if [ -d "$INSTALL_DIR/.git" ]; then
  log "Updating existing install at $INSTALL_DIR"
  git -C "$INSTALL_DIR" fetch --depth 1 origin "$REF"
  git -C "$INSTALL_DIR" checkout -q FETCH_HEAD
else
  log "Cloning repository into $INSTALL_DIR"
  rm -rf "$INSTALL_DIR"
  git clone --depth 1 --branch "$REF" "$REPO_URL" "$INSTALL_DIR"
fi

log "Creating virtual environment..."
"$PYTHON_BIN" -m venv "$INSTALL_DIR/.venv"

log "Installing Python dependencies..."
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip >/dev/null
"$INSTALL_DIR/.venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

if [ -x "$INSTALL_DIR/.venv/bin/playwright" ]; then
  log "Installing Playwright Chromium runtime (used as fallback STTM controller)..."
  "$INSTALL_DIR/.venv/bin/playwright" install chromium
fi

log "Creating launcher at $LOCAL_BIN_DIR/sttm-automate"
mkdir -p "$LOCAL_BIN_DIR"
cat > "$LOCAL_BIN_DIR/sttm-automate" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "$INSTALL_DIR/.venv/bin/python" -m src.main "\$@"
EOF
chmod +x "$LOCAL_BIN_DIR/sttm-automate"

cat <<EOF

Install complete.

Next steps:
1. Add $LOCAL_BIN_DIR to PATH if needed.
2. Run: sttm-automate dashboard
3. Open: http://localhost:8080

If audio capture is not ready yet, run:
  "$INSTALL_DIR/.venv/bin/python" scripts/setup_audio.py
EOF
