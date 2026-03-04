#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
APP_NAME="STTM Automate"
ENTRY_SCRIPT="${ROOT_DIR}/scripts/mac_app_entry.py"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[build] Missing virtual environment at ${VENV_DIR}" >&2
  echo "[build] Run install first: curl -fsSL https://raw.githubusercontent.com/surindersingh1699/sttm-automate/master/scripts/install.sh | bash" >&2
  exit 1
fi

echo "[build] Installing PyInstaller in project venv..."
"${VENV_DIR}/bin/pip" install --upgrade pyinstaller >/dev/null

echo "[build] Building macOS app bundle..."
cd "${ROOT_DIR}"
"${VENV_DIR}/bin/pyinstaller" \
  --noconfirm \
  --windowed \
  --name "${APP_NAME}" \
  --paths "${ROOT_DIR}" \
  --add-data "${ROOT_DIR}/static:static" \
  --collect-all faster_whisper \
  --collect-all webview \
  "${ENTRY_SCRIPT}"

echo

echo "[build] Build complete. App bundle:"
echo "  ${ROOT_DIR}/dist/${APP_NAME}.app"
