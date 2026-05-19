"""Per-install token auth for the controller WebSocket + dashboard.

The token is generated once and persisted under the project root (or
the path in ``STTM_TOKEN_PATH``). Clients must present it as
``?token=...`` on the ws:// upgrade request, or via the
``X-STTM-Token`` header, or the ``sttm_token`` cookie. Without a valid
token the WebSocket handshake is refused with HTTP 403 (matching
Starlette's pre-accept reject) and HTTP routes return 401.

This is **not** a replacement for TLS; on a hostile network the token
travels in the clear. It exists to stop a curious LAN-mate from
hijacking your dashboard or flipping the engine mid-kirtan, not to
defend against MITM. Pair it with localhost-only binding for the
strongest default; opt into LAN exposure only when needed.
"""

from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_token_path() -> Path:
    """Honour ``STTM_TOKEN_PATH`` (used by the Tauri sidecar to point at a
    user-data dir), otherwise fall back to the project-root file used in
    development."""
    override = os.environ.get("STTM_TOKEN_PATH")
    if override:
        p = Path(override).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return p
    return _PROJECT_ROOT / ".controller_token"


_TOKEN_PATH = _resolve_token_path()
_TOKEN_LOCK = threading.Lock()
_CACHED: str | None = None


def get_or_create_token() -> str:
    """Return the persistent controller token, generating one if absent."""
    global _CACHED
    if _CACHED:
        return _CACHED
    with _TOKEN_LOCK:
        if _CACHED:
            return _CACHED
        if _TOKEN_PATH.exists():
            tok = _TOKEN_PATH.read_text(encoding="utf-8").strip()
            if tok:
                _CACHED = tok
                return _CACHED
        tok = secrets.token_urlsafe(32)
        _TOKEN_PATH.write_text(tok, encoding="utf-8")
        try:
            os.chmod(_TOKEN_PATH, 0o600)
        except OSError:
            pass
        _CACHED = tok
        return _CACHED


def verify(token: str | None) -> bool:
    if not token:
        return False
    return secrets.compare_digest(token, get_or_create_token())


def reset_token() -> str:
    """Discard the current token and generate a fresh one."""
    global _CACHED
    with _TOKEN_LOCK:
        _CACHED = None
        if _TOKEN_PATH.exists():
            try:
                _TOKEN_PATH.unlink()
            except OSError:
                pass
    return get_or_create_token()
