"""macOS desktop window wrapper for the STTM Automate dashboard."""

from __future__ import annotations

import fcntl
import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from src.api.server import app
from src.config import config


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _pick_port(host: str, preferred: int, attempts: int = 12) -> int:
    for port in range(preferred, preferred + attempts):
        if not _is_port_open(host, port):
            return port
    raise RuntimeError(
        f"No free port available near {preferred}. Stop any old STTM Automate instance and retry."
    )


class DashboardServer:
    """Run FastAPI/uvicorn in a background thread for the desktop window."""

    def __init__(self):
        self.host = "127.0.0.1"
        self.port = _pick_port(self.host, int(config.dashboard.port))
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout_seconds: float = 20.0) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None
        with httpx.Client(timeout=0.5) as client:
            while time.time() < deadline:
                try:
                    resp = client.get(f"{self.url}/api/status")
                    if resp.status_code == 200:
                        return
                except Exception as exc:
                    last_error = exc
                time.sleep(0.15)

        self.stop()
        raise RuntimeError(
            f"Dashboard server did not start in time at {self.url}. Last error: {last_error}"
        )

    def _run(self) -> None:
        cfg = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            reload=False,
            log_level="info",
        )
        self._server = uvicorn.Server(cfg)
        # Uvicorn signals are only valid on main thread.
        self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._server.run()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=6.0)


class SingleInstanceLock:
    """Prevent multiple desktop app instances from running at once."""

    def __init__(self, lock_path: str = "/tmp/sttm-automate-mac-app.lock"):
        self.lock_path = Path(lock_path)
        self._file = None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.lock_path.open("w")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._file.write(str(self.lock_path))
            self._file.flush()
            return True
        except BlockingIOError:
            self._file.close()
            self._file = None
            return False

    def release(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None


def run_mac_app() -> None:
    """Start the local dashboard in a native desktop window."""
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'pywebview'. Run: pip install pywebview"
        ) from exc

    lock = SingleInstanceLock()
    if not lock.acquire():
        print("[MacApp] Another STTM Automate app instance is already running.")
        return

    server = DashboardServer()
    server.start()
    print(f"[MacApp] Dashboard ready at {server.url}")

    try:
        window = webview.create_window(
            title="STTM Automate",
            url=server.url,
            width=1280,
            height=860,
            min_size=(1024, 700),
        )

        if window is not None:
            window.events.closed += lambda: server.stop()

        webview.start(debug=False)
    finally:
        server.stop()
        lock.release()
