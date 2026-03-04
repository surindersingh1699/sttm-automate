"""macOS desktop window wrapper for the STTM Automate dashboard."""

from __future__ import annotations

import fcntl
import os
import re
import socket
import subprocess
import threading
import time
import traceback
import webbrowser
from pathlib import Path

import httpx
import uvicorn

from src.api.server import app
from src.config import config

LOG_PATH = Path.home() / "Library/Logs/STTM-Automate.log"


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


def _find_running_dashboard_url(host: str, preferred: int, attempts: int = 12) -> str | None:
    with httpx.Client(timeout=0.4) as client:
        for port in range(preferred, preferred + attempts):
            url = f"http://{host}:{port}"
            try:
                resp = client.get(f"{url}/api/status")
                if resp.status_code == 200:
                    return url
            except Exception:
                pass
    return None


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None


def _ports_listening_for_pid(pid: int) -> list[int]:
    try:
        proc = subprocess.run(
            ["lsof", "-nP", "-a", f"-p{pid}", "-iTCP", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    ports: list[int] = []
    for line in proc.stdout.splitlines():
        match = re.search(r":(\d+)\s+\(LISTEN\)$", line)
        if match:
            ports.append(int(match.group(1)))
    return ports


def _find_existing_instance_url(lock_path: Path, host: str, preferred: int) -> str | None:
    pid = _read_lock_pid(lock_path)
    if pid is not None:
        with httpx.Client(timeout=0.4) as client:
            for port in _ports_listening_for_pid(pid):
                url = f"http://{host}:{port}"
                try:
                    resp = client.get(f"{url}/api/status")
                    if resp.status_code == 200:
                        return url
                except Exception:
                    pass
    return _find_running_dashboard_url(host, preferred)


def _activate_existing_app_window() -> bool:
    """Bring an already-running app instance to the foreground."""
    try:
        proc = subprocess.run(
            ["osascript", "-e", 'tell application "STTM Automate" to activate'],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0


def _write_startup_error(exc: Exception) -> None:
    """Persist startup errors so Finder launches do not fail silently."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    details = "".join(traceback.format_exception(exc))
    with LOG_PATH.open("a", encoding="utf-8") as logf:
        logf.write(f"\n--- STTM Automate startup error ---\n{details}\n")


def _show_startup_error_dialog() -> None:
    message = (
        f'STTM Automate failed to start. '
        f'Open log for details:\\n{LOG_PATH}'
    )
    script = (
        f'display dialog "{message}" buttons {{"OK"}} '
        f'default button "OK" with icon caution'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=False)
    except OSError:
        pass


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
        self._file = self.lock_path.open("a+")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._file.seek(0)
            self._file.truncate()
            self._file.write(str(os.getpid()))
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
        activated = _activate_existing_app_window()
        existing_url = _find_existing_instance_url(
            lock.lock_path,
            "127.0.0.1",
            int(config.dashboard.port),
        )
        if existing_url:
            print(
                f"[MacApp] STTM Automate is already running. Reusing existing instance at {existing_url}"
            )
            if not activated:
                try:
                    webbrowser.open(existing_url)
                except Exception:
                    pass
        else:
            print("[MacApp] Another STTM Automate app instance is already running.")
        return

    server = DashboardServer()
    try:
        server.start()
        print(f"[MacApp] Dashboard ready at {server.url}")

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
    except Exception as exc:
        print(f"[MacApp] Startup failed: {exc}")
        _write_startup_error(exc)
        _show_startup_error_dialog()
    finally:
        server.stop()
        lock.release()
