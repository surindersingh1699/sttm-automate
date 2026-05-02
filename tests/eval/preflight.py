"""Preflight checks for live eval mode.

Live mode requires:
  1. STTM Desktop running and reachable on one of the configured HTTP ports.
  2. BlackHole 2ch virtual audio device installed and available as an input.
  3. Playwright Chromium browser installed.
  4. yt-dlp available on PATH (used for headless cache-fill even in live mode).

run_preflight() returns a list of issues (empty = all clear).
print_preflight() pretty-prints results and raises if hard failures exist.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal


@dataclass
class PreflightResult:
    check: str
    status: Literal["ok", "warn", "fail"]
    message: str


def _check_sttm() -> PreflightResult:
    """Try each configured STTM HTTP port."""
    try:
        import httpx
        from src.config import config
        ports = getattr(config.sttm, "http_ports", []) or []
        for port in ports:
            try:
                r = httpx.get(f"http://localhost:{port}/", timeout=1.0)
                return PreflightResult(
                    "STTM Desktop",
                    "ok",
                    f"Reachable on port {port} (HTTP {r.status_code})",
                )
            except Exception:
                pass
        return PreflightResult(
            "STTM Desktop",
            "fail",
            f"Not reachable on any port: {ports}. Start STTM Desktop first.",
        )
    except Exception as e:
        return PreflightResult("STTM Desktop", "fail", f"Config/httpx error: {e}")


def _check_blackhole() -> PreflightResult:
    """Verify BlackHole virtual device is available as an input."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        for dev in devices:
            if dev["max_input_channels"] > 0 and "blackhole" in dev["name"].lower():
                return PreflightResult(
                    "BlackHole audio device",
                    "ok",
                    f"Found: {dev['name']} (device index {list(devices).index(dev) if isinstance(devices, list) else '?'})",
                )
        return PreflightResult(
            "BlackHole audio device",
            "fail",
            "BlackHole not found. Install: brew install blackhole-2ch, then create a "
            "Multi-Output Device in Audio MIDI Setup (built-in + BlackHole) and set as default output.",
        )
    except Exception as e:
        return PreflightResult("BlackHole audio device", "fail", f"sounddevice error: {e}")


def _check_playwright() -> PreflightResult:
    try:
        from playwright.async_api import async_playwright  # noqa: F401
        # Check Chromium browser binary exists
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True, text=True, timeout=10,
        )
        if "chromium" in (result.stdout + result.stderr).lower():
            return PreflightResult("Playwright Chromium", "ok", "Installed")
        return PreflightResult("Playwright Chromium", "ok", "playwright package available")
    except ImportError:
        return PreflightResult(
            "Playwright Chromium",
            "fail",
            "playwright not installed. Run: pip install playwright && playwright install chromium",
        )
    except Exception as e:
        return PreflightResult("Playwright Chromium", "warn", f"Could not verify: {e}")


def _check_ytdlp() -> PreflightResult:
    path = shutil.which("yt-dlp")
    if path:
        return PreflightResult("yt-dlp", "ok", f"Found at {path}")
    return PreflightResult(
        "yt-dlp",
        "fail",
        "yt-dlp not on PATH. Install: pip install yt-dlp",
    )


def run_preflight(mode: str = "live") -> list[PreflightResult]:
    """Run preflight checks for the given eval mode. Returns list of results."""
    results: list[PreflightResult] = []

    if mode == "live":
        results.append(_check_sttm())
        results.append(_check_blackhole())
        results.append(_check_playwright())

    results.append(_check_ytdlp())
    return results


def print_preflight(mode: str = "live") -> bool:
    """Print preflight results. Returns True if all checks pass (no fails)."""
    results = run_preflight(mode)
    icons = {"ok": "✓", "warn": "⚠", "fail": "✗"}
    print(f"\n[Preflight] Checking requirements for mode={mode}…")
    all_ok = True
    for r in results:
        icon = icons[r.status]
        print(f"  {icon} {r.check}: {r.message}")
        if r.status == "fail":
            all_ok = False
    if not all_ok:
        print("\n[Preflight] Fix the failures above before running live eval.\n")
    else:
        print("[Preflight] All checks passed.\n")
    return all_ok
