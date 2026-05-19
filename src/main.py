"""STTM Automate — Entry point."""

import asyncio
import sys

from src.controller.sttm_http import STTMHttpController


def main():
    """Start the STTM Automate pipeline."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "dashboard"

    if mode == "pipeline":
        # Run pipeline only (no dashboard, console output)
        asyncio.run(run_pipeline_only())
    elif mode == "dashboard":
        # Run with web dashboard (default)
        run_with_dashboard()
    else:
        print(f"Usage: python -m src.main [pipeline|dashboard]")
        print(f"  pipeline  - Run pipeline with console output (no web UI)")
        print(f"  dashboard - Run with web dashboard at http://localhost:8080")
        sys.exit(1)


async def run_pipeline_only():
    """Run the pipeline with console output only (no dashboard)."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    # Try HTTP controller first, fall back to Playwright (lazy import — optional dep)
    controller = STTMHttpController()
    if not await controller.connect():
        print("HTTP controller failed. Trying Playwright...")
        await controller.disconnect()
        from src.controller.sttm_playwright import STTMPlaywrightController
        controller = STTMPlaywrightController()

    pipeline = PipelineOrchestrator(controller=controller)

    try:
        await pipeline.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await pipeline.stop()


def run_with_dashboard():
    """Run the pipeline with the web dashboard."""
    import os
    import uvicorn
    from src.api.auth import get_or_create_token
    from src.config import config

    # LAN mode is opt-in via env var or runtime setting; default is loopback.
    if os.environ.get("STTM_LAN_MODE", "").lower() in ("1", "true", "yes") or config.dashboard.lan_mode:
        host = "0.0.0.0"
        config.dashboard.lan_mode = True
    else:
        host = config.dashboard.host or "127.0.0.1"

    token = get_or_create_token()
    auth_url = f"http://127.0.0.1:{config.dashboard.port}/auth?token={token}"
    bind_label = "LAN" if host == "0.0.0.0" else "loopback only"
    print()
    print("─" * 72)
    print(f"  STTM controller   bind: {host}:{config.dashboard.port}  ({bind_label})")
    print(f"  Dashboard URL     {auth_url}")
    print(f"  WebSocket URL     ws://127.0.0.1:{config.dashboard.port}/ws?token={token}")
    print("─" * 72)
    print()

    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=config.dashboard.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
