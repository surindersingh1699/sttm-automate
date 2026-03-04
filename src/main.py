"""STTM Automate — Entry point."""

import asyncio
import sys

from src.controller.sttm_http import STTMHttpController
from src.controller.sttm_playwright import STTMPlaywrightController


def main():
    """Start the STTM Automate pipeline."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "dashboard"

    if mode == "pipeline":
        # Run pipeline only (no dashboard, console output)
        asyncio.run(run_pipeline_only())
    elif mode == "mac-app":
        # Run as a native macOS window (embeds dashboard web UI)
        run_mac_app()
    elif mode == "dashboard":
        # Run with web dashboard (default)
        run_with_dashboard()
    else:
        print(f"Usage: python -m src.main [pipeline|dashboard|mac-app]")
        print(f"  pipeline  - Run pipeline with console output (no web UI)")
        print(f"  dashboard - Run with web dashboard at http://localhost:8080")
        print(f"  mac-app   - Run native macOS app window")
        sys.exit(1)


async def run_pipeline_only():
    """Run the pipeline with console output only (no dashboard)."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    # Try HTTP controller first, fall back to Playwright
    controller = STTMHttpController()
    if not await controller.connect():
        print("HTTP controller failed. Trying Playwright...")
        await controller.disconnect()
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
    import uvicorn
    from src.config import config

    # The FastAPI app handles pipeline lifecycle
    uvicorn.run(
        "src.api.server:app",
        host=config.dashboard.host,
        port=config.dashboard.port,
        reload=False,
    )


def run_mac_app():
    """Run the pipeline/dashboard in a native macOS window."""
    from src.mac_app import run_mac_app as run_native

    run_native()


if __name__ == "__main__":
    main()
