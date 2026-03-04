"""Entry point for building a macOS app bundle."""

import multiprocessing

from src.mac_app import run_mac_app


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_mac_app()
