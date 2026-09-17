"""
One-command launcher: starts the FastAPI backend (port 8000) and the Gradio
dispatcher dashboard (port 7860).

Usage:  python src/run.py   (from the project root)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BACKEND = "http://127.0.0.1:8000"
SHARE = os.environ.get("SWR_SHARE", "0") == "1" or "--share" in sys.argv


def wait_for_backend(timeout: float = 300.0) -> bool:
    """Wait for the backend to become healthy. The first run may trigger a
    retrain (up to a couple of minutes), so use a generous timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{BACKEND}/health", timeout=2).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    return False


def main() -> None:
    print("Starting FastAPI backend ...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT / "src",
    )
    try:
        if wait_for_backend():
            print(f"✅ Backend online at {BACKEND}")
        else:
            print("⚠ Backend still starting (first run may retrain models for a "
                  "minute or two). The dashboard will connect once it is ready.")

        from app_frontend import build_demo, launch
        print("Launching Gradio dashboard at http://0.0.0.0:7860 ...")
        print("  RailPulse web app (login + dark/light + network map): http://127.0.0.1:8000")
        print("  Gradio dispatcher console: http://127.0.0.1:7860")
        if SHARE:
            print("  ⚡ SHARE mode ON - a public link will appear below once Gradio starts.")
        from config import AUTH_PASS, AUTH_USER
        print(f"  Login credentials ->  username: {AUTH_USER}   password: {AUTH_PASS}")
        print("  (override with the SWR_USER / SWR_PASS environment variables)")
        launch(build_demo(), share=SHARE)
    finally:
        backend.terminate()


if __name__ == "__main__":
    main()
