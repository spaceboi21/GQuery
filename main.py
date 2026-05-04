"""
XIA — Excel Intelligence Agent
Entry point. Run from the project root:

    python main.py
"""

import os
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from src.config import settings


def _should_open_browser() -> bool:
    val = os.environ.get("XIA_OPEN_BROWSER", "1").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    # Auto-detect headless containers / CI
    if Path("/.dockerenv").exists() or os.environ.get("CI"):
        return False
    return True


if __name__ == "__main__":
    url = f"http://{settings.app_host}:{settings.app_port}"
    print(f"\n  XIA is running → {url}\n")
    if _should_open_browser():
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(
        "src.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level="warning",
    )
