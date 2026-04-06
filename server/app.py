"""
FastAPI HTTP server for CalendarSchedulingEnv — server/app.py
Mirrors app.py but lives in the server/ package as required by openenv validate.

Run:
    python -m server.app
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os

# Ensure project root is on path when run from server/ sub-package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from app import app  # re-export the FastAPI app


def main() -> None:
    """Entry point for openenv validate and the 'server' console script."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
