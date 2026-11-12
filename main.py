"""Babel Application Server Entrypoint.

Run with:
    uvicorn main:app --host 127.0.0.1 --port 8000
"""

from server.app import app, create_app

__all__ = ["app", "create_app"]
