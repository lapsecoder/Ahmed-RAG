"""Vercel serverless entrypoint.

Vercel imports this module and looks for a callable named ``app``. Re-exporting
the existing ASGI application is the whole file: the FastAPI app, its routes,
its exception handlers and its security layers are unchanged, so the API
contract is identical on Vercel and locally.
"""

from __future__ import annotations

from app.main import app

__all__ = ["app"]
