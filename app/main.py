"""ASGI entry point.

``uvicorn app.main:app`` or ``ahmed-rag`` starts the service. The embedding model
and the FAISS index are loaded once, during startup.
"""

from __future__ import annotations

from app.api.routes import app, create_app, run
from app.container import Container, build_container

__all__ = ["Container", "app", "build_container", "create_app", "run"]
