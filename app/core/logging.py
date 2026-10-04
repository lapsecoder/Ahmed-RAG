"""Logging setup.

Uses a single stream handler and never configures the root logger, so the
library stays friendly to a host application (uvicorn, pytest, ...).
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Install a stderr handler on the ``app`` logger exactly once."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger = logging.getLogger("app")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the ``app`` logger."""
    return logging.getLogger(f"app.{name}" if not name.startswith("app") else name)
