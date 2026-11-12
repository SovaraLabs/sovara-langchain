"""Logging for the LangChain integration: metadata only, never payloads or tokens."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("sovara_langchain")

_warned: set[str] = set()
_warned_lock = threading.Lock()


def warn_once(key: str, message: str) -> None:
    """Log a warning the first time ``key`` is seen in this process."""
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message)
