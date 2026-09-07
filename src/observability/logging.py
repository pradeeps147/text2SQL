"""Structured backend logging with a rotating file and an in-memory tail."""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_PATH = LOG_DIR / "tenarai-backend.log"
LOGGER_NAME = "tenarai"

_LOCK = threading.Lock()
_RECORDS: deque[dict[str, Any]] = deque(maxlen=2000)
_CONFIGURED = False


class _MemoryJsonHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "request_id", "duration_ms", "details"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        with _LOCK:
            _RECORDS.append(entry)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "request_id", "duration_ms", "details"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    """Configure the Tenarai logger once per process and return it."""
    global _CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    if _CONFIGURED:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    memory_handler = _MemoryJsonHandler()
    memory_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_JsonFormatter())

    logger.addHandler(memory_handler)
    logger.addHandler(file_handler)
    _CONFIGURED = True
    return logger


def get_recent_logs(limit: int = 200, level: str | None = None) -> list[dict[str, Any]]:
    """Return newest backend records first, optionally filtered by level."""
    normalized_level = level.upper() if level else None
    with _LOCK:
        records = list(_RECORDS)
    if normalized_level:
        records = [record for record in records if record["level"] == normalized_level]
    return list(reversed(records[-limit:]))
