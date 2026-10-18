"""Structured logging helpers shared by training, telemetry, and scripts."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects.

    The formatter keeps container and CI logs machine-parseable while preserving
    exception tracebacks in a dedicated field.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record.

        Args:
            record (logging.LogRecord): Record emitted by a logger.

        Returns:
            str: JSON document containing timestamp, level, logger, message, and
                an optional exception traceback.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single JSON handler on the root logger.

    Args:
        level (int): Minimum severity forwarded to the handler.

    Returns:
        None: The root logger is reconfigured in place.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
