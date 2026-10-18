"""Tests for structured logging."""

from __future__ import annotations

import json
import logging

from eos.utils.logging import JsonFormatter, configure_logging


def test_json_formatter_emits_parseable_record() -> None:
    """A formatted record must be valid JSON with the expected fields."""
    record = logging.LogRecord(
        name="eos.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="training step %d",
        args=(3,),
        exc_info=None,
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "eos.test"
    assert payload["message"] == "training step 3"


def test_configure_logging_installs_single_handler() -> None:
    """Configuring logging twice must not duplicate handlers."""
    configure_logging(logging.DEBUG)
    configure_logging(logging.INFO)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.INFO
