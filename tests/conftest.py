"""Shared pytest fixtures for the earth observation engine test suite."""

from __future__ import annotations

import torch


def pytest_configure() -> None:
    """Disable hardware-specific CPU backends that crash on virtualized runners.

    Returns:
        None: Torch backends are adjusted in place when available.
    """
    try:
        torch.backends.mkldnn.enabled = False
    except (AttributeError, RuntimeError):
        return
