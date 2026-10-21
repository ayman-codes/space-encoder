"""Shared pytest fixtures for the earth observation engine test suite."""

from __future__ import annotations

from eos.tempdir import use_repo_scratch_space

use_repo_scratch_space()

import torch  # noqa: E402


def pytest_configure() -> None:
    """Disable hardware-specific CPU backends that crash on virtualized runners.

    Returns:
        None: Torch backends are adjusted in place when available.
    """
    try:
        torch.backends.mkldnn.enabled = False
    except (AttributeError, RuntimeError):
        return
