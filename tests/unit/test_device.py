"""Tests for the GPU memory budget guard."""

from __future__ import annotations

import pytest
import torch

from eos.device import DEFAULT_VRAM_LIMIT_GIB, cap_vram


def test_cap_vram_is_noop_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a CUDA device the budget guard must leave the process untouched."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cap_vram() == 0.0


def test_cap_vram_sets_fraction_from_total_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ceiling must translate into an allocator fraction of the card."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: type("Props", (), {"total_memory": 16 * 1024**3}),
    )
    recorded: list[float] = []
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        recorded.append,
    )
    assert cap_vram(DEFAULT_VRAM_LIMIT_GIB) == pytest.approx(0.75)
    assert recorded == [pytest.approx(0.75)]


def test_cap_vram_rejects_nonpositive_budget() -> None:
    """A nonpositive budget must fail loudly rather than silently disable the guard."""
    with pytest.raises(ValueError, match="must be positive"):
        cap_vram(0.0)
