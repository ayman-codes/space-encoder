"""Tests for runtime precision resolution."""

from __future__ import annotations

from eos.training.precision import resolve_precision


def test_cpu_downgrades_mixed_bfloat16() -> None:
    """A CPU runner cannot execute the configured mixed-precision path."""
    assert resolve_precision("bf16-mixed", cuda_available=False) == "32-true"


def test_gpu_keeps_mixed_bfloat16() -> None:
    """The ROCm device keeps native bfloat16 mixed precision."""
    assert resolve_precision("bf16-mixed", cuda_available=True) == "bf16-mixed"


def test_explicit_float32_is_never_upgraded() -> None:
    """A deterministic float32 run must stay float32 even on a GPU."""
    assert resolve_precision("32-true", cuda_available=True) == "32-true"
