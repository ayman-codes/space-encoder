"""Runtime resolution of the configured numerical precision.

``TrainConfig`` declares an intent, not a promise: ``bf16-mixed`` is only valid
when an accelerator with native bfloat16 support is present. Resolving the value
at runtime keeps the configuration honest on CPU runners while preserving full
throughput on the ROCm GPU.
"""

from __future__ import annotations

from typing import Literal

Precision = Literal["bf16-mixed", "32-true"]


def resolve_precision(configured: Precision, cuda_available: bool) -> Precision:
    """Return the precision actually usable on the current device.

    Mixed bfloat16 is selected only when the requested precision is
    ``bf16-mixed`` and a CUDA-compatible device is available. An explicit
    ``32-true`` request is never upgraded, because that would silently change
    the numerical contract of a deterministic run.

    Args:
        configured (Precision): Precision declared in ``TrainConfig``.
        cuda_available (bool): Whether ``torch.cuda.is_available()`` is true.

    Returns:
        Precision: ``bf16-mixed`` on an available accelerator, otherwise
            ``32-true``.
    """
    if configured == "bf16-mixed" and cuda_available:
        return "bf16-mixed"
    return "32-true"
