"""Deterministic seeding for the Python, NumPy, and Torch runtimes.

The training stack consumes four independent random streams: Python's ``random``
module, NumPy's legacy global generator, Torch's CPU generator, and the
accelerator generators. Seeding all four keeps data shuffling, augmentation, and
weight initialization reproducible across identical runs.

Full bitwise determinism is a *best-effort* property on ROCm. Several MIOpen and
HIP kernels do not expose deterministic implementations, so
``torch.use_deterministic_algorithms`` is enabled in warning mode: unsupported
operations emit a warning instead of aborting a long training job. CPU runs are
deterministic for the operations used by this project.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed every random number generator used by training.

    The helper also sets the cuDNN/MIOpen determinism flags when ``deterministic``
    is true. ``PYTHONHASHSEED`` is intentionally not modified because changing it
    after interpreter startup has no effect on the current process.

    Args:
        seed (int): Non-negative seed forwarded to every generator.
        deterministic (bool): Whether to request deterministic kernels. Defaults
            to true; the request is advisory on ROCm because some kernels lack a
            deterministic implementation.

    Returns:
        int: The seed that was applied, matching the Lightning convention.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
