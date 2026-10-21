"""GPU memory budget enforcement for every entry point that touches a device.

Training, compression benchmarking, and drift calibration share one physical
card with the desktop session, so every process must stay inside an explicit
VRAM ceiling. The ceiling is enforced through the PyTorch caching allocator,
which refuses allocations beyond the configured fraction of total device
memory instead of starving the host and crashing the driver.
"""

from __future__ import annotations

import torch

DEFAULT_VRAM_LIMIT_GIB = 12.0


def cap_vram(limit_gib: float = DEFAULT_VRAM_LIMIT_GIB) -> float:
    """Restrict this process to a fixed VRAM budget on the active GPU.

    The fraction is derived from the total device memory, so the same ceiling
    applies to any card. On CPU-only hosts the call is a no-op, which keeps
    every entry point safe to run unchanged on GitHub Actions or Colab CPU.

    Args:
        limit_gib (float): VRAM budget in gibibytes. Must be positive.

    Returns:
        float: The fraction of total device memory this process may allocate,
        or 0.0 when no CUDA device is present.

    Raises:
        ValueError: If ``limit_gib`` is not positive.
    """
    if limit_gib <= 0:
        raise ValueError(f"the VRAM budget must be positive, received {limit_gib}")
    if not torch.cuda.is_available():
        return 0.0
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    fraction = min(limit_gib / total_gib, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction)
    return fraction
