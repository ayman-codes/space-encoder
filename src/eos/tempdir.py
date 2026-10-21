"""In-repository scratch redirection for deterministic repository hygiene.

MSYS shells advertise POSIX temporary paths that Windows Python cannot write
to, so ``tempfile.gettempdir()`` falls back to the current working directory and
tooling drops pytest basetemp folders and torch Inductor caches into the tracked
tree. Routing every temporary artifact into ``.cache/tmp`` keeps the repository
pristine without depending on the invoking shell.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_ENV_TEMP_VARIABLES: tuple[str, ...] = ("TMPDIR", "TMP", "TEMP")
_TORCHINDUCTOR_VARIABLE = "TORCHINDUCTOR_CACHE_DIR"


def repo_scratch_root() -> Path:
    """Return the repository scratch directory reserved for temporary artifacts.

    Returns:
        Path: Absolute path of the ``.cache/tmp`` directory at the repository root.
    """
    return Path(__file__).resolve().parents[2] / ".cache" / "tmp"


def use_repo_scratch_space() -> Path:
    """Redirect Python, pytest, and torch scratch output into the repository cache.

    The function pins ``tempfile.tempdir`` and the ``TMPDIR``/``TMP``/``TEMP``
    environment variables to ``.cache/tmp`` and points torch Inductor at a
    sibling cache directory, so temporary artifacts can never land in the
    tracked tree regardless of the invoking shell. An operator-provided
    Inductor cache location is preserved. The operation is idempotent.

    Returns:
        Path: The scratch directory that temporary artifacts now resolve to.
    """
    scratch = repo_scratch_root()
    scratch.mkdir(parents=True, exist_ok=True)
    for variable in _ENV_TEMP_VARIABLES:
        os.environ[variable] = str(scratch)
    os.environ.setdefault(_TORCHINDUCTOR_VARIABLE, str(scratch / "torchinductor"))
    tempfile.tempdir = str(scratch)
    return scratch
