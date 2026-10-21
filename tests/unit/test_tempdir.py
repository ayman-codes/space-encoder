"""Tests for the in-repository scratch space redirection."""

from __future__ import annotations

import os
import tempfile

import pytest

from eos.tempdir import repo_scratch_root, use_repo_scratch_space


def test_repo_scratch_root_is_inside_the_cache() -> None:
    """The scratch root must live under the repository cache directory."""
    root = repo_scratch_root()
    assert root.name == "tmp"
    assert root.parent.name == ".cache"


def test_use_repo_scratch_space_pins_python_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every temporary artifact must resolve to the repository scratch directory."""
    for variable in ("TMPDIR", "TMP", "TEMP", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)
    scratch = use_repo_scratch_space()
    assert scratch == repo_scratch_root()
    assert scratch.is_dir()
    assert tempfile.tempdir == str(scratch)
    for variable in ("TMPDIR", "TMP", "TEMP"):
        assert os.environ[variable] == str(scratch)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(scratch / "torchinductor")


def test_use_repo_scratch_space_preserves_existing_torchinductor_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator-provided Inductor cache location must never be overwritten."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "X:\\custom\\cache")
    use_repo_scratch_space()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "X:\\custom\\cache"


def test_use_repo_scratch_space_is_idempotent() -> None:
    """Repeated redirection must keep resolving to the same directory."""
    assert use_repo_scratch_space() == use_repo_scratch_space()
