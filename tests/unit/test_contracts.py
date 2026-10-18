"""Tests for the zero-comment and docstring contract scanner."""

from __future__ import annotations

import json
from pathlib import Path

from eos.contracts import scan_notebook, scan_source, scan_tree, validate_configs

COMPLIANT_SOURCE = '''"""Module docstring."""


def add(left: int, right: int) -> int:
    """Add two integers.

    Args:
        left (int): Left operand.
        right (int): Right operand.

    Returns:
        int: The sum of both operands.
    """
    return left + right
'''


def test_compliant_source_passes() -> None:
    """A documented module with no comments must produce no violations."""
    assert scan_source(COMPLIANT_SOURCE, "sample.py") == []


def test_hash_comment_is_rejected() -> None:
    """A plain hash comment must be reported as CMT001."""
    source = COMPLIANT_SOURCE + "\n# explain the result\nvalue = add(1, 2)\n"
    violations = scan_source(source, "sample.py")
    assert [violation.code for violation in violations] == ["CMT001"]


def test_machine_directives_are_allowed() -> None:
    """Shebangs, type suppressions, and noqa directives stay legal."""
    source = (
        "#!/usr/bin/env python3\n"
        '"""Module docstring."""\n'
        "\n"
        "def identity(value: int) -> int:\n"
        '    """Return the argument unchanged.\n'
        "\n"
        "    Args:\n"
        "        value (int): Arbitrary integer.\n"
        "\n"
        "    Returns:\n"
        "        int: The same integer.\n"
        '    """\n'
        "    return value  # noqa: D401  # type: ignore[return-value]\n"
    )
    assert scan_source(source, "sample.py") == []


def test_missing_docstring_is_rejected() -> None:
    """An undocumented function must be reported as DOC001."""
    source = "def undocumented(value: int) -> int:\n    return value\n"
    violations = scan_source(source, "sample.py")
    assert "DOC001" in {violation.code for violation in violations}


def test_notebook_comments_are_rejected(tmp_path: Path) -> None:
    """Hash comments inside notebook code cells must be reported."""
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# a heading is fine"]},
            {"cell_type": "code", "source": ["value = 1\n", "# increment\n", "value += 1\n"]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / "notebook.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    violations = scan_notebook(path)
    assert len(violations) == 1
    assert violations[0].code == "CMT001"
    assert violations[0].line == 2


def test_invalid_config_is_reported(tmp_path: Path) -> None:
    """A configuration that violates its contract must be reported."""
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "broken.yaml").write_text("model:\n  latent_dim: 4\n", encoding="utf-8")
    violations = validate_configs(configs)
    assert [violation.code for violation in violations] == ["CFG001"]


def test_scan_tree_skips_references() -> None:
    """The reference corpus must never be scanned for repository contracts."""
    root = Path(__file__).resolve().parents[2]
    paths = {violation.path for violation in scan_tree(root)}
    assert not any("references" in path for path in paths)
