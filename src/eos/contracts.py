"""Static contract enforcement for Python sources, notebooks, and YAML configs.

Three rules are enforced: documentation must be expressed through Google-style
docstrings rather than hash comments, every module, class, and function must
carry a docstring, and every configuration file must satisfy its Pydantic
contract.
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from eos.config import ExperimentConfig
from eos.dataset.contract import DatasetContract

SKIP_DIRECTORIES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".opencode",
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        ".vscode",
        "__pycache__",
        "checkpoints",
        "data",
        "docker",
        "mlruns",
        "node_modules",
        "references",
        "reports",
    }
)

SKIP_PREFIXES = ("pytest-of-", "torchinductor_")

ALLOWED_COMMENT_PATTERNS = (
    re.compile(r"^#!\s*\S"),
    re.compile(r"^#\s*type:\s*ignore"),
    re.compile(r"^#\s*noqa(?::\s*\S+)?"),
)


@dataclass(frozen=True, slots=True)
class Violation:
    """A single contract violation.

    Args:
        path (str): Repository-relative path of the offending file.
        line (int): One-based line number, or 1 when unavailable.
        code (str): Stable violation identifier.
        message (str): Human-readable explanation.
    """

    path: str
    line: int
    code: str
    message: str


def _is_allowed_comment(text: str, line_number: int) -> bool:
    """Report whether a comment token is an approved machine directive.

    Args:
        text (str): Raw comment token including the leading hash.
        line_number (int): One-based line number of the token.

    Returns:
        bool: True for shebangs, type suppressions, and linter suppressions.
    """
    if line_number == 1 and text.startswith("#!"):
        return True
    return any(pattern.match(text) for pattern in ALLOWED_COMMENT_PATTERNS)


def _comment_violations(source: str, display_path: str) -> list[Violation]:
    """Collect forbidden hash comments from Python source text.

    Args:
        source (str): Python source code.
        display_path (str): Path used in the emitted violations.

    Returns:
        list[Violation]: One violation per disallowed comment token.
    """
    violations: list[Violation] = []
    stream = io.StringIO(source)
    try:
        tokens: Iterable[tokenize.TokenInfo] = tokenize.generate_tokens(stream.readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            if _is_allowed_comment(token.string, token.start[0]):
                continue
            violations.append(
                Violation(
                    path=display_path,
                    line=token.start[0],
                    code="CMT001",
                    message="hash comments are forbidden; use a Google-style docstring",
                )
            )
    except tokenize.TokenError as error:
        violations.append(
            Violation(
                path=display_path,
                line=int(str(error.args[1][0])) if error.args else 1,
                code="SYN001",
                message=f"unable to tokenize source: {error.args[0]}",
            )
        )
    return violations


def _docstring_violations(tree: ast.AST, display_path: str) -> list[Violation]:
    """Collect nodes that are missing a docstring.

    Args:
        tree (ast.AST): Parsed module.
        display_path (str): Path used in the emitted violations.

    Returns:
        list[Violation]: One violation per undocumented module, class, or function.
    """
    violations: list[Violation] = []
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, documented):
            continue
        if ast.get_docstring(node) is not None:
            continue
        line = int(getattr(node, "lineno", 1))
        name = getattr(node, "name", "<module>")
        violations.append(
            Violation(
                path=display_path,
                line=line,
                code="DOC001",
                message=f"missing Google-style docstring for {name}",
            )
        )
    return violations


def scan_source(source: str, display_path: str, require_docstrings: bool = True) -> list[Violation]:
    """Validate a single Python source string.

    Args:
        source (str): Python source code.
        display_path (str): Path used in the emitted violations.
        require_docstrings (bool): Whether module, class, and function
            docstrings are mandatory.

    Returns:
        list[Violation]: All detected violations, ordered by discovery.

    Raises:
        SyntaxError: If the source cannot be parsed as Python.
    """
    violations = _comment_violations(source, display_path)
    tree = ast.parse(source, filename=display_path)
    if require_docstrings:
        violations.extend(_docstring_violations(tree, display_path))
    return violations


def scan_python_file(path: Path) -> list[Violation]:
    """Validate one Python file on disk.

    Args:
        path (Path): File to validate.

    Returns:
        list[Violation]: All detected violations.

    Raises:
        SyntaxError: If the file cannot be parsed as Python.
    """
    source = path.read_text(encoding="utf-8")
    return scan_source(source, path.as_posix())


def scan_notebook(path: Path) -> list[Violation]:
    """Validate the code cells of a Jupyter notebook.

    Args:
        path (Path): Notebook file to validate.

    Returns:
        list[Violation]: Comment violations from every code cell. Docstrings are
            not required inside notebook cells.

    Raises:
        json.JSONDecodeError: If the notebook is not valid JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    violations: list[Violation] = []
    if not isinstance(payload, dict):
        return [
            Violation(
                path=path.as_posix(),
                line=1,
                code="NB001",
                message="notebook root must be a JSON object",
            )
        ]
    cells = payload.get("cells", [])
    if not isinstance(cells, list):
        return violations
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        raw_source = cell.get("source", "")
        source = raw_source if isinstance(raw_source, str) else "".join(raw_source)
        violations.extend(_comment_violations(source, path.as_posix()))
    return violations


def _iter_python_files(root: Path) -> Iterable[Path]:
    """Yield Python files below a root, skipping generated directories.

    Args:
        root (Path): Repository root.

    Yields:
        Path: Each candidate Python file.
    """
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).parts
        if any(part in SKIP_DIRECTORIES or part.startswith(SKIP_PREFIXES) for part in relative):
            continue
        yield path


def scan_tree(root: Path) -> list[Violation]:
    """Validate every Python file and notebook in the repository.

    Args:
        root (Path): Repository root.

    Returns:
        list[Violation]: Aggregated violations from all scanned files.
    """
    violations: list[Violation] = []
    for path in _iter_python_files(root):
        violations.extend(scan_python_file(path))
    notebooks = (
        sorted((root / "notebooks").glob("*.ipynb")) if (root / "notebooks").is_dir() else []
    )
    for notebook in notebooks:
        violations.extend(scan_notebook(notebook))
    return violations


def validate_configs(directory: Path) -> list[Violation]:
    """Validate every YAML configuration file in a directory.

    Args:
        directory (Path): Directory containing ``*.yaml`` experiment files.

    Returns:
        list[Violation]: One violation per invalid configuration file.
    """
    if not directory.is_dir():
        return []
    violations: list[Violation] = []
    for path in sorted(directory.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            if not isinstance(payload, dict):
                raise ValueError("configuration root must be a mapping")
            ExperimentConfig.model_validate(payload)
        except (ValidationError, ValueError) as error:
            violations.append(
                Violation(
                    path=path.as_posix(),
                    line=1,
                    code="CFG001",
                    message=str(error).replace("\n", " "),
                )
            )
    return violations


def validate_dataset_contracts(directory: Path) -> list[Violation]:
    """Validate every YAML file against the dataset contract.

    Args:
        directory (Path): Directory containing dataset contract files.

    Returns:
        list[Violation]: One violation per invalid contract file.
    """
    if not directory.is_dir():
        return []
    violations: list[Violation] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("dataset contract root must be a mapping")
            DatasetContract.model_validate(payload)
        except (ValidationError, ValueError) as error:
            violations.append(
                Violation(
                    path=path.as_posix(),
                    line=1,
                    code="CTR001",
                    message=str(error).replace("\n", " "),
                )
            )
    return violations
