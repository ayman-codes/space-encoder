"""Command line entry point for the repository contract gate."""

from __future__ import annotations

import sys
from pathlib import Path

from eos.contracts import scan_tree, validate_configs, validate_dataset_contracts


def main() -> int:
    """Run every contract check against the repository root.

    Returns:
        int: Process exit code, zero when the repository is compliant.
    """
    root = Path(__file__).resolve().parents[1]
    violations = scan_tree(root)
    violations.extend(validate_configs(root / "configs"))
    violations.extend(validate_dataset_contracts(root / "data" / "contracts"))
    for violation in violations:
        print(f"{violation.path}:{violation.line}: [{violation.code}] {violation.message}")
    if violations:
        print(f"contract gate failed with {len(violations)} violation(s)")
        return 1
    print("contract gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
