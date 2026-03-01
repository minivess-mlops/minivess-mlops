"""MinIVess CI wrapper for the standalone SDD validator.

Thin wrapper that imports the standalone validator and points it
at the MinIVess PRD directory. Used in CI/CD pipelines and
pre-commit hooks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types

# Import the standalone validator from the template directory
_TEMPLATE_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "probabilistic-sdd-template"
)
_VALIDATOR_PATH = _TEMPLATE_DIR / "validate.py"


def _import_validator() -> types.ModuleType:
    """Dynamically import the standalone validator module."""
    spec = importlib.util.spec_from_file_location("sdd_validate", _VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        msg = f"Cannot load validator from {_VALIDATOR_PATH}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sdd_validate"] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_minivess_sdd() -> Any:
    """Run the standalone SDD validator against the MinIVess PRD.

    Returns
    -------
    ValidationReport
        Structured validation report with pass/fail status.
    """
    validator = _import_validator()

    # MinIVess PRD lives at docs/prd/ relative to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    sdd_root = repo_root / "docs" / "prd"

    return validator.validate_sdd(sdd_root)


if __name__ == "__main__":
    report = validate_minivess_sdd()
    report.print_report()
    sys.exit(0 if report.overall_pass else 1)
