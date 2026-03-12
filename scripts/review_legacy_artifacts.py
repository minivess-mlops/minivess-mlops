"""Legacy artifact detector — finds v0.1-era patterns that should be removed.

Usage:
  uv run python scripts/review_legacy_artifacts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def _grep_for_pattern(
    directory: Path, pattern: str, glob_pattern: str = "**/*.py"
) -> list[Path]:
    """Search files for a string pattern without regex."""
    matches = []
    for f in (
        directory.rglob(glob_pattern)
        if glob_pattern.startswith("**")
        else directory.glob(glob_pattern)
    ):
        if not f.is_file():
            continue
        try:
            content = f.read_text(encoding="utf-8")
            if pattern in content:
                matches.append(f)
        except (UnicodeDecodeError, PermissionError):
            continue
    return matches


def _check_no_poetry() -> list[dict]:
    """Check 1: No files reference Poetry."""
    pyproject = REPO_ROOT / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    ok = "[tool.poetry]" not in content
    return [
        {
            "check": "no_poetry",
            "severity": "ERROR",
            "ok": ok,
            "message": "OK: No Poetry config"
            if ok
            else "ERROR: [tool.poetry] found in pyproject.toml",
        }
    ]


def _check_no_pip_install() -> list[dict]:
    """Check 2: No files reference pip install or requirements.txt.

    Excludes error/help messages that document external package installation
    instructions (e.g., SAM3 from-source install guidance).
    """
    checks = []
    # Patterns that indicate actual pip usage, not documentation strings
    # Error messages in adapters that tell users how to install SAM3 are OK
    excluded_dirs = {"adapters"}  # Adapter error messages are not legacy patterns

    for pattern in ["pip install", "requirements.txt"]:
        matches = _grep_for_pattern(SRC_DIR, pattern)
        # Filter out known false positives in adapter error messages
        filtered = [m for m in matches if m.parent.name not in excluded_dirs]
        ok = len(filtered) == 0
        checks.append(
            {
                "check": f"no_{pattern.replace(' ', '_').replace('.', '_')}",
                "severity": "ERROR",
                "ok": ok,
                "message": f"OK: No '{pattern}' in src/"
                if ok
                else f"ERROR: '{pattern}' found in {[str(m.relative_to(REPO_ROOT)) for m in filtered]}",
            }
        )
    return checks


def _check_no_old_imports() -> list[dict]:
    """Check 3: No old import paths."""
    checks = []
    old_patterns = [
        "from src.training import",
        "from src.log_ML import",
        "from src.datasets import",
    ]
    for pattern in old_patterns:
        matches = _grep_for_pattern(SRC_DIR, pattern)
        ok = len(matches) == 0
        checks.append(
            {
                "check": f"no_old_import:{pattern[:30]}",
                "severity": "ERROR",
                "ok": ok,
                "message": f"OK: No '{pattern}'"
                if ok
                else f"ERROR: '{pattern}' found in {[str(m.relative_to(REPO_ROOT)) for m in matches]}",
            }
        )
    return checks


def _check_wiki_deleted() -> list[dict]:
    """Check 4: wiki/ directory does not exist."""
    ok = not (REPO_ROOT / "wiki").exists()
    return [
        {
            "check": "wiki_deleted",
            "severity": "WARN",
            "ok": ok,
            "message": "OK: wiki/ deleted"
            if ok
            else "WARN: wiki/ directory still exists",
        }
    ]


def _check_legacy_config_deleted() -> list[dict]:
    """Check 5: _legacy_v01_defaults.yaml does not exist."""
    ok = not (REPO_ROOT / "configs" / "_legacy_v01_defaults.yaml").exists()
    return [
        {
            "check": "legacy_config_deleted",
            "severity": "WARN",
            "ok": ok,
            "message": "OK: legacy config deleted"
            if ok
            else "WARN: configs/_legacy_v01_defaults.yaml still exists",
        }
    ]


def _check_no_wandb_in_src() -> list[dict]:
    """Check 6: No active wandb references in src/."""
    matches = _grep_for_pattern(SRC_DIR, "import wandb")
    ok = len(matches) == 0
    return [
        {
            "check": "no_wandb",
            "severity": "WARN",
            "ok": ok,
            "message": "OK: No wandb imports"
            if ok
            else f"WARN: wandb imported in {[str(m.relative_to(REPO_ROOT)) for m in matches]}",
        }
    ]


def _check_no_airflow_in_src() -> list[dict]:
    """Check 7: No active airflow references in src/."""
    matches = _grep_for_pattern(SRC_DIR, "import airflow")
    ok = len(matches) == 0
    return [
        {
            "check": "no_airflow",
            "severity": "WARN",
            "ok": ok,
            "message": "OK: No airflow imports"
            if ok
            else f"WARN: airflow imported in {[str(m.relative_to(REPO_ROOT)) for m in matches]}",
        }
    ]


def main() -> dict:
    """Run all legacy checks."""
    all_checks: list[dict] = []
    all_checks.extend(_check_no_poetry())
    all_checks.extend(_check_no_pip_install())
    all_checks.extend(_check_no_old_imports())
    all_checks.extend(_check_wiki_deleted())
    all_checks.extend(_check_legacy_config_deleted())
    all_checks.extend(_check_no_wandb_in_src())
    all_checks.extend(_check_no_airflow_in_src())

    failures = sum(1 for c in all_checks if not c["ok"] and c["severity"] == "ERROR")
    warnings = sum(1 for c in all_checks if not c["ok"] and c["severity"] == "WARN")

    return {
        "agent_name": "legacy_detector",
        "failures": failures,
        "warnings": warnings,
        "total_checks": len(all_checks),
        "checks": all_checks,
    }


if __name__ == "__main__":
    result = main()
    print(f"\n{'=' * 60}")
    print("Legacy Artifact Detector")
    print(f"{'=' * 60}")
    print(f"Total checks: {result['total_checks']}")
    print(f"Failures (ERROR): {result['failures']}")
    print(f"Warnings (WARN):  {result['warnings']}")
    for check in result["checks"]:
        if not check["ok"]:
            print(f"  [{check['severity']}] {check['message']}")
    if result["failures"] > 0:
        print(f"\n{result['failures']} ERROR(s) — legacy artifacts detected")
        sys.exit(1)
    else:
        print("\nAll checks passed!")
        sys.exit(0)
