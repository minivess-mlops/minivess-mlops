"""Health regression gate — deterministic pre-commit check.

Blocks commits that introduce:
- New ruff lint errors (zero-error policy)
- Decreased test collection count (tests deleted without updating baseline)

Reads baseline from tests/health_baseline.json.
Update baseline with: uv run python scripts/update_health_baseline.py

This is Layer 1 of the three-layer defense:
  L1 (pre-commit): THIS SCRIPT — deterministic, cannot be rationalized away
  L2 (session start): scripts/session_health_check.sh
  L3 (prompt rules): CLAUDE.md Rules 20/25/28

See: docs/planning/v0-2_archive/critical-failure-fixing-and-silent-failure-fix.md
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _load_baseline() -> dict:
    """Load health baseline from tests/health_baseline.json."""
    baseline_path = Path(__file__).resolve().parent.parent / "tests" / "health_baseline.json"
    if not baseline_path.exists():
        print(f"WARNING: {baseline_path} not found — skipping health regression gate")
        sys.exit(0)
    with baseline_path.open(encoding="utf-8") as f:
        return json.load(f)


def _count_ruff_errors() -> int:
    """Run ruff check and count errors."""
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "src/", "tests/", "scripts/", "--output-format", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        return 0
    try:
        errors = json.loads(result.stdout)
        return len(errors)
    except (json.JSONDecodeError, TypeError):
        # If ruff output is not JSON, count lines in stderr
        return result.stdout.count("\n")


def _count_collected_tests() -> int:
    """Run pytest --collect-only and count collected tests."""
    result = subprocess.run(
        [
            "uv", "run", "pytest", "--collect-only", "-q",
            "tests/v2/unit/", "tests/unit/",
            "--timeout=30",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**__import__("os").environ, "MINIVESS_ALLOW_HOST": "1"},
    )
    # Parse "N tests collected" or "N/M tests collected" from output
    for line in result.stdout.splitlines():
        if "tests collected" in line or "test collected" in line:
            # Format: "7031 tests collected in 14.19s" or "6691/7420 tests collected"
            first_token = line.strip().split()[0]
            # Handle "N/M" format
            num_str = first_token.split("/")[0]
            try:
                return int(num_str)
            except ValueError:
                pass
    return -1  # Unknown


def main() -> int:
    """Run health regression checks. Returns 0 on pass, 1 on regression."""
    baseline = _load_baseline()
    failures = []

    # Check 1: Ruff errors
    ruff_errors = _count_ruff_errors()
    baseline_ruff = baseline.get("ruff", {}).get("error_count", 0)
    if ruff_errors > baseline_ruff:
        failures.append(
            f"RUFF REGRESSION: {ruff_errors} errors (baseline: {baseline_ruff}). "
            f"Fix all lint errors before committing."
        )
    elif ruff_errors > 0:
        failures.append(
            f"RUFF ERRORS: {ruff_errors} errors. Zero-error policy — fix before committing."
        )

    # Check 2: Test collection count
    collected = _count_collected_tests()
    baseline_collected = baseline.get("test_staging", {}).get("collected", 0)
    if collected >= 0 and collected < baseline_collected - 10:
        # Allow small fluctuations (±10) due to parametrize changes
        failures.append(
            f"TEST COUNT REGRESSION: {collected} collected (baseline: {baseline_collected}). "
            f"Tests may have been deleted. Update baseline if intentional: "
            f"uv run python scripts/update_health_baseline.py"
        )

    if failures:
        print("=" * 60)
        print("HEALTH REGRESSION GATE — COMMIT BLOCKED")
        print("=" * 60)
        for f in failures:
            print(f"  ✗ {f}")
        print()
        print("Fix the issues above, then retry your commit.")
        print("If the baseline needs updating: uv run python scripts/update_health_baseline.py")
        return 1

    print(f"Health gate: ruff={ruff_errors} errors, tests={collected} collected — OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
