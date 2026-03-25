"""Validate .env.example format is compatible with SkyPilot's --env-file parser.

SkyPilot's env-file parser expects strict KEY=value lines:
- No quoting (double or single)
- No `export ` prefix
- No inline comments (# after value)
- POSIX-compliant key names [A-Z_][A-Z0-9_]*
- No duplicate keys

TDD Task 2.F from cloud robustness plan (#954).
Rule #16: No regex — uses str.partition, str.strip, character checks via `string` module.
"""

from __future__ import annotations

import string
from pathlib import Path

ENV_EXAMPLE = Path(__file__).resolve().parents[4] / ".env.example"

# Imported from scripts/preflight_gcp.py (canonical source of truth).
# Duplicated here to avoid importing from scripts/ (test CLAUDE.md rule).
REQUIRED_ENV_VARS = ["HF_TOKEN", "MLFLOW_TRACKING_URI"]

# Valid characters for POSIX env var keys
_POSIX_FIRST_CHARS = string.ascii_letters + "_"
_POSIX_CHARS = string.ascii_letters + string.digits + "_"


def _parse_env_lines() -> list[tuple[str, str]]:
    """Parse .env.example into (key, value) pairs, skipping comments and blanks."""
    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    pairs: list[tuple[str, str]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition("=")
        if sep:
            pairs.append((key, value))
    return pairs


class TestEnvSkypilotFormat:
    """Ensure .env.example is compatible with SkyPilot --env-file parsing."""

    def test_env_example_exists(self) -> None:
        """Precondition: .env.example must exist."""
        assert ENV_EXAMPLE.exists(), f".env.example not found at {ENV_EXAMPLE}"

    def test_all_keys_valid_posix(self) -> None:
        """Each key must match POSIX [A-Z_][A-Z0-9_]* via character checks."""
        pairs = _parse_env_lines()
        assert len(pairs) > 0, ".env.example has no key=value lines"

        invalid_keys: list[str] = []
        for key, _value in pairs:
            if not key:
                invalid_keys.append("<empty key>")
                continue
            # First character must be letter or underscore
            if key[0] not in _POSIX_FIRST_CHARS:
                invalid_keys.append(key)
                continue
            # Remaining characters must be alphanumeric or underscore
            if not all(c in _POSIX_CHARS for c in key):
                invalid_keys.append(key)

        assert invalid_keys == [], (
            f"Non-POSIX env var keys found: {invalid_keys}"
        )

    def test_no_quoted_values_double(self) -> None:
        """No line should have KEY="value" (double-quoted)."""
        pairs = _parse_env_lines()
        double_quoted: list[str] = []
        for key, value in pairs:
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                double_quoted.append(f"{key}={value}")

        assert double_quoted == [], (
            f"Double-quoted values found (SkyPilot incompatible): {double_quoted}"
        )

    def test_no_quoted_values_single(self) -> None:
        """No line should have KEY='value' (single-quoted)."""
        pairs = _parse_env_lines()
        single_quoted: list[str] = []
        for key, value in pairs:
            if value.startswith("'") and value.endswith("'") and len(value) >= 2:
                single_quoted.append(f"{key}={value}")

        assert single_quoted == [], (
            f"Single-quoted values found (SkyPilot incompatible): {single_quoted}"
        )

    def test_no_export_prefix(self) -> None:
        """No line should start with 'export '."""
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        export_lines: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                export_lines.append(stripped)

        assert export_lines == [], (
            f"Lines with 'export ' prefix found (SkyPilot incompatible): {export_lines}"
        )

    def test_no_inline_comments(self) -> None:
        """No value should contain ' #' inline comment after the value.

        Heuristic: if value contains ' # ' (space-hash-space) or ends with
        ' #word', it is likely an inline comment. We check for ' #' preceded
        by a space, which catches 'value # comment' patterns.
        """
        pairs = _parse_env_lines()
        inline_comment_lines: list[str] = []
        for key, value in pairs:
            # Look for ' #' pattern in the value — indicates inline comment
            if " #" in value:
                inline_comment_lines.append(f"{key}={value}")

        assert inline_comment_lines == [], (
            f"Inline comments found (SkyPilot incompatible): {inline_comment_lines}"
        )

    def test_no_duplicate_keys(self) -> None:
        """All keys must be unique."""
        pairs = _parse_env_lines()
        keys = [key for key, _value in pairs]
        seen: dict[str, int] = {}
        duplicates: list[str] = []
        for key in keys:
            seen[key] = seen.get(key, 0) + 1
        for key, count in seen.items():
            if count > 1:
                duplicates.append(f"{key} (appears {count}x)")

        assert duplicates == [], (
            f"Duplicate keys found: {duplicates}"
        )

    def test_required_vars_have_non_empty_placeholder(self) -> None:
        """REQUIRED_ENV_VARS from preflight must each have a non-empty value."""
        pairs = _parse_env_lines()
        env_dict = dict(pairs)

        missing: list[str] = []
        empty: list[str] = []
        for var in REQUIRED_ENV_VARS:
            if var not in env_dict:
                missing.append(var)
            elif not env_dict[var].strip():
                empty.append(var)

        assert missing == [], (
            f"Required env vars missing from .env.example: {missing}"
        )
        assert empty == [], (
            f"Required env vars have empty values in .env.example: {empty}"
        )
