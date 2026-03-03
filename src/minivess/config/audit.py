"""Hydra/YAML config audit utilities.

Provides functions to discover, validate, and report on experiment
YAML config files. Used by scripts/audit_hydra_configs.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Required fields for experiment configs
_REQUIRED_EXPERIMENT_FIELDS: list[str] = [
    "experiment_name",
    "model_family",
]


def discover_config_files(
    config_dir: Path | str,
    *,
    recursive: bool = True,
) -> list[Path]:
    """Discover YAML config files in a directory.

    Parameters
    ----------
    config_dir:
        Directory to search.
    recursive:
        If True, search subdirectories recursively.

    Returns
    -------
    Sorted list of YAML file paths.
    """
    from pathlib import Path as _Path

    config_dir = _Path(config_dir)
    pattern = "**/*.yaml" if recursive else "*.yaml"
    files = sorted(config_dir.glob(pattern))
    return [f for f in files if f.is_file()]


def validate_experiment_config(
    config: dict[str, Any],
    *,
    config_path: Path,
) -> list[dict[str, Any]]:
    """Validate an experiment config against required fields.

    Parameters
    ----------
    config:
        Parsed YAML config dict.
    config_path:
        Path to the config file (for error messages).

    Returns
    -------
    List of issue dicts with ``file``, ``field``, ``severity``, ``message``.
    """
    issues: list[dict[str, Any]] = []

    for field in _REQUIRED_EXPERIMENT_FIELDS:
        if field not in config:
            issues.append(
                {
                    "file": str(config_path.name),
                    "field": field,
                    "severity": "error",
                    "message": f"Missing required field: {field}",
                }
            )

    # Check for empty losses list
    losses = config.get("losses")
    if losses is not None and len(losses) == 0:
        issues.append(
            {
                "file": str(config_path.name),
                "field": "losses",
                "severity": "warning",
                "message": "Empty losses list",
            }
        )

    return issues


def generate_audit_report(issues: list[dict[str, Any]]) -> str:
    """Generate a human-readable audit report from validation issues.

    Parameters
    ----------
    issues:
        List of issue dicts from validate_experiment_config().

    Returns
    -------
    Markdown-formatted report string.
    """
    if not issues:
        return "# Config Audit Report\n\nAll configs valid. No issues found."

    lines = ["# Config Audit Report", ""]
    for issue in issues:
        severity = issue.get("severity", "info").upper()
        file_name = issue.get("file", "unknown")
        field = issue.get("field", "")
        message = issue.get("message", "")
        lines.append(f"- [{severity}] **{file_name}** → `{field}`: {message}")

    error_count = sum(1 for i in issues if i.get("severity") == "error")
    warn_count = sum(1 for i in issues if i.get("severity") == "warning")
    lines.extend(
        [
            "",
            f"## Summary: {error_count} errors, {warn_count} warnings",
        ]
    )

    return "\n".join(lines)
