#!/usr/bin/env python
"""Audit Hydra/YAML experiment configs for missing fields and issues.

Usage::

    uv run python scripts/audit_hydra_configs.py
    uv run python scripts/audit_hydra_configs.py --configs-dir configs/experiments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minivess.config.audit import (
    discover_config_files,
    generate_audit_report,
    validate_experiment_config,
)


def main() -> None:
    """Run config audit."""
    parser = argparse.ArgumentParser(description="Audit Hydra experiment configs")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments",
        help="Directory containing experiment YAML files",
    )
    args = parser.parse_args()

    configs_dir = args.configs_dir
    if not configs_dir.exists():
        print(f"Config directory not found: {configs_dir}")
        sys.exit(1)

    files = discover_config_files(configs_dir)
    print(f"Found {len(files)} config files in {configs_dir}")

    all_issues: list[dict] = []
    for config_path in files:
        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            all_issues.append(
                {
                    "file": config_path.name,
                    "field": "yaml",
                    "severity": "error",
                    "message": f"YAML parse error: {e}",
                }
            )
            continue

        issues = validate_experiment_config(config, config_path=config_path)
        all_issues.extend(issues)

    report = generate_audit_report(all_issues)
    print(report)

    # Exit with error if any errors found
    error_count = sum(1 for i in all_issues if i.get("severity") == "error")
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
