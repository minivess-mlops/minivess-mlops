"""Guard test: detect orphan config keys (Task 2.24).

Config keys defined in YAML/env files but never consumed by Python code
are maintenance debt. This test flags them so they can be wired or deleted.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src" / "minivess"


def _source_contains(needle: str) -> bool:
    """Check if any Python file in src/ references the given string."""
    for py_file in SRC_DIR.rglob("*.py"):
        try:
            if needle in py_file.read_text(encoding="utf-8"):
                return True
        except (OSError, UnicodeDecodeError):
            continue
    return False


class TestDashboardHealthThresholdsConsumed:
    """configs/dashboard/health_thresholds.yaml keys must be consumed."""

    def test_health_threshold_keys_documented_as_orphans(self) -> None:
        """Flag orphan keys in health_thresholds.yaml.

        These keys were created for QA monitoring but the threshold logic
        was never implemented. The health adapter only does binary checks.
        """
        yaml_path = REPO_ROOT / "configs" / "dashboard" / "health_thresholds.yaml"
        if not yaml_path.exists():
            return  # File was deleted — test passes

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        orphans = []
        for key in raw:
            if not _source_contains(key):
                orphans.append(key)

        # Document known orphans — when wired, remove from this list.
        known_orphans = {
            "drift_psi_threshold",
            "model_alpha_min",
            "flow_failure_window_hours",
            "latency_p99_ms_max",
        }
        unexpected_orphans = set(orphans) - known_orphans
        assert not unexpected_orphans, (
            f"New orphan keys in health_thresholds.yaml: {unexpected_orphans}. "
            f"Either wire them in Python code or add to known_orphans."
        )


class TestEnvExampleKeysConsumed:
    """Critical .env.example keys should be consumed somewhere."""

    def test_dashboard_refresh_interval_not_consumed(self) -> None:
        """DASHBOARD_REFRESH_INTERVAL_S is defined but never read."""
        # This is a known orphan — the dashboard has no polling logic.
        assert not _source_contains("DASHBOARD_REFRESH_INTERVAL_S")
