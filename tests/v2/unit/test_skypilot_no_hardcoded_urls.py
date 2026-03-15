"""Tests for no hardcoded URLs in SkyPilot YAML configs.

T0.6: All service URLs must use ${ENV_VAR} references, not hardcoded values.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SKYPILOT_DIR = Path("deployment/skypilot")


def _all_skypilot_yamls() -> list[Path]:
    """Find all SkyPilot YAML files."""
    return sorted(_SKYPILOT_DIR.glob("*.yaml"))


class TestNoHardcodedUrls:
    """SkyPilot YAMLs must not contain hardcoded service URLs."""

    def test_no_hardcoded_mlflow_urls(self) -> None:
        """No SkyPilot YAML should hardcode an MLflow tracking URI."""
        for yaml_path in _all_skypilot_yamls():
            content = yaml_path.read_text(encoding="utf-8")
            config = yaml.safe_load(content)
            envs = config.get("envs", {})
            tracking_uri = envs.get("MLFLOW_TRACKING_URI", "")
            tracking_str = str(tracking_uri)
            # Must be a ${VAR} reference, not a hardcoded URL
            assert not tracking_str.startswith("http"), (
                f"{yaml_path.name}: MLFLOW_TRACKING_URI is hardcoded to "
                f"'{tracking_str}'. Use ${{ENV_VAR}} reference instead."
            )

    def test_no_hardcoded_cloud_run_urls(self) -> None:
        """No SkyPilot YAML should contain hardcoded Cloud Run URLs."""
        for yaml_path in _all_skypilot_yamls():
            content = yaml_path.read_text(encoding="utf-8")
            assert (
                "run.app" not in content
                or "${" in content.split("run.app")[0].split("\n")[-1]
            ), (
                f"{yaml_path.name}: Contains hardcoded Cloud Run URL. "
                f"Use ${{ENV_VAR}} reference instead."
            )
