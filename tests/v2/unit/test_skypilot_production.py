"""Tests for production SkyPilot training YAMLs.

Validates that train_production.yaml and train_hpo.yaml use Docker image_id,
have no banned commands, and include all required credentials and recovery.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SKYPILOT_DIR = Path("deployment/skypilot")
_PRODUCTION_YAML = _SKYPILOT_DIR / "train_production.yaml"
_HPO_YAML = _SKYPILOT_DIR / "train_hpo.yaml"

# Commands banned in setup: sections (Docker mandate)
_BANNED_COMMANDS = ["apt-get", "uv sync", "git clone", "pip install", "conda install"]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestProductionTrainingYaml:
    """Validate train_production.yaml for full training runs."""

    def test_production_yaml_exists(self) -> None:
        """train_production.yaml must exist."""
        assert _PRODUCTION_YAML.exists(), (
            "Missing deployment/skypilot/train_production.yaml"
        )

    def test_production_yaml_uses_docker_image(self) -> None:
        """Resources must specify Docker image via image_id."""
        config = _load(_PRODUCTION_YAML)
        resources = config.get("resources", {})
        image_id = resources.get("image_id", "")
        assert str(image_id).startswith("docker:"), (
            f"Must use Docker image_id, got: {image_id}"
        )

    def test_production_yaml_has_no_banned_commands(self) -> None:
        """Setup must NOT contain banned bare-VM commands."""
        config = _load(_PRODUCTION_YAML)
        setup = config.get("setup", "")
        for cmd in _BANNED_COMMANDS:
            assert cmd not in setup, (
                f"BANNED command '{cmd}' in setup. Setup is DATA ONLY."
            )

    def test_production_yaml_has_spot_recovery(self) -> None:
        """Resources must use spot instances."""
        config = _load(_PRODUCTION_YAML)
        resources = config.get("resources", {})
        assert resources.get("use_spot") is True, (
            "Must use spot instances for cost savings"
        )

    def test_production_yaml_has_dvc_remote(self) -> None:
        """Envs must use remote_storage DVC remote (AWS S3 public — UpCloud archived 2026-03-16)."""
        config = _load(_PRODUCTION_YAML)
        envs = config.get("envs", {})
        assert "DVC_REMOTE" in envs, "Missing DVC_REMOTE in envs"
        assert envs["DVC_REMOTE"] == "remote_storage", (
            f"DVC_REMOTE should be 'remote_storage', got: {envs['DVC_REMOTE']}"
        )
        # Stale UpCloud credentials must be absent
        for stale in ("DVC_S3_ENDPOINT_URL", "DVC_S3_ACCESS_KEY", "DVC_S3_SECRET_KEY"):
            assert stale not in envs, (
                f"Stale UpCloud var {stale} in production YAML — UpCloud archived 2026-03-16"
            )

    def test_production_yaml_has_mlflow_tracking_uri(self) -> None:
        """Envs must include MLflow tracking URI (file-based — UpCloud server archived 2026-03-16)."""
        config = _load(_PRODUCTION_YAML)
        envs = config.get("envs", {})
        assert "MLFLOW_TRACKING_URI" in envs, "Missing MLFLOW_TRACKING_URI in envs"
        # UpCloud remote server credentials no longer needed
        for stale in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
            assert stale not in envs, (
                f"Stale UpCloud MLflow credential {stale} in production YAML — "
                "UpCloud archived 2026-03-16, file-based MLflow needs no auth"
            )

    def test_production_yaml_has_full_training_params(self) -> None:
        """Envs must support full training (not just smoke test)."""
        config = _load(_PRODUCTION_YAML)
        envs = config.get("envs", {})
        assert "CHECKPOINT_DIR" in envs
        assert "SPLITS_DIR" in envs
        assert "LOGS_DIR" in envs

    def test_production_yaml_uses_full_splits(self) -> None:
        """Setup must use full 3-fold splits, not smoke test splits."""
        config = _load(_PRODUCTION_YAML)
        setup = config.get("setup", "")
        assert "smoke_test_1fold_4vol" not in setup, (
            "Production YAML must not use smoke test splits"
        )


class TestHpoTrainingYaml:
    """Validate train_hpo.yaml for hyperparameter optimization."""

    def test_hpo_yaml_exists(self) -> None:
        """train_hpo.yaml must exist."""
        assert _HPO_YAML.exists(), "Missing deployment/skypilot/train_hpo.yaml"

    def test_hpo_yaml_uses_docker_image(self) -> None:
        """Resources must specify Docker image via image_id."""
        config = _load(_HPO_YAML)
        resources = config.get("resources", {})
        image_id = resources.get("image_id", "")
        assert str(image_id).startswith("docker:"), (
            f"Must use Docker image_id, got: {image_id}"
        )

    def test_hpo_yaml_has_no_banned_commands(self) -> None:
        """Setup must NOT contain banned bare-VM commands."""
        config = _load(_HPO_YAML)
        setup = config.get("setup", "")
        for cmd in _BANNED_COMMANDS:
            assert cmd not in setup, (
                f"BANNED command '{cmd}' in setup. Setup is DATA ONLY."
            )
