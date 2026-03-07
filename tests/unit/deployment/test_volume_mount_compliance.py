"""Tests for Docker Compose volume mount compliance (T-09).

Parses docker-compose.flows.yml and verifies:
- Train service has checkpoint and mlruns volumes
- All ENV paths map to volume-mounted directories
- No service uses /tmp for persistent output

Uses yaml.safe_load (CLAUDE.md Rule #16 — no regex for structured data).

References:
  - docs/planning/minivess-vision-enforcement-plan-execution.xml (T-09)
  - CLAUDE.md Rule #18 (volume mounts)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_FILE = (
    Path(__file__).resolve().parents[3] / "deployment" / "docker-compose.flows.yml"
)


@pytest.fixture()
def compose_config() -> dict:
    """Load docker-compose.flows.yml as a dict."""
    if not COMPOSE_FILE.exists():
        pytest.skip("docker-compose.flows.yml not found")
    with COMPOSE_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestTrainServiceVolumes:
    """Verify train service has required volume mounts."""

    def test_train_service_has_checkpoint_volume(self, compose_config: dict) -> None:
        """Train service must mount a volume for checkpoints."""
        train = compose_config["services"]["train"]
        volumes = train.get("volumes", [])
        volume_strs = [str(v) for v in volumes]
        has_checkpoint = any("/app/checkpoints" in v for v in volume_strs)
        assert has_checkpoint, (
            f"Train service missing checkpoint volume mount. Volumes: {volume_strs}"
        )

    def test_train_service_has_mlruns_volume(self, compose_config: dict) -> None:
        """Train service must mount a volume for MLflow runs."""
        train = compose_config["services"]["train"]
        volumes = train.get("volumes", [])
        volume_strs = [str(v) for v in volumes]
        has_mlruns = any("/app/mlruns" in v or "mlruns" in v for v in volume_strs)
        assert has_mlruns, (
            f"Train service missing mlruns volume mount. Volumes: {volume_strs}"
        )

    def test_train_service_has_data_volume(self, compose_config: dict) -> None:
        """Train service must mount a volume for data."""
        train = compose_config["services"]["train"]
        volumes = train.get("volumes", [])
        volume_strs = [str(v) for v in volumes]
        has_data = any("/app/data" in v for v in volume_strs)
        assert has_data, (
            f"Train service missing data volume mount. Volumes: {volume_strs}"
        )

    def test_train_service_has_logs_volume(self, compose_config: dict) -> None:
        """Train service must mount a volume for logs."""
        train = compose_config["services"]["train"]
        volumes = train.get("volumes", [])
        volume_strs = [str(v) for v in volumes]
        has_logs = any("/app/logs" in v for v in volume_strs)
        assert has_logs, (
            f"Train service missing logs volume mount. Volumes: {volume_strs}"
        )


class TestEnvPathsAreMounted:
    """Verify all ENV-declared paths map to volume mounts."""

    def test_all_env_paths_are_mounted(self, compose_config: dict) -> None:
        """Every *_DIR env var in train service must have a corresponding volume."""
        train = compose_config["services"]["train"]
        env = train.get("environment", {})
        volumes = [str(v) for v in train.get("volumes", [])]

        # Collect all env vars that look like directory paths
        dir_envs = {
            k: v
            for k, v in env.items()
            if isinstance(v, str) and k.endswith("_DIR") and v.startswith("/app/")
        }

        unmounted: list[str] = []
        for env_name, env_path in dir_envs.items():
            if not any(env_path in vol for vol in volumes):
                unmounted.append(f"{env_name}={env_path}")

        assert not unmounted, (
            f"ENV paths without volume mounts in train service: {unmounted}"
        )


class TestNoTmpInEnvPaths:
    """Verify no service uses /tmp for persistent output."""

    def test_no_tmp_in_env_paths(self, compose_config: dict) -> None:
        """No service should use /tmp in env path variables."""
        violations: list[str] = []
        for svc_name, svc in compose_config.get("services", {}).items():
            env = svc.get("environment", {})
            for k, v in env.items():
                if isinstance(v, str) and v.startswith("/tmp"):
                    violations.append(f"{svc_name}.{k}={v}")
        assert not violations, (
            f"Services using /tmp for paths (forbidden — CLAUDE.md Rule #18): {violations}"
        )
