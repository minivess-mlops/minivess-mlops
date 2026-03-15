"""Tests for GPU benchmark Docker infrastructure (T3.2, #651).

Validates:
- Dockerfile.benchmark exists
- docker-compose.flows.yml has gpu-benchmark service
- gpu-benchmark service has GPU device
- gpu-benchmark service has volume mount
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[4]
DOCKER_DIR = REPO_ROOT / "deployment" / "docker"
COMPOSE_FILE = REPO_ROOT / "deployment" / "docker-compose.flows.yml"


class TestDockerfileBenchmark:
    """Dockerfile.benchmark exists and is well-formed."""

    def test_dockerfile_exists(self) -> None:
        dockerfile = DOCKER_DIR / "Dockerfile.benchmark"
        assert dockerfile.exists(), f"Missing {dockerfile}"

    def test_dockerfile_has_from(self) -> None:
        dockerfile = DOCKER_DIR / "Dockerfile.benchmark"
        content = dockerfile.read_text(encoding="utf-8")
        assert "FROM minivess-base:latest" in content


class TestComposeService:
    """docker-compose.flows.yml has gpu-benchmark service."""

    def _load_compose(self) -> dict:
        return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))

    def test_compose_has_benchmark_service(self) -> None:
        compose = self._load_compose()
        assert "gpu-benchmark" in compose.get("services", {}), (
            "gpu-benchmark service missing from docker-compose.flows.yml"
        )

    def test_benchmark_service_has_gpu_device(self) -> None:
        compose = self._load_compose()
        service = compose["services"]["gpu-benchmark"]
        devices = service.get("devices", [])
        assert any("nvidia" in str(d) for d in devices), (
            "gpu-benchmark service must have NVIDIA GPU device"
        )

    def test_benchmark_service_has_volume_mount(self) -> None:
        compose = self._load_compose()
        service = compose["services"]["gpu-benchmark"]
        volumes = service.get("volumes", [])
        volume_str = str(volumes)
        assert "benchmark" in volume_str.lower(), (
            "gpu-benchmark service must mount benchmark_data volume"
        )
