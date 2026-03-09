"""Tests for MONAI Label service configuration (T-ANN.1.1).

Validates that the MONAI Label Docker service is correctly configured:
- Port in .env.example
- Service in docker-compose.flows.yml
- MONAI Label app directory structure
- Dockerfile exists and installs monailabel
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]


class TestMonaiLabelConfig:
    """Verify MONAI Label service configuration files."""

    def test_monai_label_port_in_env_example(self) -> None:
        """MONAI_LABEL_PORT must be defined in .env.example."""
        env_example = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "MONAI_LABEL_PORT=" in env_example

    def test_monai_label_service_in_compose(self) -> None:
        """docker-compose.flows.yml must have a monai-label service."""
        compose_path = _REPO_ROOT / "deployment" / "docker-compose.flows.yml"
        with compose_path.open(encoding="utf-8") as f:
            compose = yaml.safe_load(f)
        services = compose.get("services", {})
        assert "monai-label" in services, (
            f"monai-label service missing from compose. Found: {list(services)}"
        )

    def test_monailabel_app_dir_exists(self) -> None:
        """deployment/monailabel/minivess_app/ directory must exist."""
        app_dir = _REPO_ROOT / "deployment" / "monailabel" / "minivess_app"
        assert app_dir.is_dir(), f"Missing: {app_dir}"

    def test_monailabel_app_has_main(self) -> None:
        """deployment/monailabel/minivess_app/main.py must exist."""
        main_py = _REPO_ROOT / "deployment" / "monailabel" / "minivess_app" / "main.py"
        assert main_py.is_file(), f"Missing: {main_py}"

    def test_monailabel_app_has_infer_module(self) -> None:
        """Champion infer module must exist."""
        infer_py = (
            _REPO_ROOT
            / "deployment"
            / "monailabel"
            / "minivess_app"
            / "lib"
            / "infers"
            / "champion_infer.py"
        )
        assert infer_py.is_file(), f"Missing: {infer_py}"


class TestMonaiLabelDockerfile:
    """Verify MONAI Label Dockerfile."""

    def test_monailabel_dockerfile_exists(self) -> None:
        """Dockerfile.monailabel must exist."""
        dockerfile = _REPO_ROOT / "deployment" / "docker" / "Dockerfile.monailabel"
        assert dockerfile.is_file(), f"Missing: {dockerfile}"

    def test_monailabel_dockerfile_installs_monailabel(self) -> None:
        """Dockerfile must install monailabel."""
        dockerfile = _REPO_ROOT / "deployment" / "docker" / "Dockerfile.monailabel"
        content = dockerfile.read_text(encoding="utf-8")
        assert "monailabel" in content.lower()

    def test_monailabel_port_not_hardcoded_in_dockerfile(self) -> None:
        """Port must come from env var, not hardcoded EXPOSE."""
        dockerfile = _REPO_ROOT / "deployment" / "docker" / "Dockerfile.monailabel"
        content = dockerfile.read_text(encoding="utf-8")
        # Should not have a hardcoded EXPOSE 8000
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("EXPOSE") and "8000" in stripped:
                pytest.fail(
                    "EXPOSE 8000 hardcoded in Dockerfile — port should come "
                    "from MONAI_LABEL_PORT env var via docker-compose"
                )
