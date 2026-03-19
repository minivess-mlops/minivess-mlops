"""Tests for CopilotKit Docker Compose service configuration (T4.1).

Validates that the docker-compose.yml contains a copilotkit service with
correct configuration for the agentic dashboard.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_compose(filename: str = "docker-compose.yml") -> dict:
    """Load and parse a docker-compose YAML file from the deployment directory."""
    compose_path = Path(__file__).resolve().parents[3] / "deployment" / filename
    return yaml.safe_load(compose_path.read_text(encoding="utf-8"))


class TestCopilotKitService:
    """Validate CopilotKit service in docker-compose.yml."""

    def test_copilotkit_service_exists(self) -> None:
        """CopilotKit service must exist in docker-compose.yml."""
        compose = _load_compose()
        services = compose.get("services", {})
        assert "copilotkit" in services, (
            "copilotkit service not found in docker-compose.yml"
        )

    def test_copilotkit_has_correct_profile(self) -> None:
        """CopilotKit service must be in the 'full' profile."""
        compose = _load_compose()
        service = compose["services"]["copilotkit"]
        profiles = service.get("profiles", [])
        assert "full" in profiles, (
            f"copilotkit must be in 'full' profile, got {profiles}"
        )

    def test_copilotkit_exposes_port(self) -> None:
        """CopilotKit service must expose a port via .env.example variable."""
        compose = _load_compose()
        service = compose["services"]["copilotkit"]
        ports = service.get("ports", [])
        assert len(ports) > 0, "copilotkit must expose at least one port"

    def test_copilotkit_on_minivess_network(self) -> None:
        """CopilotKit service must be on minivess network."""
        compose = _load_compose()
        service = compose["services"]["copilotkit"]
        networks = service.get("networks", [])
        assert "minivess" in networks, (
            f"copilotkit must be on minivess network, got {networks}"
        )

    def test_copilotkit_has_container_name(self) -> None:
        """CopilotKit service must have a container_name."""
        compose = _load_compose()
        service = compose["services"]["copilotkit"]
        assert "container_name" in service, "copilotkit must have container_name"
        assert "minivess" in service["container_name"], (
            "container_name must contain 'minivess'"
        )

    def test_copilotkit_has_restart_policy(self) -> None:
        """CopilotKit service must have a restart policy."""
        compose = _load_compose()
        service = compose["services"]["copilotkit"]
        assert "restart" in service, "copilotkit must have restart policy"

    def test_copilotkit_has_common_labels(self) -> None:
        """CopilotKit service must use common labels."""
        compose = _load_compose()
        service = compose["services"]["copilotkit"]
        labels = service.get("labels", {})
        assert labels.get("project") == "minivess-mlops", (
            "copilotkit must have project label"
        )
