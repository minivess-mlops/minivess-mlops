"""Tests for Falco runtime security monitoring — T-05.1 / issue #551.

Falco is an optional eBPF-based runtime security monitor activated via
docker compose --profile security. It requires privileged: true (documented).

Verifies:
- deployment/falco/minivess_rules.yaml exists and is valid YAML
- Rules target minivess containers and cover ML-specific threat scenarios
- 'falco' service in docker-compose.yml under profile 'security'
- No other service becomes privileged (regression guard)
- deployment/CLAUDE.md documents the Falco security profile

Rule #16: yaml.safe_load(), pathlib.Path — no regex.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
FALCO_RULES = ROOT / "deployment" / "falco" / "minivess_rules.yaml"
COMPOSE_YML = ROOT / "deployment" / "docker-compose.yml"
DEPLOYMENT_CLAUDE_MD = ROOT / "deployment" / "CLAUDE.md"
FALCO_COMPOSE_SERVICE = "falco"


def test_falco_rules_file_exists() -> None:
    assert FALCO_RULES.exists(), (
        "deployment/falco/minivess_rules.yaml not found. "
        "Create ML-specific Falco rules for model exfiltration, shell spawns, etc."
    )


def test_falco_rules_is_valid_yaml() -> None:
    rules = yaml.safe_load(FALCO_RULES.read_text(encoding="utf-8"))
    assert rules is not None, (
        "deployment/falco/minivess_rules.yaml is empty or invalid YAML."
    )


def test_falco_rules_has_minivess_specific_rules() -> None:
    content = FALCO_RULES.read_text(encoding="utf-8")
    assert "minivess" in content.lower(), (
        "Falco rules must target minivess containers "
        "(condition referencing container.name startswith 'minivess-flow')."
    )
    assert "checkpoints" in content or "checkpoint" in content, (
        "Falco rules must cover model weight exfiltration (checkpoint path monitoring)."
    )
    assert "spawned" in content.lower() or "shell" in content.lower(), (
        "Falco rules must cover shell spawn detection (deserialization attack vector)."
    )


def test_falco_service_in_compose() -> None:
    compose = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    services = compose.get("services", {})
    assert FALCO_COMPOSE_SERVICE in services, (
        "docker-compose.yml missing 'falco' service. "
        "Add under profile 'security' with privileged: true."
    )
    falco = services[FALCO_COMPOSE_SERVICE]
    assert "security" in falco.get("profiles", []), (
        "falco service must be under profile 'security'. "
        "Default docker compose up must NOT start it."
    )


def test_falco_service_is_the_only_privileged_service() -> None:
    """Regression: adding Falco (privileged) must not cause other services to be privileged."""
    compose = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    for name, svc in compose.get("services", {}).items():
        if name == FALCO_COMPOSE_SERVICE:
            continue  # Falco intentionally requires privileged: true for eBPF
        assert not svc.get("privileged", False), (
            f"Service '{name}' is unexpectedly privileged after Falco addition. "
            "Only the falco service may use privileged: true."
        )


def test_falco_deployment_claude_md_documents_security_profile() -> None:
    content = DEPLOYMENT_CLAUDE_MD.read_text(encoding="utf-8")
    assert "falco" in content.lower(), (
        "deployment/CLAUDE.md must document the Falco runtime security monitoring service."
    )
    assert "--profile security" in content or "profile security" in content.lower(), (
        "deployment/CLAUDE.md must document 'docker compose --profile security up falco'."
    )
