"""Knowledge Graph YAML consistency tests (PR-3, T3.5).

Validates that specific KG data fixes are correct:
- agent_architecture.yaml: Pydantic AI posterior >= 0.85, LangGraph deprecated
- operations.yaml: compliance_posture or CISO Assistant note present
- All KG YAML files parse without errors

Closes: #848, #843
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
KG_ROOT = REPO_ROOT / "knowledge-graph"
DECISIONS_DIR = KG_ROOT / "decisions"
DOMAINS_DIR = KG_ROOT / "domains"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file safely."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# T3.5a: agent_architecture.yaml correctness
# ---------------------------------------------------------------------------


class TestAgentArchitectureKG:
    """Validate agent_architecture.yaml reflects Pydantic AI as winner."""

    @pytest.fixture()
    def agent_arch(self) -> dict:
        path = DECISIONS_DIR / "L2-architecture" / "agent_architecture.yaml"
        assert path.exists(), f"Missing: {path}"
        return _load_yaml(path)

    def test_resolved_option_is_pydantic_ai(self, agent_arch: dict) -> None:
        """resolved_option must be pydantic_ai_prefect."""
        assert agent_arch["resolved_option"] == "pydantic_ai_prefect"

    def test_pydantic_ai_posterior_high(self, agent_arch: dict) -> None:
        """Pydantic AI posterior must be >= 0.85."""
        options = {opt["id"]: opt for opt in agent_arch["options"]}
        assert options["pydantic_ai_prefect"]["posterior_probability"] >= 0.85

    def test_langgraph_posterior_low(self, agent_arch: dict) -> None:
        """LangGraph posterior must be <= 0.10 (deprecated)."""
        options = {opt["id"]: opt for opt in agent_arch["options"]}
        assert options["langgraph_stateful"]["posterior_probability"] <= 0.10

    def test_status_resolved(self, agent_arch: dict) -> None:
        """Status must be 'resolved'."""
        assert agent_arch["status"] == "resolved"

    def test_probabilities_sum_to_one(self, agent_arch: dict) -> None:
        """Posterior probabilities must sum to ~1.0."""
        total = sum(opt["posterior_probability"] for opt in agent_arch["options"])
        assert abs(total - 1.0) < 0.05, f"Posterior sum = {total}"


# ---------------------------------------------------------------------------
# T3.5b: operations.yaml CISO Assistant note
# ---------------------------------------------------------------------------


class TestOperationsKGCISOAssistant:
    """Validate CISO Assistant Community note in operations.yaml."""

    @pytest.fixture()
    def operations(self) -> dict:
        path = DOMAINS_DIR / "operations.yaml"
        assert path.exists(), f"Missing: {path}"
        return _load_yaml(path)

    def test_compliance_posture_has_ciso_note(self, operations: dict) -> None:
        """operations.yaml must reference CISO Assistant Community."""
        # Check in decisions or top-level notes
        raw_text = (DOMAINS_DIR / "operations.yaml").read_text(encoding="utf-8")
        assert "ciso" in raw_text.lower(), (
            "operations.yaml must mention CISO Assistant Community"
        )


# ---------------------------------------------------------------------------
# T3.5c: All KG YAML files parse correctly
# ---------------------------------------------------------------------------


class TestAllKGYAMLParseable:
    """Validate all KG YAML files are syntactically valid."""

    @pytest.fixture(scope="class")
    def all_yaml_files(self) -> list[Path]:
        return sorted(KG_ROOT.rglob("*.yaml"))

    def test_at_least_one_yaml_exists(self, all_yaml_files: list[Path]) -> None:
        """KG must have at least one YAML file."""
        assert len(all_yaml_files) > 0

    def test_all_files_parse(self, all_yaml_files: list[Path]) -> None:
        """Every KG YAML file must parse without error."""
        failures = []
        for path in all_yaml_files:
            try:
                _load_yaml(path)
            except yaml.YAMLError as exc:
                failures.append(f"{path.relative_to(KG_ROOT)}: {exc}")
        assert not failures, f"YAML parse errors: {failures}"
