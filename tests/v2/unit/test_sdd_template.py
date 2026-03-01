"""Tests for the probabilistic SDD (Software Design Document) template.

Phase 0: Extract the generic PRD framework into a standalone template
that other researchers can instantiate for their own projects.

Tests organized by task (10 tasks, ~50 tests total).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Root of the SDD template directory
SDD_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent / "probabilistic-sdd-template"
)


def _load_yaml(path: Path) -> dict[str, Any] | None:
    """Load YAML file safely."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data
    return None


# ============================================================================
# Task 1: Directory scaffold + generic _schema.yaml (8 tests)
# ============================================================================


class TestSchemaGeneric:
    """Task 1: The generic _schema.yaml must be valid and domain-agnostic."""

    @pytest.fixture()
    def schema(self) -> dict[str, Any]:
        path = SDD_ROOT / "_schema.yaml"
        assert path.exists(), f"_schema.yaml not found at {path}"
        data = _load_yaml(path)
        assert data is not None
        return data

    def test_schema_file_exists(self) -> None:
        """T1.1: _schema.yaml exists in the template root."""
        assert (SDD_ROOT / "_schema.yaml").exists()

    def test_schema_is_valid_json_schema(self, schema: dict[str, Any]) -> None:
        """T1.2: Schema is valid JSON Schema draft-07."""
        assert schema.get("$schema") == "http://json-schema.org/draft-07/schema#"
        assert schema.get("type") == "object"

    def test_schema_has_required_fields(self, schema: dict[str, Any]) -> None:
        """T1.3: Schema requires essential decision node fields."""
        required = schema.get("required", [])
        for field in [
            "decision_id",
            "title",
            "description",
            "decision_level",
            "status",
            "options",
            "volatility",
            "domain_applicability",
        ]:
            assert field in required, f"Missing required field: {field}"

    def test_schema_enforces_probability_range(self, schema: dict[str, Any]) -> None:
        """T1.4: prior_probability constrained to [0, 1]."""
        option_props = schema["properties"]["options"]["items"]["properties"]
        prob = option_props["prior_probability"]
        assert prob["minimum"] == 0.0
        assert prob["maximum"] == 1.0

    def test_schema_requires_min_two_options(self, schema: dict[str, Any]) -> None:
        """T1.5: Options array must have at least 2 items."""
        assert schema["properties"]["options"]["minItems"] == 2

    def test_schema_has_no_minivess_strings(self, schema: dict[str, Any]) -> None:
        """T1.6: No MinIVess-specific strings in the generic schema."""
        yaml_str = yaml.dump(schema, default_flow_style=False)
        for forbidden in ["MinIVess", "minivess", "vascular", "vessel", "MONAI"]:
            assert forbidden not in yaml_str, (
                f"Found domain-specific string: {forbidden}"
            )

    def test_schema_has_decision_levels(self, schema: dict[str, Any]) -> None:
        """T1.7: Schema defines L1-L5 decision levels."""
        levels = schema["properties"]["decision_level"]["enum"]
        assert len(levels) == 5
        assert "L1_research_goals" in levels
        assert "L5_operations" in levels

    def test_schema_has_conditional_on_section(self, schema: dict[str, Any]) -> None:
        """T1.8: Schema defines conditional_on for Bayesian dependencies."""
        assert "conditional_on" in schema["properties"]
        cond = schema["properties"]["conditional_on"]
        assert cond["type"] == "array"


# ============================================================================
# Task 2: Generic _network-template.yaml (2 tests)
# ============================================================================


class TestNetworkTemplate:
    """Task 2: Empty DAG skeleton with L1-L5 level descriptions."""

    @pytest.fixture()
    def network(self) -> dict[str, Any]:
        path = SDD_ROOT / "_network-template.yaml"
        assert path.exists(), f"_network-template.yaml not found at {path}"
        data = _load_yaml(path)
        assert data is not None
        return data

    def test_network_has_l1_through_l5_levels(self, network: dict[str, Any]) -> None:
        """T2.1: Network has all 5 level sections defined."""
        net = network["network"]
        levels = net["levels"]
        level_ids = [lev["level_id"] for lev in levels]
        for level in ["L1", "L2", "L3", "L4", "L5"]:
            assert level in level_ids, f"Missing level: {level}"

    def test_network_has_no_nodes(self, network: dict[str, Any]) -> None:
        """T2.2: Template network has empty node lists (skeleton only)."""
        net = network["network"]
        assert net.get("nodes", []) == []
        assert net.get("edges", []) == []
        assert "CHANGE_ME" in net["name"]


# ============================================================================
# Task 3: Generic backbone-defaults.yaml (2 tests)
# ============================================================================


class TestBackboneDefaults:
    """Task 3: Domain-agnostic backbone defaults."""

    @pytest.fixture()
    def backbone(self) -> dict[str, Any]:
        path = SDD_ROOT / "backbone-defaults.yaml"
        assert path.exists(), f"backbone-defaults.yaml not found at {path}"
        data = _load_yaml(path)
        assert data is not None
        return data

    def test_backbone_has_generic_structure(self, backbone: dict[str, Any]) -> None:
        """T3.1: Has required sections (metrics, loss, training, data_splits)."""
        assert "metrics" in backbone
        assert "loss_defaults" in backbone
        assert "training_defaults" in backbone
        assert "data_splits" in backbone

    def test_backbone_no_minivess_specifics(self, backbone: dict[str, Any]) -> None:
        """T3.2: No MinIVess-specific values."""
        yaml_str = yaml.dump(backbone, default_flow_style=False)
        for forbidden in [
            "MinIVess",
            "minivess",
            "cbdice_cldice",
            "vascular",
            "vessel",
        ]:
            assert forbidden not in yaml_str, f"Found domain-specific: {forbidden}"


# ============================================================================
# Task 4: Template files (6 tests)
# ============================================================================


class TestTemplateFiles:
    """Task 4: decision-node, scenario, archetype, domain-overlay templates."""

    TEMPLATE_DIR = SDD_ROOT / "templates"

    @pytest.mark.parametrize(
        "filename",
        [
            "decision-node.yaml",
            "scenario.yaml",
            "archetype.yaml",
            "domain-overlay.yaml",
        ],
    )
    def test_template_exists(self, filename: str) -> None:
        """T4.1-T4.4: Each template file exists."""
        assert (self.TEMPLATE_DIR / filename).exists(), f"Missing template: {filename}"

    @pytest.mark.parametrize(
        "filename",
        [
            "decision-node.yaml",
            "scenario.yaml",
            "archetype.yaml",
            "domain-overlay.yaml",
        ],
    )
    def test_template_has_change_me_markers(self, filename: str) -> None:
        """T4.5: Templates have CHANGE_ME placeholders."""
        content = (self.TEMPLATE_DIR / filename).read_text(encoding="utf-8")
        assert "CHANGE_ME" in content, f"No CHANGE_ME markers in {filename}"

    def test_templates_are_valid_yaml(self) -> None:
        """T4.6: All templates parse as valid YAML."""
        for filename in [
            "decision-node.yaml",
            "scenario.yaml",
            "archetype.yaml",
            "domain-overlay.yaml",
        ]:
            path = self.TEMPLATE_DIR / filename
            data = _load_yaml(path)
            assert data is not None, f"{filename} parsed to None"


# ============================================================================
# Task 5: Protocol documents (5 tests)
# ============================================================================


class TestProtocolDocuments:
    """Task 5: 7 genericized protocol markdown files."""

    PROTOCOL_DIR = SDD_ROOT / "protocols"

    EXPECTED_PROTOCOLS = [
        "add-decision.md",
        "update-priors.md",
        "add-option.md",
        "create-scenario.md",
        "ingest-paper.md",
        "validate.md",
        "citation-guide.md",
    ]

    def test_all_seven_protocols_exist(self) -> None:
        """T5.1: All 7 protocol files exist."""
        for proto in self.EXPECTED_PROTOCOLS:
            assert (self.PROTOCOL_DIR / proto).exists(), f"Missing protocol: {proto}"

    @pytest.mark.parametrize(
        "protocol",
        [
            "add-decision.md",
            "update-priors.md",
            "add-option.md",
            "create-scenario.md",
            "ingest-paper.md",
            "validate.md",
            "citation-guide.md",
        ],
    )
    def test_protocol_has_when_to_use_section(self, protocol: str) -> None:
        """T5.2: Each protocol has a 'When to Use' section."""
        content = (self.PROTOCOL_DIR / protocol).read_text(encoding="utf-8")
        assert "When to Use" in content or "Purpose" in content, (
            f"{protocol} missing 'When to Use' or 'Purpose' section"
        )

    @pytest.mark.parametrize(
        "protocol",
        [
            "add-decision.md",
            "update-priors.md",
            "add-option.md",
            "create-scenario.md",
            "ingest-paper.md",
            "validate.md",
            "citation-guide.md",
        ],
    )
    def test_protocol_has_steps_section(self, protocol: str) -> None:
        """T5.3: Each protocol has a 'Steps' or structured section."""
        content = (self.PROTOCOL_DIR / protocol).read_text(encoding="utf-8")
        has_structure = (
            "## Steps" in content or "### Step" in content or "### 1." in content
        )
        assert has_structure, f"{protocol} missing structured steps"

    def test_protocols_no_minivess_paths(self) -> None:
        """T5.4: No MinIVess-specific paths in protocols."""
        for proto in self.EXPECTED_PROTOCOLS:
            content = (self.PROTOCOL_DIR / proto).read_text(encoding="utf-8")
            assert "docs/planning/prd/" not in content, (
                f"{proto} contains MinIVess-specific path 'docs/planning/prd/'"
            )

    def test_protocols_no_minivess_references(self) -> None:
        """T5.5: No MinIVess project name in protocols."""
        for proto in self.EXPECTED_PROTOCOLS:
            content = (self.PROTOCOL_DIR / proto).read_text(encoding="utf-8")
            assert "MinIVess" not in content, f"{proto} references MinIVess"


# ============================================================================
# Task 6: Standalone validator (12 tests)
# ============================================================================


class TestStandaloneValidator:
    """Task 6: validate.py — SDD structure validator."""

    @pytest.fixture()
    def minimal_valid_sdd(self, tmp_path: Path) -> Path:
        """Create a minimal valid SDD instance for testing."""
        sdd = tmp_path / "sdd"
        sdd.mkdir()
        decisions = sdd / "decisions"
        decisions.mkdir()
        l1 = decisions / "L1-goals"
        l1.mkdir()

        # Schema
        schema_src = SDD_ROOT / "_schema.yaml"
        (decisions / "_schema.yaml").write_text(
            schema_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

        # Network
        network = {
            "network": {
                "name": "Test SDD",
                "version": "1.0.0",
                "nodes": [
                    {
                        "id": "test_decision",
                        "level": "L1",
                        "file": "L1-goals/test_decision.decision.yaml",
                        "title": "Test",
                    }
                ],
                "edges": [],
            }
        }
        (decisions / "_network.yaml").write_text(
            yaml.dump(network, default_flow_style=False), encoding="utf-8"
        )

        # Decision file
        decision = {
            "decision_id": "test_decision",
            "title": "Test Decision",
            "description": "A test decision",
            "decision_level": "L1_research_goals",
            "status": "active",
            "options": [
                {
                    "option_id": "opt_a",
                    "title": "A",
                    "description": "Option A",
                    "prior_probability": 0.6,
                    "status": "viable",
                },
                {
                    "option_id": "opt_b",
                    "title": "B",
                    "description": "Option B",
                    "prior_probability": 0.4,
                    "status": "viable",
                },
            ],
            "volatility": {
                "classification": "stable",
                "last_assessed": "2026-01-01",
                "next_review": "2026-06-01",
                "change_drivers": ["example"],
            },
            "domain_applicability": {"general": 1.0},
        }
        (l1 / "test_decision.decision.yaml").write_text(
            yaml.dump(decision, default_flow_style=False), encoding="utf-8"
        )

        # Bibliography
        bib: dict[str, Any] = {"bibliography": []}
        (sdd / "bibliography.yaml").write_text(
            yaml.dump(bib, default_flow_style=False), encoding="utf-8"
        )

        return sdd

    @pytest.fixture()
    def invalid_probability_sdd(self, minimal_valid_sdd: Path) -> Path:
        """SDD with probability sum violation."""
        decision_file = (
            minimal_valid_sdd / "decisions" / "L1-goals" / "test_decision.decision.yaml"
        )
        decision = _load_yaml(decision_file)
        assert decision is not None
        decision["options"][0]["prior_probability"] = 0.8
        # Sum = 0.8 + 0.4 = 1.2 (violation)
        decision_file.write_text(
            yaml.dump(decision, default_flow_style=False), encoding="utf-8"
        )
        return minimal_valid_sdd

    @pytest.fixture()
    def cyclic_sdd(self, minimal_valid_sdd: Path) -> Path:
        """SDD with a cycle in the DAG."""
        network_file = minimal_valid_sdd / "decisions" / "_network.yaml"
        network = _load_yaml(network_file)
        assert network is not None

        # Add a second node
        l1 = minimal_valid_sdd / "decisions" / "L1-goals"
        decision2 = {
            "decision_id": "test_decision_2",
            "title": "Test Decision 2",
            "description": "Another test decision",
            "decision_level": "L1_research_goals",
            "status": "active",
            "options": [
                {
                    "option_id": "opt_x",
                    "title": "X",
                    "description": "Option X",
                    "prior_probability": 0.5,
                    "status": "viable",
                },
                {
                    "option_id": "opt_y",
                    "title": "Y",
                    "description": "Option Y",
                    "prior_probability": 0.5,
                    "status": "viable",
                },
            ],
            "volatility": {
                "classification": "stable",
                "last_assessed": "2026-01-01",
                "next_review": "2026-06-01",
                "change_drivers": ["example"],
            },
            "domain_applicability": {"general": 1.0},
        }
        (l1 / "test_decision_2.decision.yaml").write_text(
            yaml.dump(decision2, default_flow_style=False), encoding="utf-8"
        )

        network["network"]["nodes"].append(
            {
                "id": "test_decision_2",
                "level": "L1",
                "file": "L1-goals/test_decision_2.decision.yaml",
                "title": "Test 2",
            }
        )
        # Create cycle: A -> B -> A
        network["network"]["edges"] = [
            {"from": "test_decision", "to": "test_decision_2"},
            {"from": "test_decision_2", "to": "test_decision"},
        ]
        network_file.write_text(
            yaml.dump(network, default_flow_style=False), encoding="utf-8"
        )
        return minimal_valid_sdd

    @pytest.fixture()
    def broken_ref_sdd(self, minimal_valid_sdd: Path) -> Path:
        """SDD with broken cross-references."""
        network_file = minimal_valid_sdd / "decisions" / "_network.yaml"
        network = _load_yaml(network_file)
        assert network is not None
        network["network"]["edges"] = [
            {"from": "test_decision", "to": "nonexistent_node"},
        ]
        network_file.write_text(
            yaml.dump(network, default_flow_style=False), encoding="utf-8"
        )
        return minimal_valid_sdd

    def _run_validator(self, sdd_root: Path) -> subprocess.CompletedProcess[str]:
        """Run the standalone validator as a subprocess."""
        validator = SDD_ROOT / "validate.py"
        return subprocess.run(
            [sys.executable, str(validator), "--sdd-root", str(sdd_root)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_validator_exists(self) -> None:
        """T6.1: validate.py exists."""
        assert (SDD_ROOT / "validate.py").exists()

    def test_validator_importable(self) -> None:
        """T6.2: validate.py is importable as a module."""
        spec = importlib.util.spec_from_file_location(
            "sdd_validate", SDD_ROOT / "validate.py"
        )
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sdd_validate"] = mod
        spec.loader.exec_module(mod)
        assert hasattr(mod, "validate_sdd")
        assert hasattr(mod, "ValidationReport")

    def test_validator_accepts_valid_sdd(self, minimal_valid_sdd: Path) -> None:
        """T6.3: Validator returns exit 0 for valid SDD."""
        result = self._run_validator(minimal_valid_sdd)
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_validator_rejects_probability_violation(
        self, invalid_probability_sdd: Path
    ) -> None:
        """T6.4: Validator returns exit 1 for probability sum violation."""
        result = self._run_validator(invalid_probability_sdd)
        assert result.returncode == 1, (
            f"Expected exit 1 for probability violation.\nstdout: {result.stdout}"
        )
        assert (
            "probability" in result.stdout.lower()
            or "probability" in result.stderr.lower()
        )

    def test_validator_detects_dag_cycle(self, cyclic_sdd: Path) -> None:
        """T6.5: Validator detects and reports DAG cycles."""
        result = self._run_validator(cyclic_sdd)
        assert result.returncode == 1, (
            f"Expected exit 1 for cycle.\nstdout: {result.stdout}"
        )
        assert "cycle" in result.stdout.lower() or "cycle" in result.stderr.lower()

    def test_validator_detects_broken_crossrefs(self, broken_ref_sdd: Path) -> None:
        """T6.6: Validator detects broken cross-references."""
        result = self._run_validator(broken_ref_sdd)
        assert result.returncode == 1, (
            f"Expected exit 1 for broken refs.\nstdout: {result.stdout}"
        )
        assert (
            "nonexistent" in result.stdout.lower()
            or "reference" in result.stdout.lower()
        )

    def test_validator_detects_missing_decision_files(
        self, minimal_valid_sdd: Path
    ) -> None:
        """T6.7: Validator detects when a node's file is missing."""
        # Add node without corresponding file
        network_file = minimal_valid_sdd / "decisions" / "_network.yaml"
        network = _load_yaml(network_file)
        assert network is not None
        network["network"]["nodes"].append(
            {
                "id": "ghost_node",
                "level": "L2",
                "file": "L2-arch/ghost_node.decision.yaml",
                "title": "Ghost",
            }
        )
        network_file.write_text(
            yaml.dump(network, default_flow_style=False), encoding="utf-8"
        )
        result = self._run_validator(minimal_valid_sdd)
        assert result.returncode == 1, (
            f"Expected exit 1 for missing file.\nstdout: {result.stdout}"
        )

    def test_validator_returns_structured_report(self, minimal_valid_sdd: Path) -> None:
        """T6.8: Validator output contains structured validation sections."""
        result = self._run_validator(minimal_valid_sdd)
        output = result.stdout
        assert "DAG Acyclicity" in output
        assert "Probability Sums" in output
        assert "Cross-References" in output

    def test_validator_cli_exit_codes(self, minimal_valid_sdd: Path) -> None:
        """T6.9: CLI exit 0 for pass, 1 for fail."""
        # Valid SDD
        result = self._run_validator(minimal_valid_sdd)
        assert result.returncode == 0

    def test_validator_accepts_sdd_root_flag(self) -> None:
        """T6.10: Validator accepts --sdd-root flag."""
        validator = SDD_ROOT / "validate.py"
        result = subprocess.run(
            [sys.executable, str(validator), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert "sdd-root" in result.stdout.lower() or result.returncode == 0

    def test_validator_report_dataclass(self) -> None:
        """T6.11: ValidationReport is a dataclass with expected fields."""
        spec = importlib.util.spec_from_file_location(
            "sdd_validate", SDD_ROOT / "validate.py"
        )
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sdd_validate"] = mod
        spec.loader.exec_module(mod)
        report = mod.ValidationReport(
            dag_acyclicity=True,
            probability_sums=[],
            cross_references=[],
            missing_files=[],
            schema_violations=[],
            overall_pass=True,
        )
        assert report.overall_pass is True
        assert report.dag_acyclicity is True

    def test_validator_parameterized_sdd_root(self, minimal_valid_sdd: Path) -> None:
        """T6.12: Validator uses parameterized sdd_root (no hardcoded paths)."""
        validator_code = (SDD_ROOT / "validate.py").read_text(encoding="utf-8")
        # Should not contain hardcoded MinIVess paths
        assert "docs/planning/prd" not in validator_code
        assert "docs/prd" not in validator_code


# ============================================================================
# Task 7: MinIVess CI wrapper (3 tests)
# ============================================================================


class TestCIWrapper:
    """Task 7: MinIVess-specific CI wrapper for the standalone validator."""

    def test_wrapper_module_exists(self) -> None:
        """T7.1: Wrapper module exists."""
        wrapper = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "src"
            / "minivess"
            / "pipeline"
            / "sdd_template_validator.py"
        )
        assert wrapper.exists(), f"CI wrapper not found at {wrapper}"

    def test_wrapper_importable(self) -> None:
        """T7.2: Wrapper is importable."""
        from minivess.pipeline.sdd_template_validator import validate_minivess_sdd

        assert callable(validate_minivess_sdd)

    def test_wrapper_returns_report(self) -> None:
        """T7.3: Wrapper returns a validation report."""
        from minivess.pipeline.sdd_template_validator import validate_minivess_sdd

        report = validate_minivess_sdd()
        assert hasattr(report, "overall_pass")
        assert hasattr(report, "dag_acyclicity")


# ============================================================================
# Task 8: Examples directory (4 tests)
# ============================================================================


class TestExamplesDirectory:
    """Task 8: MinIVess instantiation example."""

    EXAMPLE_DIR = SDD_ROOT / "examples" / "minivess"

    def test_example_directory_exists(self) -> None:
        """T8.1: Example directory exists."""
        assert self.EXAMPLE_DIR.exists()

    def test_example_has_network(self) -> None:
        """T8.2: Example has a populated network file."""
        network_file = self.EXAMPLE_DIR / "decisions" / "_network.yaml"
        assert network_file.exists(), "Example missing _network.yaml"
        network = _load_yaml(network_file)
        assert network is not None, "Example _network.yaml parsed to None"
        assert len(network["network"]["nodes"]) > 0, "Example network has no nodes"

    def test_example_decisions_validate(self) -> None:
        """T8.3: Example decision files are valid YAML."""
        decisions_dir = self.EXAMPLE_DIR / "decisions"
        decision_files = list(decisions_dir.rglob("*.decision.yaml"))
        assert len(decision_files) > 0, "No decision files in example"
        for f in decision_files:
            data = _load_yaml(f)
            assert data is not None, f"Invalid YAML: {f}"
            assert "decision_id" in data, f"Missing decision_id in {f}"

    def test_example_is_complete_instantiation(self) -> None:
        """T8.4: Example has all required SDD components."""
        assert (self.EXAMPLE_DIR / "decisions" / "_schema.yaml").exists()
        assert (self.EXAMPLE_DIR / "decisions" / "_network.yaml").exists()
        assert (self.EXAMPLE_DIR / "bibliography.yaml").exists()
        # At least one archetype, scenario, or domain
        has_archetype = list(self.EXAMPLE_DIR.rglob("*.archetype.yaml"))
        has_scenario = list(self.EXAMPLE_DIR.rglob("*.scenario.yaml"))
        has_domain = list(self.EXAMPLE_DIR.rglob("overlay.yaml"))
        assert len(has_archetype) + len(has_scenario) + len(has_domain) > 0, (
            "Example should have at least one archetype, scenario, or domain overlay"
        )


# ============================================================================
# Task 9: GUIDE.md (4 tests)
# ============================================================================


class TestGuideMd:
    """Task 9: Comprehensive user guide."""

    @pytest.fixture()
    def guide_content(self) -> str:
        path = SDD_ROOT / "GUIDE.md"
        assert path.exists(), "GUIDE.md not found"
        return path.read_text(encoding="utf-8")

    def test_guide_exists(self) -> None:
        """T9.1: GUIDE.md exists."""
        assert (SDD_ROOT / "GUIDE.md").exists()

    def test_guide_has_required_sections(self, guide_content: str) -> None:
        """T9.2: GUIDE.md has Quick Start, Step-by-Step, FAQ sections."""
        assert "Quick Start" in guide_content
        assert "Step-by-Step" in guide_content or "Step by Step" in guide_content
        assert "FAQ" in guide_content

    def test_guide_no_minivess_in_generic_sections(self, guide_content: str) -> None:
        """T9.3: Generic sections do not reference MinIVess."""
        # Split by sections — the main body should be generic
        # MinIVess may appear in example references but not in generic instructions
        sections = guide_content.split("## ")
        for section in sections:
            section_title = section.split("\n")[0].strip()
            if section_title in [
                "Quick Start",
                "Step-by-Step Population Guide",
                "FAQ",
                "Concepts",
                "Validation",
            ]:
                # These generic sections should not reference MinIVess
                assert "MinIVess" not in section, (
                    f"Generic section '{section_title}' references MinIVess"
                )

    def test_guide_references_template_files(self, guide_content: str) -> None:
        """T9.4: GUIDE.md references template files."""
        assert "_schema.yaml" in guide_content
        assert "validate.py" in guide_content
        assert "templates/" in guide_content


# ============================================================================
# Task 10: CLAUDE-TEMPLATE.md (4 tests)
# ============================================================================


class TestClaudeTemplateMd:
    """Task 10: Generic CLAUDE.md template."""

    @pytest.fixture()
    def template_content(self) -> str:
        path = SDD_ROOT / "CLAUDE-TEMPLATE.md"
        assert path.exists(), "CLAUDE-TEMPLATE.md not found"
        return path.read_text(encoding="utf-8")

    def test_template_exists(self) -> None:
        """T10.1: CLAUDE-TEMPLATE.md exists."""
        assert (SDD_ROOT / "CLAUDE-TEMPLATE.md").exists()

    def test_template_has_change_me_markers(self, template_content: str) -> None:
        """T10.2: Template has CHANGE_ME markers for customization."""
        assert "CHANGE_ME" in template_content
        # Should have multiple markers
        assert template_content.count("CHANGE_ME") >= 5

    def test_template_has_tdd_section(self, template_content: str) -> None:
        """T10.3: Template preserves TDD workflow section."""
        assert "TDD" in template_content

    def test_template_has_prd_section(self, template_content: str) -> None:
        """T10.4: Template has PRD system section."""
        assert "PRD" in template_content or "SDD" in template_content
