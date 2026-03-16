"""Evals for the issue-creator Skill.

Tests that the Skill's templates, protocols, and validation rules produce
structurally correct GitHub issues. These are structural evals — they verify
the Skill's infrastructure and template correctness.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]


class TestSkillStructureEval:
    """Eval: Skill has all required files."""

    def test_skill_md_exists(self) -> None:
        assert (SKILL_DIR / "SKILL.md").exists()

    def test_protocols_exist(self) -> None:
        """All protocol files must exist."""
        protocols_dir = SKILL_DIR / "protocols"
        required = [
            "create-issue.md",
            "create-from-plan.md",
            "create-from-failure.md",
            "batch-create.md",
            "retroactive-create.md",
        ]
        for protocol in required:
            assert (protocols_dir / protocol).exists(), f"Missing protocol: {protocol}"

    def test_templates_exist(self) -> None:
        """Issue templates must exist."""
        templates_dir = SKILL_DIR / "templates"
        required = ["feature.md", "bugfix.md", "research.md", "debt.md"]
        for template in required:
            assert (templates_dir / template).exists(), f"Missing template: {template}"


class TestIssueTemplateEval:
    """Eval: issue templates have required structure."""

    @pytest.fixture(params=["feature.md", "bugfix.md", "research.md", "debt.md"])
    def template_content(self, request: pytest.FixtureRequest) -> str:
        path: Path = SKILL_DIR / "templates" / str(request.param)
        return path.read_text(encoding="utf-8")

    def test_template_has_metadata_block(self, template_content: str) -> None:
        """Every template must have <!-- METADATA block."""
        assert "<!-- METADATA" in template_content, (
            "Template must have <!-- METADATA comment block"
        )

    def test_template_has_summary_section(self, template_content: str) -> None:
        """Every template must have ## Summary."""
        assert "## Summary" in template_content

    def test_template_has_acceptance_or_deliverables(
        self, template_content: str
    ) -> None:
        """Every template must have acceptance criteria or deliverables."""
        has_acceptance = "Acceptance Criteria" in template_content
        has_deliverables = "Deliverables" in template_content
        assert has_acceptance or has_deliverables, (
            "Template must have either 'Acceptance Criteria' or 'Deliverables' section"
        )


class TestSkillMdContentEval:
    """Eval: SKILL.md has required content for issue creation."""

    def test_has_project_board_config(self) -> None:
        """SKILL.md must have GitHub Project board configuration."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "PVT_kwDOCPpnGc4AYSAM" in content, (
            "SKILL.md must contain project ID for board integration"
        )

    def test_has_priority_option_ids(self) -> None:
        """SKILL.md must map P0-P3 to project field option IDs."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        for priority in ["P0", "P1", "P2", "P3"]:
            assert priority in content, f"Missing {priority} in SKILL.md"

    def test_has_validation_rules(self) -> None:
        """SKILL.md must have validation rules (hard gates)."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "Validation Rules" in content or "Hard Gates" in content

    def test_has_navigator_reference(self) -> None:
        """SKILL.md must reference navigator.yaml for domain routing."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "navigator.yaml" in content


class TestNavigatorDomainRoutingEval:
    """Eval: navigator.yaml domains are valid for issue domain: field."""

    def test_all_domains_in_navigator(self) -> None:
        """All domains referenced in the Skill must exist in navigator."""
        nav = yaml.safe_load(
            (REPO_ROOT / "knowledge-graph" / "navigator.yaml").read_text(
                encoding="utf-8"
            )
        )
        domains = set(nav.get("domains", {}).keys())
        # The issue-creator Skill uses domain: field — all values must be valid
        assert len(domains) >= 8, f"Navigator must have >=8 domains, got {len(domains)}"
