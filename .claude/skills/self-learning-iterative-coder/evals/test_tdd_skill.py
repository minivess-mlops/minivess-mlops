"""Evals for the self-learning-iterative-coder Skill.

Tests that the TDD state schema is valid, that the Skill's protocols exist,
and that state file management works correctly. These are structural evals
that verify the Skill's infrastructure, not its runtime behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]


class TestSkillStructureEval:
    """Eval: Skill has all required files for the TDD loop."""

    def test_skill_md_exists(self) -> None:
        """SKILL.md is the master specification."""
        assert (SKILL_DIR / "SKILL.md").exists()

    def test_activation_checklist_exists(self) -> None:
        """ACTIVATION-CHECKLIST.md must exist for session startup."""
        assert (SKILL_DIR / "ACTIVATION-CHECKLIST.md").exists()

    def test_all_protocols_exist(self) -> None:
        """All 6 TDD protocols must exist."""
        protocols_dir = SKILL_DIR / "protocols"
        required = [
            "red-phase.md",
            "green-phase.md",
            "verify-phase.md",
            "fix-phase.md",
            "checkpoint.md",
            "convergence.md",
        ]
        for protocol in required:
            assert (protocols_dir / protocol).exists(), f"Missing protocol: {protocol}"

    def test_state_schema_exists(self) -> None:
        """JSON schema for state file must exist."""
        assert (SKILL_DIR / "state" / "tdd-state.schema.json").exists()

    def test_example_state_exists(self) -> None:
        """Example state file for reference."""
        assert (SKILL_DIR / "state" / "example-state.json").exists()


class TestStateSchemaEval:
    """Eval: TDD state schema is valid JSON Schema."""

    def test_schema_is_valid_json(self) -> None:
        """Schema file must be parseable JSON."""
        schema_path = SKILL_DIR / "state" / "tdd-state.schema.json"
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        assert "$schema" in data or "type" in data

    def test_schema_requires_tasks(self) -> None:
        """Schema must define a 'tasks' property."""
        schema_path = SKILL_DIR / "state" / "tdd-state.schema.json"
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        props = data.get("properties", {})
        assert "tasks" in props, "State schema must have 'tasks' property"

    def test_example_state_is_valid_json(self) -> None:
        """Example state file must be parseable JSON."""
        state_path = SKILL_DIR / "state" / "example-state.json"
        data = json.loads(state_path.read_text(encoding="utf-8"))
        assert "tasks" in data, "Example state must have 'tasks' key"


class TestSkillMdContentEval:
    """Eval: SKILL.md contains critical TDD rules."""

    def test_skill_md_has_tdd_mandate(self) -> None:
        """SKILL.md must reference TDD as mandatory."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "TESTS FIRST" in content or "RED" in content

    def test_skill_md_has_verify_phase(self) -> None:
        """SKILL.md must reference the VERIFY phase (no ghost completions)."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "VERIFY" in content
        assert "ghost" in content.lower() or "Ghost" in content

    def test_skill_md_has_no_placeholders_rule(self) -> None:
        """SKILL.md must ban placeholder code."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "NO PLACEHOLDERS" in content or "NotImplementedError" in content

    def test_skill_md_has_learnings_rule(self) -> None:
        """SKILL.md must reference LEARNINGS.md accumulation."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "LEARNINGS.md" in content

    def test_skill_md_has_session_budget(self) -> None:
        """SKILL.md must define iteration budget limits."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "Max inner iterations" in content or "FORCE_STOP" in content
