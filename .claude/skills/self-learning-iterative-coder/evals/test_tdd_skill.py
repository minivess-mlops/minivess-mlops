"""Evals for the self-learning-iterative-coder Skill v3.0.0.

Structural evals (file existence, schema validity) + behavioral evals
(content rules, v3.0 features like failure triage and zero tolerance).

Skills 2.0 compatible: these tests can be used as assertions in
evals/evals.json for the skill-creator eval framework.
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
        """All 9 TDD protocols must exist (8 core + failure-triage)."""
        protocols_dir = SKILL_DIR / "protocols"
        required = [
            "red-phase.md",
            "green-phase.md",
            "verify-phase.md",
            "fix-phase.md",
            "checkpoint.md",
            "convergence.md",
            "task-selection.md",
            "spec-adaptation.md",
            "failure-triage.md",
        ]
        for protocol in required:
            assert (protocols_dir / protocol).exists(), f"Missing protocol: {protocol}"

    def test_state_schema_exists(self) -> None:
        """JSON schema for state file must exist."""
        assert (SKILL_DIR / "state" / "tdd-state.schema.json").exists()

    def test_example_state_exists(self) -> None:
        """Example state file for reference."""
        assert (SKILL_DIR / "state" / "example-state.json").exists()

    def test_self_correction_principles_exists(self) -> None:
        """Philosophical foundation document must exist."""
        assert (SKILL_DIR / "prompts" / "self-correction-principles.md").exists()


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


class TestV3ZeroToleranceEval:
    """Eval: v3.0.0 features — zero tolerance and failure triage."""

    def test_skill_md_has_zero_tolerance_rule(self) -> None:
        """SKILL.md Rule #9: zero tolerance for observed failures."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "ZERO TOLERANCE" in content
        assert "pre-existing" in content.lower() or "Pre-existing" in content

    def test_skill_md_has_failure_triage_rule(self) -> None:
        """SKILL.md Rule #10: failure triage before fixing."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "FAILURE TRIAGE" in content or "failure-triage" in content
        assert "GATHER" in content

    def test_skill_md_has_tokens_upfront_rule(self) -> None:
        """SKILL.md Rule #11: tokens upfront quality philosophy."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "TOKENS UPFRONT" in content or "tokens upfront" in content

    def test_skill_md_has_silent_dismissal_antipattern(self) -> None:
        """Anti-patterns table must include silent dismissal."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "Silent dismissal" in content or "silent dismissal" in content

    def test_skill_md_has_whac_a_mole_antipattern(self) -> None:
        """Anti-patterns table must include whac-a-mole."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "Whac-a-mole" in content or "whac-a-mole" in content

    def test_failure_triage_protocol_has_gather_step(self) -> None:
        """failure-triage.md must define GATHER step with --maxfail=200."""
        content = (SKILL_DIR / "protocols" / "failure-triage.md").read_text(
            encoding="utf-8"
        )
        assert "GATHER" in content
        assert "maxfail" in content

    def test_failure_triage_protocol_has_categorize_step(self) -> None:
        """failure-triage.md must define CATEGORIZE step."""
        content = (SKILL_DIR / "protocols" / "failure-triage.md").read_text(
            encoding="utf-8"
        )
        assert "CATEGORIZE" in content
        assert "root cause" in content.lower()

    def test_verify_phase_has_failure_gate(self) -> None:
        """verify-phase.md must have FAILURE GATE at the top."""
        content = (SKILL_DIR / "protocols" / "verify-phase.md").read_text(
            encoding="utf-8"
        )
        assert "FAILURE GATE" in content

    def test_fix_phase_has_pre_check(self) -> None:
        """fix-phase.md must have PRE-CHECK for triage completion."""
        content = (SKILL_DIR / "protocols" / "fix-phase.md").read_text(encoding="utf-8")
        assert "PRE-CHECK" in content
        assert "failure-triage" in content.lower() or "Failure Triage" in content

    def test_activation_checklist_handles_non_green_baseline(self) -> None:
        """ACTIVATION-CHECKLIST.md must address non-green baselines."""
        content = (SKILL_DIR / "ACTIVATION-CHECKLIST.md").read_text(encoding="utf-8")
        assert "NOT green" in content or "not green" in content
        assert "Failure Triage" in content or "failure-triage" in content


class TestSkillVersionEval:
    """Eval: Skill version and metadata are current."""

    def test_skill_version_is_3_or_higher(self) -> None:
        """SKILL.md frontmatter must have version >= 3.0.0."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "version: 3." in content or "version: 4." in content

    def test_skill_has_yaml_frontmatter(self) -> None:
        """SKILL.md must start with YAML frontmatter."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert content.startswith("---")

    def test_skill_description_is_pushy(self) -> None:
        """Skill description should be broad enough for auto-triggering."""
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        # Per Skills 2.0: descriptions should list explicit trigger contexts
        assert "executable plan" in content.lower() or "XML" in content
        assert "test-driven" in content.lower() or "TDD" in content
