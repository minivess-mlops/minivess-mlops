"""Evals for the knowledge-reviewer Skill.

Tests that the 4 reviewer scripts (link checker, PRD auditor, legacy detector,
staleness scanner) produce correct structured output and catch known issues.
These are outcome-based evals: they run the actual scripts and verify results.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _run_reviewer(script_name: str, *args: str) -> dict[str, str | int]:
    """Run a reviewer script and return parsed output."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


class TestLinkCheckerEval:
    """Eval: link checker must find 0 errors on current repo state."""

    def test_quick_mode_no_errors(self) -> None:
        """Quick mode (pre-commit equivalent) must pass with 0 errors."""
        result = _run_reviewer("review_knowledge_links.py", "--quick")
        assert result["returncode"] == 0, (
            f"Link checker --quick failed with errors:\n{result['stdout']}"
        )

    def test_quick_mode_reports_total_checks(self) -> None:
        """Output must contain 'Total checks:' line."""
        result = _run_reviewer("review_knowledge_links.py", "--quick")
        stdout = str(result["stdout"])
        assert "Total checks:" in stdout

    def test_quick_mode_reports_failure_count(self) -> None:
        """Output must contain 'Failures (ERROR):' line."""
        result = _run_reviewer("review_knowledge_links.py", "--quick")
        stdout = str(result["stdout"])
        assert "Failures (ERROR):" in stdout


class TestLegacyDetectorEval:
    """Eval: legacy detector must find no v0.1-era patterns."""

    def test_no_legacy_errors(self) -> None:
        """Legacy detector must pass (no Poetry, no pip install, no old imports)."""
        result = _run_reviewer("review_legacy_artifacts.py")
        assert result["returncode"] == 0, (
            f"Legacy detector found v0.1 patterns:\n{result['stdout']}"
        )


class TestOrchestratorEval:
    """Eval: orchestrator --quick must pass (link checker + legacy)."""

    def test_orchestrator_quick_passes(self) -> None:
        """Orchestrator --quick is the pre-commit equivalent; must pass."""
        result = _run_reviewer("review_knowledge.py", "--quick")
        assert result["returncode"] == 0, (
            f"Orchestrator --quick failed:\n{result['stdout']}"
        )


class TestNavigatorIntegrityEval:
    """Eval: navigator.yaml must have correct structure for the Skill to work."""

    def test_navigator_has_all_domains(self) -> None:
        """Navigator must route to all knowledge domains."""
        import yaml

        nav = yaml.safe_load(
            (REPO_ROOT / "knowledge-graph" / "navigator.yaml").read_text(
                encoding="utf-8"
            )
        )
        domains = nav.get("domains", {})
        required = {
            "architecture",
            "training",
            "infrastructure",
            "cloud",
            "data",
            "models",
            "observability",
            "operations",
            "testing",
            "manuscript",
        }
        actual = set(domains.keys())
        missing = required - actual
        assert not missing, f"Navigator missing domains: {missing}"

    def test_navigator_has_invariants(self) -> None:
        """Navigator must have invariants section with guardrails."""
        import yaml

        nav = yaml.safe_load(
            (REPO_ROOT / "knowledge-graph" / "navigator.yaml").read_text(
                encoding="utf-8"
            )
        )
        invariants = nav.get("invariants", [])
        assert len(invariants) >= 4, (
            f"Navigator must have >=4 invariants, got {len(invariants)}"
        )
        invariant_ids = {i["id"] for i in invariants}
        assert "two_providers_only" in invariant_ids, (
            "Navigator must have 'two_providers_only' invariant"
        )

    def test_navigator_has_cloud_domain(self) -> None:
        """Navigator MUST have a cloud domain (prevents AWS S3 repeat)."""
        import yaml

        nav = yaml.safe_load(
            (REPO_ROOT / "knowledge-graph" / "navigator.yaml").read_text(
                encoding="utf-8"
            )
        )
        domains = nav.get("domains", {})
        assert "cloud" in domains, (
            "Navigator MUST have 'cloud' domain. Without it, cloud architecture "
            "is invisible and agents make unauthorized infrastructure changes. "
            "See: .claude/metalearning/2026-03-16-unauthorized-aws-s3-architecture-migration.md"
        )
