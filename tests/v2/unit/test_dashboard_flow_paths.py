"""Tests for T-13: dashboard_flow output paths and FlowContract wiring.

Verifies that run_dashboard_flow() uses DASHBOARD_OUTPUT env var for output_dir,
creates expected files, and has FlowContract wiring.

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
from pathlib import Path

_DASHBOARD_FLOW_SRC = Path("src/minivess/orchestration/flows/dashboard_flow.py")


# ---------------------------------------------------------------------------
# AST-level: DASHBOARD_OUTPUT env var must appear in source
# ---------------------------------------------------------------------------


class TestDashboardOutputEnvVar:
    def test_dashboard_flow_references_dashboard_output(self) -> None:
        """dashboard_flow.py must reference DASHBOARD_OUTPUT env var."""
        source = _DASHBOARD_FLOW_SRC.read_text(encoding="utf-8")
        assert "DASHBOARD_OUTPUT" in source, (
            "dashboard_flow.py must read DASHBOARD_OUTPUT env var. "
            "Change output_dir from required positional to optional with default: "
            "Path(os.environ.get('DASHBOARD_OUTPUT', '/app/outputs/dashboard'))"
        )

    def test_dashboard_output_default_absolute(self, monkeypatch) -> None:
        """Default DASHBOARD_OUTPUT must be absolute."""
        import os

        monkeypatch.delenv("DASHBOARD_OUTPUT", raising=False)
        resolved = Path(os.environ.get("DASHBOARD_OUTPUT", "/app/outputs/dashboard"))
        assert resolved.is_absolute(), (
            f"Default DASHBOARD_OUTPUT is not absolute: {resolved}"
        )

    def test_dashboard_output_default_value(self, monkeypatch) -> None:
        """Default DASHBOARD_OUTPUT must be /app/outputs/dashboard."""
        import os

        monkeypatch.delenv("DASHBOARD_OUTPUT", raising=False)
        resolved = Path(os.environ.get("DASHBOARD_OUTPUT", "/app/outputs/dashboard"))
        assert str(resolved) == "/app/outputs/dashboard"

    def test_dashboard_output_from_env(self, monkeypatch) -> None:
        """DASHBOARD_OUTPUT env var must control output_dir default."""
        import os

        monkeypatch.setenv("DASHBOARD_OUTPUT", "/test/dash")
        resolved = Path(os.environ.get("DASHBOARD_OUTPUT", "/app/outputs/dashboard"))
        assert str(resolved) == "/test/dash"


# ---------------------------------------------------------------------------
# Functional: output files created
# ---------------------------------------------------------------------------


class TestDashboardOutputFiles:
    def test_dashboard_creates_report(self, monkeypatch, tmp_path) -> None:
        """run_dashboard_flow() must create everything_dashboard_report.md."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("DASHBOARD_OUTPUT", str(tmp_path / "dash"))

        from minivess.orchestration.flows.dashboard_flow import run_dashboard_flow

        result = run_dashboard_flow()
        report_path = result["report_path"]
        assert Path(report_path).exists(), (
            f"everything_dashboard_report.md not found at {report_path}. "
            "run_dashboard_flow() must create the report file."
        )

    def test_dashboard_default_uses_absolute_path(self) -> None:
        """Default output_dir must use /app/outputs/dashboard (absolute, not /tmp)."""
        source = _DASHBOARD_FLOW_SRC.read_text(encoding="utf-8")
        # Ensure the default value in the source is the absolute Docker path
        assert "/app/outputs/dashboard" in source, (
            "dashboard_flow.py default output_dir must be /app/outputs/dashboard. "
            "Use: Path(os.environ.get('DASHBOARD_OUTPUT', '/app/outputs/dashboard'))"
        )

    def test_dashboard_metadata_json_created(self, monkeypatch, tmp_path) -> None:
        """run_dashboard_flow() must create everything_dashboard_metadata.json."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("DASHBOARD_OUTPUT", str(tmp_path / "dash"))

        from minivess.orchestration.flows.dashboard_flow import run_dashboard_flow

        result = run_dashboard_flow()
        metadata_path = result["metadata_path"]
        assert Path(metadata_path).exists(), (
            f"everything_dashboard_metadata.json not found at {metadata_path}. "
            "run_dashboard_flow() must create the JSON metadata file."
        )


# ---------------------------------------------------------------------------
# FlowContract wiring
# ---------------------------------------------------------------------------


class TestDashboardFlowContract:
    def test_dashboard_flow_references_flow_contract(self) -> None:
        """dashboard_flow.py must reference FlowContract."""
        source = _DASHBOARD_FLOW_SRC.read_text(encoding="utf-8")
        assert "FlowContract" in source, (
            "dashboard_flow.py must use FlowContract. "
            "Add: from minivess.orchestration.flow_contract import FlowContract"
        )

    def test_dashboard_flow_references_log_flow_completion(self) -> None:
        """dashboard_flow.py must call log_flow_completion."""
        source = _DASHBOARD_FLOW_SRC.read_text(encoding="utf-8")
        assert "log_flow_completion" in source, (
            "dashboard_flow.py must call FlowContract.log_flow_completion(). "
            "Add the call near the end of run_dashboard_flow()."
        )

    def test_dashboard_flow_tags_flow_name(self) -> None:
        """dashboard_flow.py must contain 'dashboard' as a flow_name tag value."""
        source = _DASHBOARD_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "dashboard":
                found = True
                break
        assert found, (
            "dashboard_flow.py must tag MLflow run with flow_name='dashboard'. "
            "Add flow_name='dashboard' tag when opening MLflow run."
        )
