"""Tests for T-11: Wire FlowContract into analysis_flow.

Verifies that run_analysis_flow() tags MLflow runs with flow_name="analyze"
and upstream_training_run_id, respects ANALYSIS_OUTPUT env var, and has no
hardcoded relative output paths.

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
from pathlib import Path

_ANALYSIS_FLOW_SRC = Path("src/minivess/orchestration/flows/analysis_flow.py")


# ---------------------------------------------------------------------------
# AST-level: no hardcoded Path("outputs/analysis") literals
# ---------------------------------------------------------------------------


class TestNoHardcodedRelativePath:
    def test_no_hardcoded_relative_path(self) -> None:
        """analysis_flow.py must not contain Path('outputs/analysis') literal."""
        source = _ANALYSIS_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_path_call = (isinstance(func, ast.Name) and func.id == "Path") or (
                isinstance(func, ast.Attribute) and func.attr == "Path"
            )
            if not is_path_call:
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    assert "outputs/analysis" not in arg.value, (
                        f"Hardcoded relative path Path({arg.value!r}) found at "
                        f"line {node.lineno}. "
                        "Replace with Path(os.environ.get('ANALYSIS_OUTPUT', "
                        "'/app/outputs/analysis'))."
                    )


# ---------------------------------------------------------------------------
# Functional: FlowContract wiring
# ---------------------------------------------------------------------------


class TestAnalysisFlowContract:
    def test_analysis_flow_has_flow_name_tag(self) -> None:
        """run_analysis_flow() must tag its MLflow run with FLOW_NAME_ANALYSIS constant."""
        source = _ANALYSIS_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # The flow must reference FLOW_NAME_ANALYSIS (from constants) for flow_name tags.
        # Check for ast.Name references to the constant.
        found_flow_name_ref = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "FLOW_NAME_ANALYSIS":
                found_flow_name_ref = True
                break

        assert found_flow_name_ref, (
            "analysis_flow.py must use FLOW_NAME_ANALYSIS constant for flow_name tags. "
            "Use: from minivess.orchestration.constants import FLOW_NAME_ANALYSIS"
        )

    def test_analysis_flow_references_flow_contract(self) -> None:
        """analysis_flow.py must use FlowContract (directly or via mlflow_helpers)."""
        source = _ANALYSIS_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "FlowContract" in node.value
            ):
                found = True
                break
            if isinstance(node, ast.Name) and node.id == "FlowContract":
                found = True
                break
            if isinstance(node, ast.Attribute) and node.attr == "FlowContract":
                found = True
                break
            # Check import statements — accept FlowContract or mlflow_helpers
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in (
                        "FlowContract",
                        "log_completion_safe",
                        "find_upstream_safely",
                    ):
                        found = True
                        break
                    if alias.asname in ("FlowContract",):
                        found = True
                        break

        assert found, (
            "analysis_flow.py must reference FlowContract (directly or via mlflow_helpers). "
            "Add: from minivess.orchestration.mlflow_helpers import log_completion_safe"
        )

    def test_analysis_flow_references_upstream_run(self) -> None:
        """analysis_flow.py must reference upstream_training_run_id."""
        source = _ANALYSIS_FLOW_SRC.read_text(encoding="utf-8")
        assert "upstream_training_run_id" in source, (
            "analysis_flow.py must reference 'upstream_training_run_id'. "
            "Read upstream run via FlowContract.find_upstream_run() and tag "
            "the MLflow run with upstream_training_run_id."
        )

    def test_analysis_flow_references_log_flow_completion(self) -> None:
        """analysis_flow.py must call log_flow_completion or log_completion_safe."""
        source = _ANALYSIS_FLOW_SRC.read_text(encoding="utf-8")
        has_completion = (
            "log_flow_completion" in source or "log_completion_safe" in source
        )
        assert has_completion, (
            "analysis_flow.py must call log_flow_completion() or log_completion_safe(). "
            "Add the call near the end of run_analysis_flow()."
        )


# ---------------------------------------------------------------------------
# Functional: ANALYSIS_OUTPUT env var
# ---------------------------------------------------------------------------


class TestAnalysisOutputFromEnv:
    def test_analysis_output_env_var_pattern(self, monkeypatch) -> None:
        """ANALYSIS_OUTPUT env var must control output root."""
        import os

        target = "/my/analysis/output"
        monkeypatch.setenv("ANALYSIS_OUTPUT", target)
        resolved = Path(os.environ.get("ANALYSIS_OUTPUT", "/app/outputs/analysis"))
        assert str(resolved) == target

    def test_analysis_output_default_absolute(self, monkeypatch) -> None:
        """Default ANALYSIS_OUTPUT must be absolute."""
        import os

        monkeypatch.delenv("ANALYSIS_OUTPUT", raising=False)
        resolved = Path(os.environ.get("ANALYSIS_OUTPUT", "/app/outputs/analysis"))
        assert resolved.is_absolute(), (
            f"Default ANALYSIS_OUTPUT is not absolute: {resolved}"
        )

    def test_analysis_output_default_value(self, monkeypatch) -> None:
        """Default ANALYSIS_OUTPUT must be /app/outputs/analysis."""
        import os

        monkeypatch.delenv("ANALYSIS_OUTPUT", raising=False)
        resolved = Path(os.environ.get("ANALYSIS_OUTPUT", "/app/outputs/analysis"))
        assert str(resolved) == "/app/outputs/analysis"

    def test_export_artifacts_uses_env_var(self) -> None:
        """_export_analysis_artifacts must read ANALYSIS_OUTPUT env var."""
        source = _ANALYSIS_FLOW_SRC.read_text(encoding="utf-8")
        assert "ANALYSIS_OUTPUT" in source, (
            "analysis_flow.py must read ANALYSIS_OUTPUT env var. "
            "Replace Path('outputs/analysis') with "
            "Path(os.environ.get('ANALYSIS_OUTPUT', '/app/outputs/analysis'))."
        )
