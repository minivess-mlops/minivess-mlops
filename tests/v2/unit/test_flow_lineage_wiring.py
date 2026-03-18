"""Tests that verify OpenLineage is WIRED into actual Prefect flow modules.

Unlike test_flow_lineage_emission.py (tests LineageEmitter in isolation),
these tests verify that the flow SOURCE CODE actually imports and uses
the lineage helper. This is the "flow-level wiring" test from Issue #799/#821.

Phase 1 (P0) of fda-insights-second-pass-executable.xml.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Flow files to check for lineage wiring
_FLOWS_DIR = Path("src/minivess/orchestration/flows")
_WIRED_FLOWS = [
    "train_flow.py",
    "post_training_flow.py",
    "analysis_flow.py",
    "biostatistics_flow.py",
    "deploy_flow.py",
]


def _get_imported_names(filepath: Path) -> set[str]:
    """Parse a Python file's AST and return all imported names."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)
            names.add(module)
    return names


def _source_contains(filepath: Path, substring: str) -> bool:
    """Check if a file's source code contains a substring."""
    return substring in filepath.read_text(encoding="utf-8")


class TestFlowsImportLineageHelper:
    """Every wired flow must import the lineage emission helper."""

    def test_train_flow_imports_lineage(self) -> None:
        names = _get_imported_names(_FLOWS_DIR / "train_flow.py")
        assert "emit_flow_lineage" in names or "LineageEmitter" in names, (
            "train_flow.py must import emit_flow_lineage or LineageEmitter"
        )

    def test_post_training_flow_imports_lineage(self) -> None:
        names = _get_imported_names(_FLOWS_DIR / "post_training_flow.py")
        assert "emit_flow_lineage" in names or "LineageEmitter" in names, (
            "post_training_flow.py must import emit_flow_lineage or LineageEmitter"
        )

    def test_analysis_flow_imports_lineage(self) -> None:
        names = _get_imported_names(_FLOWS_DIR / "analysis_flow.py")
        assert "emit_flow_lineage" in names or "LineageEmitter" in names, (
            "analysis_flow.py must import emit_flow_lineage or LineageEmitter"
        )

    def test_biostatistics_flow_imports_lineage(self) -> None:
        names = _get_imported_names(_FLOWS_DIR / "biostatistics_flow.py")
        assert "emit_flow_lineage" in names or "LineageEmitter" in names, (
            "biostatistics_flow.py must import emit_flow_lineage or LineageEmitter"
        )

    def test_deploy_flow_imports_lineage(self) -> None:
        names = _get_imported_names(_FLOWS_DIR / "deploy_flow.py")
        assert "emit_flow_lineage" in names or "LineageEmitter" in names, (
            "deploy_flow.py must import emit_flow_lineage or LineageEmitter"
        )


class TestFlowsCallLineageEmission:
    """Every wired flow must call emit_flow_lineage() or pipeline_run()."""

    def test_train_flow_calls_lineage(self) -> None:
        assert _source_contains(
            _FLOWS_DIR / "train_flow.py", "emit_flow_lineage"
        ) or _source_contains(_FLOWS_DIR / "train_flow.py", "pipeline_run"), (
            "train_flow.py must call emit_flow_lineage() or pipeline_run()"
        )

    def test_post_training_flow_calls_lineage(self) -> None:
        assert _source_contains(
            _FLOWS_DIR / "post_training_flow.py", "emit_flow_lineage"
        ) or _source_contains(_FLOWS_DIR / "post_training_flow.py", "pipeline_run"), (
            "post_training_flow.py must call emit_flow_lineage() or pipeline_run()"
        )

    def test_analysis_flow_calls_lineage(self) -> None:
        assert _source_contains(
            _FLOWS_DIR / "analysis_flow.py", "emit_flow_lineage"
        ) or _source_contains(_FLOWS_DIR / "analysis_flow.py", "pipeline_run"), (
            "analysis_flow.py must call emit_flow_lineage() or pipeline_run()"
        )

    def test_biostatistics_flow_calls_lineage(self) -> None:
        assert _source_contains(
            _FLOWS_DIR / "biostatistics_flow.py", "emit_flow_lineage"
        ) or _source_contains(_FLOWS_DIR / "biostatistics_flow.py", "pipeline_run"), (
            "biostatistics_flow.py must call emit_flow_lineage() or pipeline_run()"
        )

    def test_deploy_flow_calls_lineage(self) -> None:
        assert _source_contains(
            _FLOWS_DIR / "deploy_flow.py", "emit_flow_lineage"
        ) or _source_contains(_FLOWS_DIR / "deploy_flow.py", "pipeline_run"), (
            "deploy_flow.py must call emit_flow_lineage() or pipeline_run()"
        )


class TestAnalysisFlowTestSetAudit:
    """Analysis flow must log test set evaluations to AuditTrail (Issue #821)."""

    def test_analysis_flow_imports_audit_trail(self) -> None:
        names = _get_imported_names(_FLOWS_DIR / "analysis_flow.py")
        assert "AuditTrail" in names or "log_test_evaluation" in names, (
            "analysis_flow.py must import AuditTrail or log_test_evaluation "
            "for FDA test set firewall (Issue #821)"
        )

    def test_analysis_flow_calls_test_evaluation_logging(self) -> None:
        assert _source_contains(
            _FLOWS_DIR / "analysis_flow.py", "log_test_evaluation"
        ), (
            "analysis_flow.py must call log_test_evaluation() for FDA "
            "test set access logging"
        )


class TestEmitFlowLineageHelper:
    """The emit_flow_lineage helper must exist and work correctly."""

    def test_helper_exists(self) -> None:
        from minivess.observability.lineage import emit_flow_lineage

        assert callable(emit_flow_lineage)

    def test_helper_emits_start_and_complete(self) -> None:
        from openlineage.client.event_v2 import RunState

        from minivess.observability.lineage import LineageEmitter, emit_flow_lineage

        emitter = LineageEmitter()
        emit_flow_lineage(
            emitter=emitter,
            job_name="test-flow",
            inputs=[{"namespace": "minivess", "name": "input_data"}],
            outputs=[{"namespace": "minivess", "name": "output_data"}],
        )
        events = emitter.get_events_for_job("test-flow")
        assert len(events) == 2
        assert events[0].eventType == RunState.START
        assert events[1].eventType == RunState.COMPLETE
