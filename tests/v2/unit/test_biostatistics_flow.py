"""Tests for biostatistics Prefect flow (Phase 8, Task 8.1)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

# Set MINIVESS_TESTING=1 BEFORE importing the flow (Docker context check)
os.environ["MINIVESS_TESTING"] = "1"

if TYPE_CHECKING:
    from pathlib import Path


class TestFlowRunsWithMockData:
    # All Prefect tasks + helper functions in the flow MUST be mocked.
    # Missing mocks cause real code to run, which caused a 62 GB RAM crash.
    # See: .claude/metalearning/2026-03-21-ram-crash-biostatistics-test.md
    @patch("minivess.orchestration.flows.biostatistics_flow.task_log_mlflow")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_build_lineage")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_generate_tables")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_generate_figures")
    @patch(
        "minivess.orchestration.flows.biostatistics_flow.task_compute_rank_concordance"
    )
    @patch(
        "minivess.orchestration.flows.biostatistics_flow.task_compute_specification_curve"
    )
    @patch("minivess.orchestration.flows.biostatistics_flow.task_compute_rankings")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_compute_variance")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_compute_pairwise")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_compute_bayesian")
    @patch(
        "minivess.orchestration.flows.biostatistics_flow.task_compute_factorial_anova"
    )
    @patch("minivess.orchestration.flows.biostatistics_flow.task_build_duckdb")
    @patch(
        "minivess.orchestration.flows.biostatistics_flow.task_validate_source_completeness"
    )
    @patch("minivess.orchestration.flows.biostatistics_flow.task_discover_source_runs")
    @patch("minivess.orchestration.flows.biostatistics_flow._resolve_factor_names")
    @patch("minivess.orchestration.flows.biostatistics_flow._load_config")
    @patch("minivess.orchestration.flows.biostatistics_flow._build_per_volume_data")
    def test_flow_runs_with_mock_data(
        self,
        mock_build_pvd: MagicMock,
        mock_load_config: MagicMock,
        mock_resolve_factors: MagicMock,
        mock_discover: MagicMock,
        mock_validate: MagicMock,
        mock_build_db: MagicMock,
        mock_factorial_anova: MagicMock,
        mock_bayesian: MagicMock,
        mock_pairwise: MagicMock,
        mock_variance: MagicMock,
        mock_rankings: MagicMock,
        mock_spec_curve: MagicMock,
        mock_rank_concordance: MagicMock,
        mock_figures: MagicMock,
        mock_tables: MagicMock,
        mock_lineage: MagicMock,
        mock_mlflow: MagicMock,
        tmp_path: Path,
    ) -> None:
        from minivess.config.biostatistics_config import BiostatisticsConfig
        from minivess.orchestration.flows.biostatistics_flow import (
            run_biostatistics_flow,
        )
        from minivess.pipeline.biostatistics_rank_stability import (
            RankConcordanceResult,
        )
        from minivess.pipeline.biostatistics_specification_curve import (
            SpecificationCurveResult,
        )
        from minivess.pipeline.biostatistics_types import (
            SourceRun,
            SourceRunManifest,
            ValidationResult,
        )

        # Use real BiostatisticsConfig with test-appropriate defaults.
        # NEVER use bare MagicMock() for config — MagicMock arithmetic is
        # unpredictable and caused a 62 GB RAM crash (H1 in report).
        config = BiostatisticsConfig(
            mlruns_dir=tmp_path / "mlruns",
            output_dir=tmp_path / "output",
            experiment_names=["test"],
            metrics=["dsc"],
            primary_metric="dsc",
            n_bootstrap=100,
        )
        mock_load_config.return_value = config
        mock_resolve_factors.return_value = ["model_family", "loss_name"]

        manifest = SourceRunManifest.from_runs(
            [
                SourceRun("r1", "1", "test", "dice_ce", 0, "FINISHED"),
            ]
        )
        mock_discover.return_value = manifest
        mock_validate.return_value = ValidationResult(
            valid=True, warnings=[], errors=[], n_conditions=2, n_folds_per_condition=3
        )
        mock_build_db.return_value = tmp_path / "biostatistics.duckdb"
        mock_build_pvd.return_value = {}
        mock_pairwise.return_value = []
        mock_bayesian.return_value = []
        mock_variance.return_value = []
        mock_rankings.return_value = []
        mock_spec_curve.return_value = SpecificationCurveResult(
            specifications=[],
            median_effect=0.0,
            fraction_significant=0.0,
        )
        mock_rank_concordance.return_value = RankConcordanceResult(
            tau_matrix=[],
            condition_ranks={},
            n_inversions=0,
            n_pairs=0,
        )
        mock_figures.return_value = []
        mock_tables.return_value = []
        mock_lineage.return_value = {"fingerprint": manifest.fingerprint}
        mock_mlflow.return_value = "mock_run_id"

        result = run_biostatistics_flow.fn(config_path=str(tmp_path / "config.yaml"))
        assert result is not None


class TestMockCoverage:
    """Validate that ALL callables used in the flow are mocked in tests.

    Prevents regressions like the 62 GB RAM crash caused by missing mocks.
    Uses AST introspection (Rule #16: no regex for structured data).
    """

    def test_all_tasks_mocked(self) -> None:
        """Every callable invoked in run_biostatistics_flow must be mocked."""
        import ast
        from pathlib import Path

        flow_path = Path("src/minivess/orchestration/flows/biostatistics_flow.py")
        tree = ast.parse(flow_path.read_text(encoding="utf-8"))

        # Find all @task-decorated function names in the module
        task_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    if (isinstance(dec, ast.Name) and dec.id == "task") or (
                        isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Name)
                        and dec.func.id == "task"
                    ):
                        task_names.add(node.name)

        # Find the flow function body and extract all called names
        called_in_flow: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "run_biostatistics_flow"
            ):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        called_in_flow.add(child.func.id)

        # Only tasks/helpers that are actually called in the flow need mocking
        helpers_called = {
            "_load_config",
            "_build_per_volume_data",
            "_resolve_factor_names",
        } & called_in_flow
        tasks_called = task_names & called_in_flow
        all_must_mock = tasks_called | helpers_called

        # Read our test file and find all @patch targets
        test_path = Path("tests/v2/unit/test_biostatistics_flow.py")
        test_tree = ast.parse(test_path.read_text(encoding="utf-8"))

        patched_names: set[str] = set()
        for node in ast.walk(test_tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "patch"
                and node.args
            ):
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    patched_names.add(arg.value.split(".")[-1])

        missing = all_must_mock - patched_names
        assert not missing, (
            f"Flow callables not mocked in test: {missing}. "
            f"Missing mocks cause real code to run, which can crash the system. "
            f"See: .claude/metalearning/2026-03-21-ram-crash-biostatistics-test.md"
        )


class TestFlowRaisesOutsideDocker:
    def test_flow_raises_outside_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flow must raise RuntimeError when not in Docker and escape hatches unset."""
        # Clear all Docker gate escape hatches
        monkeypatch.delenv("MINIVESS_TESTING", raising=False)
        monkeypatch.delenv("MINIVESS_ALLOW_HOST", raising=False)
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)

        from minivess.orchestration.flows.biostatistics_flow import (
            _require_docker_context,
        )

        with pytest.raises(RuntimeError, match="Docker"):
            _require_docker_context()
