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
    @patch("minivess.orchestration.flows.biostatistics_flow.task_log_mlflow")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_build_lineage")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_generate_tables")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_generate_figures")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_compute_rankings")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_compute_variance")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_compute_pairwise")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_compute_bayesian")
    @patch("minivess.orchestration.flows.biostatistics_flow.task_build_duckdb")
    @patch(
        "minivess.orchestration.flows.biostatistics_flow.task_validate_source_completeness"
    )
    @patch("minivess.orchestration.flows.biostatistics_flow.task_discover_source_runs")
    @patch("minivess.orchestration.flows.biostatistics_flow._load_config")
    @patch("minivess.orchestration.flows.biostatistics_flow._build_per_volume_data")
    def test_flow_runs_with_mock_data(
        self,
        mock_build_pvd: MagicMock,
        mock_load_config: MagicMock,
        mock_discover: MagicMock,
        mock_validate: MagicMock,
        mock_build_db: MagicMock,
        mock_bayesian: MagicMock,
        mock_pairwise: MagicMock,
        mock_variance: MagicMock,
        mock_rankings: MagicMock,
        mock_figures: MagicMock,
        mock_tables: MagicMock,
        mock_lineage: MagicMock,
        mock_mlflow: MagicMock,
        tmp_path: Path,
    ) -> None:
        from minivess.orchestration.flows.biostatistics_flow import (
            run_biostatistics_flow,
        )
        from minivess.pipeline.biostatistics_types import (
            SourceRun,
            SourceRunManifest,
            ValidationResult,
        )

        # Setup config mock
        mock_config = MagicMock()
        mock_config.mlruns_dir = tmp_path / "mlruns"
        mock_config.output_dir = tmp_path / "output"
        mock_config.experiment_names = ["test"]
        mock_config.min_folds_per_condition = 3
        mock_config.min_conditions = 2
        mock_config.metrics = ["val_dice"]
        mock_config.primary_metric = "val_dice"
        mock_config.alpha = 0.05
        mock_config.n_bootstrap = 100
        mock_config.seed = 42
        mock_config.rope_values = {"val_dice": 0.01}
        mock_load_config.return_value = mock_config

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
        mock_figures.return_value = []
        mock_tables.return_value = []
        mock_lineage.return_value = {"fingerprint": manifest.fingerprint}
        mock_mlflow.return_value = "mock_run_id"

        result = run_biostatistics_flow.fn(config_path=str(tmp_path / "config.yaml"))
        assert result is not None


class TestFlowRaisesOutsideDocker:
    def test_flow_raises_outside_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flow must raise RuntimeError when not in Docker and MINIVESS_TESTING is unset."""
        # Temporarily unset MINIVESS_TESTING
        monkeypatch.delenv("MINIVESS_TESTING", raising=False)

        from minivess.orchestration.flows.biostatistics_flow import (
            _require_docker_context,
        )

        with pytest.raises(RuntimeError, match="Docker"):
            _require_docker_context()
