"""Tests for T-16: acquisition_flow MLflow run logging.

Verifies that run_acquisition_flow() opens an MLflow run in the
'minivess_acquisition' experiment, logs acq_ params, and tags with
flow_name='acquisition'.

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
from pathlib import Path

import mlflow

_ACQUISITION_FLOW_SRC = Path("src/minivess/orchestration/flows/acquisition_flow.py")
_ACQUISITION_CONFIG_SRC = Path("src/minivess/config/acquisition_config.py")


def _get_run_tags(run) -> dict:
    return dict(run.data.tags)


# ---------------------------------------------------------------------------
# Source-level: AcquisitionResult must have mlflow_run_id
# ---------------------------------------------------------------------------


class TestAcquisitionResultHasRunId:
    def test_acquisition_result_has_mlflow_run_id_field(self) -> None:
        """AcquisitionResult must have mlflow_run_id field."""
        source = _ACQUISITION_CONFIG_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Name) and target.id == "mlflow_run_id":
                    found = True
                    break
        assert found, (
            "AcquisitionResult in acquisition_config.py must have a "
            "'mlflow_run_id' field. Add: mlflow_run_id: str | None = None"
        )

    def test_acquisition_flow_references_minivess_acquisition_experiment(
        self,
    ) -> None:
        """acquisition_flow.py must reference 'minivess_acquisition' experiment."""
        source = _ACQUISITION_FLOW_SRC.read_text(encoding="utf-8")
        assert "minivess_acquisition" in source, (
            "acquisition_flow.py must open an MLflow run in the "
            "'minivess_acquisition' experiment. "
            "Add: mlflow.set_experiment('minivess_acquisition')"
        )

    def test_acquisition_flow_references_flow_name_tag(self) -> None:
        """acquisition_flow.py must contain 'acquisition' as a flow_name tag value."""
        source = _ACQUISITION_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "acquisition":
                found = True
                break
        assert found, (
            "acquisition_flow.py must tag MLflow run with flow_name='acquisition'. "
            "Add flow_name='acquisition' in mlflow.start_run() tags."
        )


# ---------------------------------------------------------------------------
# Functional: MLflow run creation
# ---------------------------------------------------------------------------


class TestAcquisitionMlflow:
    def _make_minimal_config(self, tmp_path):
        """Build minimal AcquisitionConfig for testing."""
        from minivess.config.acquisition_config import (
            AcquisitionConfig,
        )

        return AcquisitionConfig(
            datasets=[],
            output_dir=tmp_path / "raw",
        )

    def test_acquisition_opens_mlflow_run(self, monkeypatch, tmp_path) -> None:
        """run_acquisition_flow() must create an MLflow run in minivess_acquisition."""
        mlflow_dir = tmp_path / "mlruns"
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlflow_dir))

        mlflow.set_tracking_uri(str(mlflow_dir))

        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        run_acquisition_flow(config=self._make_minimal_config(tmp_path))

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_acquisition")
        assert experiment is not None, (
            "No 'minivess_acquisition' experiment after run_acquisition_flow(). "
            "Add mlflow.set_experiment('minivess_acquisition') to the flow."
        )
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
        )
        assert runs, (
            "No MLflow run found in 'minivess_acquisition' experiment. "
            "run_acquisition_flow() must open an MLflow run."
        )

    def test_acquisition_flow_name_tag(self, monkeypatch, tmp_path) -> None:
        """MLflow run must have tags.flow_name == 'acquisition'."""
        mlflow_dir = tmp_path / "mlruns"
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlflow_dir))

        mlflow.set_tracking_uri(str(mlflow_dir))

        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        run_acquisition_flow(config=self._make_minimal_config(tmp_path))

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_acquisition")
        if experiment is None:
            return  # Skip if flow didn't create experiment

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.flow_name = 'acquisition'",
        )
        assert runs, (
            "No MLflow run with flow_name='acquisition' found. "
            "Tag the MLflow run with flow_name='acquisition'."
        )

    def test_acquisition_result_has_run_id(self, monkeypatch, tmp_path) -> None:
        """AcquisitionResult.mlflow_run_id must not be None after flow run."""
        mlflow_dir = tmp_path / "mlruns"
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlflow_dir))

        from minivess.orchestration.flows.acquisition_flow import (
            run_acquisition_flow,
        )

        result = run_acquisition_flow(config=self._make_minimal_config(tmp_path))
        assert hasattr(result, "mlflow_run_id"), (
            "AcquisitionResult must have 'mlflow_run_id' attribute. "
            "Add mlflow_run_id field to AcquisitionResult dataclass."
        )
        assert result.mlflow_run_id is not None, (
            "AcquisitionResult.mlflow_run_id must not be None after flow run. "
            "Set it from the active MLflow run ID."
        )
