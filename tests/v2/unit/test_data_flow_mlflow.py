"""Tests for T-24: MLflow run logging in data_flow.py.

Verifies that run_data_flow() opens a run in 'minivess_data' experiment
and logs data_n_volumes, data_hash, flow_name tag, and that DataFlowResult
includes mlflow_run_id.

Uses source-level inspection (ast.parse) and minimal functional tests
with a temp mlruns directory. No subprocess — all MLflow calls via real API
with file:// tracking URI.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

_DATA_FLOW_SRC = Path("src/minivess/orchestration/flows/data_flow.py")


# ---------------------------------------------------------------------------
# Source-level tests (fast, no I/O)
# ---------------------------------------------------------------------------


class TestDataFlowMlflowSource:
    def test_data_flow_opens_mlflow_experiment(self) -> None:
        """data_flow.py must open an MLflow run in 'minivess_data' experiment."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "minivess_data" in source, (
            "run_data_flow() must call mlflow.set_experiment('minivess_data') to "
            "ensure data engineering runs are discoverable by training_flow via FlowContract."
        )

    def test_data_flow_logs_n_volumes(self) -> None:
        """data_flow.py must log data_n_volumes param to MLflow."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "data_n_volumes" in source, (
            "run_data_flow() must log 'data_n_volumes' param to MLflow. "
            "Training flow needs this to verify input data is consistent."
        )

    def test_data_flow_logs_data_hash(self) -> None:
        """data_flow.py must log data_hash param to MLflow."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "data_hash" in source, (
            "run_data_flow() must log 'data_hash' param to MLflow. "
            "This fingerprints the dataset so downstream flows can detect data drift."
        )

    def test_data_flow_tags_flow_name(self) -> None:
        """data_flow.py must tag MLflow run with flow_name='data'."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "flow_name" in source, (
            "run_data_flow() must tag the MLflow run with flow_name='data'. "
            "FlowContract.find_upstream_run() searches by this tag."
        )

    def test_data_flow_calls_flow_contract(self) -> None:
        """data_flow.py must call FlowContract.log_flow_completion()."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "FlowContract" in source, (
            "run_data_flow() must use FlowContract to mark the run as FLOW_COMPLETE. "
            "This is how training_flow discovers the upstream data run."
        )

    def test_data_flow_result_has_mlflow_run_id_field(self) -> None:
        """DataFlowResult dataclass must have an mlflow_run_id field."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Find DataFlowResult class and check its fields
        result_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DataFlowResult":
                result_class = node
                break
        assert result_class is not None, (
            "DataFlowResult class not found in data_flow.py"
        )

        # Collect all annotated field names in the class body
        field_names: list[str] = []
        for stmt in result_class.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                field_names.append(stmt.target.id)

        assert "mlflow_run_id" in field_names, (
            f"DataFlowResult must have an 'mlflow_run_id' field. Found: {field_names}. "
            "Add: mlflow_run_id: str | None = None"
        )


# ---------------------------------------------------------------------------
# Functional tests with temp mlruns
# ---------------------------------------------------------------------------


class TestDataFlowMlflowFunctional:
    def _make_data_dir(self, tmp: Path) -> Path:
        """Create a minimal data directory with images/labels subdirs."""
        data_dir = tmp / "data"
        (data_dir / "images").mkdir(parents=True)
        (data_dir / "labels").mkdir(parents=True)
        # Create a few fake NIfTI files
        for i in range(3):
            (data_dir / "images" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")
            (data_dir / "labels" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")
        return data_dir

    def test_data_flow_result_has_run_id(self) -> None:
        """DataFlowResult.mlflow_run_id must not be None after run_data_flow()."""
        import os

        from minivess.orchestration.flows.data_flow import run_data_flow

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            tracking_uri = f"file://{tmp}/mlruns"
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            data_dir = self._make_data_dir(tmp)
            try:
                result = run_data_flow(data_dir=data_dir, n_folds=2)
            finally:
                del os.environ["MLFLOW_TRACKING_URI"]

            assert result.mlflow_run_id is not None, (
                "DataFlowResult.mlflow_run_id must be set after run_data_flow(). "
                "Open an MLflow run in run_data_flow() and store the run_id."
            )

    def test_data_flow_run_in_correct_experiment(self) -> None:
        """MLflow run from run_data_flow() must be in 'minivess_data' experiment."""
        import os

        import mlflow

        from minivess.orchestration.flows.data_flow import run_data_flow

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            tracking_uri = f"file://{tmp}/mlruns"
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            data_dir = self._make_data_dir(tmp)
            try:
                result = run_data_flow(data_dir=data_dir, n_folds=2)
            finally:
                del os.environ["MLFLOW_TRACKING_URI"]

            run_id = result.mlflow_run_id
            mlflow.set_tracking_uri(tracking_uri)
            run = mlflow.get_run(run_id)
            experiment = mlflow.get_experiment(run.info.experiment_id)
            assert experiment.name == "minivess_data", (
                f"Run must be in 'minivess_data' experiment, got: {experiment.name!r}"
            )

    def test_data_flow_logs_n_volumes_param(self) -> None:
        """MLflow run must have data_n_volumes logged as a param."""
        import os

        import mlflow

        from minivess.orchestration.flows.data_flow import run_data_flow

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            tracking_uri = f"file://{tmp}/mlruns"
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            data_dir = self._make_data_dir(tmp)
            try:
                result = run_data_flow(data_dir=data_dir, n_folds=2)
            finally:
                del os.environ["MLFLOW_TRACKING_URI"]

            mlflow.set_tracking_uri(tracking_uri)
            run = mlflow.get_run(result.mlflow_run_id)
            assert "data_n_volumes" in run.data.params, (
                f"MLflow run must have data_n_volumes param. "
                f"Found params: {list(run.data.params.keys())}"
            )

    def test_data_flow_run_has_flow_name_tag(self) -> None:
        """MLflow run from run_data_flow() must have flow_name='data' tag."""
        import os

        import mlflow

        from minivess.orchestration.flows.data_flow import run_data_flow

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            tracking_uri = f"file://{tmp}/mlruns"
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            data_dir = self._make_data_dir(tmp)
            try:
                result = run_data_flow(data_dir=data_dir, n_folds=2)
            finally:
                del os.environ["MLFLOW_TRACKING_URI"]

            mlflow.set_tracking_uri(tracking_uri)
            run = mlflow.get_run(result.mlflow_run_id)
            assert run.data.tags.get("flow_name") == "data", (
                f"MLflow run must have flow_name='data' tag. "
                f"Found tags: {run.data.tags}"
            )
