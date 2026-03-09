"""Tests for T-24: MLflow run logging in data_flow.py.

Verifies that run_data_flow() opens a run in 'minivess_data' experiment
and logs data_n_volumes, data_hash, flow_name tag, and that DataFlowResult
includes mlflow_run_id.

Uses source-level inspection (ast.parse) and minimal functional tests
with a temp mlruns directory. No subprocess — all MLflow calls via real API
with file:// tracking URI.

Refactored (Issue #558): TestDataFlowMlflowFunctional now uses tmp_path fixture
and monkeypatch.setenv() instead of tempfile.TemporaryDirectory() + direct
os.environ manipulation, preventing SQLite telemetry lock collisions.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_DATA_FLOW_SRC = Path("src/minivess/orchestration/flows/data_flow.py")


# ---------------------------------------------------------------------------
# Source-level tests (fast, no I/O)
# ---------------------------------------------------------------------------


class TestDataFlowMlflowSource:
    def test_data_flow_opens_mlflow_experiment(self) -> None:
        """data_flow.py must open an MLflow run in the minivess_data experiment.

        The experiment name is set via resolve_experiment_name(EXPERIMENT_DATA) so that
        debug runs land in a separate experiment. We check for the constant name, not
        the raw string (which lives only in constants.py, not data_flow.py).
        """
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "EXPERIMENT_DATA" in source, (
            "run_data_flow() must use EXPERIMENT_DATA constant (from constants.py) "
            "via resolve_experiment_name(EXPERIMENT_DATA) so debug runs land in a "
            "separate experiment. Hard-coding 'minivess_data' string is forbidden "
            "— use the constant."
        )
        assert "resolve_experiment_name" in source, (
            "run_data_flow() must call resolve_experiment_name(EXPERIMENT_DATA) to "
            "support MINIVESS_DEBUG_SUFFIX-based experiment isolation."
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
    """Functional tests using isolated tmp_path + monkeypatch for each test.

    Using tmp_path fixture + monkeypatch.setenv() instead of
    tempfile.TemporaryDirectory() + direct os.environ manipulation:
    - pytest guarantees cleanup of env vars even on test failure
    - each test gets its own mlruns/ directory (no shared SQLite state)
    - prevents Prefect/MLflow telemetry SQLite lock collisions (Issue #558)
    """

    def _make_data_dir(self, tmp_path: Path) -> Path:
        """Create a minimal data directory with images/labels subdirs."""
        data_dir = tmp_path / "data"
        (data_dir / "images").mkdir(parents=True)
        (data_dir / "labels").mkdir(parents=True)
        for i in range(3):
            (data_dir / "images" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")
            (data_dir / "labels" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")
        return data_dir

    def test_data_flow_result_has_run_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DataFlowResult.mlflow_run_id must not be None after run_data_flow()."""
        import mlflow

        from minivess.orchestration.flows.data_flow import run_data_flow

        tracking_uri = str(tmp_path / "mlruns")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        mlflow.set_tracking_uri(tracking_uri)

        data_dir = self._make_data_dir(tmp_path)
        result = run_data_flow(data_dir=data_dir, n_folds=2)

        assert result.mlflow_run_id is not None, (
            "DataFlowResult.mlflow_run_id must be set after run_data_flow(). "
            "Open an MLflow run in run_data_flow() and store the run_id."
        )

    def test_data_flow_run_in_correct_experiment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLflow run from run_data_flow() must be in 'minivess_data' experiment."""
        import mlflow

        from minivess.orchestration.flows.data_flow import run_data_flow

        tracking_uri = str(tmp_path / "mlruns")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        # Ensure no debug suffix so experiment name is exactly 'minivess_data'
        monkeypatch.delenv("MINIVESS_DEBUG_SUFFIX", raising=False)
        mlflow.set_tracking_uri(tracking_uri)

        data_dir = self._make_data_dir(tmp_path)
        result = run_data_flow(data_dir=data_dir, n_folds=2)

        run = mlflow.get_run(result.mlflow_run_id)
        experiment = mlflow.get_experiment(run.info.experiment_id)
        assert experiment.name == "minivess_data", (
            f"Run must be in 'minivess_data' experiment, got: {experiment.name!r}"
        )

    def test_data_flow_logs_n_volumes_param(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLflow run must have data_n_volumes logged as a param."""
        import mlflow

        from minivess.orchestration.flows.data_flow import run_data_flow

        tracking_uri = str(tmp_path / "mlruns")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        mlflow.set_tracking_uri(tracking_uri)

        data_dir = self._make_data_dir(tmp_path)
        result = run_data_flow(data_dir=data_dir, n_folds=2)

        run = mlflow.get_run(result.mlflow_run_id)
        assert "data_n_volumes" in run.data.params, (
            f"MLflow run must have data_n_volumes param. "
            f"Found params: {list(run.data.params.keys())}"
        )

    def test_data_flow_run_has_flow_name_tag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLflow run from run_data_flow() must have flow_name='data-flow' tag."""
        import mlflow

        from minivess.orchestration.flows.data_flow import run_data_flow

        tracking_uri = str(tmp_path / "mlruns")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        mlflow.set_tracking_uri(tracking_uri)

        data_dir = self._make_data_dir(tmp_path)
        result = run_data_flow(data_dir=data_dir, n_folds=2)

        run = mlflow.get_run(result.mlflow_run_id)
        assert run.data.tags.get("flow_name") == "data-flow", (
            f"MLflow run must have flow_name='data-flow' tag (FLOW_NAME_DATA constant). "
            f"Found tags: {run.data.tags}"
        )
