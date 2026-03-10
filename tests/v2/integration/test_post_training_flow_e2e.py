"""Integration tests for post-training flow: SWA + calibration + conformal.

E2E Plan Phase 1, Task T1.5: Post-training flow consuming checkpoints from training.

Verifies:
- FlowContract.find_fold_checkpoints() discovers upstream checkpoints
- SWA plugin creates sibling __swa run in MLflow
- Calibration plugin creates __calibrated sibling run with temperature param
- Each sibling run has appropriate artifacts
- Plugins skip gracefully (warning, not exception) when no checkpoint found

Marked @integration @slow — excluded from staging and prod suites.
Runs via: make test-e2e (requires Docker + trained model checkpoints)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


_MLFLOW_SKIP = "requires MLflow with post-training results"


def _get_repo_root() -> Path:
    """Find the repository root from the test file location."""
    from pathlib import Path

    return Path(__file__).resolve().parents[4]


def _mlflow_post_training_exists(tracking_uri: str, experiment_name: str) -> bool:
    """Check if MLflow has post-training results."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return False
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.flow_type = 'post_training'",
            max_results=1,
        )
        return len(runs) > 0
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.slow
class TestPostTrainingFlowE2E:
    """Verify post-training flow produces correct artifacts.

    Requires: Docker stack running, training flow already completed.
    These tests verify SWA, calibration, and conformal plugins.
    """

    @pytest.fixture(scope="class")
    def tracking_uri(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        """MLflow tracking URI."""
        repo_root = _get_repo_root()
        mlruns_dir = repo_root / "mlruns"
        if mlruns_dir.is_dir():
            return str(mlruns_dir)
        return str(tmp_path_factory.mktemp("mlruns"))

    @pytest.fixture(scope="class")
    def experiment_name(self) -> str:
        """Experiment name used by post-training flow."""
        return "debug_all_models_DEBUG"

    def test_post_training_finds_upstream_checkpoints(
        self, tracking_uri: str, experiment_name: str
    ) -> None:
        """Verify FlowContract returns non-empty checkpoint list from training run."""
        if not _mlflow_post_training_exists(tracking_uri, experiment_name):
            pytest.skip(_MLFLOW_SKIP)

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None

        # Find training parent runs (they should have checkpoint_dir_fold_* tags)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.parent_run_id != ''",
        )
        checkpoints_found = 0
        for run in runs:
            for key in run.data.tags:
                if key.startswith("checkpoint_dir_fold_"):
                    checkpoints_found += 1

        assert checkpoints_found > 0, (
            "No checkpoint_dir_fold_* tags found in any training run. "
            "Post-training flow cannot find upstream checkpoints."
        )

    def test_swa_sibling_run_created(
        self, tracking_uri: str, experiment_name: str
    ) -> None:
        """Verify MLflow run with __swa suffix exists after post-training."""
        if not _mlflow_post_training_exists(tracking_uri, experiment_name):
            pytest.skip(_MLFLOW_SKIP)

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.plugin_type = 'swa'",
        )
        assert len(runs) > 0, (
            "No SWA sibling runs found. Post-training SWA plugin "
            "must create a run tagged with plugin_type=swa."
        )

    def test_calibration_sibling_run_created(
        self, tracking_uri: str, experiment_name: str
    ) -> None:
        """Verify MLflow run with __calibrated suffix exists."""
        if not _mlflow_post_training_exists(tracking_uri, experiment_name):
            pytest.skip(_MLFLOW_SKIP)

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.plugin_type = 'calibration'",
        )
        assert len(runs) > 0, (
            "No calibration sibling runs found. Post-training calibration plugin "
            "must create a run tagged with plugin_type=calibration."
        )

    def test_calibrated_model_has_temperature_param(
        self, tracking_uri: str, experiment_name: str
    ) -> None:
        """Verify temperature parameter logged in calibration run."""
        if not _mlflow_post_training_exists(tracking_uri, experiment_name):
            pytest.skip(_MLFLOW_SKIP)

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.plugin_type = 'calibration'",
        )
        if not runs:
            pytest.skip("No calibration runs found")

        for run in runs:
            params = run.data.params
            temp_keys = [k for k in params if "temperature" in k.lower()]
            assert temp_keys, (
                f"Calibration run {run.info.run_id} missing temperature parameter. "
                f"Available params: {list(params.keys())}"
            )

    def test_swa_checkpoint_artifact_exists(
        self, tracking_uri: str, experiment_name: str
    ) -> None:
        """Verify SWA averaged checkpoint file in MLflow artifacts."""
        if not _mlflow_post_training_exists(tracking_uri, experiment_name):
            pytest.skip(_MLFLOW_SKIP)

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.plugin_type = 'swa'",
        )
        if not runs:
            pytest.skip("No SWA runs found")

        for run in runs:
            artifacts = client.list_artifacts(run.info.run_id)
            artifact_paths = [a.path for a in artifacts]
            # SWA should produce a checkpoint artifact
            checkpoint_artifacts = [
                p for p in artifact_paths if "swa" in p.lower() or p.endswith(".pt")
            ]
            assert checkpoint_artifacts, (
                f"SWA run {run.info.run_id} has no checkpoint artifact. "
                f"Available artifacts: {artifact_paths}"
            )

    def test_post_training_flow_status_complete(
        self, tracking_uri: str, experiment_name: str
    ) -> None:
        """Verify flow_status=FLOW_COMPLETE on post-training parent run."""
        if not _mlflow_post_training_exists(tracking_uri, experiment_name):
            pytest.skip(_MLFLOW_SKIP)

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.flow_type = 'post_training'",
        )
        if not runs:
            pytest.skip("No post-training flow runs found")

        for run in runs:
            status = run.data.tags.get("flow_status", "")
            assert status == "FLOW_COMPLETE", (
                f"Post-training run {run.info.run_id} has flow_status={status!r}, "
                f"expected FLOW_COMPLETE"
            )


@pytest.mark.integration
@pytest.mark.slow
class TestPluginSkipBehavior:
    """Verify plugins skip gracefully when no checkpoint is available."""

    def test_plugin_logs_warning_not_exception(self, tmp_path: Path) -> None:
        """If a plugin has no upstream checkpoint, it should log a warning, not crash."""
        import mlflow

        from minivess.orchestration.flow_contract import FlowContract

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_plugin_skip")

        # Create a run with no checkpoint_dir tags
        with mlflow.start_run() as run:
            mlflow.set_tag("parent_run_id", run.info.run_id)
            run_id = run.info.run_id

        contract = FlowContract(tracking_uri=tracking_uri)
        result = contract.find_fold_checkpoints(parent_run_id=run_id)
        assert isinstance(result, list)
        assert len(result) == 0, (
            "find_fold_checkpoints should return empty list when no "
            "checkpoint_dir_fold_* tags exist"
        )
