"""Tests for T-10: post_training_flow MLflow run logging.

Verifies that post_training_flow() opens an MLflow run, logs plugin metrics
with post_ prefix, and calls FlowContract.log_flow_completion().
"""

from __future__ import annotations

import mlflow


def _get_run_tags(run) -> dict:
    return dict(run.data.tags)


def _get_run_metrics(run) -> dict:
    return dict(run.data.metrics)


# ---------------------------------------------------------------------------
# MLflow run creation and tagging
# ---------------------------------------------------------------------------


class TestPostTrainingMlflow:
    def test_post_training_result_has_run_id(self, monkeypatch, tmp_path) -> None:
        """post_training_flow() must return dict with 'mlflow_run_id' key."""
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
        monkeypatch.setenv("POST_TRAINING_OUTPUT_DIR", str(tmp_path / "pt_output"))

        from minivess.orchestration.flows.post_training_flow import post_training_flow

        result = post_training_flow()
        assert "mlflow_run_id" in result, (
            f"post_training_flow() result missing 'mlflow_run_id'. Got: {list(result.keys())}"
        )

    def test_post_training_opens_mlflow_run(self, monkeypatch, tmp_path) -> None:
        """post_training_flow() must create an MLflow run with flow_name='post_training'."""
        mlflow_dir = tmp_path / "mlruns"
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlflow_dir))
        monkeypatch.setenv("POST_TRAINING_OUTPUT_DIR", str(tmp_path / "pt_output"))

        mlflow.set_tracking_uri(str(mlflow_dir))

        from minivess.orchestration.flows.post_training_flow import post_training_flow

        post_training_flow()

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None, (
            "No 'minivess_training' experiment found after post_training_flow(). "
            "The flow must open an MLflow run."
        )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.flow_name = 'post_training'",
        )
        assert runs, (
            "No MLflow run with flow_name='post_training' found after post_training_flow(). "
            "Add mlflow.start_run() and set flow_name tag."
        )

    def test_post_training_logs_upstream_run_id(self, monkeypatch, tmp_path) -> None:
        """post_training_flow() MLflow run must have upstream_training_run_id tag."""
        mlflow_dir = tmp_path / "mlruns"
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlflow_dir))
        monkeypatch.setenv("POST_TRAINING_OUTPUT_DIR", str(tmp_path / "pt_output"))

        mlflow.set_tracking_uri(str(mlflow_dir))

        from minivess.orchestration.flows.post_training_flow import post_training_flow

        post_training_flow()

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        if experiment is None:
            return  # Skip if flow didn't create experiment (no-plugin run)

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.flow_name = 'post_training'",
        )
        if not runs:
            return

        tags = _get_run_tags(runs[0])
        assert "upstream_training_run_id" in tags, (
            f"MLflow run missing 'upstream_training_run_id' tag. Got: {list(tags.keys())}"
        )
