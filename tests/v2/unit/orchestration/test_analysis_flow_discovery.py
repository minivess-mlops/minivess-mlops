"""Analysis flow training run discovery — Phase 6 Task 6.1.

Verifies that EnsembleBuilder.discover_training_runs() finds runs with
the correct metric format (eval/fold2/dsc, NOT eval_fold2_dsc).

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix-2.xml
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from minivess.observability.metric_keys import MetricKeys


@pytest.fixture()
def mlflow_dir(tmp_path: Path) -> Path:
    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir()
    return tracking_dir


@pytest.fixture()
def client(mlflow_dir: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=str(mlflow_dir))


def _create_training_run(
    client: MlflowClient,
    experiment_name: str,
    loss_type: str,
    *,
    num_folds: int = 3,
    with_eval_metrics: bool = True,
) -> str:
    """Create a realistic training run with tags and eval metrics."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    run = client.create_run(experiment_id)
    run_id = run.info.run_id

    # Tags that builder.py filters on
    client.set_tag(run_id, "loss_function", loss_type)
    client.set_tag(run_id, "flow_name", "training-flow")
    client.set_tag(run_id, "num_folds", str(num_folds))

    if with_eval_metrics:
        # Log eval metrics in SLASH format (what tracking.py produces)
        for fold in range(num_folds):
            eval_key = f"{MetricKeys.EVAL_FOLD_PREFIX}{fold}/dsc"
            client.log_metric(run_id, eval_key, 0.85 + fold * 0.01)
            client.log_metric(
                run_id, f"{MetricKeys.EVAL_FOLD_PREFIX}{fold}/cldice", 0.70
            )

    # Log standard training metrics
    client.log_metric(run_id, "val/dice", 0.88)
    client.log_metric(run_id, "val/loss", 0.12)

    client.set_terminated(run_id, status="FINISHED")
    return run_id


def _make_eval_config(experiment_name: str) -> object:
    """Create a minimal EvaluationConfig for testing."""
    from minivess.config.evaluation_config import EvaluationConfig

    return EvaluationConfig(mlflow_training_experiment=experiment_name)


class TestAnalysisFlowDiscovery:
    """Analysis flow must discover training runs with slash-format metrics."""

    def test_discovers_run_with_slash_metrics(
        self, mlflow_dir: Path, client: MlflowClient
    ) -> None:
        """Builder should find runs that have eval/fold2/dsc metrics."""
        experiment_name = "test_analysis_discovery"
        run_id = _create_training_run(client, experiment_name, "cbdice_cldice")

        from minivess.ensemble.builder import EnsembleBuilder

        eval_cfg = _make_eval_config(experiment_name)
        builder = EnsembleBuilder(
            eval_config=eval_cfg,
            model_config=None,
            tracking_uri=str(mlflow_dir),
        )

        runs = builder.discover_training_runs(require_eval_metrics=True)
        assert len(runs) >= 1, (
            f"Builder should find the training run with eval/fold2/dsc metrics, "
            f"but found {len(runs)} runs"
        )
        assert runs[0]["run_id"] == run_id

    def test_skips_runs_without_eval_metrics(
        self, mlflow_dir: Path, client: MlflowClient
    ) -> None:
        """Builder should skip runs that lack eval fold metrics."""
        experiment_name = "test_skip_incomplete"
        _create_training_run(
            client, experiment_name, "cbdice_cldice", with_eval_metrics=False
        )

        from minivess.ensemble.builder import EnsembleBuilder

        eval_cfg = _make_eval_config(experiment_name)
        builder = EnsembleBuilder(
            eval_config=eval_cfg,
            model_config=None,
            tracking_uri=str(mlflow_dir),
        )

        runs = builder.discover_training_runs(require_eval_metrics=True)
        assert len(runs) == 0, (
            "Builder should skip runs without eval metrics when require_eval_metrics=True"
        )

    def test_eval_metric_key_matches_tracking_format(self) -> None:
        """The eval gate key must match what tracking.py produces."""
        # tracking.py logs: f"{MetricKeys.EVAL_FOLD_PREFIX}{fold_id}/{metric_name}"
        # builder.py checks: f"{MetricKeys.EVAL_FOLD_PREFIX}2/dsc"
        expected = "eval/fold2/dsc"
        actual = f"{MetricKeys.EVAL_FOLD_PREFIX}2/dsc"
        assert actual == expected, (
            f"Metric key mismatch: builder produces '{actual}', expected '{expected}'"
        )
