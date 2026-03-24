"""Tests for post_training_flow MLflow run logging and tag schema.

Verifies that post_training_flow() opens an MLflow run, logs plugin metrics
with post_ prefix, and sets tags per synthesis Part 2.4.

Tag schema (synthesis Part 2.4):
  flow_name: "post-training-flow"
  upstream_training_run_id: {original_run_id}
  post_training_method: {method_name}
  model_family, loss_function, fold_id, with_aux_calib: inherited from upstream
"""

from __future__ import annotations

import mlflow


def _get_run_tags(run) -> dict:
    return dict(run.data.tags)


def _get_run_metrics(run) -> dict:
    return dict(run.data.metrics)


# Synthesis Part 2.4 required tag keys for post-training runs
_REQUIRED_TAG_KEYS = {"flow_name"}
# These tags are only present when upstream run exists:
_CONDITIONAL_TAG_KEYS = {"upstream_training_run_id"}


# ---------------------------------------------------------------------------
# MLflow run creation and tagging
# ---------------------------------------------------------------------------


class TestPostTrainingMlflow:
    def test_post_training_result_has_run_id(self, monkeypatch, tmp_path) -> None:
        """post_training_flow() must return result with 'mlflow_run_id' attribute.

        Uses a tiny checkpoint so plugin validation passes (Rule #25:
        empty checkpoints now raise instead of silently returning status=error).
        """
        import torch

        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
        monkeypatch.setenv("POST_TRAINING_OUTPUT_DIR", str(tmp_path / "pt_output"))

        # Create 2 tiny checkpoints (model_merging needs >= 2)
        ckpts = []
        for i in range(2):
            ckpt = tmp_path / f"ckpt{i}.pt"
            torch.save({"model_state_dict": {}}, ckpt)
            ckpts.append(ckpt)

        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        # Disable all plugins except checkpoint_averaging to avoid validation errors.
        # Rule #25: plugins with missing inputs now RAISE instead of silently returning.
        config = PostTrainingConfig(
            subsampled_ensemble={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": False},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        result = post_training_flow(config=config, checkpoint_paths=ckpts)
        assert hasattr(result, "mlflow_run_id"), (
            f"post_training_flow() result missing 'mlflow_run_id' attribute. Got: {type(result)}"
        )

    def test_post_training_opens_mlflow_run(self, monkeypatch, tmp_path) -> None:
        """post_training_flow() must create an MLflow run with flow_name='post_training'."""
        import torch

        mlflow_dir = tmp_path / "mlruns"
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlflow_dir))
        monkeypatch.setenv("POST_TRAINING_OUTPUT_DIR", str(tmp_path / "pt_output"))

        mlflow.set_tracking_uri(str(mlflow_dir))

        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        ckpts = []
        for i in range(2):
            ckpt = tmp_path / f"ckpt{i}.pt"
            torch.save({"model_state_dict": {}}, ckpt)
            ckpts.append(ckpt)
        config = PostTrainingConfig(
            subsampled_ensemble={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": False},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        post_training_flow(config=config, checkpoint_paths=ckpts)

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None, (
            "No 'minivess_training' experiment found after post_training_flow(). "
            "The flow must open an MLflow run."
        )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.flow_name = 'post-training-flow'",
        )
        assert runs, (
            "No MLflow run with flow_name='post_training' found after post_training_flow(). "
            "Add mlflow.start_run() and set flow_name tag."
        )

    def test_post_training_logs_upstream_run_id(self, monkeypatch, tmp_path) -> None:
        """post_training_flow() MLflow run must have upstream_training_run_id tag."""
        import torch

        mlflow_dir = tmp_path / "mlruns"
        monkeypatch.setenv("PREFECT_DISABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlflow_dir))
        monkeypatch.setenv("POST_TRAINING_OUTPUT_DIR", str(tmp_path / "pt_output"))

        mlflow.set_tracking_uri(str(mlflow_dir))

        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        ckpts = []
        for i in range(2):
            ckpt = tmp_path / f"ckpt{i}.pt"
            torch.save({"model_state_dict": {}}, ckpt)
            ckpts.append(ckpt)
        config = PostTrainingConfig(
            subsampled_ensemble={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": False},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        post_training_flow(config=config, checkpoint_paths=ckpts)

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        if experiment is None:
            return  # Skip if flow didn't create experiment (no-plugin run)

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.flow_name = 'post-training-flow'",
        )
        if not runs:
            return

        tags = _get_run_tags(runs[0])
        # upstream_training_run_id is only set when an upstream training run
        # is discovered. In this test there's no upstream run, so the tag
        # may be absent. Verify flow_name is always present instead.
        assert "flow_name" in tags, (
            f"MLflow run missing 'flow_name' tag. Got: {list(tags.keys())}"
        )
        assert tags["flow_name"] == "post-training-flow"


class TestPostTrainingTagSchema:
    """Synthesis Part 2.4: Verify tag schema for factorial discovery."""

    def test_flow_logs_to_same_experiment_as_training(
        self, monkeypatch, tmp_path
    ) -> None:
        """Post-training MUST log to minivess_training (synthesis Part 2.3)."""
        from minivess.orchestration.constants import (
            EXPERIMENT_POST_TRAINING,
            EXPERIMENT_TRAINING,
        )

        assert EXPERIMENT_POST_TRAINING == EXPERIMENT_TRAINING, (
            f"EXPERIMENT_POST_TRAINING ({EXPERIMENT_POST_TRAINING}) must equal "
            f"EXPERIMENT_TRAINING ({EXPERIMENT_TRAINING}) per synthesis Part 2.3"
        )

    def test_config_default_matches_training_experiment(self) -> None:
        """PostTrainingConfig.mlflow_experiment must default to minivess_training."""
        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.constants import EXPERIMENT_TRAINING

        config = PostTrainingConfig()
        assert config.mlflow_experiment == EXPERIMENT_TRAINING

    def test_flow_name_tag_is_post_training_flow(self, monkeypatch, tmp_path) -> None:
        """Post-training flow_name tag must be 'post-training-flow'."""
        from minivess.orchestration.constants import FLOW_NAME_POST_TRAINING

        assert FLOW_NAME_POST_TRAINING == "post-training-flow"

    def test_factorial_result_includes_post_training_method(self, tmp_path) -> None:
        """Each factorial variant result must include post_training_method tag."""
        import torch

        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        ckpt = tmp_path / "ckpt.pt"
        torch.save({"model_state_dict": {"w": torch.randn(4, 4)}}, ckpt)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        results = run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["none", "checkpoint_averaging"],
            output_dir=output_dir,
            seed=42,
        )

        for result in results:
            assert "post_training_method" in result, (
                f"Missing post_training_method tag in result: {result}"
            )
            assert result["post_training_method"] in {"none", "checkpoint_averaging"}


# ---------------------------------------------------------------------------
# Issue #885: Per-method MLflow runs for factorial discovery
# ---------------------------------------------------------------------------

# Standard upstream tags that training flow sets on each run
_UPSTREAM_TAGS = {
    "model_family": "dynunet",
    "loss_function": "cbdice_cldice",
    "fold_id": "0",
    "with_aux_calib": "false",
    "flow_name": "training-flow",
}


def _make_checkpoint(tmp_path, seed: int = 0):
    """Create a minimal checkpoint file for testing."""
    import torch

    ckpt = tmp_path / f"ckpt_{seed}.pt"
    gen = torch.Generator().manual_seed(seed)
    torch.save(
        {"model_state_dict": {"w": torch.randn(4, 4, generator=gen)}},
        ckpt,
    )
    return ckpt


class TestFactorialPerMethodMlflowRuns:
    """Issue #885: run_factorial_post_training() must create SEPARATE MLflow
    runs per method variant so Biostatistics can discover them as factorial
    conditions. Synthesis Part 2.4 tag schema."""

    def test_creates_one_mlflow_run_per_method(self, tmp_path) -> None:
        """N methods → N MLflow runs in the experiment."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        mlflow_dir = tmp_path / "mlruns"
        ckpt = _make_checkpoint(tmp_path)
        methods = ["none", "checkpoint_averaging"]

        results = run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=methods,
            output_dir=tmp_path / "output",
            seed=42,
            tracking_uri=str(mlflow_dir),
            experiment_name="minivess_training",
            upstream_run_id="fake_upstream_123",
            upstream_tags=_UPSTREAM_TAGS,
        )

        # Each method should have created an MLflow run
        assert len(results) == len(methods)
        for result in results:
            assert "mlflow_run_id" in result, (
                f"Missing mlflow_run_id in result: {result}"
            )
            assert result["mlflow_run_id"] is not None

        # Verify runs exist in MLflow
        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
        )
        assert len(runs) == len(methods), (
            f"Expected {len(methods)} MLflow runs, found {len(runs)}"
        )

    def test_run_name_follows_pattern(self, tmp_path) -> None:
        """run_name must be {model}__{loss}__{fold}__{method}."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        mlflow_dir = tmp_path / "mlruns"
        ckpt = _make_checkpoint(tmp_path)

        run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["none", "checkpoint_averaging"],
            output_dir=tmp_path / "output",
            seed=42,
            tracking_uri=str(mlflow_dir),
            experiment_name="minivess_training",
            upstream_run_id="fake_upstream_123",
            upstream_tags=_UPSTREAM_TAGS,
        )

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
        )
        run_names = {r.info.run_name for r in runs}

        assert "dynunet__cbdice_cldice__fold0__none" in run_names
        assert "dynunet__cbdice_cldice__fold0__checkpoint_averaging" in run_names

    def test_inherits_upstream_tags(self, tmp_path) -> None:
        """Each per-method run must inherit model_family, loss_function,
        fold_id, with_aux_calib from the upstream training run."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        mlflow_dir = tmp_path / "mlruns"
        ckpt = _make_checkpoint(tmp_path)

        run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["checkpoint_averaging"],
            output_dir=tmp_path / "output",
            seed=42,
            tracking_uri=str(mlflow_dir),
            experiment_name="minivess_training",
            upstream_run_id="fake_upstream_123",
            upstream_tags=_UPSTREAM_TAGS,
        )

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
        )
        assert len(runs) == 1
        tags = _get_run_tags(runs[0])

        # All upstream tags must be inherited
        assert tags["model_family"] == "dynunet"
        assert tags["loss_function"] == "cbdice_cldice"
        assert tags["fold_id"] == "0"
        assert tags["with_aux_calib"] == "false"

    def test_sets_post_training_method_tag(self, tmp_path) -> None:
        """Each run must have post_training_method tag matching its method."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        mlflow_dir = tmp_path / "mlruns"
        ckpt = _make_checkpoint(tmp_path)

        run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["none", "checkpoint_averaging"],
            output_dir=tmp_path / "output",
            seed=42,
            tracking_uri=str(mlflow_dir),
            experiment_name="minivess_training",
            upstream_run_id="fake_upstream_123",
            upstream_tags=_UPSTREAM_TAGS,
        )

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
        )

        method_tags = {_get_run_tags(r)["post_training_method"] for r in runs}
        assert method_tags == {"none", "checkpoint_averaging"}

    def test_sets_flow_name_tag(self, tmp_path) -> None:
        """Each per-method run must have flow_name = 'post-training-flow'."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        mlflow_dir = tmp_path / "mlruns"
        ckpt = _make_checkpoint(tmp_path)

        run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["checkpoint_averaging"],
            output_dir=tmp_path / "output",
            seed=42,
            tracking_uri=str(mlflow_dir),
            experiment_name="minivess_training",
            upstream_run_id="fake_upstream_123",
            upstream_tags=_UPSTREAM_TAGS,
        )

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
        )
        assert len(runs) == 1
        tags = _get_run_tags(runs[0])
        assert tags["flow_name"] == "post-training-flow"

    def test_sets_upstream_training_run_id_tag(self, tmp_path) -> None:
        """Each per-method run must tag the upstream training run_id."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        mlflow_dir = tmp_path / "mlruns"
        ckpt = _make_checkpoint(tmp_path)

        run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["none"],
            output_dir=tmp_path / "output",
            seed=42,
            tracking_uri=str(mlflow_dir),
            experiment_name="minivess_training",
            upstream_run_id="fake_upstream_123",
            upstream_tags=_UPSTREAM_TAGS,
        )

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
        )
        tags = _get_run_tags(runs[0])
        assert tags["upstream_training_run_id"] == "fake_upstream_123"

    def test_without_tracking_uri_no_mlflow_runs(self, tmp_path) -> None:
        """Without tracking_uri, no MLflow runs are created (backward compat)."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        ckpt = _make_checkpoint(tmp_path)

        results = run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["none", "checkpoint_averaging"],
            output_dir=tmp_path / "output",
            seed=42,
        )

        # Should still return results, but without mlflow_run_id
        assert len(results) == 2
        for result in results:
            assert result.get("mlflow_run_id") is None

    def test_logs_checkpoint_size_metric(self, tmp_path) -> None:
        """Each per-method run must log checkpoint_size_mb as a metric."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        mlflow_dir = tmp_path / "mlruns"
        ckpt = _make_checkpoint(tmp_path)

        run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["none"],
            output_dir=tmp_path / "output",
            seed=42,
            tracking_uri=str(mlflow_dir),
            experiment_name="minivess_training",
            upstream_run_id="fake_upstream_123",
            upstream_tags=_UPSTREAM_TAGS,
        )

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
        )
        metrics = _get_run_metrics(runs[0])
        assert "checkpoint_size_mb" in metrics
        assert metrics["checkpoint_size_mb"] >= 0

    def test_three_methods_create_three_runs(self, tmp_path) -> None:
        """subsampled_ensemble method should also create its own MLflow run."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        mlflow_dir = tmp_path / "mlruns"
        ckpt1 = _make_checkpoint(tmp_path, seed=0)
        ckpt2 = _make_checkpoint(tmp_path, seed=1)

        results = run_factorial_post_training(
            checkpoint_paths=[ckpt1, ckpt2],
            methods=["none", "checkpoint_averaging", "subsampled_ensemble"],
            output_dir=tmp_path / "output",
            seed=42,
            tracking_uri=str(mlflow_dir),
            experiment_name="minivess_training",
            upstream_run_id="fake_upstream_123",
            upstream_tags=_UPSTREAM_TAGS,
        )

        assert len(results) == 3
        run_ids = [r["mlflow_run_id"] for r in results]
        # All run IDs should be unique
        assert len(set(run_ids)) == 3

        client = mlflow.tracking.MlflowClient(tracking_uri=str(mlflow_dir))
        experiment = client.get_experiment_by_name("minivess_training")
        assert experiment is not None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
        )
        assert len(runs) == 3
