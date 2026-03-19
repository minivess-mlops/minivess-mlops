"""Tests for T-22: MLflow epoch-level curve logging in training_flow().

Verifies that:
- SegmentationTrainer.fit() with an ExperimentTracker logs epoch metrics
  to MLflow at the correct step numbers (step=1, step=2, step=3...)
- val_loss and train_loss both appear in the metric history
- learning_rate is logged at each epoch
- train_one_fold_task source wires ExperimentTracker to SegmentationTrainer

NO subprocess invocation — uses temp mlruns directories and mock data.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from minivess.pipeline.loss_functions import build_loss_function

# ---------------------------------------------------------------------------
# Helpers: minimal model + fake DataLoader for trainer construction
# ---------------------------------------------------------------------------


class _Output:
    """Wraps a tensor so trainer code can access .logits."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _TinyConvModel(nn.Module):
    """Minimal Conv3d model that passes trainer's output.logits access."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> _Output:
        return _Output(logits=self.conv(x))


def _fake_loader(n_batches: int = 2) -> list[dict[str, torch.Tensor]]:
    """Return a list of dicts acting as a DataLoader.

    label shape: [B, 1, H, W, D] with class indices (MONAI DiceCE convention).
    """
    return [
        {
            "image": torch.zeros(1, 1, 8, 8, 4),
            # MONAI DiceCELoss expects [B, 1, H, W, D] class index labels
            "label": torch.zeros(1, 1, 8, 8, 4),
        }
        for _ in range(n_batches)
    ]


def _build_tracker(tracking_uri: str, max_epochs: int = 3) -> Any:
    """Build a minimal ExperimentTracker pointing to temp mlruns dir."""
    from minivess.config.models import (
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        ModelFamily,
        TrainingConfig,
    )
    from minivess.observability.tracking import ExperimentTracker

    data_config = DataConfig(
        dataset_name="test",
        data_dir=Path("data/raw"),
        patch_size=(8, 8, 4),
        num_workers=0,
    )
    model_config = ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="dynunet",
        in_channels=1,
        out_channels=2,
    )
    training_config = TrainingConfig(
        max_epochs=max_epochs,
        batch_size=1,
        warmup_epochs=0,
        early_stopping_patience=max_epochs + 1,
    )
    exp_config = ExperimentConfig(
        experiment_name="test_epoch_curves",
        run_name="test_run",
        data=data_config,
        model=model_config,
        training=training_config,
    )
    return ExperimentTracker(exp_config, tracking_uri=tracking_uri)


# ---------------------------------------------------------------------------
# Functional integration tests
# ---------------------------------------------------------------------------


class TestEpochCurveLogging:
    def test_val_loss_at_each_epoch(self) -> None:
        """After fit() with max_epochs=3, val_loss history has 3 entries at steps 1,2,3."""
        import mlflow
        from mlflow import MlflowClient

        from minivess.config.models import TrainingConfig
        from minivess.pipeline.metrics import SegmentationMetrics
        from minivess.pipeline.trainer import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmp:
            tracking_uri = f"file://{tmp}/mlruns"
            tracker = _build_tracker(tracking_uri, max_epochs=3)

            train_loader = _fake_loader()
            val_loader = _fake_loader()
            model = _TinyConvModel()
            criterion = build_loss_function("dice_ce")
            training_config = TrainingConfig(
                max_epochs=3,
                batch_size=1,
                warmup_epochs=0,
                early_stopping_patience=10,
            )
            metrics = SegmentationMetrics(num_classes=2, device="cpu")

            trainer = SegmentationTrainer(
                model,
                training_config,
                device="cpu",
                metrics=metrics,
                criterion=criterion,
                val_roi_size=(8, 8, 4),
                sw_batch_size=1,
                tracker=tracker,
            )

            with tracker.start_run():
                result = trainer.fit(train_loader, val_loader)

            run_id = result.get("mlflow_run_id")
            assert run_id is not None, (
                "fit() must return mlflow_run_id when tracker is set"
            )

            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)
            history = client.get_metric_history(run_id, "val/loss")

            assert len(history) >= 3, (
                f"Expected ≥3 val/loss entries, got {len(history)}. "
                "Trainer must log val/loss at every epoch via tracker.log_epoch_metrics()."
            )

    def test_epoch_steps_are_sequential(self) -> None:
        """Epoch metric steps must be 1, 2, 3 (not 0, 1, 2 or random)."""
        import mlflow
        from mlflow import MlflowClient

        from minivess.config.models import TrainingConfig
        from minivess.pipeline.metrics import SegmentationMetrics
        from minivess.pipeline.trainer import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmp:
            tracking_uri = f"file://{tmp}/mlruns"
            tracker = _build_tracker(tracking_uri, max_epochs=3)

            training_config = TrainingConfig(
                max_epochs=3,
                batch_size=1,
                warmup_epochs=0,
                early_stopping_patience=10,
            )
            trainer = SegmentationTrainer(
                _TinyConvModel(),
                training_config,
                device="cpu",
                metrics=SegmentationMetrics(num_classes=2, device="cpu"),
                criterion=build_loss_function("dice_ce"),
                val_roi_size=(8, 8, 4),
                sw_batch_size=1,
                tracker=tracker,
            )

            with tracker.start_run():
                result = trainer.fit(_fake_loader(), _fake_loader())

            run_id = result["mlflow_run_id"]
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)
            history = client.get_metric_history(run_id, "val/loss")

            steps = [h.step for h in history]
            assert steps == list(range(1, len(steps) + 1)), (
                f"Steps must be sequential starting at 1, got: {steps}"
            )

    def test_train_loss_logged_at_each_epoch(self) -> None:
        """train_loss must appear in MLflow metric history."""
        import mlflow
        from mlflow import MlflowClient

        from minivess.config.models import TrainingConfig
        from minivess.pipeline.metrics import SegmentationMetrics
        from minivess.pipeline.trainer import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmp:
            tracking_uri = f"file://{tmp}/mlruns"
            tracker = _build_tracker(tracking_uri, max_epochs=2)

            training_config = TrainingConfig(
                max_epochs=2,
                batch_size=1,
                warmup_epochs=0,
                early_stopping_patience=10,
            )
            trainer = SegmentationTrainer(
                _TinyConvModel(),
                training_config,
                device="cpu",
                metrics=SegmentationMetrics(num_classes=2, device="cpu"),
                criterion=build_loss_function("dice_ce"),
                val_roi_size=(8, 8, 4),
                sw_batch_size=1,
                tracker=tracker,
            )

            with tracker.start_run():
                result = trainer.fit(_fake_loader(), _fake_loader())

            run_id = result["mlflow_run_id"]
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)
            history = client.get_metric_history(run_id, "train/loss")

            assert len(history) >= 2, (
                f"Expected ≥2 train/loss entries, got {len(history)}. "
                "train/loss must be logged at every epoch."
            )

    def test_learning_rate_logged(self) -> None:
        """learning_rate must appear in the MLflow run metrics."""
        import mlflow
        from mlflow import MlflowClient

        from minivess.config.models import TrainingConfig
        from minivess.pipeline.metrics import SegmentationMetrics
        from minivess.pipeline.trainer import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmp:
            tracking_uri = f"file://{tmp}/mlruns"
            tracker = _build_tracker(tracking_uri, max_epochs=1)

            training_config = TrainingConfig(
                max_epochs=1,
                batch_size=1,
                warmup_epochs=0,
                early_stopping_patience=10,
            )
            trainer = SegmentationTrainer(
                _TinyConvModel(),
                training_config,
                device="cpu",
                metrics=SegmentationMetrics(num_classes=2, device="cpu"),
                criterion=build_loss_function("dice_ce"),
                val_roi_size=(8, 8, 4),
                sw_batch_size=1,
                tracker=tracker,
            )

            with tracker.start_run():
                result = trainer.fit(_fake_loader(), _fake_loader())

            run_id = result["mlflow_run_id"]
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)
            all_metrics = client.get_run(run_id).data.metrics
            # Trainer logs as optim/lr (slash-prefix convention, #790)
            assert "optim/lr" in all_metrics, (
                f"optim/lr not found in MLflow metrics: {list(all_metrics.keys())}. "
                "Trainer must log optim/lr each epoch via tracker.log_epoch_metrics()."
            )


# ---------------------------------------------------------------------------
# Source-level: train_one_fold_task must wire tracker to trainer
# ---------------------------------------------------------------------------


class TestTrainOneFoldTaskWiring:
    _TRAIN_FLOW_SRC = Path("src/minivess/orchestration/flows/train_flow.py")

    def _parse_train_flow(self) -> ast.Module:
        return ast.parse(self._TRAIN_FLOW_SRC.read_text(encoding="utf-8"))

    def test_train_one_fold_task_references_experiment_tracker(self) -> None:
        """train_one_fold_task must reference ExperimentTracker for MLflow run context."""
        source = self._TRAIN_FLOW_SRC.read_text(encoding="utf-8")
        assert "ExperimentTracker" in source, (
            "train_flow.py must reference ExperimentTracker. "
            "train_one_fold_task() must create an ExperimentTracker and pass it to "
            "SegmentationTrainer so epoch metrics are logged to MLflow."
        )

    def test_train_one_fold_task_passes_tracker_to_trainer(self) -> None:
        """train_one_fold_task must pass tracker= kwarg to SegmentationTrainer."""
        source = self._TRAIN_FLOW_SRC.read_text(encoding="utf-8")
        assert "tracker=" in source, (
            "train_flow.py must pass tracker=tracker to SegmentationTrainer. "
            "Without this, trainer.tracker is None and no epoch metrics are logged."
        )

    def test_train_one_fold_task_uses_start_run(self) -> None:
        """train_one_fold_task must open a tracker run context for epoch logging."""
        source = self._TRAIN_FLOW_SRC.read_text(encoding="utf-8")
        assert "start_run" in source, (
            "train_flow.py must call tracker.start_run() to open an MLflow run. "
            "Epoch metrics (val_loss, train_loss) are logged within this context."
        )
