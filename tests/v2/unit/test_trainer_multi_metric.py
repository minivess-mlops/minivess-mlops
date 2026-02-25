"""Tests for Phase 2: MultiMetricTracker integration into SegmentationTrainer.

RED phase — all tests written before implementation.
Tests cover:
- CheckpointConfig / TrackedMetricConfig models
- TrainingConfig.checkpoint field
- Trainer builds MultiMetricTracker from config
- fit() saves best_<metric>.pth per improved metric
- fit() saves last.pth when save_last=True
- fit() saves metric_history.json when save_history=True
- fit() uses MultiMetricTracker.should_stop() for early stopping
- fit() backward-compat return dict (best_val_loss, history)
- Saved checkpoint files are self-contained (full metric dict)
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import Any
from unittest.mock import MagicMock

import torch
from monai.losses import DiceCELoss

from minivess.adapters.base import SegmentationOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_model(num_classes: int = 2) -> MagicMock:
    """Return a mock ModelAdapter that yields random logits."""
    model = MagicMock()
    model.parameters.return_value = [torch.randn(2, 2, requires_grad=True)]
    model.train.return_value = None
    model.eval.return_value = None
    model.to.return_value = model

    def _forward(images: torch.Tensor) -> SegmentationOutput:
        b = images.shape[0]
        logits = torch.randn(b, num_classes, 4, 4, 4, requires_grad=True)
        pred = torch.softmax(logits, dim=1)
        return SegmentationOutput(prediction=pred, logits=logits, metadata={})

    model.side_effect = _forward
    model.__call__ = _forward

    # save_checkpoint writes a torch file
    def _save_checkpoint(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"dummy": torch.tensor(1.0)}, path)

    model.save_checkpoint.side_effect = _save_checkpoint
    model.state_dict.return_value = {"dummy": torch.tensor(1.0)}
    return model


def _make_loader(
    num_batches: int = 2, batch_size: int = 1
) -> list[dict[str, torch.Tensor]]:
    """Return a synthetic data loader (list of batch dicts)."""
    return [
        {
            "image": torch.randn(batch_size, 1, 4, 4, 4),
            "label": torch.randint(0, 2, (batch_size, 1, 4, 4, 4)),
        }
        for _ in range(num_batches)
    ]


def _make_training_config(**overrides: Any):
    """Return a minimal TrainingConfig for testing."""
    from minivess.config.models import TrainingConfig

    defaults: dict[str, Any] = {
        "max_epochs": 3,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "warmup_epochs": 0,
        "mixed_precision": False,
        "gradient_clip_val": 0.0,
        "early_stopping_patience": 10,
        "num_folds": 2,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# T1: Config model tests
# ---------------------------------------------------------------------------


class TestCheckedMetricConfig:
    """Tests for TrackedMetricConfig."""

    def test_defaults(self) -> None:
        """TrackedMetricConfig should have sensible defaults."""
        from minivess.config.models import TrackedMetricConfig

        cfg = TrackedMetricConfig(name="val_loss")
        assert cfg.name == "val_loss"
        assert cfg.direction == "minimize"
        assert cfg.patience == 10

    def test_maximize_direction(self) -> None:
        """TrackedMetricConfig allows 'maximize' direction."""
        from minivess.config.models import TrackedMetricConfig

        cfg = TrackedMetricConfig(name="val_dice", direction="maximize", patience=20)
        assert cfg.direction == "maximize"
        assert cfg.patience == 20


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_defaults(self) -> None:
        """Default CheckpointConfig has a single val_loss minimize tracker."""
        from minivess.config.models import CheckpointConfig

        cfg = CheckpointConfig()
        assert len(cfg.tracked_metrics) == 1
        assert cfg.tracked_metrics[0].name == "val_loss"
        assert cfg.tracked_metrics[0].direction == "minimize"
        assert cfg.primary_metric == "val_loss"
        assert cfg.early_stopping_strategy == "all"
        assert cfg.save_last is True
        assert cfg.save_history is True

    def test_multi_metric(self) -> None:
        """CheckpointConfig can hold multiple tracked metrics."""
        from minivess.config.models import CheckpointConfig, TrackedMetricConfig

        cfg = CheckpointConfig(
            tracked_metrics=[
                TrackedMetricConfig(name="val_loss", direction="minimize", patience=15),
                TrackedMetricConfig(name="val_dice", direction="maximize", patience=20),
            ],
            primary_metric="val_loss",
        )
        assert len(cfg.tracked_metrics) == 2
        names = {m.name for m in cfg.tracked_metrics}
        assert names == {"val_loss", "val_dice"}

    def test_min_delta_and_min_epochs(self) -> None:
        """CheckpointConfig exposes min_delta and min_epochs."""
        from minivess.config.models import CheckpointConfig

        cfg = CheckpointConfig(min_delta=1e-3, min_epochs=5)
        assert cfg.min_delta == 1e-3
        assert cfg.min_epochs == 5


class TestTrainingConfigCheckpointField:
    """TrainingConfig should include a checkpoint field."""

    def test_has_checkpoint(self) -> None:
        """TrainingConfig default includes a CheckpointConfig."""
        from minivess.config.models import CheckpointConfig, TrainingConfig

        cfg = TrainingConfig()
        assert isinstance(cfg.checkpoint, CheckpointConfig)

    def test_checkpoint_is_configurable(self) -> None:
        """TrainingConfig.checkpoint can be overridden."""
        from minivess.config.models import (
            CheckpointConfig,
            TrackedMetricConfig,
            TrainingConfig,
        )

        custom_ckpt = CheckpointConfig(
            tracked_metrics=[
                TrackedMetricConfig(name="val_loss", direction="minimize"),
                TrackedMetricConfig(name="val_dice", direction="maximize"),
            ],
            primary_metric="val_loss",
        )
        cfg = TrainingConfig(checkpoint=custom_ckpt)
        assert len(cfg.checkpoint.tracked_metrics) == 2


# ---------------------------------------------------------------------------
# T2: Trainer init tests
# ---------------------------------------------------------------------------


class TestTrainerBuildsMultiTracker:
    """SegmentationTrainer should build a MultiMetricTracker in __init__."""

    def test_has_multi_tracker_attribute(self) -> None:
        """Trainer.__init__ creates a _multi_tracker attribute."""
        from minivess.pipeline.multi_metric_tracker import MultiMetricTracker
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        assert hasattr(trainer, "_multi_tracker")
        assert isinstance(trainer._multi_tracker, MultiMetricTracker)

    def test_has_metric_history_attribute(self) -> None:
        """Trainer.__init__ creates a _metric_history attribute."""
        from minivess.pipeline.multi_metric_tracker import MetricHistory
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        assert hasattr(trainer, "_metric_history")
        assert isinstance(trainer._metric_history, MetricHistory)

    def test_no_legacy_attributes(self) -> None:
        """Trainer should NOT have _best_val_loss or _patience_counter attributes."""
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        assert not hasattr(trainer, "_best_val_loss"), (
            "_best_val_loss should be removed from trainer"
        )
        assert not hasattr(trainer, "_patience_counter"), (
            "_patience_counter should be removed from trainer"
        )

    def test_tracker_built_from_config_metrics(self) -> None:
        """_multi_tracker should contain trackers matching config.checkpoint.tracked_metrics."""
        from minivess.config.models import CheckpointConfig, TrackedMetricConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        custom_ckpt = CheckpointConfig(
            tracked_metrics=[
                TrackedMetricConfig(name="val_loss", direction="minimize", patience=5),
                TrackedMetricConfig(name="val_dice", direction="maximize", patience=7),
            ],
            primary_metric="val_loss",
        )
        config = _make_training_config(checkpoint=custom_ckpt)
        model = _make_fake_model()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        tracker_names = {t.name for t in trainer._multi_tracker.trackers}
        assert tracker_names == {"val_loss", "val_dice"}


# ---------------------------------------------------------------------------
# T3: fit() checkpoint saving tests
# ---------------------------------------------------------------------------


class TestFitSavesMultipleCheckpoints:
    """fit() saves best_<metric>.pth files for each improved metric."""

    def test_saves_best_val_loss_checkpoint(self, tmp_path: Path) -> None:
        """fit() saves best_val_loss.pth when val_loss improves."""
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config(max_epochs=2)
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        # At least one best checkpoint should be saved
        checkpoint_files = list(tmp_path.glob("best_*.pth"))
        assert len(checkpoint_files) > 0, f"No best_*.pth found in {tmp_path}"

    def test_checkpoint_named_for_metric(self, tmp_path: Path) -> None:
        """Checkpoint file is named best_<metric_name>.pth."""
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config(max_epochs=2)
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        # val_loss is default primary — best_val_loss.pth should exist
        assert (tmp_path / "best_val_loss.pth").exists(), (
            "Expected best_val_loss.pth checkpoint file"
        )

    def test_multi_metric_saves_separate_files(self, tmp_path: Path) -> None:
        """With two tracked metrics, fit() saves two separate best files."""
        from minivess.config.models import CheckpointConfig, TrackedMetricConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        custom_ckpt = CheckpointConfig(
            tracked_metrics=[
                TrackedMetricConfig(name="val_loss", direction="minimize", patience=10),
                TrackedMetricConfig(
                    name="train_loss", direction="minimize", patience=10
                ),
            ],
            primary_metric="val_loss",
        )
        config = _make_training_config(max_epochs=2, checkpoint=custom_ckpt)
        model = _make_fake_model()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        assert (tmp_path / "best_val_loss.pth").exists()
        assert (tmp_path / "best_train_loss.pth").exists()


class TestFitSavesLastCheckpoint:
    """fit() saves last.pth after each epoch when save_last=True."""

    def test_saves_last_pth(self, tmp_path: Path) -> None:
        """last.pth should exist after fit() when save_last=True."""
        from minivess.config.models import CheckpointConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        config = _make_training_config(
            max_epochs=2,
            checkpoint=CheckpointConfig(save_last=True),
        )
        model = _make_fake_model()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        assert (tmp_path / "last.pth").exists(), (
            "last.pth should be saved when save_last=True"
        )

    def test_no_last_pth_when_disabled(self, tmp_path: Path) -> None:
        """last.pth should NOT exist when save_last=False."""
        from minivess.config.models import CheckpointConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        config = _make_training_config(
            max_epochs=2,
            checkpoint=CheckpointConfig(save_last=False),
        )
        model = _make_fake_model()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        assert not (tmp_path / "last.pth").exists(), (
            "last.pth should NOT exist when save_last=False"
        )


class TestFitSavesMetricHistory:
    """fit() saves metric_history.json when save_history=True."""

    def test_saves_metric_history_json(self, tmp_path: Path) -> None:
        """metric_history.json should exist after fit() when save_history=True."""
        from minivess.config.models import CheckpointConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        config = _make_training_config(
            max_epochs=2,
            checkpoint=CheckpointConfig(save_history=True),
        )
        model = _make_fake_model()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        history_path = tmp_path / "metric_history.json"
        assert history_path.exists(), "metric_history.json should be saved"

    def test_metric_history_json_structure(self, tmp_path: Path) -> None:
        """metric_history.json should contain an 'epochs' list."""
        from minivess.config.models import CheckpointConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        config = _make_training_config(
            max_epochs=2,
            checkpoint=CheckpointConfig(save_history=True),
        )
        model = _make_fake_model()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        history_path = tmp_path / "metric_history.json"
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert "epochs" in data
        assert len(data["epochs"]) == 2  # 2 epochs ran

    def test_no_history_json_when_disabled(self, tmp_path: Path) -> None:
        """metric_history.json should NOT exist when save_history=False."""
        from minivess.config.models import CheckpointConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        config = _make_training_config(
            max_epochs=2,
            checkpoint=CheckpointConfig(save_history=False),
        )
        model = _make_fake_model()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        assert not (tmp_path / "metric_history.json").exists()


# ---------------------------------------------------------------------------
# T4: Early stopping via MultiMetricTracker
# ---------------------------------------------------------------------------


class TestFitEarlyStopsViaMultiTracker:
    """fit() uses MultiMetricTracker.should_stop() for early stopping."""

    def test_early_stop_triggered(self) -> None:
        """With tiny patience, fit() stops before max_epochs."""
        from minivess.config.models import CheckpointConfig, TrackedMetricConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        # patience=1 → stop after 1 epoch without improvement
        custom_ckpt = CheckpointConfig(
            tracked_metrics=[
                TrackedMetricConfig(name="val_loss", direction="minimize", patience=1),
            ],
            primary_metric="val_loss",
            early_stopping_strategy="all",
        )
        config = _make_training_config(max_epochs=20, checkpoint=custom_ckpt)
        model = _make_fake_model()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        result = trainer.fit(_make_loader(), _make_loader())

        # Should stop well before 20 epochs
        assert result["final_epoch"] < 20, (
            f"Expected early stopping before epoch 20, got {result['final_epoch']}"
        )


# ---------------------------------------------------------------------------
# T5: Backward-compatibility return dict
# ---------------------------------------------------------------------------


class TestFitBackwardCompatReturns:
    """fit() return dict must still contain best_val_loss and history keys."""

    def test_returns_best_val_loss(self) -> None:
        """fit() return dict contains 'best_val_loss' key for backward compat."""
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config(max_epochs=2)
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        result = trainer.fit(_make_loader(), _make_loader())

        assert "best_val_loss" in result, (
            "fit() must return 'best_val_loss' for backward compat"
        )
        assert isinstance(result["best_val_loss"], float)

    def test_returns_history(self) -> None:
        """fit() return dict contains 'history' key with train/val loss lists."""
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config(max_epochs=2)
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        result = trainer.fit(_make_loader(), _make_loader())

        assert "history" in result
        assert "train_loss" in result["history"]
        assert "val_loss" in result["history"]
        assert len(result["history"]["train_loss"]) == 2
        assert len(result["history"]["val_loss"]) == 2

    def test_returns_final_epoch(self) -> None:
        """fit() return dict contains 'final_epoch' key."""
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config(max_epochs=2)
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        result = trainer.fit(_make_loader(), _make_loader())

        assert "final_epoch" in result
        assert result["final_epoch"] == 2

    def test_returns_best_metrics(self) -> None:
        """fit() return dict contains 'best_metrics' with all tracked metrics."""
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config(max_epochs=2)
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        result = trainer.fit(_make_loader(), _make_loader())

        assert "best_metrics" in result
        assert "val_loss" in result["best_metrics"]


# ---------------------------------------------------------------------------
# T6: Self-contained checkpoint files
# ---------------------------------------------------------------------------


class TestCheckpointFilesAreSelfContained:
    """Saved checkpoint files contain full metric dict via save_metric_checkpoint."""

    def test_checkpoint_has_metadata(self, tmp_path: Path) -> None:
        """best_val_loss.pth should contain 'checkpoint_metadata' with full metrics dict."""
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config(max_epochs=2)
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        ckpt_path = tmp_path / "best_val_loss.pth"
        assert ckpt_path.exists()

        payload = torch.load(ckpt_path, weights_only=True)
        assert "checkpoint_metadata" in payload, (
            "checkpoint must contain checkpoint_metadata"
        )
        meta = payload["checkpoint_metadata"]
        assert "metrics" in meta, "checkpoint_metadata must contain 'metrics' dict"
        assert "val_loss" in meta["metrics"], "metrics dict must contain 'val_loss'"

    def test_checkpoint_has_model_state_dict(self, tmp_path: Path) -> None:
        """best_val_loss.pth should contain 'model_state_dict'."""
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _make_fake_model()
        config = _make_training_config(max_epochs=2)
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        ckpt_path = tmp_path / "best_val_loss.pth"
        payload = torch.load(ckpt_path, weights_only=True)
        assert "model_state_dict" in payload

    def test_last_checkpoint_loadable(self, tmp_path: Path) -> None:
        """last.pth should be loadable via save_metric_checkpoint format."""
        from minivess.config.models import CheckpointConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        config = _make_training_config(
            max_epochs=2,
            checkpoint=CheckpointConfig(save_last=True),
        )
        model = _make_fake_model()
        trainer = SegmentationTrainer(
            model,
            config,
            criterion=DiceCELoss(softmax=True, to_onehot_y=True),
        )
        trainer.fit(_make_loader(), _make_loader(), checkpoint_dir=tmp_path)

        last_path = tmp_path / "last.pth"
        assert last_path.exists()
        payload = torch.load(last_path, weights_only=True)
        assert "model_state_dict" in payload
        assert "checkpoint_metadata" in payload
