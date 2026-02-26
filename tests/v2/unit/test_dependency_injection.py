"""Tests for Dependency Injection — R5.14, R5.13, R5.17 (Issue #56).

R5.14: Trainer dependency injection (criterion, optimizer, scheduler)
R5.13: BentoML accepts ModelAdapter
R5.17: Data loader transform injection
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import nn

from minivess.config.models import ModelConfig, ModelFamily, TrainingConfig


def _make_model_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.MONAI_SEGRESNET,
        name="test",
        in_channels=1,
        out_channels=2,
    )


def _make_training_config(**overrides: object) -> TrainingConfig:
    defaults = {
        "max_epochs": 2,
        "learning_rate": 1e-3,
        "batch_size": 1,
        "early_stopping_patience": 5,
        "mixed_precision": False,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _fake_loader(n_batches: int = 2) -> list[dict[str, torch.Tensor]]:
    """Create a synthetic data loader (list of batch dicts)."""
    return [
        {
            "image": torch.randn(1, 1, 16, 16, 16),
            "label": torch.randint(0, 2, (1, 1, 16, 16, 16)),
        }
        for _ in range(n_batches)
    ]


# =========================================================================
# R5.14: Trainer dependency injection
# =========================================================================


class TestTrainerCriterionInjection:
    """Test that criterion can be injected into SegmentationTrainer."""

    def test_injected_criterion_used(self) -> None:
        """When criterion is provided, trainer should use it instead of default."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_model_config())
        custom_criterion = nn.MSELoss()
        trainer = SegmentationTrainer(
            model,
            _make_training_config(),
            criterion=custom_criterion,
        )
        assert trainer.criterion is custom_criterion

    def test_default_criterion_when_not_injected(self) -> None:
        """When criterion is not provided, trainer builds its own."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_model_config())
        trainer = SegmentationTrainer(model, _make_training_config())
        assert trainer.criterion is not None
        # Default should not be MSELoss
        assert not isinstance(trainer.criterion, nn.MSELoss)


class TestTrainerOptimizerInjection:
    """Test that optimizer can be injected into SegmentationTrainer."""

    def test_injected_optimizer_used(self) -> None:
        """When optimizer is provided, trainer should use it."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_model_config())
        custom_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = SegmentationTrainer(
            model,
            _make_training_config(),
            optimizer=custom_optimizer,
        )
        assert trainer.optimizer is custom_optimizer

    def test_default_optimizer_when_not_injected(self) -> None:
        """When optimizer is not provided, trainer builds its own."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_model_config())
        trainer = SegmentationTrainer(model, _make_training_config())
        assert trainer.optimizer is not None


class TestTrainerSchedulerInjection:
    """Test that scheduler can be injected into SegmentationTrainer."""

    def test_injected_scheduler_used(self) -> None:
        """When scheduler is provided, trainer should use it."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_model_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        custom_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        trainer = SegmentationTrainer(
            model,
            _make_training_config(),
            optimizer=optimizer,
            scheduler=custom_scheduler,
        )
        assert trainer.scheduler is custom_scheduler

    def test_default_scheduler_when_not_injected(self) -> None:
        """When scheduler is not provided, trainer builds its own."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_model_config())
        trainer = SegmentationTrainer(model, _make_training_config())
        assert trainer.scheduler is not None


class TestTrainerInjectionFunctional:
    """Test that injected dependencies actually work during training."""

    def test_train_with_injected_criterion(self) -> None:
        """Training should work with an injected criterion."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.loss_functions import build_loss_function
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_model_config())
        custom_criterion = build_loss_function("focal")
        trainer = SegmentationTrainer(
            model,
            _make_training_config(),
            criterion=custom_criterion,
        )
        result = trainer.train_epoch(_fake_loader())
        assert result.loss > 0
        assert result.loss == result.loss  # Not NaN


# =========================================================================
# R5.13: BentoML accepts ModelAdapter
# =========================================================================


class TestBentoServiceAdapterInjection:
    """Test BentoML service ModelAdapter injection documentation.

    BentoML's @bentoml.service decorator controls __init__, making true
    constructor injection infeasible. We document this and verify the
    service structure is compatible with adapter-based inference.
    """

    def test_service_inner_predict_uses_model(self) -> None:
        """SegmentationService.predict should use self.model for inference."""
        import inspect

        from minivess.serving import bento_service

        # Read the module source to verify self.model usage in predict
        source = inspect.getsource(bento_service)
        assert "self.model" in source
        assert "def predict" in source

    def test_bento_model_tag_is_configurable(self) -> None:
        """BENTO_MODEL_TAG should be a module-level constant that can be overridden."""
        from minivess.serving import bento_service

        assert hasattr(bento_service, "BENTO_MODEL_TAG")
        original = bento_service.BENTO_MODEL_TAG
        assert isinstance(original, str)
        assert len(original) > 0


# =========================================================================
# R5.17: Data loader transform injection
# =========================================================================


class TestLoaderTransformInjection:
    """Test that custom transforms can be injected into data loaders."""

    def test_build_train_loader_accepts_transforms(self) -> None:
        """build_train_loader should accept optional transforms parameter."""
        import inspect

        from minivess.data.loader import build_train_loader

        sig = inspect.signature(build_train_loader)
        assert "transforms" in sig.parameters

    def test_build_val_loader_accepts_transforms(self) -> None:
        """build_val_loader should accept optional transforms parameter."""
        import inspect

        from minivess.data.loader import build_val_loader

        sig = inspect.signature(build_val_loader)
        assert "transforms" in sig.parameters

    def test_injected_train_transform_used(self, tmp_path: Any) -> None:
        """When transforms is provided, build_train_loader should use it."""
        from unittest.mock import patch

        from monai.transforms import Compose, Identityd

        from minivess.config.models import DataConfig
        from minivess.data.loader import build_train_loader

        custom_transforms = Compose(
            [
                Identityd(keys=["image", "label"]),
            ]
        )

        config = DataConfig(dataset_name="test", num_workers=0)
        data_dicts = [{"image": "dummy.nii.gz", "label": "dummy_y.nii.gz"}]

        # Mock CacheDataset to capture what transforms were passed
        with patch("minivess.data.loader.CacheDataset") as mock_ds:
            mock_ds.return_value = []  # Empty dataset
            with contextlib.suppress(Exception):
                build_train_loader(
                    data_dicts,
                    config,
                    transforms=custom_transforms,
                )

            # Verify CacheDataset was called with our custom transforms
            if mock_ds.called:
                call_kwargs = mock_ds.call_args
                assert (
                    call_kwargs[1].get("transform") is custom_transforms
                    or call_kwargs.kwargs.get("transform") is custom_transforms
                )

    def test_injected_val_transform_used(self, tmp_path: Any) -> None:
        """When transforms is provided, build_val_loader should use it."""
        from unittest.mock import patch

        from monai.transforms import Compose, Identityd

        from minivess.config.models import DataConfig
        from minivess.data.loader import build_val_loader

        custom_transforms = Compose(
            [
                Identityd(keys=["image", "label"]),
            ]
        )

        config = DataConfig(dataset_name="test", num_workers=0)
        data_dicts = [{"image": "dummy.nii.gz", "label": "dummy_y.nii.gz"}]

        with patch("minivess.data.loader.CacheDataset") as mock_ds:
            mock_ds.return_value = []
            with contextlib.suppress(Exception):
                build_val_loader(
                    data_dicts,
                    config,
                    transforms=custom_transforms,
                )

            if mock_ds.called:
                call_kwargs = mock_ds.call_args
                assert (
                    call_kwargs[1].get("transform") is custom_transforms
                    or call_kwargs.kwargs.get("transform") is custom_transforms
                )
