"""Integration test: synthetic NIfTI files → MONAI loader → training step.

Exercises the full data pipeline with real files on disk, not just
tensor fixtures. This catches mismatches between NIfTI I/O, transforms,
and the training loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path


class TestNiftiToTrainingStep:
    """End-to-end: NIfTI files → discover → load → forward pass."""

    def test_nifti_to_training_step(self, tmp_path: Path) -> None:
        """Full pipeline: create NIfTI, discover, load, run one training step."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.config.models import DataConfig, ModelConfig, ModelFamily
        from minivess.data.loader import build_train_loader, discover_nifti_pairs
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        # Create synthetic NIfTI dataset
        data_dir = create_synthetic_nifti_dataset(
            tmp_path, n_volumes=2, spatial_size=(32, 32, 16)
        )

        # Discover pairs
        pairs = discover_nifti_pairs(data_dir)
        assert len(pairs) == 2

        # Create loader
        config = DataConfig(
            dataset_name="test-integration",
            data_dir=data_dir,
            patch_size=(16, 16, 8),
            voxel_spacing=(1.0, 1.0, 1.0),
            num_workers=0,
        )
        loader = build_train_loader(pairs, config, batch_size=1, cache_rate=1.0)

        # Get a batch
        batch = next(iter(loader))
        images = batch["image"]
        labels = batch["label"]

        # Create model and run forward pass
        model_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="integration-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = SegResNetAdapter(model_config)
        model = adapter.net
        model.train()

        # Forward pass
        output = model(images)
        assert output.shape[0] == images.shape[0]  # Batch dim matches
        assert output.shape[1] == 2  # num_classes

        # Compute loss (verify gradients flow)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, labels[:, 0].long())
        loss.backward()

        # Verify gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, "Gradients should flow through the model"

    def test_nifti_to_validation_metrics(self, tmp_path: Path) -> None:
        """Validation path: load → predict → compute dice-like metric."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.config.models import DataConfig, ModelConfig, ModelFamily
        from minivess.data.loader import build_val_loader, discover_nifti_pairs
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        data_dir = create_synthetic_nifti_dataset(
            tmp_path, n_volumes=1, spatial_size=(32, 32, 16)
        )
        pairs = discover_nifti_pairs(data_dir)

        config = DataConfig(
            dataset_name="test-validation",
            data_dir=data_dir,
            patch_size=(16, 16, 8),
            voxel_spacing=(1.0, 1.0, 1.0),
            num_workers=0,
        )
        loader = build_val_loader(pairs, config, cache_rate=1.0)
        batch = next(iter(loader))

        model_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="val-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = SegResNetAdapter(model_config)
        model = adapter.net
        model.eval()

        with torch.no_grad():
            output = model(batch["image"])
            pred = output.argmax(dim=1, keepdim=True)

        # Basic shape and range checks
        assert pred.shape == batch["label"].shape
        assert pred.min() >= 0
        assert pred.max() <= 1  # Binary segmentation


class TestEbrainsLayoutPipeline:
    """EBRAINS raw/seg layout → discover → load pipeline."""

    def test_ebrains_layout_to_loader(self, tmp_path: Path) -> None:
        from minivess.config.models import DataConfig
        from minivess.data.loader import build_val_loader, discover_nifti_pairs
        from tests.v2.fixtures.synthetic_nifti import (
            create_synthetic_nifti_dataset_ebrains,
        )

        data_dir = create_synthetic_nifti_dataset_ebrains(
            tmp_path, n_volumes=2, spatial_size=(32, 32, 16)
        )
        pairs = discover_nifti_pairs(data_dir)
        assert len(pairs) == 2

        config = DataConfig(
            dataset_name="ebrains-test",
            data_dir=data_dir,
            patch_size=(16, 16, 8),
            voxel_spacing=(1.0, 1.0, 1.0),
            num_workers=0,
        )
        loader = build_val_loader(pairs, config, cache_rate=1.0)
        batch = next(iter(loader))

        assert batch["image"].ndim == 5
        assert batch["label"].ndim == 5
