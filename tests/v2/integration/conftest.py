"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.dynunet import DynUNetAdapter
from minivess.config.models import ModelConfig, ModelFamily

# Spatial size for synthetic volumes — small enough for fast CPU tests.
SPATIAL_SIZE = (16, 16, 16)
IN_CHANNELS = 1
NUM_CLASSES = 2
BATCH_SIZE = 2


@pytest.fixture()
def model_config() -> ModelConfig:
    """Minimal DynUNet model config for integration tests."""
    return ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="integration-dynunet",
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
    )


@pytest.fixture()
def dynunet_adapter(model_config: ModelConfig) -> DynUNetAdapter:
    """Pre-built DynUNet adapter for integration tests."""
    return DynUNetAdapter(model_config)


@pytest.fixture()
def synthetic_batch() -> dict[str, torch.Tensor]:
    """Single synthetic batch: image (B,1,D,H,W) + label (B,1,D,H,W)."""
    images = torch.randn(BATCH_SIZE, IN_CHANNELS, *SPATIAL_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, 1, *SPATIAL_SIZE))
    return {"image": images, "label": labels}


@pytest.fixture()
def synthetic_loader(
    synthetic_batch: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    """Minimal 'dataloader' as a list of one batch (for trainer.fit)."""
    return [synthetic_batch]
