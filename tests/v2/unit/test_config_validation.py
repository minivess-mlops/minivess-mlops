"""Negative config validation tests (Issue #54 — R5.9).

Tests that invalid configurations are properly rejected by Pydantic.
Complements test_config_models.py with additional edge cases.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from minivess.config.models import (
    DataConfig,
    EnsembleConfig,
    ModelConfig,
    ModelFamily,
    ServingConfig,
    TrainingConfig,
)


class TestTrainingConfigNegative:
    """TrainingConfig should reject invalid values."""

    def test_batch_size_zero(self) -> None:
        """batch_size=0 should be rejected (ge=1)."""
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_batch_size_negative(self) -> None:
        """batch_size=-1 should be rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=-1)

    def test_learning_rate_zero(self) -> None:
        """learning_rate=0 should be rejected (gt=0)."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)

    def test_learning_rate_negative(self) -> None:
        """Negative learning rate should be rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-0.001)

    def test_gradient_clip_val_negative(self) -> None:
        """Negative gradient_clip_val should be rejected (ge=0)."""
        with pytest.raises(ValidationError):
            TrainingConfig(gradient_clip_val=-1.0)

    def test_invalid_optimizer_name(self) -> None:
        """Unknown optimizer string should be rejected by validator."""
        with pytest.raises(ValidationError, match="Optimizer must be one of"):
            TrainingConfig(optimizer="rmsprop")

    def test_valid_optimizer_names(self) -> None:
        """All valid optimizer names should be accepted."""
        for opt in ("adam", "adamw", "sgd", "lamb"):
            cfg = TrainingConfig(optimizer=opt)
            assert cfg.optimizer == opt

    def test_negative_warmup_epochs(self) -> None:
        """Negative warmup_epochs should be rejected (ge=0)."""
        with pytest.raises(ValidationError):
            TrainingConfig(warmup_epochs=-1)


class TestDataConfigNegative:
    """DataConfig should reject invalid values."""

    def test_negative_num_workers(self) -> None:
        """Negative num_workers should be rejected (ge=0)."""
        with pytest.raises(ValidationError):
            DataConfig(dataset_name="test", num_workers=-1)

    def test_zero_prefetch_factor(self) -> None:
        """prefetch_factor=0 should be rejected (ge=1)."""
        with pytest.raises(ValidationError):
            DataConfig(dataset_name="test", prefetch_factor=0)


class TestModelConfigNegative:
    """ModelConfig should reject invalid values."""

    def test_zero_in_channels(self) -> None:
        """in_channels=0 should be rejected (ge=1)."""
        with pytest.raises(ValidationError):
            ModelConfig(family=ModelFamily.MONAI_SEGRESNET, name="test", in_channels=0)

    def test_zero_out_channels(self) -> None:
        """out_channels=0 should be rejected (ge=1)."""
        with pytest.raises(ValidationError):
            ModelConfig(family=ModelFamily.MONAI_SEGRESNET, name="test", out_channels=0)

    def test_lora_rank_zero(self) -> None:
        """lora_rank=0 should be rejected (ge=1)."""
        with pytest.raises(ValidationError):
            ModelConfig(
                family=ModelFamily.SAM3_LORA,
                name="test",
                lora_rank=0,
            )


class TestServingConfigNegative:
    """ServingConfig should reject invalid values."""

    def test_port_zero(self) -> None:
        """Port 0 should be rejected (ge=1)."""
        with pytest.raises(ValidationError):
            ServingConfig(port=0)

    def test_port_above_max(self) -> None:
        """Port above 65535 should be rejected."""
        with pytest.raises(ValidationError):
            ServingConfig(port=70000)


class TestEnsembleConfigNegative:
    """EnsembleConfig should reject invalid values."""

    def test_conformal_alpha_zero(self) -> None:
        """conformal_alpha=0 should be rejected (gt=0)."""
        with pytest.raises(ValidationError):
            EnsembleConfig(conformal_alpha=0.0)

    def test_conformal_alpha_one(self) -> None:
        """conformal_alpha=1.0 should be rejected (lt=1)."""
        with pytest.raises(ValidationError):
            EnsembleConfig(conformal_alpha=1.0)
