"""Tests for factorial post-training config schema.

Validates the factorial_methods field in PostTrainingConfig that enables
systematic application of post-training methods across all checkpoints.
"""

from __future__ import annotations

import pytest


class TestFactorialPostTrainingConfigDefault:
    """T3: Default config has no factorial_methods."""

    def test_factorial_post_training_config_default(self) -> None:
        from minivess.config.post_training_config import PostTrainingConfig

        config = PostTrainingConfig()
        assert config.factorial_methods == [], (
            "Default factorial_methods should be empty list"
        )


class TestFactorialPostTrainingConfigMethods:
    """T3: Config accepts valid factorial methods."""

    def test_factorial_post_training_config_methods(self) -> None:
        from minivess.config.post_training_config import PostTrainingConfig

        config = PostTrainingConfig(
            factorial_methods=["none", "swa", "multi_swa"],
        )
        assert config.factorial_methods == ["none", "swa", "multi_swa"]


class TestFactorialPostTrainingConfigValidation:
    """T3: Invalid methods raise ValidationError."""

    def test_factorial_post_training_config_validates_methods(self) -> None:
        from pydantic import ValidationError

        from minivess.config.post_training_config import PostTrainingConfig

        with pytest.raises(ValidationError):
            PostTrainingConfig(factorial_methods=["none", "invalid_method"])


class TestFactorialCheckpointNaming:
    """T3: Checkpoint naming convention is deterministic."""

    def test_factorial_checkpoint_naming(self) -> None:
        from minivess.config.post_training_config import (
            factorial_checkpoint_name,
        )

        assert factorial_checkpoint_name("abc123", "none") == "abc123_none.pt"
        assert factorial_checkpoint_name("abc123", "swa") == "abc123_swa.pt"
        assert factorial_checkpoint_name("abc123", "multi_swa") == "abc123_multi_swa.pt"


class TestFactorialConfigFromYaml:
    """T3: Config can be serialized to/from dict (YAML-compatible)."""

    def test_factorial_config_from_yaml(self) -> None:
        from minivess.config.post_training_config import PostTrainingConfig

        config = PostTrainingConfig(
            factorial_methods=["none", "swa", "multi_swa"],
        )

        # Round-trip through dict (simulates YAML load)
        config_dict = config.model_dump()
        restored = PostTrainingConfig(**config_dict)

        assert restored.factorial_methods == ["none", "swa", "multi_swa"]
