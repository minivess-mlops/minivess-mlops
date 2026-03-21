"""Tests for TrainingConfig zero-shot support (Glitch #12).

max_epochs=0 must be valid for zero-shot evaluation paths
(sam3_vanilla, vesselfm). max_epochs < 0 must still be rejected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from minivess.config.models import TrainingConfig


@pytest.mark.model_construction
class TestTrainingConfigZeroShot:
    """TrainingConfig must accept max_epochs=0 for zero-shot."""

    def test_accepts_zero_epochs(self) -> None:
        """max_epochs=0 is valid for zero-shot evaluation."""
        config = TrainingConfig(max_epochs=0)
        assert config.max_epochs == 0

    def test_accepts_positive_epochs(self) -> None:
        """max_epochs > 0 is valid for training."""
        config = TrainingConfig(max_epochs=50)
        assert config.max_epochs == 50

    def test_accepts_one_epoch(self) -> None:
        """max_epochs=1 is valid (boundary)."""
        config = TrainingConfig(max_epochs=1)
        assert config.max_epochs == 1

    def test_rejects_negative_epochs(self) -> None:
        """max_epochs < 0 is invalid."""
        with pytest.raises(ValidationError):
            TrainingConfig(max_epochs=-1)

    def test_rejects_negative_batch_size(self) -> None:
        """batch_size must be >= 1."""
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)
