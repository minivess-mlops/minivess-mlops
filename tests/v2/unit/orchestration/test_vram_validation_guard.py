"""Tests for VRAM-aware validation guard.

T1.2: Verify _should_skip_validation() uses VRAM budget to decide
whether validation is safe, replacing fragile val_interval sentinel.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestShouldSkipValidation:
    """_should_skip_validation decides based on available VRAM."""

    def test_skip_when_vram_insufficient(self) -> None:
        """Should skip validation when available VRAM < required."""
        from minivess.pipeline.vram_guard import should_skip_validation

        # 8 GB GPU, sam3_hybrid needs ~11 GB for validation
        result = should_skip_validation(
            model_family="sam3_hybrid",
            available_vram_mb=8000,
        )
        assert result is True

    def test_run_when_vram_sufficient(self) -> None:
        """Should NOT skip validation when available VRAM >= required."""
        from minivess.pipeline.vram_guard import should_skip_validation

        # 24 GB GPU, sam3_hybrid fits fine
        result = should_skip_validation(
            model_family="sam3_hybrid",
            available_vram_mb=24000,
        )
        assert result is False

    def test_never_skip_for_small_models(self) -> None:
        """Should never skip validation for DynUNet (fits on any GPU)."""
        from minivess.pipeline.vram_guard import should_skip_validation

        result = should_skip_validation(
            model_family="dynunet",
            available_vram_mb=4000,
        )
        assert result is False

    def test_cpu_always_skips_sam3_hybrid(self) -> None:
        """On CPU (no VRAM), should skip validation for heavy models."""
        from minivess.pipeline.vram_guard import should_skip_validation

        result = should_skip_validation(
            model_family="sam3_hybrid",
            available_vram_mb=0,
        )
        assert result is True

    def test_returns_false_for_unknown_model(self) -> None:
        """Unknown models default to NOT skipping validation."""
        from minivess.pipeline.vram_guard import should_skip_validation

        result = should_skip_validation(
            model_family="future_model",
            available_vram_mb=8000,
        )
        assert result is False


class TestGetAvailableVram:
    """get_available_vram_mb() queries GPU or returns 0."""

    def test_returns_zero_on_cpu(self) -> None:
        """Returns 0 when CUDA is not available."""
        from minivess.pipeline.vram_guard import get_available_vram_mb

        with patch("torch.cuda.is_available", return_value=False):
            assert get_available_vram_mb() == 0

    def test_returns_total_memory_on_gpu(self) -> None:
        """Returns GPU total memory in MB when CUDA available."""
        from minivess.pipeline.vram_guard import get_available_vram_mb

        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24 GB in bytes

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            vram = get_available_vram_mb()
            assert vram == pytest.approx(24 * 1024, rel=0.01)


class TestCloudSmokeConfig:
    """T1.1: Cloud smoke config exists with val_interval=1."""

    def test_cloud_config_exists(self) -> None:
        """smoke_sam3_hybrid_cloud.yaml must exist."""
        path = Path("configs/experiment/smoke_sam3_hybrid_cloud.yaml")
        assert path.exists()

    def test_cloud_config_has_val_interval_1(self) -> None:
        """Cloud config must have val_interval=1 (not sentinel)."""
        path = Path("configs/experiment/smoke_sam3_hybrid_cloud.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["val_interval"] == 1

    def test_cloud_config_has_more_epochs(self) -> None:
        """Cloud config should have more epochs than local smoke test."""
        path = Path("configs/experiment/smoke_sam3_hybrid_cloud.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["max_epochs"] >= 3
