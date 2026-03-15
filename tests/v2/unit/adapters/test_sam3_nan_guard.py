"""Tests for Sam3Backbone NaN guard in extract_features().

Phase 1+2 from docs/planning/sam3-nan-loss-fix.md:
  H2: FP16 overflow in encoder → NaN in features → NaN val_loss.
  Fix: detect NaN after encoder, replace with 0.0 via nan_to_num.

Issue: #715
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest  # noqa: TC002 — used at runtime for LogCaptureFixture
import torch
from torch import Tensor, nn


class _NaNEncoder(nn.Module):
    """Mock encoder that returns features with NaN values."""

    def __init__(self, nan_fraction: float = 0.1) -> None:
        super().__init__()
        self._nan_fraction = nan_fraction

    def forward(self, x: Tensor) -> Tensor:
        out = torch.randn(x.shape[0], 256, 8, 8, dtype=x.dtype, device=x.device)
        mask = torch.rand_like(out) < self._nan_fraction
        out[mask] = float("nan")
        return out


class _AllNaNEncoder(nn.Module):
    """Mock encoder that returns ALL NaN values (worst case)."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.full(
            (x.shape[0], 256, 8, 8), float("nan"), dtype=x.dtype, device=x.device
        )


class _CleanEncoder(nn.Module):
    """Mock encoder that returns clean (finite) features."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.randn(x.shape[0], 256, 8, 8, dtype=x.dtype, device=x.device)


def _make_backbone(encoder: nn.Module, *, frozen: bool = True) -> nn.Module:
    """Create a Sam3Backbone with a mock encoder (no HF weights needed).

    Patches _load_sam3_encoder to return the mock encoder + Identity neck,
    so the real __init__ runs but doesn't try to download SAM3 weights.
    """
    from minivess.adapters.sam3_backbone import Sam3Backbone
    from minivess.config.models import ModelConfig

    config = ModelConfig(
        name="test_sam3",
        family="sam3_vanilla",
        in_channels=1,
        out_channels=2,
        architecture_params={"sam3_input_size": 64},
    )

    def _mock_load(self_: nn.Module) -> tuple[nn.Module, nn.Module]:
        return encoder, nn.Identity()

    with patch.object(Sam3Backbone, "_load_sam3_encoder", _mock_load):
        backbone = Sam3Backbone(config=config, freeze=frozen, input_size=64)

    return backbone


class TestExtractFeaturesNaNGuard:
    """extract_features() must return finite tensors even when encoder produces NaN."""

    def test_nan_in_encoder_output_replaced_with_zeros(self) -> None:
        """When encoder produces NaN, output should be finite (NaN → 0.0)."""
        backbone = _make_backbone(_NaNEncoder(nan_fraction=0.3))
        x = torch.randn(1, 1, 64, 64)

        result = backbone.extract_features(x)

        assert torch.isfinite(result).all(), (
            f"Expected all-finite output but got {torch.isnan(result).sum()} NaN values"
        )

    def test_all_nan_encoder_output_replaced(self) -> None:
        """Even if encoder returns all NaN, output should be all zeros."""
        backbone = _make_backbone(_AllNaNEncoder())
        x = torch.randn(1, 1, 64, 64)

        result = backbone.extract_features(x)

        assert torch.isfinite(result).all(), (
            "All-NaN input should produce all-zero output"
        )
        assert (result == 0.0).all(), "NaN should be replaced with 0.0"

    def test_clean_encoder_output_unchanged(self) -> None:
        """When encoder produces clean output, it should pass through unchanged."""
        backbone = _make_backbone(_CleanEncoder())
        x = torch.randn(1, 1, 64, 64)

        result = backbone.extract_features(x)

        assert torch.isfinite(result).all(), "Clean output should remain finite"

    def test_nan_detection_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """When NaN is detected, a warning should be logged."""
        backbone = _make_backbone(_NaNEncoder(nan_fraction=0.5))
        x = torch.randn(1, 1, 64, 64)

        with caplog.at_level(logging.WARNING, logger="minivess.adapters.sam3_backbone"):
            backbone.extract_features(x)

        assert any(
            "NaN" in record.message and "encoder" in record.message.lower()
            for record in caplog.records
        ), f"Expected NaN warning in logs, got: {[r.message for r in caplog.records]}"

    def test_clean_encoder_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """When encoder output is clean, no NaN warning should be logged."""
        backbone = _make_backbone(_CleanEncoder())
        x = torch.randn(1, 1, 64, 64)

        with caplog.at_level(logging.WARNING, logger="minivess.adapters.sam3_backbone"):
            backbone.extract_features(x)

        nan_warnings = [
            r
            for r in caplog.records
            if "NaN" in r.message and "encoder" in r.message.lower()
        ]
        assert not nan_warnings, f"Unexpected NaN warning: {nan_warnings}"

    def test_unfrozen_encoder_nan_guard_also_works(self) -> None:
        """NaN guard should work for unfrozen encoder path too."""
        backbone = _make_backbone(_NaNEncoder(nan_fraction=0.3), frozen=False)
        x = torch.randn(1, 1, 64, 64)

        result = backbone.extract_features(x)

        assert torch.isfinite(result).all(), (
            "Unfrozen path should also guard against NaN"
        )

    def test_inf_in_encoder_output_replaced(self) -> None:
        """Inf values (FP16 overflow) should also be replaced."""

        class _InfEncoder(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                out = torch.randn(x.shape[0], 256, 8, 8, dtype=x.dtype, device=x.device)
                out[0, 0, 0, 0] = float("inf")
                out[0, 1, 0, 0] = float("-inf")
                return out

        backbone = _make_backbone(_InfEncoder())
        x = torch.randn(1, 1, 64, 64)

        result = backbone.extract_features(x)

        assert torch.isfinite(result).all(), "Inf values should be clamped to finite"
