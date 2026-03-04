"""Tests for MambaAdapter (U-shaped Mamba for 3D segmentation).

Based on UlikeMamba (Wang 2025) — O(n) complexity alternative to DynUNet
using 3D depthwise convolution and tri-directional scanning.

Issue: #312 | Phase 8 | Plan: T8.1 (RED)
"""

from __future__ import annotations

from pathlib import Path

import torch


class TestMambaAdapter:
    """Unit tests for MambaAdapter."""

    def test_mamba_adapter_creation(self) -> None:
        """MambaAdapter should be instantiable with a valid ModelConfig."""
        from minivess.adapters.mamba import MambaAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.ULIKE_MAMBA,
            name="ulike_mamba_test",
            architecture_params={
                "in_channels": 1,
                "out_channels": 2,
            },
        )
        adapter = MambaAdapter(config)
        assert adapter is not None

    def test_mamba_adapter_forward_shape(self) -> None:
        """Forward pass should produce (B, C, D, H, W) output."""
        from minivess.adapters.mamba import MambaAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.ULIKE_MAMBA,
            name="ulike_mamba_test",
            architecture_params={
                "in_channels": 1,
                "out_channels": 2,
            },
        )
        adapter = MambaAdapter(config)
        adapter.eval()

        x = torch.randn(1, 1, 16, 16, 16)
        with torch.no_grad():
            output = adapter(x)

        assert output.prediction.shape == (1, 2, 16, 16, 16)
        assert output.logits.shape == (1, 2, 16, 16, 16)
        assert torch.isfinite(output.prediction).all()

    def test_mamba_adapter_implements_model_adapter(self) -> None:
        """MambaAdapter should implement the ModelAdapter ABC."""
        from minivess.adapters.base import ModelAdapter
        from minivess.adapters.mamba import MambaAdapter

        assert issubclass(MambaAdapter, ModelAdapter)

    def test_mamba_model_profile_exists(self) -> None:
        """configs/model_profiles/mamba.yaml should exist."""
        profile_path = Path("configs/model_profiles/mamba.yaml")
        assert profile_path.exists(), f"Missing model profile: {profile_path}"

    def test_mamba_registered_in_model_builder(self) -> None:
        """build_adapter() should support ULIKE_MAMBA family."""
        from minivess.adapters.model_builder import build_adapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.ULIKE_MAMBA,
            name="ulike_mamba_test",
            architecture_params={
                "in_channels": 1,
                "out_channels": 2,
            },
        )
        adapter = build_adapter(config)
        assert adapter is not None

    def test_mamba_get_config(self) -> None:
        """MambaAdapter.get_config() should return valid AdapterConfigInfo."""
        from minivess.adapters.mamba import MambaAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.ULIKE_MAMBA,
            name="ulike_mamba_test",
            architecture_params={
                "in_channels": 1,
                "out_channels": 2,
            },
        )
        adapter = MambaAdapter(config)
        info = adapter.get_config()

        assert info.family == "ulike_mamba"
        assert info.in_channels == 1
        assert info.out_channels == 2
