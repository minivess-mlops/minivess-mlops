"""Tests for VesselFM adapter fixes (#289).

Covers:
- Adapter creation (pretrained=False, no HF download)
- Pretrained flag passthrough from build_adapter()
- Forward shape (B, 2, D, H, W) from (B, 1, D, H, W)
- Binary-to-2class logit conversion correctness
- Data leakage warning when pretrained=True
- State dict prefix stripping
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest  # noqa: TCH002 — used at runtime for fixtures
import torch

from minivess.config.models import ModelConfig, ModelFamily


class TestVesselFMAdapterCreation:
    """Test VesselFMAdapter instantiation."""

    def test_vesselfm_adapter_creation_no_pretrained(self) -> None:
        """Basic instantiation with pretrained=False (no HF download)."""
        from minivess.adapters.vesselfm import VesselFMAdapter

        config = ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="vesselfm-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = VesselFMAdapter(config, pretrained=False)
        assert adapter is not None
        assert hasattr(adapter, "net")

    def test_vesselfm_forward_shape(self) -> None:
        """Output shape should be (B, 2, D, H, W) from (B, 1, D, H, W)."""
        from minivess.adapters.vesselfm import VesselFMAdapter

        config = ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="vesselfm-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = VesselFMAdapter(config, pretrained=False)

        # Must be large enough for 5 stride-2 levels + instance norm (>1 spatial)
        x = torch.randn(1, 1, 64, 64, 64)
        output = adapter(x)
        assert output.logits.shape == (1, 2, 64, 64, 64)

    def test_vesselfm_binary_to_2class(self) -> None:
        """Verify binary → 2-class conversion: logits[:, 0] = -binary, logits[:, 1] = binary."""
        from minivess.adapters.vesselfm import VesselFMAdapter

        config = ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="vesselfm-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = VesselFMAdapter(config, pretrained=False)
        x = torch.randn(1, 1, 64, 64, 64)

        with torch.no_grad():
            output = adapter(x)

        # Channel 0 (background) should be opposite sign of channel 1 (foreground)
        bg = output.logits[:, 0]
        fg = output.logits[:, 1]
        assert torch.allclose(bg, -fg, atol=1e-6)


class TestVesselFMPretrainedPassthrough:
    """Test that build_adapter passes pretrained flag correctly."""

    def test_build_adapter_passes_pretrained_true(self) -> None:
        """build_adapter should pass pretrained=True from architecture_params."""
        config = ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="vesselfm-test",
            in_channels=1,
            out_channels=2,
            architecture_params={"pretrained": True},
        )

        with patch(
            "minivess.adapters.vesselfm.VesselFMAdapter._load_pretrained"
        ) as mock_load:
            from minivess.adapters.model_builder import build_adapter

            adapter = build_adapter(config)
            mock_load.assert_called_once()
            assert adapter is not None

    def test_build_adapter_passes_pretrained_false(self) -> None:
        """build_adapter with pretrained=False should NOT call _load_pretrained."""
        config = ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="vesselfm-test",
            in_channels=1,
            out_channels=2,
            architecture_params={"pretrained": False},
        )

        with patch(
            "minivess.adapters.vesselfm.VesselFMAdapter._load_pretrained"
        ) as mock_load:
            from minivess.adapters.model_builder import build_adapter

            build_adapter(config)
            mock_load.assert_not_called()

    def test_build_adapter_default_pretrained_false(self) -> None:
        """Without architecture_params.pretrained, default should be False."""
        config = ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="vesselfm-test",
            in_channels=1,
            out_channels=2,
        )

        with patch(
            "minivess.adapters.vesselfm.VesselFMAdapter._load_pretrained"
        ) as mock_load:
            from minivess.adapters.model_builder import build_adapter

            build_adapter(config)
            mock_load.assert_not_called()


class TestVesselFMDataLeakageWarning:
    """Test that pretrained=True emits data leakage warning."""

    def test_pretrained_emits_leakage_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When pretrained=True, a data leakage warning should be logged."""
        from minivess.adapters.vesselfm import VesselFMAdapter

        config = ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="vesselfm-test",
            in_channels=1,
            out_channels=2,
        )

        with (
            patch.object(VesselFMAdapter, "_load_pretrained"),
            caplog.at_level(logging.WARNING, logger="minivess.adapters.vesselfm"),
        ):
            VesselFMAdapter(config, pretrained=True)

        assert any("pre-trained on MiniVess" in msg for msg in caplog.messages), (
            f"Expected data leakage warning, got: {caplog.messages}"
        )

    def test_no_leakage_warning_when_not_pretrained(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When pretrained=False, no leakage warning."""
        from minivess.adapters.vesselfm import VesselFMAdapter

        config = ModelConfig(
            family=ModelFamily.VESSEL_FM,
            name="vesselfm-test",
            in_channels=1,
            out_channels=2,
        )

        with caplog.at_level(logging.WARNING, logger="minivess.adapters.vesselfm"):
            VesselFMAdapter(config, pretrained=False)

        assert not any("pre-trained on MiniVess" in msg for msg in caplog.messages)


class TestVesselFMStateDictStripping:
    """Test state dict prefix stripping for checkpoint loading."""

    def test_strip_state_dict_prefix(self) -> None:
        """_strip_state_dict_prefix should remove module./model. prefixes."""
        from minivess.adapters.vesselfm import _strip_state_dict_prefix

        state_dict = {
            "module.encoder.0.weight": torch.randn(3, 3),
            "module.decoder.0.bias": torch.randn(3),
        }
        stripped = _strip_state_dict_prefix(state_dict)
        assert "encoder.0.weight" in stripped
        assert "decoder.0.bias" in stripped
        assert "module.encoder.0.weight" not in stripped

    def test_strip_no_prefix(self) -> None:
        """When no prefix, state dict should be unchanged."""
        from minivess.adapters.vesselfm import _strip_state_dict_prefix

        state_dict = {
            "encoder.0.weight": torch.randn(3, 3),
        }
        stripped = _strip_state_dict_prefix(state_dict)
        assert "encoder.0.weight" in stripped
