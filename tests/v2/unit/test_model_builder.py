"""Tests for model builder factory.

Validates build_adapter() dispatches correctly for all model families.
SAM3 tests are skipped when SAM3 is not installed.
SAM3 TopoLoRA tests are skipped when GPU VRAM < 16 GB.
"""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.model_builder import _sam3_package_available
from minivess.config.models import ModelConfig, ModelFamily

_sam3_skip = pytest.mark.skipif(
    not _sam3_package_available(), reason="SAM3 not installed"
)


def _gpu_vram_gb() -> float:
    """Return total VRAM of GPU 0 in GB, or 0.0 if no CUDA GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


_vram_16gb_skip = pytest.mark.skipif(
    _gpu_vram_gb() < 16.0,
    reason=f"SAM3 TopoLoRA requires >= 16 GB VRAM (detected {_gpu_vram_gb():.1f} GB)",
)


class TestBuildAdapter:
    """build_adapter() dispatches on ModelFamily enum."""

    def test_build_dynunet(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="dynunet-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config)
        cfg = adapter.get_config()
        assert cfg.family == "dynunet"

    @_sam3_skip
    def test_build_sam3_vanilla(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="vanilla-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config)
        cfg = adapter.get_config()
        assert cfg.family == "sam3_vanilla"

    @_sam3_skip
    @_vram_16gb_skip
    @pytest.mark.gpu
    def test_build_sam3_topolora(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_TOPOLORA,
            name="topolora-test",
            in_channels=1,
            out_channels=2,
            lora_rank=2,
        )
        adapter = build_adapter(config)
        cfg = adapter.get_config()
        assert cfg.family == "sam3_topolora"

    @_sam3_skip
    def test_build_sam3_hybrid(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_HYBRID,
            name="hybrid-test",
            in_channels=1,
            out_channels=2,
            architecture_params={"filters": [16, 32, 64]},
        )
        adapter = build_adapter(config)
        cfg = adapter.get_config()
        assert cfg.family == "sam3_hybrid"

    def test_build_unknown_raises(self) -> None:
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.CUSTOM,
            name="unknown-test",
        )
        with pytest.raises(ValueError, match="Unsupported model family"):
            build_adapter(config)

    def test_build_sam3_raises_without_installation(self) -> None:
        """build_adapter raises RuntimeError for SAM3 when not installed."""
        from unittest.mock import patch

        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="vanilla-no-sam3",
            in_channels=1,
            out_channels=2,
        )
        with (
            patch(
                "minivess.adapters.model_builder._sam3_package_available",
                return_value=False,
            ),
            pytest.raises(RuntimeError, match="SAM3"),
        ):
            build_adapter(config)
