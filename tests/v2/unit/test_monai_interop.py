"""MONAI interoperability tests for all registered adapters (T-02.8).

Verifies that:
- Each MONAI adapter outputs SegmentationOutput compatible with MONAI losses
- MONAI sliding window inferer works with each adapter
- Loss factory includes MONAI losses
- All adapters return SegmentationOutput dataclass

Closes: #474 (MONAI ecosystem audit), #343 (ModelAdapter audit)
"""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.base import SegmentationOutput
from minivess.config.models import ModelConfig, ModelFamily

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MONAI_FAMILIES = [
    ModelFamily.MONAI_DYNUNET,
    ModelFamily.MONAI_SEGRESNET,
    ModelFamily.MONAI_SWINUNETR,
    ModelFamily.MONAI_UNETR,
    ModelFamily.MONAI_ATTENTIONUNET,
]

# Must be divisible by 32 (SwinUNETR 2^5 constraint) and 16 (UNETR patch size)
_TEST_SPATIAL = (64, 64, 32)


def _build_adapter(family: ModelFamily) -> object:
    from minivess.adapters.model_builder import build_adapter

    arch: dict[str, object] = {}
    if family == ModelFamily.MONAI_UNETR:
        arch["img_size"] = _TEST_SPATIAL
        arch["hidden_size"] = 192  # small ViT for fast test
        arch["feature_size"] = 8
    config = ModelConfig(
        family=family,
        name=f"test-{family.value}",
        in_channels=1,
        out_channels=2,
        architecture_params=arch,
    )
    return build_adapter(config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMonaiLossCompatibility:
    """MONAI losses must work with adapter logit tensors."""

    @pytest.mark.parametrize("family", _MONAI_FAMILIES)
    def test_monai_dice_loss_works_with_adapters(self, family: ModelFamily) -> None:
        """DiceLoss accepts adapter logit output."""
        from monai.losses import DiceLoss  # type: ignore[import-untyped]

        adapter = _build_adapter(family)
        x = torch.randn(1, 1, *_TEST_SPATIAL)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)

        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        label = torch.zeros(1, 1, *_TEST_SPATIAL, dtype=torch.long)
        loss = loss_fn(output.logits, label)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    @pytest.mark.parametrize("family", _MONAI_FAMILIES)
    def test_monai_dice_ce_loss_works(self, family: ModelFamily) -> None:
        """DiceCELoss accepts adapter logit output."""
        from monai.losses import DiceCELoss  # type: ignore[import-untyped]

        adapter = _build_adapter(family)
        x = torch.randn(1, 1, *_TEST_SPATIAL)
        output = adapter(x)

        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        label = torch.zeros(1, 1, *_TEST_SPATIAL, dtype=torch.long)
        loss = loss_fn(output.logits, label)
        assert loss.ndim == 0
        assert loss.item() >= 0.0


class TestMonaiSlidingWindowInferer:
    """SlidingWindowInferer must work with each adapter."""

    @pytest.mark.parametrize("family", _MONAI_FAMILIES)
    def test_monai_sliding_window_inferer(self, family: ModelFamily) -> None:
        """SlidingWindowInferer produces correct output shape."""
        from monai.inferers import SlidingWindowInferer  # type: ignore[import-untyped]

        # ROI must also satisfy per-model constraints
        roi_size = _TEST_SPATIAL
        arch: dict[str, object] = {}
        if family == ModelFamily.MONAI_UNETR:
            arch["img_size"] = roi_size
            arch["hidden_size"] = 192
            arch["feature_size"] = 8
        config = ModelConfig(
            family=family,
            name=f"sw-{family.value}",
            in_channels=1,
            out_channels=2,
            architecture_params=arch,
        )
        from minivess.adapters.model_builder import build_adapter

        adapter = build_adapter(config)
        adapter.eval()

        # Volume 2x roi_size to exercise sliding window tiling
        vol_spatial = tuple(d * 2 for d in roi_size)
        volume = torch.randn(1, 1, *vol_spatial)

        inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.25)

        # SlidingWindowInferer calls adapter(patch) and expects a Tensor.
        # Our adapters return SegmentationOutput; wrap to extract logits.
        def _logits_fn(x: torch.Tensor) -> torch.Tensor:
            return adapter(x).logits  # type: ignore[union-attr]

        with torch.no_grad():
            result = inferer(volume, _logits_fn)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 2, *vol_spatial)


class TestAllAdaptersReturnSegmentationOutput:
    """Every registered non-SAM3 adapter must return SegmentationOutput."""

    @pytest.mark.parametrize("family", _MONAI_FAMILIES)
    def test_all_adapters_output_segmentation_output(self, family: ModelFamily) -> None:
        """Forward returns SegmentationOutput dataclass."""
        adapter = _build_adapter(family)
        x = torch.randn(1, 1, *_TEST_SPATIAL)
        output = adapter(x)
        assert isinstance(output, SegmentationOutput)
        assert hasattr(output, "prediction")
        assert hasattr(output, "logits")
        assert hasattr(output, "metadata")


class TestLossFactoryMonaiIntegration:
    """Loss factory must include registered MONAI losses."""

    def test_loss_factory_includes_dice(self) -> None:
        """build_loss_function('dice') must work."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss = build_loss_function("dice")
        assert loss is not None

    def test_loss_factory_includes_dice_ce(self) -> None:
        """build_loss_function('dice_ce') must work."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss = build_loss_function("dice_ce")
        assert loss is not None
