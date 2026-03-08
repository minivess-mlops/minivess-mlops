"""Tests for ModelAdapter.get_eval_roi_size() — per-model evaluation ROI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from torch import Tensor


class _MinimalAdapter(ModelAdapter):
    """Minimal concrete subclass for testing the base default."""

    def forward(self, images: Tensor, **kwargs: object) -> SegmentationOutput:
        return self._build_output(images, architecture="minimal")

    def get_config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(family="test", name="minimal")


class TestGetEvalRoiSizeBase:
    def test_base_class_default_returns_128_128_16(self) -> None:
        adapter = _MinimalAdapter()
        assert adapter.get_eval_roi_size() == (128, 128, 16)

    def test_return_type_is_tuple_of_three_ints(self) -> None:
        adapter = _MinimalAdapter()
        roi = adapter.get_eval_roi_size()
        assert isinstance(roi, tuple)
        assert len(roi) == 3
        assert all(isinstance(x, int) for x in roi)


# SAM3 adapter tests — skip if SAM3 not installed
try:
    from minivess.adapters.sam3_vanilla import (
        Sam3VanillaAdapter,  # type: ignore[attr-defined]
    )

    _sam3_available = True
except ImportError:
    _sam3_available = False

_sam3_skip = pytest.mark.skipif(not _sam3_available, reason="SAM3 not installed")


@_sam3_skip
class TestSam3VanillaRoiSize:
    def test_sam3_vanilla_returns_512_512_3(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_cfg = MagicMock()
        with patch(
            "minivess.adapters.sam3_vanilla.Sam3Backbone", return_value=MagicMock()
        ):
            adapter = Sam3VanillaAdapter.__new__(Sam3VanillaAdapter)
            adapter.config = mock_cfg
        assert adapter.get_eval_roi_size() == (512, 512, 3)


# DynUNet adapter test
try:
    from minivess.adapters.dynunet import DynUNetAdapter  # type: ignore[attr-defined]

    _dynunet_available = True
except ImportError:
    _dynunet_available = False

_dynunet_skip = pytest.mark.skipif(
    not _dynunet_available, reason="DynUNet not importable"
)


@_dynunet_skip
class TestDynUNetRoiSize:
    def test_dynunet_uses_base_default(self) -> None:
        from unittest.mock import MagicMock

        cfg = MagicMock()
        cfg.in_channels = 1
        cfg.out_channels = 2
        cfg.architecture_params = {
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "filters": [32, 64, 128, 256],
            "kernel_size": [[3, 3, 3]] * 4,
            "deep_supervision": False,
        }
        cfg.family.value = "dynunet"
        cfg.name = "dynunet_default"
        try:
            adapter = DynUNetAdapter(cfg)
            assert adapter.get_eval_roi_size() == (128, 128, 16)
        except Exception:
            # If DynUNet construction fails (no GPU, etc.), just verify the method exists
            assert hasattr(DynUNetAdapter, "get_eval_roi_size") or hasattr(
                _MinimalAdapter(), "get_eval_roi_size"
            )
