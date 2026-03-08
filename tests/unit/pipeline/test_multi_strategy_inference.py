"""Tests for MultiStrategyInferenceRunner — MONAI sliding_window for all models."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.config.evaluation_config import InferenceStrategyConfig
from minivess.pipeline.multi_strategy_inference import MultiStrategyInferenceRunner


def _make_strategy(
    name: str = "test",
    roi_size: list[int] | str = None,  # type: ignore[assignment]
    is_primary: bool = False,
) -> InferenceStrategyConfig:
    if roi_size is None:
        roi_size = [128, 128, 16]
    return InferenceStrategyConfig(
        name=name,
        roi_size=roi_size,
        is_primary=is_primary,
    )


class _MockAdapter(ModelAdapter):
    """Minimal adapter returning SegmentationOutput."""

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        b = images.shape[0]
        spatial = images.shape[2:]  # (H, W, D)
        logits = torch.zeros(b, 2, *spatial)
        return self._build_output(logits, "mock")

    def get_config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(family="mock", name="mock")


class _MockAdapterCustomRoi(ModelAdapter):
    """Adapter with custom get_eval_roi_size."""

    def get_eval_roi_size(self) -> tuple[int, int, int]:
        return (64, 64, 8)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        b = images.shape[0]
        spatial = images.shape[2:]
        logits = torch.zeros(b, 2, *spatial)
        return self._build_output(logits, "mock_custom")

    def get_config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(family="mock_custom", name="mock_custom")


class TestResolveRoiSize:
    def test_resolve_roi_size_per_model_calls_adapter(self) -> None:
        runner = MultiStrategyInferenceRunner(strategies=[], num_classes=2)
        adapter = _MockAdapterCustomRoi()
        strategy = _make_strategy(roi_size="per_model")
        # Volume shape (B, C, H, W, D) = (1,1,16,16,8)
        roi = runner._resolve_roi_size(strategy, adapter, (1, 1, 16, 16, 8))
        assert roi == (64, 64, 8)

    def test_resolve_roi_size_literal(self) -> None:
        runner = MultiStrategyInferenceRunner(strategies=[], num_classes=2)
        adapter = _MockAdapter()
        strategy = _make_strategy(roi_size=[128, 128, 16])
        roi = runner._resolve_roi_size(strategy, adapter, (1, 1, 32, 64, 16))
        assert roi == (128, 128, 16)

    def test_resolve_roi_size_wildcard_depth(self) -> None:
        runner = MultiStrategyInferenceRunner(strategies=[], num_classes=2)
        adapter = _MockAdapter()
        strategy = _make_strategy(roi_size=[512, 512, -1])
        # Volume shape (B, C, H, W, D) — depth is dim 4
        roi = runner._resolve_roi_size(strategy, adapter, (1, 1, 64, 64, 30))
        assert roi == (512, 512, 30)

    def test_plain_nn_module_uses_fallback_roi(self) -> None:
        runner = MultiStrategyInferenceRunner(strategies=[], num_classes=2)
        plain_model = torch.nn.Identity()
        strategy = _make_strategy(roi_size="per_model")
        roi = runner._resolve_roi_size(strategy, plain_model, (1, 1, 32, 32, 16))
        assert roi == (128, 128, 16)


class TestMakePredictor:
    def test_predictor_handles_segmentation_output(self) -> None:
        runner = MultiStrategyInferenceRunner(strategies=[], num_classes=2)
        adapter = _MockAdapter()
        predictor = runner._make_predictor(adapter)
        x = torch.zeros(1, 1, 8, 8, 4)
        out = predictor(x)
        assert isinstance(out, Tensor)

    def test_predictor_handles_plain_tensor(self) -> None:
        runner = MultiStrategyInferenceRunner(strategies=[], num_classes=2)

        class _PlainModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.shape[0], 2, *x.shape[2:])

        predictor = runner._make_predictor(_PlainModel())
        x = torch.zeros(1, 1, 8, 8, 4)
        out = predictor(x)
        assert isinstance(out, Tensor)


class TestRunAllStrategies:
    def test_run_all_strategies_returns_dict_keyed_by_strategy_name(self) -> None:
        s1 = _make_strategy("standard_patch", roi_size=[16, 16, 4], is_primary=True)
        s2 = _make_strategy("fast", roi_size=[8, 8, 2], is_primary=False)
        runner = MultiStrategyInferenceRunner(strategies=[s1, s2], num_classes=2)

        adapter = _MockAdapter()
        volume = torch.zeros(1, 1, 16, 16, 4)

        fake_output = torch.zeros(1, 2, 16, 16, 4)
        with patch(
            "minivess.pipeline.multi_strategy_inference.sliding_window_inference",
            return_value=fake_output,
        ):
            results = runner.run_all_strategies(adapter, volume)

        assert set(results.keys()) == {"standard_patch", "fast"}
        assert isinstance(results["standard_patch"], dict)
        assert isinstance(results["fast"], dict)
