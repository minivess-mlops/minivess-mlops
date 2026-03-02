"""Tests for condition-to-model-and-loss factory (T4 — topology real-data plan)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from minivess.pipeline.condition_builder import (
    build_condition_loss,
    build_condition_model,
    parse_aux_head_configs,
)


def _make_stub_model() -> nn.Module:
    """Create a minimal model stub with config property and DynUNet-like structure."""

    class _Bottleneck(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(32, 32, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    class _Net(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.bottleneck = _Bottleneck()
            self.output = nn.Conv3d(32, 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.output(self.bottleneck(x))

    class _StubModel(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.net = _Net()
            self.conv = nn.Conv3d(1, 2, 3, padding=1)

        @property
        def config(self) -> Any:
            from minivess.adapters.base import AdapterConfigInfo

            return AdapterConfigInfo(
                family="test", name="stub", in_channels=1, out_channels=2
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    return _StubModel()


class TestBuildConditionModel:
    """Tests for build_condition_model()."""

    def test_baseline_no_wrappers(self) -> None:
        """Baseline condition returns same model (no wrappers)."""
        model = _make_stub_model()
        condition: dict[str, Any] = {"name": "baseline", "wrappers": []}
        result = build_condition_model(model, condition)
        assert result is model

    def test_tffm_wrapper_applied(self) -> None:
        """TFFM condition returns TFFMWrapper."""
        from minivess.adapters.tffm_wrapper import TFFMWrapper

        model = _make_stub_model()
        condition: dict[str, Any] = {
            "name": "tffm",
            "wrappers": [
                {
                    "type": "tffm",
                    "grid_size": 4,
                    "hidden_dim": 16,
                    "n_heads": 2,
                    "k_neighbors": 4,
                }
            ],
        }
        result = build_condition_model(model, condition)
        assert isinstance(result, TFFMWrapper)

    def test_multitask_wrapper_applied(self) -> None:
        """Multitask condition returns MultiTaskAdapter."""
        from minivess.adapters.multitask_adapter import MultiTaskAdapter

        model = _make_stub_model()
        condition: dict[str, Any] = {
            "name": "multitask",
            "wrappers": [
                {
                    "type": "multitask",
                    "auxiliary_heads": [
                        {
                            "name": "sdf",
                            "type": "regression",
                            "out_channels": 1,
                            "loss": "smooth_l1",
                            "weight": 0.25,
                            "gt_key": "sdf",
                        }
                    ],
                }
            ],
        }
        result = build_condition_model(model, condition)
        assert isinstance(result, MultiTaskAdapter)

    def test_both_wrappers_nested(self) -> None:
        """Full pipeline condition applies both TFFM then multitask."""
        from minivess.adapters.multitask_adapter import MultiTaskAdapter

        model = _make_stub_model()
        condition: dict[str, Any] = {
            "name": "full_pipeline",
            "wrappers": [
                {
                    "type": "tffm",
                    "grid_size": 4,
                    "hidden_dim": 16,
                    "n_heads": 2,
                    "k_neighbors": 4,
                },
                {
                    "type": "multitask",
                    "auxiliary_heads": [
                        {
                            "name": "sdf",
                            "type": "regression",
                            "out_channels": 1,
                            "loss": "smooth_l1",
                            "weight": 0.25,
                            "gt_key": "sdf",
                        },
                    ],
                },
            ],
        }
        result = build_condition_model(model, condition)
        # Outermost wrapper should be MultiTaskAdapter (applied second)
        assert isinstance(result, MultiTaskAdapter)


class TestBuildConditionLoss:
    """Tests for build_condition_loss()."""

    def test_baseline_returns_standard_criterion(self) -> None:
        """Baseline condition returns standard loss (no MultiTaskLoss)."""
        condition: dict[str, Any] = {"name": "baseline", "wrappers": []}
        loss = build_condition_loss("cbdice_cldice", condition)
        # Should NOT be MultiTaskLoss
        from minivess.pipeline.multitask_loss import MultiTaskLoss

        assert not isinstance(loss, MultiTaskLoss)

    def test_multitask_returns_multitask_loss(self) -> None:
        """Multitask condition returns MultiTaskLoss."""
        from minivess.pipeline.multitask_loss import MultiTaskLoss

        condition: dict[str, Any] = {
            "name": "multitask",
            "wrappers": [
                {
                    "type": "multitask",
                    "auxiliary_heads": [
                        {
                            "name": "sdf",
                            "type": "regression",
                            "out_channels": 1,
                            "loss": "smooth_l1",
                            "weight": 0.25,
                            "gt_key": "sdf",
                        }
                    ],
                }
            ],
        }
        loss = build_condition_loss("cbdice_cldice", condition)
        assert isinstance(loss, MultiTaskLoss)
        assert len(loss.aux_head_configs) == 1
        assert loss.aux_head_configs[0].name == "sdf"


class TestParseAuxHeadConfigs:
    """Tests for YAML dict -> AuxHeadConfig conversion."""

    def test_parses_single_head(self) -> None:
        """Parses one auxiliary head from YAML dict."""
        from minivess.adapters.multitask_adapter import AuxHeadConfig

        heads = [
            {"name": "sdf", "type": "regression", "out_channels": 1, "gt_key": "sdf"}
        ]
        result = parse_aux_head_configs(heads)
        assert len(result) == 1
        assert isinstance(result[0], AuxHeadConfig)
        assert result[0].name == "sdf"
        assert result[0].head_type == "regression"
        assert result[0].gt_key == "sdf"

    def test_gt_key_defaults_to_name(self) -> None:
        """gt_key defaults to name if not specified."""
        heads = [{"name": "centerline", "type": "regression", "out_channels": 1}]
        result = parse_aux_head_configs(heads)
        assert result[0].gt_key == "centerline"
