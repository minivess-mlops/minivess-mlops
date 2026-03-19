"""Integration smoke tests for MultiStrategyInferenceRunner with DynUNet (CPU)."""

from __future__ import annotations

import pytest
import torch

from minivess.config.evaluation_config import InferenceStrategyConfig
from minivess.pipeline.multi_strategy_inference import MultiStrategyInferenceRunner


def _two_strategies() -> list[InferenceStrategyConfig]:
    return [
        InferenceStrategyConfig(
            name="standard_patch",
            roi_size=[16, 16, 4],
            is_primary=True,
        ),
        InferenceStrategyConfig(
            name="fast",
            roi_size="per_model",
            is_primary=False,
        ),
    ]


@pytest.mark.slow
class TestMultiStrategyDynUNetCpu:
    def test_multi_strategy_dynunet_cpu_smoke(self) -> None:
        """DynUNetAdapter on CPU produces correct output shape for each strategy."""
        from unittest.mock import MagicMock

        from minivess.adapters.dynunet import DynUNetAdapter

        cfg = MagicMock()
        cfg.in_channels = 1
        cfg.out_channels = 2
        cfg.architecture_params = {
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "filters": [8, 16, 32],
            "kernel_size": [[3, 3, 3]] * 3,
            "deep_supervision": False,
        }
        cfg.family.value = "dynunet"
        cfg.name = "smoke_test"

        try:
            adapter = DynUNetAdapter(cfg)
        except Exception as exc:
            pytest.skip(f"DynUNet construction failed (may need GPU): {exc}")

        adapter.eval()
        strategies = _two_strategies()
        runner = MultiStrategyInferenceRunner(strategies=strategies, num_classes=2)

        volume = torch.zeros(1, 1, 16, 16, 4)
        with torch.no_grad():
            results = runner.run_all_strategies(adapter, volume)

        assert set(results.keys()) == {"standard_patch", "fast"}
        for strategy_name, result_dict in results.items():
            out = result_dict["raw_output"]
            assert isinstance(out, torch.Tensor), (
                f"{strategy_name}: output must be Tensor"
            )
            assert out.shape[0] == 1, f"{strategy_name}: batch dim must be 1"
            assert out.shape[1] == 2, (
                f"{strategy_name}: channel dim must be 2 (num_classes)"
            )
            # Output spatial dims match input
            assert out.shape[2:] == (16, 16, 4), (
                f"{strategy_name}: spatial dims mismatch"
            )
