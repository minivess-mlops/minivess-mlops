"""Tests for ConSeCo FP control post-training plugin.

Phase 7 of post-training plugin architecture (#321).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from minivess.pipeline.post_training_plugin import (
    PluginInput,
    PluginOutput,
    PostTrainingPlugin,
)


def _make_segmentation_data(
    n_samples: int = 20,
    spatial: tuple[int, ...] = (8, 8, 8),
) -> dict[str, Any]:
    """Create synthetic predicted masks + ground truth for FP control."""
    rng = np.random.default_rng(42)
    # Binary predictions (some false positives)
    pred_probs = rng.uniform(0, 1, size=(n_samples, *spatial)).astype(np.float32)
    pred_masks = (pred_probs > 0.4).astype(
        np.int64
    )  # Intentionally low threshold → FPs
    gt_masks = (rng.uniform(0, 1, size=(n_samples, *spatial)) > 0.6).astype(np.int64)
    return {
        "pred_probs": pred_probs,
        "pred_masks": pred_masks,
        "gt_masks": gt_masks,
    }


class TestConSeCoPlugin:
    """ConSeCo FP control should shrink masks to reduce false positives."""

    def test_implements_protocol(self) -> None:
        from minivess.pipeline.post_training_plugins.conseco_fp_control import (
            ConSeCoPlugin,
        )

        assert isinstance(ConSeCoPlugin(), PostTrainingPlugin)

    def test_name_property(self) -> None:
        from minivess.pipeline.post_training_plugins.conseco_fp_control import (
            ConSeCoPlugin,
        )

        assert ConSeCoPlugin().name == "conseco_fp_control"

    def test_requires_calibration_data(self) -> None:
        from minivess.pipeline.post_training_plugins.conseco_fp_control import (
            ConSeCoPlugin,
        )

        assert ConSeCoPlugin().requires_calibration_data is True

    def test_threshold_shrinking(self) -> None:
        from minivess.pipeline.post_training_plugins.conseco_fp_control import (
            ConSeCoPlugin,
        )

        data = _make_segmentation_data()
        pi = PluginInput(
            checkpoint_paths=[],
            config={"tolerance": 0.3, "shrink_method": "threshold"},
            calibration_data=data,
        )
        result = ConSeCoPlugin().execute(pi)
        assert isinstance(result, PluginOutput)
        assert "optimal_threshold" in result.metrics

    def test_erosion_shrinking(self) -> None:
        from minivess.pipeline.post_training_plugins.conseco_fp_control import (
            ConSeCoPlugin,
        )

        data = _make_segmentation_data()
        pi = PluginInput(
            checkpoint_paths=[],
            config={"tolerance": 0.3, "shrink_method": "erosion"},
            calibration_data=data,
        )
        result = ConSeCoPlugin().execute(pi)
        assert isinstance(result, PluginOutput)
        assert "optimal_erosion_radius" in result.metrics

    def test_tolerance_parameter(self) -> None:
        from minivess.pipeline.post_training_plugins.conseco_fp_control import (
            ConSeCoPlugin,
        )

        data = _make_segmentation_data()
        pi = PluginInput(
            checkpoint_paths=[],
            config={"tolerance": 0.05, "shrink_method": "threshold"},
            calibration_data=data,
        )
        result = ConSeCoPlugin().execute(pi)
        assert "tolerance" in result.metrics
        assert result.metrics["tolerance"] == pytest.approx(0.05)

    def test_validate_inputs_missing_data(self) -> None:
        from minivess.pipeline.post_training_plugins.conseco_fp_control import (
            ConSeCoPlugin,
        )

        pi = PluginInput(
            checkpoint_paths=[],
            config={"tolerance": 0.1},
            calibration_data=None,
        )
        errors = ConSeCoPlugin().validate_inputs(pi)
        assert len(errors) > 0
