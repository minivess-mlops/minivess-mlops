"""Tests for post-hoc calibration post-training plugin.

Phase 5 of post-training plugin architecture (#319).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from minivess.pipeline.post_training_plugin import (
    PluginInput,
    PluginOutput,
    PostTrainingPlugin,
)


def _make_calibration_data(n_samples: int = 100, n_classes: int = 2) -> dict[str, Any]:
    """Create synthetic calibration data (overconfident logits)."""
    rng = np.random.default_rng(42)
    # Create overconfident logits (high magnitude)
    logits = rng.standard_normal((n_samples, n_classes)).astype(np.float64) * 3.0
    labels = rng.integers(0, n_classes, size=n_samples).astype(np.int64)
    return {"logits": logits, "labels": labels}


class TestCalibrationPlugin:
    """Calibration plugin should wrap existing + new calibration methods."""

    def test_implements_protocol(self) -> None:
        from minivess.pipeline.post_training_plugins.calibration import (
            CalibrationPlugin,
        )

        assert isinstance(CalibrationPlugin(), PostTrainingPlugin)

    def test_name_property(self) -> None:
        from minivess.pipeline.post_training_plugins.calibration import (
            CalibrationPlugin,
        )

        assert CalibrationPlugin().name == "calibration"

    def test_requires_calibration_data(self) -> None:
        from minivess.pipeline.post_training_plugins.calibration import (
            CalibrationPlugin,
        )

        assert CalibrationPlugin().requires_calibration_data is True

    def test_validate_inputs_missing_calibration(self) -> None:
        from minivess.pipeline.post_training_plugins.calibration import (
            CalibrationPlugin,
        )

        pi = PluginInput(
            checkpoint_paths=[],
            config={"methods": ["global_temperature"]},
            calibration_data=None,
        )
        errors = CalibrationPlugin().validate_inputs(pi)
        assert len(errors) > 0
        assert "calibration" in errors[0].lower()

    def test_global_temperature(self) -> None:
        from minivess.pipeline.post_training_plugins.calibration import (
            CalibrationPlugin,
        )

        cal_data = _make_calibration_data()
        pi = PluginInput(
            checkpoint_paths=[],
            config={"methods": ["global_temperature"], "n_bins": 15},
            calibration_data=cal_data,
        )
        result = CalibrationPlugin().execute(pi)
        assert isinstance(result, PluginOutput)
        assert "global_temperature_optimal_t" in result.metrics
        assert result.metrics["global_temperature_optimal_t"] > 0

    def test_isotonic_regression(self) -> None:
        from minivess.pipeline.post_training_plugins.calibration import (
            CalibrationPlugin,
        )

        cal_data = _make_calibration_data()
        pi = PluginInput(
            checkpoint_paths=[],
            config={"methods": ["isotonic_regression"], "n_bins": 15},
            calibration_data=cal_data,
        )
        result = CalibrationPlugin().execute(pi)
        assert isinstance(result, PluginOutput)
        assert "isotonic_regression_ece_after" in result.metrics

    def test_multiple_methods(self) -> None:
        from minivess.pipeline.post_training_plugins.calibration import (
            CalibrationPlugin,
        )

        cal_data = _make_calibration_data()
        pi = PluginInput(
            checkpoint_paths=[],
            config={
                "methods": ["global_temperature", "isotonic_regression"],
                "n_bins": 15,
            },
            calibration_data=cal_data,
        )
        result = CalibrationPlugin().execute(pi)
        assert "global_temperature_optimal_t" in result.metrics
        assert "isotonic_regression_ece_after" in result.metrics

    def test_ece_before_after(self) -> None:
        from minivess.pipeline.post_training_plugins.calibration import (
            CalibrationPlugin,
        )

        cal_data = _make_calibration_data(n_samples=500)
        pi = PluginInput(
            checkpoint_paths=[],
            config={"methods": ["global_temperature"], "n_bins": 15},
            calibration_data=cal_data,
        )
        result = CalibrationPlugin().execute(pi)
        # ECE after calibration should be reported
        assert "global_temperature_ece_before" in result.metrics
        assert "global_temperature_ece_after" in result.metrics

    def test_n_bins_parameter(self) -> None:
        from minivess.pipeline.post_training_plugins.calibration import (
            CalibrationPlugin,
        )

        cal_data = _make_calibration_data()
        pi = PluginInput(
            checkpoint_paths=[],
            config={"methods": ["global_temperature"], "n_bins": 5},
            calibration_data=cal_data,
        )
        result = CalibrationPlugin().execute(pi)
        # Should complete without error with different n_bins
        assert "global_temperature_ece_before" in result.metrics
