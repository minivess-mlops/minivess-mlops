"""Tests for CRC conformal post-training plugin.

Phase 6 of post-training plugin architecture (#320).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from minivess.pipeline.post_training_plugin import (
    PluginInput,
    PostTrainingPlugin,
)


def _make_conformal_data(
    n_samples: int = 50, n_classes: int = 2, spatial: tuple[int, ...] = (4, 4, 4)
) -> dict[str, Any]:
    """Create synthetic softmax scores + labels for conformal calibration."""
    rng = np.random.default_rng(42)
    # Shape: (N, C, D, H, W) softmax scores
    raw = rng.standard_normal((n_samples, n_classes, *spatial)).astype(np.float32)
    exp = np.exp(raw - raw.max(axis=1, keepdims=True))
    scores = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
    labels = rng.integers(0, n_classes, size=(n_samples, *spatial)).astype(np.int64)
    return {"scores": scores, "labels": labels}


class TestCRCConformalPlugin:
    """CRC conformal plugin should wrap CRCPredictor + varisco heatmaps."""

    def test_implements_protocol(self) -> None:
        from minivess.pipeline.post_training_plugins.crc_conformal import (
            CRCConformalPlugin,
        )

        assert isinstance(CRCConformalPlugin(), PostTrainingPlugin)

    def test_name_property(self) -> None:
        from minivess.pipeline.post_training_plugins.crc_conformal import (
            CRCConformalPlugin,
        )

        assert CRCConformalPlugin().name == "crc_conformal"

    def test_requires_calibration_data(self) -> None:
        from minivess.pipeline.post_training_plugins.crc_conformal import (
            CRCConformalPlugin,
        )

        assert CRCConformalPlugin().requires_calibration_data is True

    def test_validate_inputs_missing_calibration(self) -> None:
        from minivess.pipeline.post_training_plugins.crc_conformal import (
            CRCConformalPlugin,
        )

        pi = PluginInput(
            checkpoint_paths=[],
            config={"alpha": 0.1},
            calibration_data=None,
        )
        errors = CRCConformalPlugin().validate_inputs(pi)
        assert len(errors) > 0

    def test_alpha_parameter(self) -> None:
        from minivess.pipeline.post_training_plugins.crc_conformal import (
            CRCConformalPlugin,
        )

        data = _make_conformal_data()
        pi = PluginInput(
            checkpoint_paths=[],
            config={"alpha": 0.2},
            calibration_data=data,
        )
        result = CRCConformalPlugin().execute(pi)
        assert result.metrics["alpha"] == pytest.approx(0.2)

    def test_coverage_guarantee(self) -> None:
        from minivess.pipeline.post_training_plugins.crc_conformal import (
            CRCConformalPlugin,
        )

        data = _make_conformal_data(n_samples=200)
        pi = PluginInput(
            checkpoint_paths=[],
            config={"alpha": 0.1},
            calibration_data=data,
        )
        result = CRCConformalPlugin().execute(pi)
        assert "quantile" in result.metrics
        assert result.metrics["quantile"] > 0

    def test_varisco_heatmap_output(self) -> None:
        from minivess.pipeline.post_training_plugins.crc_conformal import (
            CRCConformalPlugin,
        )

        data = _make_conformal_data()
        pi = PluginInput(
            checkpoint_paths=[],
            config={"alpha": 0.1},
            calibration_data=data,
        )
        result = CRCConformalPlugin().execute(pi)
        assert "varisco_heatmap" in result.artifacts
