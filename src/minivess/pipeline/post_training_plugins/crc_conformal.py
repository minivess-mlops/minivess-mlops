"""CRC conformal post-training plugin.

Wraps ``minivess.ensemble.crc_conformal.CRCPredictor`` and
``varisco_heatmap()`` for prediction sets with coverage guarantees.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from minivess.ensemble.crc_conformal import CRCPredictor, varisco_heatmap
from minivess.pipeline.post_training_plugin import PluginInput, PluginOutput

logger = logging.getLogger(__name__)


class CRCConformalPlugin:
    """CRC conformal plugin — conformalized risk control with LAC scores."""

    @property
    def name(self) -> str:
        return "crc_conformal"

    @property
    def requires_calibration_data(self) -> bool:
        return True

    def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
        errors: list[str] = []
        if plugin_input.calibration_data is None:
            errors.append("CRC conformal requires calibration_data (scores + labels)")
        else:
            if "scores" not in plugin_input.calibration_data:
                errors.append("calibration_data must contain 'scores' key")
            if "labels" not in plugin_input.calibration_data:
                errors.append("calibration_data must contain 'labels' key")
        return errors

    def execute(self, plugin_input: PluginInput) -> PluginOutput:
        config = plugin_input.config
        alpha: float = config.get("alpha", 0.1)
        cal_data = plugin_input.calibration_data
        assert cal_data is not None

        scores = np.asarray(cal_data["scores"], dtype=np.float32)
        labels = np.asarray(cal_data["labels"], dtype=np.int64)

        # Split into calibration and test halves
        n = scores.shape[0]
        n_cal = max(1, n // 2)
        cal_scores, test_scores = scores[:n_cal], scores[n_cal:]
        cal_labels = labels[:n_cal]

        # Calibrate and predict
        predictor = CRCPredictor(alpha=alpha)
        predictor.calibrate(cal_scores, cal_labels)
        result = predictor.predict(test_scores)

        # Varisco heatmap
        heatmap = varisco_heatmap(result.prediction_sets)

        metrics: dict[str, float] = {
            "alpha": alpha,
            "quantile": result.quantile,
            "mean_set_size": float(result.prediction_sets.sum(axis=1).mean()),
        }
        artifacts: dict[str, Any] = {
            "varisco_heatmap": heatmap,
            "prediction_sets_shape": list(result.prediction_sets.shape),
        }

        logger.info(
            "CRC conformal: alpha=%.2f, quantile=%.4f, mean_set_size=%.2f",
            alpha,
            result.quantile,
            metrics["mean_set_size"],
        )

        return PluginOutput(artifacts=artifacts, metrics=metrics)
