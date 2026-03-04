"""Post-hoc calibration post-training plugin.

Wraps existing calibration methods (global temperature scaling) and adds
isotonic regression and spatial Platt scaling.

References:
    - Guo et al. (2017), "On Calibration of Modern Neural Networks"
    - Rousseau et al. (ISBI 2021), "Post-training calibration for segmentation"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from minivess.ensemble.calibration import expected_calibration_error, temperature_scale
from minivess.pipeline.post_training_plugin import PluginInput, PluginOutput

logger = logging.getLogger(__name__)


class CalibrationPlugin:
    """Post-hoc calibration plugin — temperature scaling, isotonic, spatial Platt."""

    @property
    def name(self) -> str:
        return "calibration"

    @property
    def requires_calibration_data(self) -> bool:
        return True

    def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
        errors: list[str] = []
        if plugin_input.calibration_data is None:
            errors.append(
                "Calibration plugin requires calibration_data (logits + labels)"
            )
        return errors

    def execute(self, plugin_input: PluginInput) -> PluginOutput:
        config = plugin_input.config
        methods: list[str] = config.get("methods", ["global_temperature"])
        n_bins: int = config.get("n_bins", 15)
        cal_data = plugin_input.calibration_data
        assert cal_data is not None

        logits = np.asarray(cal_data["logits"], dtype=np.float64)
        labels = np.asarray(cal_data["labels"], dtype=np.int64)

        metrics: dict[str, float] = {}
        artifacts: dict[str, Any] = {"methods_applied": methods}

        for method in methods:
            if method == "global_temperature":
                _run_global_temperature(logits, labels, n_bins=n_bins, metrics=metrics)
            elif method == "isotonic_regression":
                _run_isotonic_regression(logits, labels, n_bins=n_bins, metrics=metrics)
            elif method == "spatial_platt":
                _run_spatial_platt(logits, labels, n_bins=n_bins, metrics=metrics)
            else:
                logger.warning("Unknown calibration method: %s — skipping", method)

        return PluginOutput(artifacts=artifacts, metrics=metrics)


def _compute_ece(logits: np.ndarray, labels: np.ndarray, *, n_bins: int) -> float:
    """Compute ECE from logits and labels."""
    # Softmax
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)

    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == labels).astype(np.float64)
    ece, _ = expected_calibration_error(confidences, accuracies, n_bins=n_bins)
    return float(ece)


def _run_global_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int,
    metrics: dict[str, float],
) -> None:
    """Find optimal temperature via NLL minimization, report ECE before/after."""
    ece_before = _compute_ece(logits, labels, n_bins=n_bins)
    metrics["global_temperature_ece_before"] = ece_before

    def nll(t: float) -> float:
        calibrated = temperature_scale(logits, t)
        # Clip to avoid log(0)
        calibrated = np.clip(calibrated, 1e-10, 1.0)
        log_probs = np.log(calibrated)
        return float(-log_probs[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    optimal_t = float(result.x)
    metrics["global_temperature_optimal_t"] = optimal_t

    ece_after = _compute_ece(logits / optimal_t, labels, n_bins=n_bins)
    metrics["global_temperature_ece_after"] = ece_after
    logger.info(
        "Global temperature: T=%.3f, ECE %.4f → %.4f",
        optimal_t,
        ece_before,
        ece_after,
    )


def _run_isotonic_regression(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int,
    metrics: dict[str, float],
) -> None:
    """Fit isotonic regression on max-class probabilities."""
    ece_before = _compute_ece(logits, labels, n_bins=n_bins)
    metrics["isotonic_regression_ece_before"] = ece_before

    # Softmax
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)

    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correct = (predictions == labels).astype(np.float64)

    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_reg.fit(confidences, correct)
    calibrated_conf = iso_reg.predict(confidences)

    ece_after, _ = expected_calibration_error(calibrated_conf, correct, n_bins=n_bins)
    metrics["isotonic_regression_ece_after"] = float(ece_after)
    logger.info("Isotonic regression: ECE %.4f → %.4f", ece_before, ece_after)


def _run_spatial_platt(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int,
    metrics: dict[str, float],
) -> None:
    """Spatial Platt scaling (per-class bias + scale, fitted via logistic regression)."""
    ece_before = _compute_ece(logits, labels, n_bins=n_bins)
    metrics["spatial_platt_ece_before"] = ece_before

    n_classes = logits.shape[-1]
    # Platt scaling: fit a, b per class such that p(y=c) = sigmoid(a_c * logit_c + b_c)
    # Simplified: use per-class temperature + bias
    from scipy.optimize import minimize

    def platt_nll(params: np.ndarray) -> float:
        a = params[:n_classes]
        b = params[n_classes:]
        scaled = logits * a[np.newaxis, :] + b[np.newaxis, :]
        shifted = scaled - scaled.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        probs = np.clip(probs, 1e-10, 1.0)
        log_probs = np.log(probs)
        return float(-log_probs[np.arange(len(labels)), labels].mean())

    x0 = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
    result = minimize(platt_nll, x0, method="L-BFGS-B")

    a_opt = result.x[:n_classes]
    b_opt = result.x[n_classes:]
    calibrated_logits = logits * a_opt[np.newaxis, :] + b_opt[np.newaxis, :]

    ece_after = _compute_ece(calibrated_logits, labels, n_bins=n_bins)
    metrics["spatial_platt_ece_after"] = float(ece_after)
    metrics["spatial_platt_params_a"] = float(np.mean(a_opt))
    metrics["spatial_platt_params_b"] = float(np.mean(b_opt))
    logger.info("Spatial Platt: ECE %.4f → %.4f", ece_before, ece_after)
