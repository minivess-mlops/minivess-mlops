"""Harmonized segmentation output schema — single source of truth.

Every model (deterministic, probabilistic, ensemble, conformal) returns
this schema. Missing fields are ``None`` (flat ``X | None`` pattern).
Compatible with both MLflow ColSpec and BentoML flat API.

This file is the SINGLE definition point for the serving output contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HarmonizedSegmentationOutput:
    """Standardized output schema for ALL segmentation methods.

    Deterministic models: binary_mask + probabilities (UQ fields = None)
    Probabilistic models: all fields populated
    Ensemble models: all fields populated (UQ from ensemble variance)
    Conformal models: all fields + prediction_set + coverage
    """

    # --- Always present (all methods) ---
    binary_mask: np.ndarray  # (D, H, W) uint8
    probabilities: np.ndarray  # (D, H, W) float32, [0, 1]
    volume_id: str  # e.g., "mv01"
    model_name: str  # e.g., "dynunet_cbdice_cldice"

    # --- Uncertainty Quantification (None for deterministic) ---
    uncertainty_map: np.ndarray | None = None
    aleatoric_uncertainty: np.ndarray | None = None
    epistemic_uncertainty: np.ndarray | None = None
    mutual_information: np.ndarray | None = None

    # --- Conformal Prediction (None for non-conformal) ---
    prediction_set: np.ndarray | None = None
    coverage_guarantee: float | None = None
    conformal_alpha: float | None = None

    # --- Calibration (None for uncalibrated) ---
    calibrated_probabilities: np.ndarray | None = None
    ece_before: float | None = None
    ece_after: float | None = None

    # --- Ensemble metadata (None for single models) ---
    n_ensemble_members: int | None = None
    ensemble_strategy: str | None = None
    member_model_names: list[str] | None = None

    # --- Topology metrics (optional, computed post-hoc) ---
    topology_metrics: dict[str, float] | None = None

    # --- Metadata ---
    extra: dict[str, object] = field(default_factory=dict)


def validate_output(output: HarmonizedSegmentationOutput) -> list[str]:
    """Validate schema invariants. Returns list of errors (empty = valid)."""
    errors: list[str] = []

    if output.binary_mask.dtype != np.uint8:
        errors.append(f"binary_mask dtype {output.binary_mask.dtype}, expected uint8")

    if output.probabilities.min() < 0 or output.probabilities.max() > 1:
        errors.append("probabilities outside [0, 1]")

    if output.binary_mask.shape != output.probabilities.shape:
        errors.append("binary_mask and probabilities shape mismatch")

    # Conformal consistency
    if output.prediction_set is not None and output.coverage_guarantee is None:
        errors.append("prediction_set without coverage_guarantee")

    return errors
