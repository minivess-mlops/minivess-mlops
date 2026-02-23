"""Ensemble — Model ensembling strategies (soup, voting, conformal)."""

from __future__ import annotations

from minivess.ensemble.calibration import (
    CalibrationResult,
    expected_calibration_error,
    temperature_scale,
)
from minivess.ensemble.strategies import EnsemblePredictor, greedy_soup
from minivess.ensemble.weightwatcher import WeightWatcherReport, analyze_model

__all__ = [
    "CalibrationResult",
    "EnsemblePredictor",
    "WeightWatcherReport",
    "analyze_model",
    "expected_calibration_error",
    "greedy_soup",
    "temperature_scale",
]
