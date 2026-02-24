"""Ensemble — Model ensembling strategies (soup, voting, conformal, UQ)."""

from __future__ import annotations

from minivess.ensemble.calibration import (
    CalibrationResult,
    expected_calibration_error,
    temperature_scale,
)
from minivess.ensemble.conformal import ConformalPredictor, ConformalResult
from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor
from minivess.ensemble.mc_dropout import MCDropoutPredictor, UncertaintyOutput
from minivess.ensemble.strategies import EnsemblePredictor, greedy_soup
from minivess.ensemble.weightwatcher import WeightWatcherReport, analyze_model

__all__ = [
    "CalibrationResult",
    "ConformalPredictor",
    "ConformalResult",
    "DeepEnsemblePredictor",
    "EnsemblePredictor",
    "MCDropoutPredictor",
    "UncertaintyOutput",
    "WeightWatcherReport",
    "analyze_model",
    "expected_calibration_error",
    "greedy_soup",
    "temperature_scale",
]
