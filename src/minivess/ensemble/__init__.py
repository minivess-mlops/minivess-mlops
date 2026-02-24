"""Ensemble — Model ensembling strategies (soup, voting, conformal, UQ)."""

from __future__ import annotations

from minivess.ensemble.calibration import (
    CalibrationResult,
    expected_calibration_error,
    temperature_scale,
)
from minivess.ensemble.conformal import ConformalPredictor, ConformalResult
from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor
from minivess.ensemble.mapie_conformal import (
    ConformalMetrics,
    MapieConformalSegmentation,
    compute_coverage_metrics,
)
from minivess.ensemble.mc_dropout import MCDropoutPredictor, UncertaintyOutput
from minivess.ensemble.strategies import EnsemblePredictor, greedy_soup
from minivess.ensemble.weightwatcher import WeightWatcherReport, analyze_model

__all__ = [
    "CalibrationResult",
    "ConformalMetrics",
    "ConformalPredictor",
    "ConformalResult",
    "DeepEnsemblePredictor",
    "MapieConformalSegmentation",
    "EnsemblePredictor",
    "MCDropoutPredictor",
    "UncertaintyOutput",
    "WeightWatcherReport",
    "analyze_model",
    "compute_coverage_metrics",
    "expected_calibration_error",
    "greedy_soup",
    "temperature_scale",
]
