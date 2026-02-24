"""Ensemble — Model ensembling strategies (soup, voting, conformal, UQ)."""

from __future__ import annotations

from minivess.ensemble.calibration import (
    CalibrationResult,
    expected_calibration_error,
    temperature_scale,
)
from minivess.ensemble.calibration_shift import (
    CalibrationShiftAnalyzer,
    ShiftedCalibrationResult,
    ShiftType,
    apply_synthetic_shift,
    evaluate_calibration_transfer,
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
    "CalibrationShiftAnalyzer",
    "ConformalMetrics",
    "ConformalPredictor",
    "ConformalResult",
    "DeepEnsemblePredictor",
    "MapieConformalSegmentation",
    "EnsemblePredictor",
    "MCDropoutPredictor",
    "ShiftType",
    "ShiftedCalibrationResult",
    "UncertaintyOutput",
    "WeightWatcherReport",
    "analyze_model",
    "apply_synthetic_shift",
    "compute_coverage_metrics",
    "evaluate_calibration_transfer",
    "expected_calibration_error",
    "greedy_soup",
    "temperature_scale",
]
