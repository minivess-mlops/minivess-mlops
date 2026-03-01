"""Tests for reproducibility verification module.

Verifies that training metrics match inference metrics, confirming
deterministic reproducibility of the pipeline.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from minivess.pipeline.reproducibility_check import (
    ReproducibilityResult,
    compare_metric_values,
    create_reproducibility_report,
    read_training_metric_for_fold,
    verify_cv_mean_reproducibility,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mlruns_run(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
    *,
    metrics: dict[str, list[str]] | None = None,
    tags: dict[str, str] | None = None,
) -> Path:
    """Create a mock MLflow run directory with metrics and tags."""
    run_dir = mlruns_dir / experiment_id / run_id
    metrics_dir = run_dir / "metrics"
    tags_dir = run_dir / "tags"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tags_dir.mkdir(parents=True, exist_ok=True)

    if metrics:
        for name, lines in metrics.items():
            (metrics_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if tags:
        for key, value in tags.items():
            (tags_dir / key).write_text(value, encoding="utf-8")

    return run_dir


# ---------------------------------------------------------------------------
# TestReproducibilityResult
# ---------------------------------------------------------------------------


class TestReproducibilityResult:
    """Tests for the ReproducibilityResult dataclass."""

    def test_reproducible_within_tolerance(self) -> None:
        """Result is reproducible when diff < tolerance."""
        result = ReproducibilityResult(
            run_id="abc123",
            fold_id=0,
            metric_name="val_dice",
            training_value=0.8500,
            inference_value=0.8500,
            absolute_diff=0.0,
            is_reproducible=True,
        )
        assert result.is_reproducible is True
        assert result.absolute_diff == 0.0

    def test_not_reproducible_exceeds_tolerance(self) -> None:
        """Result is NOT reproducible when diff exceeds tolerance."""
        result = ReproducibilityResult(
            run_id="abc123",
            fold_id=0,
            metric_name="val_dice",
            training_value=0.8500,
            inference_value=0.7000,
            absolute_diff=0.15,
            is_reproducible=False,
        )
        assert result.is_reproducible is False
        assert result.absolute_diff == pytest.approx(0.15)

    def test_frozen_dataclass(self) -> None:
        """ReproducibilityResult should be immutable."""
        result = ReproducibilityResult(
            run_id="abc123",
            fold_id=0,
            metric_name="val_dice",
            training_value=0.85,
            inference_value=0.85,
            absolute_diff=0.0,
            is_reproducible=True,
        )
        with pytest.raises(AttributeError):
            result.is_reproducible = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestCompareMetricValues
# ---------------------------------------------------------------------------


class TestCompareMetricValues:
    """Tests for compare_metric_values()."""

    def test_exact_match(self) -> None:
        """Exact match returns reproducible with zero diff."""
        result = compare_metric_values(
            run_id="run1",
            fold_id=0,
            metric_name="val_dice",
            training_value=0.85,
            inference_value=0.85,
            tolerance=1e-5,
        )
        assert result.is_reproducible is True
        assert result.absolute_diff == 0.0

    def test_within_tolerance(self) -> None:
        """Values within tolerance are reproducible."""
        result = compare_metric_values(
            run_id="run1",
            fold_id=0,
            metric_name="val_dice",
            training_value=0.85000,
            inference_value=0.85001,
            tolerance=1e-4,
        )
        assert result.is_reproducible is True
        assert result.absolute_diff < 1e-4

    def test_exceeds_tolerance(self) -> None:
        """Values exceeding tolerance are NOT reproducible."""
        result = compare_metric_values(
            run_id="run1",
            fold_id=0,
            metric_name="val_dice",
            training_value=0.85,
            inference_value=0.80,
            tolerance=1e-3,
        )
        assert result.is_reproducible is False
        assert result.absolute_diff == pytest.approx(0.05)

    def test_nan_training_value(self) -> None:
        """NaN training value → not reproducible."""
        result = compare_metric_values(
            run_id="run1",
            fold_id=0,
            metric_name="val_dice",
            training_value=float("nan"),
            inference_value=0.85,
            tolerance=1e-5,
        )
        assert result.is_reproducible is False
        assert math.isnan(result.absolute_diff)

    def test_nan_inference_value(self) -> None:
        """NaN inference value → not reproducible."""
        result = compare_metric_values(
            run_id="run1",
            fold_id=0,
            metric_name="val_dice",
            training_value=0.85,
            inference_value=float("nan"),
            tolerance=1e-5,
        )
        assert result.is_reproducible is False


# ---------------------------------------------------------------------------
# TestReadTrainingMetricForFold
# ---------------------------------------------------------------------------


class TestReadTrainingMetricForFold:
    """Tests for read_training_metric_for_fold()."""

    def test_reads_eval_fold_metric(self, tmp_path: Path) -> None:
        """Reads the last value from eval_fold{N}_{metric} file."""
        mlruns = tmp_path / "mlruns"
        _make_mlruns_run(
            mlruns,
            "exp1",
            "run1",
            metrics={
                "eval_fold0_dsc": ["1700000000 0.8500 99"],
                "eval_fold1_dsc": ["1700000000 0.8700 99"],
            },
        )
        val = read_training_metric_for_fold(
            mlruns, "exp1", "run1", fold_id=0, metric_name="dsc"
        )
        assert val == pytest.approx(0.85)

    def test_reads_different_fold(self, tmp_path: Path) -> None:
        """Reads correct fold when multiple exist."""
        mlruns = tmp_path / "mlruns"
        _make_mlruns_run(
            mlruns,
            "exp1",
            "run1",
            metrics={
                "eval_fold0_dsc": ["1700000000 0.8500 99"],
                "eval_fold1_dsc": ["1700000000 0.8700 99"],
                "eval_fold2_dsc": ["1700000000 0.9000 99"],
            },
        )
        val = read_training_metric_for_fold(
            mlruns, "exp1", "run1", fold_id=2, metric_name="dsc"
        )
        assert val == pytest.approx(0.90)

    def test_missing_metric_raises(self, tmp_path: Path) -> None:
        """Missing metric file raises FileNotFoundError."""
        mlruns = tmp_path / "mlruns"
        _make_mlruns_run(mlruns, "exp1", "run1", metrics={})
        with pytest.raises(FileNotFoundError):
            read_training_metric_for_fold(
                mlruns, "exp1", "run1", fold_id=0, metric_name="dsc"
            )

    def test_missing_run_raises(self, tmp_path: Path) -> None:
        """Missing run directory raises FileNotFoundError."""
        mlruns = tmp_path / "mlruns"
        mlruns.mkdir()
        with pytest.raises(FileNotFoundError):
            read_training_metric_for_fold(
                mlruns, "exp1", "nonexistent", fold_id=0, metric_name="dsc"
            )


# ---------------------------------------------------------------------------
# TestReproducibilityReport
# ---------------------------------------------------------------------------


class TestReproducibilityReport:
    """Tests for ReproducibilityReport dataclass and creation."""

    def test_all_pass(self) -> None:
        """Report with all reproducible results has all_pass=True."""
        results = [
            ReproducibilityResult("r1", 0, "val_dice", 0.85, 0.85, 0.0, True),
            ReproducibilityResult("r1", 1, "val_dice", 0.87, 0.87, 0.0, True),
        ]
        report = create_reproducibility_report(results, tolerance=1e-5)
        assert report.all_pass is True
        assert len(report.results) == 2

    def test_partial_fail(self) -> None:
        """Report with one failure has all_pass=False."""
        results = [
            ReproducibilityResult("r1", 0, "val_dice", 0.85, 0.85, 0.0, True),
            ReproducibilityResult("r1", 1, "val_dice", 0.87, 0.50, 0.37, False),
        ]
        report = create_reproducibility_report(results, tolerance=1e-5)
        assert report.all_pass is False

    def test_empty_results(self) -> None:
        """Report with no results has all_pass=True (vacuously)."""
        report = create_reproducibility_report([], tolerance=1e-5)
        assert report.all_pass is True
        assert len(report.results) == 0

    def test_summary_string(self) -> None:
        """Report summary contains key information."""
        results = [
            ReproducibilityResult("r1", 0, "val_dice", 0.85, 0.85, 0.0, True),
        ]
        report = create_reproducibility_report(results, tolerance=1e-5)
        summary = report.summary
        assert "val_dice" in summary
        assert "1/1" in summary or "pass" in summary.lower()

    def test_tolerance_propagation(self) -> None:
        """Report stores the tolerance used."""
        report = create_reproducibility_report([], tolerance=1e-3)
        assert report.tolerance == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# TestVerifyCvMeanReproducibility
# ---------------------------------------------------------------------------


class TestVerifyCvMeanReproducibility:
    """Tests for verify_cv_mean_reproducibility()."""

    def test_matching_cv_means(self) -> None:
        """CV means computed from fold values match expected values."""
        fold_results = {
            0: [ReproducibilityResult("r1", 0, "val_dice", 0.80, 0.80, 0.0, True)],
            1: [ReproducibilityResult("r1", 1, "val_dice", 0.85, 0.85, 0.0, True)],
            2: [ReproducibilityResult("r1", 2, "val_dice", 0.90, 0.90, 0.0, True)],
        }
        cv_mean_metrics = {"val_dice": 0.85}  # Mean of 0.80, 0.85, 0.90
        report = verify_cv_mean_reproducibility(
            fold_results, cv_mean_metrics, tolerance=1e-4
        )
        assert report.all_pass is True

    def test_mismatched_cv_means(self) -> None:
        """CV means that don't match expected values → report fails."""
        fold_results = {
            0: [ReproducibilityResult("r1", 0, "val_dice", 0.80, 0.80, 0.0, True)],
            1: [ReproducibilityResult("r1", 1, "val_dice", 0.85, 0.85, 0.0, True)],
            2: [ReproducibilityResult("r1", 2, "val_dice", 0.90, 0.90, 0.0, True)],
        }
        cv_mean_metrics = {"val_dice": 0.50}  # Wrong mean
        report = verify_cv_mean_reproducibility(
            fold_results, cv_mean_metrics, tolerance=1e-4
        )
        assert report.all_pass is False

    def test_nan_in_fold_results(self) -> None:
        """NaN in fold results → report fails for that metric."""
        fold_results = {
            0: [
                ReproducibilityResult(
                    "r1", 0, "val_dice", float("nan"), 0.80, float("nan"), False
                )
            ],
            1: [ReproducibilityResult("r1", 1, "val_dice", 0.85, 0.85, 0.0, True)],
        }
        cv_mean_metrics = {"val_dice": 0.825}
        report = verify_cv_mean_reproducibility(
            fold_results, cv_mean_metrics, tolerance=1e-4
        )
        assert report.all_pass is False

    def test_empty_fold_results(self) -> None:
        """Empty fold results → report passes vacuously."""
        report = verify_cv_mean_reproducibility({}, {}, tolerance=1e-4)
        assert report.all_pass is True
