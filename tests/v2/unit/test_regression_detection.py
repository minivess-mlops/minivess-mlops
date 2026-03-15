"""Tests for performance regression detection helpers (T4.2).

Validates:
- No regression when within threshold
- Regression detected when above threshold
- Empty history handled gracefully
- format_regression_report() returns human-readable string
- GPU matching uses normalized name
"""

from __future__ import annotations


class TestNoRegression:
    """Latest metric within threshold produces no regression."""

    def test_no_regression_within_threshold(self) -> None:
        from minivess.diagnostics.regression_detection import (
            BaselineStats,
            detect_regression,
        )

        baseline = BaselineStats(
            metric_name="prof_cuda_peak_allocated_mb",
            count=5,
            median=3500.0,
            min=3400.0,
            max=3600.0,
            std=50.0,
            gpu_model="rtx_2070_super",
        )

        result = detect_regression(baseline, latest_value=3600.0, threshold_pct=5.0)
        assert result.is_regression is False
        assert abs(result.delta_pct - 2.857) < 0.1  # (3600-3500)/3500 * 100


class TestRegressionDetected:
    """Latest metric above threshold triggers regression."""

    def test_regression_detected_above_threshold(self) -> None:
        from minivess.diagnostics.regression_detection import (
            BaselineStats,
            detect_regression,
        )

        baseline = BaselineStats(
            metric_name="prof_cuda_peak_allocated_mb",
            count=5,
            median=3500.0,
            min=3400.0,
            max=3600.0,
            std=50.0,
            gpu_model="rtx_2070_super",
        )

        result = detect_regression(baseline, latest_value=3850.0, threshold_pct=5.0)
        assert result.is_regression is True
        assert abs(result.delta_pct - 10.0) < 0.1  # (3850-3500)/3500 * 100


class TestEmptyHistory:
    """No prior runs means no baseline — skip gracefully."""

    def test_baseline_from_empty_history(self) -> None:
        from minivess.diagnostics.regression_detection import (
            BaselineStats,
            detect_regression,
        )

        empty_baseline = BaselineStats(
            metric_name="prof_cuda_peak_allocated_mb",
            count=0,
            median=0.0,
            min=0.0,
            max=0.0,
            std=0.0,
            gpu_model="rtx_2070_super",
        )

        result = detect_regression(empty_baseline, latest_value=3500.0)
        assert result.is_regression is False


class TestReportFormat:
    """format_regression_report() returns human-readable string."""

    def test_regression_report_format(self) -> None:
        from minivess.diagnostics.regression_detection import (
            RegressionResult,
            format_regression_report,
        )

        result = RegressionResult(
            is_regression=True,
            baseline_value=3500.0,
            latest_value=3850.0,
            delta_pct=10.0,
            metric_name="prof_cuda_peak_allocated_mb",
            gpu_model="rtx_2070_super",
        )

        report = format_regression_report(result)
        assert "prof_cuda_peak_allocated_mb" in report
        assert "rtx_2070_super" in report
        assert "3500" in report
        assert "3850" in report
        assert "10.0%" in report


class TestGpuMatching:
    """Regression detection uses normalized GPU name."""

    def test_gpu_matching_uses_normalized_name(self) -> None:
        from minivess.diagnostics.regression_detection import (
            BaselineStats,
            detect_regression,
        )

        baseline = BaselineStats(
            metric_name="epoch_time_seconds",
            count=3,
            median=120.0,
            min=115.0,
            max=125.0,
            std=5.0,
            gpu_model="rtx_2070_super",
        )

        result = detect_regression(baseline, latest_value=130.0, threshold_pct=5.0)
        assert result.gpu_model == "rtx_2070_super"
        assert result.is_regression is True
