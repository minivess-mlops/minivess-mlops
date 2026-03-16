"""Tests for Evidently 0.7+ Workspace service integration.

TDD RED phase for Task T-B1 (Issue #762).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def reference_features() -> pd.DataFrame:
    """Reference feature distribution (simulates MiniVess training set)."""
    rng = np.random.default_rng(42)
    n = 47  # MiniVess training set size
    return pd.DataFrame(
        {
            "mean": rng.normal(0.3, 0.05, n),
            "std": rng.normal(0.15, 0.02, n),
            "min": rng.uniform(0.0, 0.05, n),
            "max": rng.uniform(0.85, 1.0, n),
            "p5": rng.normal(0.05, 0.02, n),
            "p95": rng.normal(0.65, 0.05, n),
            "snr": rng.normal(5.0, 1.0, n),
            "contrast": rng.normal(0.6, 0.05, n),
            "entropy": rng.normal(3.5, 0.3, n),
        }
    )


@pytest.fixture
def drifted_features() -> pd.DataFrame:
    """Features with intentional drift (simulates VesselNN)."""
    rng = np.random.default_rng(99)
    n = 4  # small batch
    return pd.DataFrame(
        {
            "mean": rng.normal(0.6, 0.05, n),  # shifted up
            "std": rng.normal(0.25, 0.02, n),
            "min": rng.uniform(0.1, 0.2, n),
            "max": rng.uniform(0.95, 1.0, n),
            "p5": rng.normal(0.15, 0.02, n),
            "p95": rng.normal(0.85, 0.05, n),
            "snr": rng.normal(2.0, 0.5, n),  # lower SNR
            "contrast": rng.normal(0.7, 0.05, n),
            "entropy": rng.normal(4.5, 0.3, n),
        }
    )


class TestEvidentlyDriftReporter:
    """Test the Evidently drift report generation."""

    def test_reporter_instantiation(self, reference_features: pd.DataFrame) -> None:
        from minivess.observability.evidently_service import EvidentlyDriftReporter

        reporter = EvidentlyDriftReporter(reference_data=reference_features)
        assert reporter is not None

    def test_generate_drift_report(
        self,
        reference_features: pd.DataFrame,
        drifted_features: pd.DataFrame,
    ) -> None:
        from minivess.observability.evidently_service import EvidentlyDriftReporter

        reporter = EvidentlyDriftReporter(reference_data=reference_features)
        result = reporter.generate_report(current_data=drifted_features)
        assert result is not None
        assert "drift_detected" in result
        assert "n_drifted_features" in result
        assert "feature_scores" in result

    def test_drift_detected_on_shifted_data(
        self,
        reference_features: pd.DataFrame,
        drifted_features: pd.DataFrame,
    ) -> None:
        from minivess.observability.evidently_service import EvidentlyDriftReporter

        reporter = EvidentlyDriftReporter(reference_data=reference_features)
        result = reporter.generate_report(current_data=drifted_features)
        assert result["drift_detected"] is True
        assert result["n_drifted_features"] > 0

    def test_no_drift_on_same_distribution(
        self, reference_features: pd.DataFrame
    ) -> None:
        from minivess.observability.evidently_service import EvidentlyDriftReporter

        reporter = EvidentlyDriftReporter(reference_data=reference_features)
        # Use a subset of the same distribution
        subset = reference_features.sample(n=10, random_state=42)
        result = reporter.generate_report(current_data=subset)
        assert result["drift_detected"] is False

    def test_save_html_report(
        self,
        reference_features: pd.DataFrame,
        drifted_features: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        from minivess.observability.evidently_service import EvidentlyDriftReporter

        reporter = EvidentlyDriftReporter(reference_data=reference_features)
        html_path = reporter.save_html_report(
            current_data=drifted_features,
            output_path=tmp_path / "drift_report.html",
        )
        assert html_path.exists()
        assert html_path.stat().st_size > 100

    def test_save_json_report(
        self,
        reference_features: pd.DataFrame,
        drifted_features: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        from minivess.observability.evidently_service import EvidentlyDriftReporter

        reporter = EvidentlyDriftReporter(reference_data=reference_features)
        json_path = reporter.save_json_report(
            current_data=drifted_features,
            output_path=tmp_path / "drift_report.json",
        )
        assert json_path.exists()
        import json

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "metrics" in data or "result" in data or isinstance(data, dict)


class TestPrometheusExporter:
    """Test Prometheus metric export from drift results."""

    def test_format_prometheus_metrics(self) -> None:
        from minivess.observability.evidently_service import format_prometheus_metrics

        drift_result = {
            "drift_detected": True,
            "n_drifted_features": 3,
            "feature_scores": {"mean": 0.001, "std": 0.02, "snr": 0.005},
            "dataset_drift_score": 0.33,
        }
        metrics_text = format_prometheus_metrics(drift_result)
        assert "evidently_drift_detected" in metrics_text
        assert "evidently_n_drifted_features" in metrics_text
        assert "evidently_dataset_drift_score" in metrics_text

    def test_per_feature_prometheus_metrics(self) -> None:
        from minivess.observability.evidently_service import format_prometheus_metrics

        drift_result = {
            "drift_detected": True,
            "n_drifted_features": 2,
            "feature_scores": {"mean": 0.001, "snr": 0.005},
            "dataset_drift_score": 0.22,
        }
        metrics_text = format_prometheus_metrics(drift_result)
        assert 'feature="mean"' in metrics_text
        assert 'feature="snr"' in metrics_text
