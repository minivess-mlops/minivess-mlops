"""Tests for Evidently DataDriftPreset Tier 1 integration (#574 T2, #600).

Verifies that FeatureDriftDetector generates Evidently DataDriftPreset reports
alongside existing KS tests. Evidently supplements (does NOT replace) the
scipy-based KS implementation with rich HTML reports and JSON export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _make_reference_features(*, n_samples: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate reference feature DataFrame matching the 9 volume features."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mean": rng.normal(100.0, 10.0, n_samples),
            "std": rng.normal(30.0, 5.0, n_samples),
            "min": rng.normal(0.0, 2.0, n_samples),
            "max": rng.normal(255.0, 5.0, n_samples),
            "p5": rng.normal(20.0, 5.0, n_samples),
            "p95": rng.normal(230.0, 10.0, n_samples),
            "snr": rng.normal(3.5, 0.5, n_samples),
            "contrast": rng.normal(210.0, 12.0, n_samples),
            "entropy": rng.normal(7.0, 0.3, n_samples),
        }
    )


def _make_shifted_features(
    reference: pd.DataFrame, *, shift_mean: float = 50.0, seed: int = 99
) -> pd.DataFrame:
    """Create current features with synthetic intensity drift."""
    rng = np.random.default_rng(seed)
    shifted = reference.copy()
    # Shift intensity-related features
    shifted["mean"] = shifted["mean"] + shift_mean
    shifted["p5"] = shifted["p5"] + shift_mean * 0.5
    shifted["p95"] = shifted["p95"] + shift_mean * 0.8
    shifted["contrast"] = shifted["contrast"] + rng.normal(20.0, 5.0, len(shifted))
    shifted["entropy"] = shifted["entropy"] + rng.normal(1.0, 0.2, len(shifted))
    return shifted


FEATURE_COLUMNS = [
    "mean",
    "std",
    "min",
    "max",
    "p5",
    "p95",
    "snr",
    "contrast",
    "entropy",
]


class TestEvidentlyTier1:
    """Verify Evidently DataDriftPreset integration in FeatureDriftDetector."""

    def test_evidently_report_generated(self) -> None:
        """DataDriftPreset produces a Snapshot object from reference + current."""
        from minivess.observability.drift import FeatureDriftDetector

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)
        detector = FeatureDriftDetector(ref)
        snapshot = detector.generate_evidently_report(cur)
        assert snapshot is not None

    def test_evidently_report_html_export(self, tmp_path: Path) -> None:
        """save_html() produces valid HTML file."""
        from minivess.observability.drift import FeatureDriftDetector

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)
        detector = FeatureDriftDetector(ref)
        html_path = tmp_path / "drift_report.html"
        detector.save_evidently_html(cur, output_path=html_path)
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "<html" in content.lower()
        assert len(content) > 1000  # Non-trivial Evidently report

    def test_evidently_report_json_export(self) -> None:
        """Snapshot dict() returns drift metrics."""
        from minivess.observability.drift import FeatureDriftDetector

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)
        detector = FeatureDriftDetector(ref)
        report_dict = detector.get_evidently_dict(cur)
        assert isinstance(report_dict, dict)
        assert "metrics" in report_dict

    def test_evidently_detects_synthetic_intensity_drift(self) -> None:
        """Inject intensity shift, verify dataset drift detected by Evidently."""
        from minivess.observability.drift import FeatureDriftDetector

        ref = _make_reference_features()
        cur = _make_shifted_features(ref, shift_mean=80.0)
        detector = FeatureDriftDetector(ref)
        report_dict = detector.get_evidently_dict(cur)
        # DriftedColumnsCount metric has share >= drift_share → dataset drift
        assert _extract_dataset_drift(report_dict) is True

    def test_evidently_no_drift_on_same_distribution(self) -> None:
        """Same distribution, different seeds → no dataset drift."""
        from minivess.observability.drift import FeatureDriftDetector

        ref = _make_reference_features(seed=42)
        cur = _make_reference_features(seed=43)
        detector = FeatureDriftDetector(ref)
        report_dict = detector.get_evidently_dict(cur)
        assert _extract_dataset_drift(report_dict) is False

    def test_evidently_report_contains_all_9_features(self) -> None:
        """All 9 feature columns present in Evidently report."""
        from minivess.observability.drift import FeatureDriftDetector

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)
        detector = FeatureDriftDetector(ref)
        report_dict = detector.get_evidently_dict(cur)
        columns_in_report = _extract_column_names(report_dict)
        for col in FEATURE_COLUMNS:
            assert col in columns_in_report, (
                f"Feature '{col}' missing from Evidently report"
            )

    def test_feature_drift_detector_returns_evidently_html(
        self, tmp_path: Path
    ) -> None:
        """DriftResult includes evidently_html_path when path provided."""
        from minivess.observability.drift import FeatureDriftDetector

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)
        detector = FeatureDriftDetector(ref)
        html_path = tmp_path / "evidently_report.html"
        result = detector.detect(cur, evidently_html_path=html_path)
        assert result.evidently_html_path is not None
        assert result.evidently_html_path.exists()


def _extract_dataset_drift(report_dict: dict) -> bool:
    """Extract dataset drift flag from Evidently Snapshot dict.

    Adapted from plan: Evidently 0.7.21 Snapshot.dict() structure uses
    'metric_name' and 'value' keys (not 'metric'/'result' as in older API).
    DriftedColumnsCount metric's value.share >= drift_share means dataset drift.
    """
    metrics = report_dict.get("metrics", [])
    for metric in metrics:
        metric_name = str(metric.get("metric_name", ""))
        if "DriftedColumnsCount" in metric_name:
            value = metric.get("value", {})
            if isinstance(value, dict):
                share = value.get("share", 0.0)
                # Extract drift_share from metric_name string
                # e.g., "DriftedColumnsCount(drift_share=0.5)"
                return share >= 0.5
    msg = f"DriftedColumnsCount metric not found in: {[m.get('metric_name') for m in metrics]}"
    raise KeyError(msg)


def _extract_column_names(report_dict: dict) -> set[str]:
    """Extract column names from Evidently DataDriftPreset report dict.

    Adapted from plan: ValueDrift metrics contain 'column=X' in metric_name.
    """
    columns: set[str] = set()
    metrics = report_dict.get("metrics", [])
    for metric in metrics:
        metric_name = str(metric.get("metric_name", ""))
        if "ValueDrift" in metric_name and "column=" in metric_name:
            # Parse "ValueDrift(column=a,...)" → "a"
            start = metric_name.index("column=") + len("column=")
            rest = metric_name[start:]
            end = rest.index(",") if "," in rest else rest.index(")")
            columns.add(rest[:end])
    return columns
