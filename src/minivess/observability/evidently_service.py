"""Evidently 0.7+ drift reporting service.

Generates DataDriftPreset reports, exports to HTML/JSON, and formats
Prometheus-compatible metrics for Grafana dashboard integration.

Uses Evidently 0.7+ Dataset + Report API (NOT the legacy ColumnMapping API).

Usage:
    reporter = EvidentlyDriftReporter(reference_data=train_features)
    result = reporter.generate_report(current_data=batch_features)
    reporter.save_html_report(current_data=batch_features, output_path=path)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


class EvidentlyDriftReporter:
    """Generate drift reports using Evidently 0.7+ DataDriftPreset.

    Wraps Evidently's Report API for tabular feature-level
    drift detection on extracted 3D volume features.
    """

    def __init__(self, reference_data: pd.DataFrame) -> None:
        self._reference_df = reference_data
        self._reference_ds = Dataset.from_pandas(reference_data)

    def _build_report(self, current_data: pd.DataFrame) -> Any:
        """Build and run an Evidently DataDriftPreset report snapshot."""
        current_ds = Dataset.from_pandas(current_data)
        report = Report([DataDriftPreset()])
        snapshot = report.run(
            reference_data=self._reference_ds,
            current_data=current_ds,
        )
        return snapshot

    def generate_report(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """Generate a drift detection report.

        Returns:
            Dict with keys: drift_detected, n_drifted_features,
            feature_scores, dataset_drift_score, timestamp.
        """
        snapshot = self._build_report(current_data)
        report_dict = snapshot.dict()
        drift_info = _extract_drift_info(report_dict.get("metrics", []))

        return {
            "drift_detected": drift_info["dataset_drift"],
            "n_drifted_features": drift_info["n_drifted"],
            "n_features": drift_info["n_features"],
            "feature_scores": drift_info["feature_p_values"],
            "dataset_drift_score": drift_info["share_drifted"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def save_html_report(
        self,
        current_data: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Save an Evidently HTML report to disk."""
        snapshot = self._build_report(current_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))
        return output_path

    def save_json_report(
        self,
        current_data: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Save drift report as JSON to disk."""
        snapshot = self._build_report(current_data)
        report_dict = snapshot.dict()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report_dict, indent=2, default=str),
            encoding="utf-8",
        )
        return output_path


def _extract_drift_info(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract drift info from Evidently 0.7+ report metrics.

    Evidently 0.7+ metrics structure:
    - DriftedColumnsCount: {value: {count: N, share: F}}
    - ValueDrift(column=X): {value: p_value_float, config: {column: X, threshold: 0.05}}
    """
    n_drifted = 0
    n_features = 0
    share_drifted = 0.0
    feature_p_values: dict[str, float] = {}

    for metric in metrics:
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {})
        value = metric.get("value")

        if "DriftedColumnsCount" in metric_name:
            # Overall drift summary
            if isinstance(value, dict):
                n_drifted = int(value.get("count", 0))
                share_drifted = float(value.get("share", 0.0))

        elif "ValueDrift" in metric_name:
            # Per-column drift p-value
            column = config.get("column", "")
            if column and isinstance(value, int | float):
                feature_p_values[column] = float(value)
                n_features += 1

    dataset_drift = n_drifted > 0

    return {
        "dataset_drift": dataset_drift,
        "n_drifted": n_drifted,
        "n_features": n_features,
        "share_drifted": share_drifted,
        "feature_p_values": feature_p_values,
    }


def format_prometheus_metrics(drift_result: dict[str, Any]) -> str:
    """Format drift result as Prometheus text exposition.

    Returns:
        Multi-line string in Prometheus text format.
    """
    lines: list[str] = []

    detected = 1 if drift_result.get("drift_detected") else 0
    lines.append(f"evidently_drift_detected {detected}")
    lines.append(
        f"evidently_n_drifted_features {drift_result.get('n_drifted_features', 0)}"
    )
    lines.append(
        f"evidently_dataset_drift_score {drift_result.get('dataset_drift_score', 0.0)}"
    )

    feature_scores = drift_result.get("feature_scores", {})
    for feature_name, p_value in feature_scores.items():
        lines.append(
            f'evidently_feature_drift_pvalue{{feature="{feature_name}"}} {p_value}'
        )

    return "\n".join(lines) + "\n"
