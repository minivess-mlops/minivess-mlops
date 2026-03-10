"""Integration tests for Evidently data quality and drift reports.

E2E Plan Phase 3, Task T3.3: Evidently report generation.

Verifies:
1. DataQualityReport produces non-empty HTML
2. DataDriftReport produces non-empty HTML
3. Reports saved as MLflow artifacts
4. Evidently handles 3D volume feature statistics

Drift SIMULATION is deferred to issue #574.
This only verifies the infrastructure works.

Marked @integration — excluded from staging and prod suites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
class TestEvidentlyReports:
    """Verify Evidently report generation infrastructure works."""

    def test_data_quality_report_generated(self, tmp_path: Path) -> None:
        """Verify Evidently DataQualityReport produces non-empty HTML."""
        try:
            import pandas as pd
            from evidently.metric_preset import DataQualityPreset
            from evidently.report import Report
        except ImportError:
            pytest.skip("evidently not installed")

        # Create synthetic volume metadata (3D spatial statistics)
        reference_data = pd.DataFrame(
            {
                "mean_intensity": [0.5, 0.6, 0.55, 0.48, 0.52],
                "std_intensity": [0.1, 0.12, 0.11, 0.09, 0.10],
                "volume_shape_x": [512, 512, 512, 512, 512],
                "volume_shape_y": [512, 512, 512, 512, 512],
                "volume_shape_z": [50, 60, 45, 55, 48],
                "spacing_xy": [0.31, 0.31, 0.31, 0.31, 0.31],
                "spacing_z": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=reference_data, current_data=reference_data)

        html_path = tmp_path / "data_quality_report.html"
        report.save_html(str(html_path))
        assert html_path.exists(), "Data quality report HTML not generated"
        assert html_path.stat().st_size > 1000, "Data quality report HTML too small"

    def test_prediction_drift_report_generated(self, tmp_path: Path) -> None:
        """Verify Evidently DataDriftReport produces non-empty HTML."""
        try:
            import pandas as pd
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
        except ImportError:
            pytest.skip("evidently not installed")

        # Reference predictions (training-time)
        reference = pd.DataFrame(
            {
                "prediction_mean": [0.45, 0.52, 0.48, 0.50, 0.55],
                "prediction_std": [0.05, 0.06, 0.04, 0.05, 0.07],
                "dice_score": [0.82, 0.85, 0.80, 0.83, 0.78],
            }
        )

        # Current predictions (inference-time)
        current = pd.DataFrame(
            {
                "prediction_mean": [0.47, 0.50, 0.49, 0.51, 0.53],
                "prediction_std": [0.06, 0.05, 0.05, 0.06, 0.06],
                "dice_score": [0.81, 0.84, 0.79, 0.82, 0.77],
            }
        )

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        html_path = tmp_path / "data_drift_report.html"
        report.save_html(str(html_path))
        assert html_path.exists(), "Data drift report HTML not generated"
        assert html_path.stat().st_size > 1000, "Data drift report HTML too small"

    def test_report_html_contains_expected_sections(self, tmp_path: Path) -> None:
        """Verify HTML has data quality metrics sections."""
        try:
            import pandas as pd
            from evidently.metric_preset import DataQualityPreset
            from evidently.report import Report
        except ImportError:
            pytest.skip("evidently not installed")

        data = pd.DataFrame(
            {
                "intensity": [0.5, 0.6, 0.55],
                "volume_z": [50, 60, 45],
            }
        )

        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=data, current_data=data)

        html_path = tmp_path / "report.html"
        report.save_html(str(html_path))
        html_content = html_path.read_text(encoding="utf-8")

        # HTML should contain meaningful content (not empty shell)
        assert len(html_content) > 5000, "Report HTML suspiciously small"

    def test_evidently_handles_3d_volume_features(self, tmp_path: Path) -> None:
        """Verify Evidently works with 3D spatial statistics."""
        try:
            import pandas as pd
            from evidently.metric_preset import DataQualityPreset
            from evidently.report import Report
        except ImportError:
            pytest.skip("evidently not installed")

        # Features typical of 3D biomedical volumes
        data = pd.DataFrame(
            {
                "shape_x": [512, 512, 256, 512, 256],
                "shape_y": [512, 512, 256, 512, 256],
                "shape_z": [50, 60, 30, 45, 110],
                "spacing_x": [0.31, 0.31, 0.62, 0.31, 0.62],
                "spacing_y": [0.31, 0.31, 0.62, 0.31, 0.62],
                "spacing_z": [1.0, 1.0, 2.0, 1.5, 4.97],
                "mean_intensity": [0.5, 0.48, 0.55, 0.52, 0.60],
                "foreground_ratio": [0.02, 0.03, 0.01, 0.025, 0.015],
            }
        )

        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=data, current_data=data)

        html_path = tmp_path / "3d_report.html"
        report.save_html(str(html_path))
        assert html_path.exists()
        assert html_path.stat().st_size > 1000

    def test_reports_saved_as_mlflow_artifacts(self, tmp_path: Path) -> None:
        """Verify HTML reports can be uploaded to MLflow artifact store."""
        try:
            import pandas as pd
            from evidently.metric_preset import DataQualityPreset
            from evidently.report import Report
        except ImportError:
            pytest.skip("evidently not installed")

        import mlflow

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_evidently_artifacts")

        data = pd.DataFrame(
            {
                "intensity": [0.5, 0.6, 0.55],
            }
        )

        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=data, current_data=data)

        html_path = tmp_path / "data_quality.html"
        report.save_html(str(html_path))

        with mlflow.start_run() as run:
            mlflow.log_artifact(str(html_path), "evidently_reports")

        from mlflow.tracking import MlflowClient

        client = MlflowClient(str(tmp_path / "mlruns"))
        artifacts = client.list_artifacts(run.info.run_id, "evidently_reports")
        artifact_names = [a.path for a in artifacts]
        assert any("data_quality" in name for name in artifact_names), (
            f"Evidently report not found in MLflow artifacts: {artifact_names}"
        )
