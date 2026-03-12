"""Tests for drift detection integration in Data Flow (#574 T6, #604).

Verifies that drift_detection_task is a Prefect task that runs Tier 1
drift detection on feature DataFrames and triggers triage when drift is found.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_reference_features(*, n_samples: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate reference feature DataFrame."""
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


def _make_shifted_features(reference: pd.DataFrame) -> pd.DataFrame:
    """Create features with synthetic intensity drift across majority of features."""
    shifted = reference.copy()
    shifted["mean"] = shifted["mean"] + 80.0
    shifted["std"] = shifted["std"] + 20.0
    shifted["min"] = shifted["min"] + 30.0
    shifted["max"] = shifted["max"] + 50.0
    shifted["p5"] = shifted["p5"] + 40.0
    shifted["p95"] = shifted["p95"] + 60.0
    shifted["contrast"] = shifted["contrast"] + 40.0
    return shifted


class TestDataFlowDriftIntegration:
    """Verify drift detection task in Data Flow."""

    def test_drift_detection_task_exists(self) -> None:
        """drift_detection_task is importable and has @task decorator."""
        from minivess.orchestration.flows.data_flow import drift_detection_task

        # Verify it's a Prefect task (has .fn attribute from @task decorator)
        assert hasattr(drift_detection_task, "fn") or callable(drift_detection_task)

    def test_drift_detection_task_has_task_decorator(self) -> None:
        """Verify @task decorator via AST inspection."""
        source_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "data_flow.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "drift_detection_task"
            ):
                decorator_names = []
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        decorator_names.append(dec.id)
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                        decorator_names.append(dec.func.id)
                assert "task" in decorator_names, (
                    f"drift_detection_task decorators: {decorator_names}"
                )
                return
        raise AssertionError("drift_detection_task not found in data_flow.py")  # noqa: EM101

    def test_drift_task_receives_feature_dataframe(self) -> None:
        """drift_detection_task accepts reference and current DataFrames."""
        from minivess.orchestration.flows.data_flow import drift_detection_task

        ref = _make_reference_features()
        cur = _make_reference_features(seed=43)
        # Call the underlying function (bypass Prefect wrapper)
        fn = (
            drift_detection_task.fn
            if hasattr(drift_detection_task, "fn")
            else drift_detection_task
        )
        result = fn(reference_features=ref, current_features=cur)
        assert result is not None

    def test_drift_task_calls_tier1_detector(self) -> None:
        """FeatureDriftDetector is invoked inside drift_detection_task."""
        from minivess.orchestration.flows.data_flow import drift_detection_task

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)
        fn = (
            drift_detection_task.fn
            if hasattr(drift_detection_task, "fn")
            else drift_detection_task
        )

        with patch("minivess.observability.drift.FeatureDriftDetector") as mock_cls:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = MagicMock(
                drift_detected=True,
                dataset_drift_score=0.8,
                feature_scores={"mean": 0.001},
                drifted_features=["mean"],
            )
            mock_cls.return_value = mock_detector
            fn(reference_features=ref, current_features=cur)
            mock_cls.assert_called_once()
            mock_detector.detect.assert_called_once()

    def test_drift_task_triggers_triage_on_positive(self) -> None:
        """triage_drift called when drift detected."""
        from minivess.orchestration.flows.data_flow import drift_detection_task

        ref = _make_reference_features()
        cur = _make_shifted_features(ref)
        fn = (
            drift_detection_task.fn
            if hasattr(drift_detection_task, "fn")
            else drift_detection_task
        )

        result = fn(reference_features=ref, current_features=cur)
        # When drift is detected, triage_recommendation should be populated
        assert result.drift_detected is True
        assert result.triage_recommendation is not None

    def test_drift_task_skips_triage_on_negative(self) -> None:
        """triage NOT called when no drift."""
        from minivess.orchestration.flows.data_flow import drift_detection_task

        ref = _make_reference_features(seed=42)
        cur = _make_reference_features(seed=43)
        fn = (
            drift_detection_task.fn
            if hasattr(drift_detection_task, "fn")
            else drift_detection_task
        )

        result = fn(reference_features=ref, current_features=cur)
        assert result.drift_detected is False
        assert result.triage_recommendation is None
