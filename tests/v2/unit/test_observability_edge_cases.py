"""Edge-case and error-path tests for observability modules (Code Review R2.4).

Tests gaps identified in the code review:
- Registry.promote: missing version, missing metrics defaults, version listing
- PPRM: monitor with single sample, recalibration, CI ordering invariant
- Lineage: pipeline_run with datasets, nonexistent job filter
- RunAnalytics: cross_fold_summary with no metrics, top_models edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# T1: ModelRegistry edge cases
# ---------------------------------------------------------------------------


class TestRegistryPromoteEdgeCases:
    """Test ModelRegistry.promote error paths and edge cases."""

    def test_promote_missing_version_raises(self) -> None:
        """promote on a nonexistent version should raise KeyError."""
        from minivess.observability.model_registry import (
            ModelRegistry,
            ModelStage,
            PromotionCriteria,
        )

        registry = ModelRegistry()
        registry.register_version("model_a", "1.0.0", {"dice": 0.85})
        criteria = PromotionCriteria(min_thresholds={"dice": 0.80})
        with pytest.raises(KeyError):
            registry.promote("model_a", "9.9.9", ModelStage.STAGING, criteria)

    def test_promote_missing_model_raises(self) -> None:
        """promote on a nonexistent model should raise KeyError."""
        from minivess.observability.model_registry import (
            ModelRegistry,
            ModelStage,
            PromotionCriteria,
        )

        registry = ModelRegistry()
        criteria = PromotionCriteria(min_thresholds={"dice": 0.80})
        with pytest.raises(KeyError):
            registry.promote("nonexistent", "1.0.0", ModelStage.STAGING, criteria)

    def test_criteria_missing_metric_defaults_to_zero(self) -> None:
        """Missing metric in min_thresholds check should default to 0.0."""
        from minivess.observability.model_registry import PromotionCriteria

        criteria = PromotionCriteria(min_thresholds={"hd95": 0.5})
        # Metrics dict does not contain hd95 → defaults to 0.0, which < 0.5
        result = criteria.check({"dice": 0.90})
        assert result.approved is False
        assert "hd95" in result.reason

    def test_criteria_missing_metric_defaults_to_inf_for_max(self) -> None:
        """Missing metric in max_thresholds check should default to inf."""
        from minivess.observability.model_registry import PromotionCriteria

        criteria = PromotionCriteria(max_thresholds={"hd95": 5.0})
        # Metrics dict does not contain hd95 → defaults to inf, which > 5.0
        result = criteria.check({"dice": 0.90})
        assert result.approved is False
        assert "hd95" in result.reason

    def test_list_versions_unknown_model(self) -> None:
        """list_versions for unknown model should return empty list."""
        from minivess.observability.model_registry import ModelRegistry

        registry = ModelRegistry()
        assert registry.list_versions("nonexistent") == []

    def test_get_production_model_unknown_model(self) -> None:
        """get_production_model for unknown model should return None."""
        from minivess.observability.model_registry import ModelRegistry

        registry = ModelRegistry()
        assert registry.get_production_model("nonexistent") is None

    def test_promote_history_grows(self) -> None:
        """Each promote call should add to audit history."""
        from minivess.observability.model_registry import (
            ModelRegistry,
            ModelStage,
            PromotionCriteria,
        )

        registry = ModelRegistry()
        registry.register_version("m", "1.0.0", {"dice": 0.85})
        criteria = PromotionCriteria(min_thresholds={"dice": 0.80})

        registry.promote("m", "1.0.0", ModelStage.STAGING, criteria)
        registry.promote("m", "1.0.0", ModelStage.PRODUCTION, criteria)

        # 1 REGISTER + 2 PROMOTE events
        assert len(registry._history) == 3
        assert registry._history[1]["action"] == "PROMOTE"
        assert registry._history[2]["action"] == "PROMOTE"

    def test_to_markdown_empty_registry(self) -> None:
        """to_markdown with no models should produce a report."""
        from minivess.observability.model_registry import ModelRegistry

        registry = ModelRegistry()
        md = registry.to_markdown()
        assert "No models registered" in md


# ---------------------------------------------------------------------------
# T2: PPRM edge cases
# ---------------------------------------------------------------------------


class TestPPRMEdgeCases:
    """Test PPRM detector numerical edge cases."""

    def test_monitor_single_sample(self) -> None:
        """monitor with a single deployment sample should not crash."""
        from minivess.observability.pprm import PPRMDetector, compute_prediction_risk

        detector = PPRMDetector(threshold=0.20)
        rng = np.random.default_rng(99)
        cal = rng.random((50,))
        labels = rng.random((50,))
        detector.calibrate(cal, labels, risk_fn=compute_prediction_risk)

        # Single deployment sample — ddof=1 variance could be problematic
        result = detector.monitor(np.array([0.5]))
        assert np.isfinite(result.risk)

    def test_recalibration_overwrites_state(self) -> None:
        """Calling calibrate() twice should overwrite rectifier."""
        from minivess.observability.pprm import PPRMDetector, compute_prediction_risk

        detector = PPRMDetector(threshold=0.20)
        rng = np.random.default_rng(42)

        # First calibration
        detector.calibrate(
            rng.random((30,)), rng.random((30,)), risk_fn=compute_prediction_risk
        )
        rect1 = detector._rectifier

        # Second calibration with different data
        detector.calibrate(
            rng.random((50,)), rng.random((50,)), risk_fn=compute_prediction_risk
        )
        rect2 = detector._rectifier

        assert rect1 != rect2
        assert detector._n_cal == 50

    def test_ci_ordering_invariant(self) -> None:
        """CI lower should always be <= risk <= CI upper."""
        from minivess.observability.pprm import PPRMDetector, compute_prediction_risk

        detector = PPRMDetector(threshold=0.20, alpha=0.05)
        rng = np.random.default_rng(42)
        detector.calibrate(
            rng.random((100,)), rng.random((100,)), risk_fn=compute_prediction_risk
        )

        result = detector.monitor(rng.random((200,)))
        assert result.ci_lower <= result.risk <= result.ci_upper

    def test_risk_estimate_to_dict_keys(self) -> None:
        """RiskEstimate.to_dict should contain all expected keys."""
        from minivess.observability.pprm import RiskEstimate

        est = RiskEstimate(
            risk=0.15,
            ci_lower=0.12,
            ci_upper=0.18,
            alarm=False,
            threshold=0.20,
            n_calibration=50,
            n_deployment=200,
        )
        d = est.to_dict()
        expected = {
            "pprm_risk",
            "pprm_ci_lower",
            "pprm_ci_upper",
            "pprm_alarm",
            "pprm_threshold",
            "pprm_n_calibration",
            "pprm_n_deployment",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# T3: LineageEmitter edge cases
# ---------------------------------------------------------------------------


class TestLineageEdgeCases:
    """Test LineageEmitter edge cases."""

    def test_pipeline_run_with_datasets(self) -> None:
        """pipeline_run should pass datasets to START and COMPLETE events."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        inputs = [{"namespace": "file", "name": "/data/raw"}]
        outputs = [{"namespace": "file", "name": "/data/processed"}]

        with emitter.pipeline_run("etl", inputs=inputs, outputs=outputs):
            pass

        start = emitter.events[0]
        complete = emitter.events[1]
        assert len(start.inputs) == 1
        assert len(complete.outputs) == 1

    def test_get_events_for_nonexistent_job(self) -> None:
        """get_events_for_job with no matching job should return empty list."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        emitter.emit_start("real_job")
        assert emitter.get_events_for_job("fake_job") == []

    def test_emit_start_auto_generates_run_id(self) -> None:
        """emit_start without explicit run_id should auto-generate one."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        e1 = emitter.emit_start("job_a")
        e2 = emitter.emit_start("job_b")
        assert e1.run.runId != e2.run.runId

    def test_empty_datasets(self) -> None:
        """emit_start with empty datasets should work fine."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        event = emitter.emit_start("job", inputs=[], outputs=[])
        assert event.inputs == []
        assert event.outputs == []


# ---------------------------------------------------------------------------
# T4: RunAnalytics DuckDB edge cases (no MLflow needed)
# ---------------------------------------------------------------------------


class TestRunAnalyticsDuckDB:
    """Test RunAnalytics SQL analytics using direct DuckDB operations."""

    def test_cross_fold_summary_no_metric_columns(self) -> None:
        """cross_fold_summary with no metric_ columns should return empty DataFrame."""
        from unittest.mock import patch

        from minivess.observability.analytics import RunAnalytics

        with (
            patch(
                "minivess.observability.analytics.resolve_tracking_uri",
                return_value="file:///tmp/mlruns",
            ),
            patch("minivess.observability.analytics.MlflowClient"),
        ):
            analytics = RunAnalytics()

        df = pd.DataFrame({"run_id": ["a", "b"], "param_fold": ["0", "1"]})
        result = analytics.cross_fold_summary(df)
        assert result.empty
        analytics.close()

    def test_register_and_query(self) -> None:
        """register_dataframe + query should allow SQL over DataFrames."""
        from unittest.mock import patch

        from minivess.observability.analytics import RunAnalytics

        with (
            patch(
                "minivess.observability.analytics.resolve_tracking_uri",
                return_value="file:///tmp/mlruns",
            ),
            patch("minivess.observability.analytics.MlflowClient"),
        ):
            analytics = RunAnalytics()

        df = pd.DataFrame(
            {
                "run_id": ["r1", "r2", "r3"],
                "metric_val_dice": [0.85, 0.90, 0.78],
            }
        )
        analytics.register_dataframe("runs", df)
        result = analytics.query("SELECT run_id FROM runs WHERE metric_val_dice > 0.80")
        assert len(result) == 2
        analytics.close()

    def test_top_models(self) -> None:
        """top_models should return top-N by metric."""
        from unittest.mock import patch

        from minivess.observability.analytics import RunAnalytics

        with (
            patch(
                "minivess.observability.analytics.resolve_tracking_uri",
                return_value="file:///tmp/mlruns",
            ),
            patch("minivess.observability.analytics.MlflowClient"),
        ):
            analytics = RunAnalytics()

        df = pd.DataFrame(
            {
                "run_id": ["r1", "r2", "r3"],
                "run_name": ["a", "b", "c"],
                "metric_val_dice": [0.85, 0.90, 0.78],
            }
        )
        result = analytics.top_models(df, metric="metric_val_dice", n=2)
        assert len(result) == 2
        assert result.iloc[0]["metric_val_dice"] == 0.90
        analytics.close()
