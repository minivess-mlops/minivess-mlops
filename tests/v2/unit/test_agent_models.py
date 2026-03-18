"""Tests for agent Pydantic output models (T-1.1, T-2.1, T-3.1)."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai", reason="pydantic_ai not installed")

from pydantic import ValidationError  # noqa: E402

try:
    import pydantic_ai  # noqa: F401

    _HAS_PYDANTIC_AI = True
except ImportError:
    _HAS_PYDANTIC_AI = False

pytestmark = pytest.mark.skipif(
    not _HAS_PYDANTIC_AI, reason="pydantic_ai not installed"
)


class TestExperimentSummary:
    """T-1.1: ExperimentSummary output model."""

    def test_validates_valid_data(self):
        from minivess.agents.models import ExperimentSummary

        summary = ExperimentSummary(
            narrative="Model A outperformed B on Dice.",
            best_model="dynunet_dice_ce",
            best_metric_value=0.824,
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            recommendations=["Try learning rate 1e-4"],
        )
        assert summary.best_model == "dynunet_dice_ce"
        assert summary.best_metric_value == 0.824

    def test_rejects_empty_findings(self):
        from minivess.agents.models import ExperimentSummary

        with pytest.raises(ValidationError):
            ExperimentSummary(
                narrative="Summary",
                best_model="model",
                best_metric_value=0.5,
                key_findings=[],  # min_length=1
            )

    def test_serializes(self):
        from minivess.agents.models import ExperimentSummary

        summary = ExperimentSummary(
            narrative="Test",
            best_model="m",
            best_metric_value=0.9,
            key_findings=["f1"],
        )
        d = summary.model_dump()
        assert isinstance(d, dict)
        assert d["best_model"] == "m"
        assert d["key_findings"] == ["f1"]
        assert d["recommendations"] == []

    def test_max_findings_enforced(self):
        from minivess.agents.models import ExperimentSummary

        with pytest.raises(ValidationError):
            ExperimentSummary(
                narrative="Test",
                best_model="m",
                best_metric_value=0.9,
                key_findings=["f1", "f2", "f3", "f4", "f5", "f6"],  # max=5
            )


class TestDriftTriageResult:
    """T-2.1: DriftTriageResult output model."""

    def test_validates_valid_data(self):
        from minivess.agents.models import DriftTriageResult

        result = DriftTriageResult(
            action="retrain",
            confidence=0.85,
            reasoning="Significant drift detected",
            severity="high",
            affected_features=["feature_a"],
        )
        assert result.action == "retrain"
        assert result.confidence == 0.85

    def test_confidence_bounds(self):
        from minivess.agents.models import DriftTriageResult

        with pytest.raises(ValidationError):
            DriftTriageResult(
                action="monitor",
                confidence=1.5,  # > 1.0
                reasoning="Test",
                severity="low",
            )

        with pytest.raises(ValidationError):
            DriftTriageResult(
                action="monitor",
                confidence=-0.1,  # < 0.0
                reasoning="Test",
                severity="low",
            )

    def test_action_literal(self):
        from minivess.agents.models import DriftTriageResult

        with pytest.raises(ValidationError):
            DriftTriageResult(
                action="invalid_action",  # type: ignore[arg-type]
                confidence=0.5,
                reasoning="Test",
                severity="low",
            )

    def test_severity_literal(self):
        from minivess.agents.models import DriftTriageResult

        with pytest.raises(ValidationError):
            DriftTriageResult(
                action="monitor",
                confidence=0.5,
                reasoning="Test",
                severity="critical",  # type: ignore[arg-type]
            )


class TestFigureCaption:
    """T-3.1: FigureCaption output model."""

    def test_validates_valid_data(self):
        from minivess.agents.models import FigureCaption

        caption = FigureCaption(
            caption="Comparison of 4 loss functions across 3 folds.",
            alt_text="Bar chart showing Dice scores",
        )
        assert caption.statistical_note is None

    def test_optional_stat_note(self):
        from minivess.agents.models import FigureCaption

        caption = FigureCaption(
            caption="Test caption.",
            alt_text="Alt text",
            statistical_note="p < 0.05, Wilcoxon signed-rank test",
        )
        assert caption.statistical_note is not None

    def test_serializes(self):
        from minivess.agents.models import FigureCaption

        caption = FigureCaption(
            caption="Caption",
            alt_text="Alt",
        )
        d = caption.model_dump()
        assert isinstance(d, dict)
        assert d["caption"] == "Caption"
        assert d["statistical_note"] is None
