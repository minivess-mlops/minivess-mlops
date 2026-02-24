"""Tests for CyclOps-inspired healthcare ML fairness auditing (Issue #12)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T1: SubgroupMetrics and FairnessReport dataclasses
# ---------------------------------------------------------------------------


class TestSubgroupMetrics:
    """Test per-subgroup performance metrics."""

    def test_creation(self) -> None:
        """SubgroupMetrics should store subgroup info and metrics."""
        from minivess.compliance.fairness import SubgroupMetrics

        sm = SubgroupMetrics(
            subgroup_name="age_65_plus",
            subgroup_size=50,
            metrics={"dice": 0.85, "sensitivity": 0.92},
        )
        assert sm.subgroup_name == "age_65_plus"
        assert sm.metrics["dice"] == 0.85


class TestFairnessReport:
    """Test aggregate fairness report."""

    def test_creation(self) -> None:
        """FairnessReport should contain subgroup metrics and disparity."""
        from minivess.compliance.fairness import FairnessReport, SubgroupMetrics

        report = FairnessReport(
            subgroup_metrics=[
                SubgroupMetrics("group_a", 100, {"dice": 0.9}),
                SubgroupMetrics("group_b", 80, {"dice": 0.7}),
            ],
            disparity_scores={"dice": 0.2},
            passed=False,
            threshold=0.1,
        )
        assert len(report.subgroup_metrics) == 2
        assert report.disparity_scores["dice"] == 0.2
        assert report.passed is False

    def test_to_dict(self) -> None:
        """Report should serialize to dict for MLflow."""
        from minivess.compliance.fairness import FairnessReport, SubgroupMetrics

        report = FairnessReport(
            subgroup_metrics=[
                SubgroupMetrics("group_a", 100, {"dice": 0.9}),
            ],
            disparity_scores={"dice": 0.0},
            passed=True,
            threshold=0.1,
        )
        d = report.to_dict()
        assert "disparity_dice" in d
        assert "group_a_dice" in d


# ---------------------------------------------------------------------------
# T2: evaluate_subgroup_fairness
# ---------------------------------------------------------------------------


class TestEvaluateSubgroupFairness:
    """Test fairness evaluation across subgroups."""

    def _dummy_metric(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Simple accuracy metric for testing."""
        correct = (predictions == labels).mean()
        return {"accuracy": float(correct)}

    def test_equal_performance_passes(self) -> None:
        """Equal performance across subgroups should pass."""
        from minivess.compliance.fairness import evaluate_subgroup_fairness

        predictions = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        labels = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        subgroups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = evaluate_subgroup_fairness(
            predictions=predictions,
            labels=labels,
            subgroups=subgroups,
            metric_fn=self._dummy_metric,
            threshold=0.1,
        )
        assert report.passed is True
        assert report.disparity_scores["accuracy"] == 0.0

    def test_unequal_performance_fails(self) -> None:
        """Significant disparity should fail."""
        from minivess.compliance.fairness import evaluate_subgroup_fairness

        # Group A: perfect, Group B: 50% accuracy
        predictions = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        labels = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        subgroups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = evaluate_subgroup_fairness(
            predictions=predictions,
            labels=labels,
            subgroups=subgroups,
            metric_fn=self._dummy_metric,
            threshold=0.1,
        )
        assert report.passed is False
        assert report.disparity_scores["accuracy"] > 0.1

    def test_multiple_subgroups(self) -> None:
        """Should handle 3+ subgroups."""
        from minivess.compliance.fairness import evaluate_subgroup_fairness

        predictions = np.ones(12, dtype=int)
        labels = np.ones(12, dtype=int)
        subgroups = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)

        report = evaluate_subgroup_fairness(
            predictions=predictions,
            labels=labels,
            subgroups=subgroups,
            metric_fn=self._dummy_metric,
            threshold=0.1,
        )
        assert len(report.subgroup_metrics) == 3

    def test_subgroup_sizes_correct(self) -> None:
        """Reported subgroup sizes should match actual counts."""
        from minivess.compliance.fairness import evaluate_subgroup_fairness

        predictions = np.zeros(10, dtype=int)
        labels = np.zeros(10, dtype=int)
        subgroups = np.array(["A"] * 7 + ["B"] * 3)

        report = evaluate_subgroup_fairness(
            predictions=predictions,
            labels=labels,
            subgroups=subgroups,
            metric_fn=self._dummy_metric,
            threshold=0.1,
        )
        sizes = {sm.subgroup_name: sm.subgroup_size for sm in report.subgroup_metrics}
        assert sizes["A"] == 7
        assert sizes["B"] == 3


# ---------------------------------------------------------------------------
# T3: compute_disparity
# ---------------------------------------------------------------------------


class TestComputeDisparity:
    """Test disparity computation."""

    def test_zero_disparity(self) -> None:
        """Identical scores should give zero disparity."""
        from minivess.compliance.fairness import SubgroupMetrics, compute_disparity

        metrics = [
            SubgroupMetrics("A", 50, {"dice": 0.85}),
            SubgroupMetrics("B", 50, {"dice": 0.85}),
        ]
        assert compute_disparity(metrics, "dice") == 0.0

    def test_positive_disparity(self) -> None:
        """Different scores should give positive disparity."""
        from minivess.compliance.fairness import SubgroupMetrics, compute_disparity

        metrics = [
            SubgroupMetrics("A", 50, {"dice": 0.9}),
            SubgroupMetrics("B", 50, {"dice": 0.7}),
        ]
        assert abs(compute_disparity(metrics, "dice") - 0.2) < 1e-9

    def test_missing_metric_returns_zero(self) -> None:
        """Missing metric name should return zero disparity."""
        from minivess.compliance.fairness import SubgroupMetrics, compute_disparity

        metrics = [SubgroupMetrics("A", 50, {"dice": 0.85})]
        assert compute_disparity(metrics, "nonexistent") == 0.0


# ---------------------------------------------------------------------------
# T4: generate_audit_report
# ---------------------------------------------------------------------------


class TestGenerateAuditReport:
    """Test healthcare audit report generation."""

    def test_report_is_markdown(self) -> None:
        """Audit report should be a markdown string."""
        from minivess.compliance.fairness import (
            FairnessReport,
            SubgroupMetrics,
            generate_audit_report,
        )

        report = FairnessReport(
            subgroup_metrics=[
                SubgroupMetrics("group_a", 100, {"dice": 0.85}),
                SubgroupMetrics("group_b", 80, {"dice": 0.80}),
            ],
            disparity_scores={"dice": 0.05},
            passed=True,
            threshold=0.1,
        )
        md = generate_audit_report(report, product_name="MiniVess")
        assert "# Fairness Audit Report" in md
        assert "MiniVess" in md
        assert "group_a" in md

    def test_report_includes_disparity(self) -> None:
        """Audit report should include disparity scores."""
        from minivess.compliance.fairness import (
            FairnessReport,
            SubgroupMetrics,
            generate_audit_report,
        )

        report = FairnessReport(
            subgroup_metrics=[
                SubgroupMetrics("male", 60, {"dice": 0.9}),
                SubgroupMetrics("female", 40, {"dice": 0.7}),
            ],
            disparity_scores={"dice": 0.2},
            passed=False,
            threshold=0.1,
        )
        md = generate_audit_report(report, product_name="MiniVess")
        assert "FAIL" in md or "fail" in md.lower()
        assert "disparity" in md.lower() or "Disparity" in md
