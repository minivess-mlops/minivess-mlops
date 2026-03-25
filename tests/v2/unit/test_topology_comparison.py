"""Tests for cross-approach topology comparison evaluator (T15 — #242)."""

from __future__ import annotations

from minivess.pipeline.topology_comparison import TopologyComparisonEvaluator


def _make_mock_results() -> dict[str, list[dict[str, float]]]:
    """Create synthetic 3-fold results for 3 conditions."""
    return {
        "baseline": [
            {"dice": 0.80, "cldice": 0.75, "hd95": 5.0, "nsd": 0.70},
            {"dice": 0.82, "cldice": 0.77, "hd95": 4.8, "nsd": 0.72},
            {"dice": 0.81, "cldice": 0.76, "hd95": 4.9, "nsd": 0.71},
        ],
        "d2c_only": [
            {"dice": 0.82, "cldice": 0.82, "hd95": 4.5, "nsd": 0.74},
            {"dice": 0.83, "cldice": 0.83, "hd95": 4.3, "nsd": 0.75},
            {"dice": 0.82, "cldice": 0.82, "hd95": 4.4, "nsd": 0.74},
        ],
        "multitask": [
            {"dice": 0.83, "cldice": 0.79, "hd95": 4.2, "nsd": 0.76},
            {"dice": 0.84, "cldice": 0.80, "hd95": 4.0, "nsd": 0.77},
            {"dice": 0.83, "cldice": 0.79, "hd95": 4.1, "nsd": 0.76},
        ],
        "tffm": [
            {"dice": 0.81, "cldice": 0.78, "hd95": 4.7, "nsd": 0.72},
            {"dice": 0.82, "cldice": 0.79, "hd95": 4.5, "nsd": 0.73},
            {"dice": 0.81, "cldice": 0.78, "hd95": 4.6, "nsd": 0.72},
        ],
    }


class TestTopologyComparison:
    """Tests for TopologyComparisonEvaluator."""

    def test_evaluator_loads_mock_results(self) -> None:
        """Loads synthetic results dict."""
        results = _make_mock_results()
        evaluator = TopologyComparisonEvaluator(
            results, metric_names=["dice", "cldice"]
        )
        assert len(evaluator.results) == 4

    def test_evaluator_computes_mean_std(self) -> None:
        """Correct aggregation across folds."""
        results = _make_mock_results()
        evaluator = TopologyComparisonEvaluator(
            results, metric_names=["dice", "cldice"]
        )
        summary = evaluator.compute_summary()

        assert "baseline" in summary
        assert abs(summary["baseline"]["dice"]["mean"] - 0.81) < 0.01
        assert summary["baseline"]["dice"]["std"] > 0.0

    def test_evaluator_paired_bootstrap(self) -> None:
        """p-values computed for pairs."""
        results = _make_mock_results()
        evaluator = TopologyComparisonEvaluator(results)
        result = evaluator.paired_bootstrap("d2c_only", "baseline", "cldice", seed=42)

        assert "p_value" in result
        assert "mean_diff" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["mean_diff"] > 0  # D2C should be better

    def test_evaluator_prediction_p1(self) -> None:
        """P1 pass if clDice diff > 3pp (or secondary gate)."""
        results = _make_mock_results()
        evaluator = TopologyComparisonEvaluator(results)
        predictions = evaluator.evaluate_predictions()

        assert "P1" in predictions
        # D2C clDice ~0.823 vs baseline ~0.76, diff ~6.3pp > 3pp → pass
        assert predictions["P1"]["pass"] is True

    def test_evaluator_prediction_p3(self) -> None:
        """P3 pass if clDice diff > 1pp."""
        results = _make_mock_results()
        evaluator = TopologyComparisonEvaluator(results)
        predictions = evaluator.evaluate_predictions()

        assert "P3" in predictions
        # Multitask clDice ~0.793 vs baseline ~0.76, diff ~3.3pp > 1pp → pass
        assert predictions["P3"]["pass"] is True

    def test_evaluator_prediction_p4(self) -> None:
        """P4 reports diff (informational)."""
        results = _make_mock_results()
        evaluator = TopologyComparisonEvaluator(results)
        predictions = evaluator.evaluate_predictions()

        assert "P4" in predictions
        assert predictions["P4"]["pass"] is None  # Informational
        assert "cldice_diff_pp" in predictions["P4"]

    def test_evaluator_exports_markdown(self) -> None:
        """Valid Markdown table."""
        results = _make_mock_results()
        evaluator = TopologyComparisonEvaluator(
            results, metric_names=["dice", "cldice"]
        )
        md = evaluator.export_markdown()

        assert "| Condition |" in md
        assert "baseline" in md
        assert "+/-" in md

    def test_evaluator_exports_latex(self) -> None:
        """Valid LaTeX table."""
        results = _make_mock_results()
        evaluator = TopologyComparisonEvaluator(
            results, metric_names=["dice", "cldice"]
        )
        latex = evaluator.export_latex()

        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex
        assert "baseline" in latex
        assert "\\pm" in latex
