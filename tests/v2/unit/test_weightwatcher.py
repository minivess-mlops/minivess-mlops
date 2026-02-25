"""Tests for WeightWatcher spectral analysis (Issue #53 — R5.8).

Tests analyze_model with simple models, result structure, and threshold gate.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from torch import nn

# ---------------------------------------------------------------------------
# R5.8 T1: WeightWatcherReport structure
# ---------------------------------------------------------------------------


class TestWeightWatcherReport:
    """Test WeightWatcherReport dataclass."""

    def test_report_construction(self) -> None:
        """WeightWatcherReport should capture all fields."""
        from minivess.ensemble.weightwatcher import WeightWatcherReport

        report = WeightWatcherReport(
            alpha_weighted=3.5,
            log_norm=1.2,
            num_layers=5,
            details={"alpha_weighted": 3.5},
            passed_gate=True,
        )
        assert report.alpha_weighted == 3.5
        assert report.passed_gate is True

    def test_report_gate_fail(self) -> None:
        """Report with alpha above threshold should fail gate."""
        from minivess.ensemble.weightwatcher import WeightWatcherReport

        report = WeightWatcherReport(
            alpha_weighted=6.0,
            log_norm=2.0,
            num_layers=5,
            details={},
            passed_gate=False,
        )
        assert report.passed_gate is False


# ---------------------------------------------------------------------------
# R5.8 T2: analyze_model with mocked WeightWatcher
# ---------------------------------------------------------------------------


def _make_mock_ww(
    alpha: float = 3.0,
    log_norm: float = 1.0,
    num_layers: int = 3,
) -> MagicMock:
    """Create a mock weightwatcher.WeightWatcher instance."""
    mock_watcher = MagicMock()
    mock_watcher.analyze.return_value = {}
    mock_watcher.get_summary.return_value = {
        "alpha_weighted": alpha,
        "log_norm": log_norm,
        "num_layers": num_layers,
    }
    return mock_watcher


class TestAnalyzeModel:
    """Test analyze_model function with mock WeightWatcher backend."""

    def test_analyze_returns_report(self) -> None:
        """analyze_model should return a WeightWatcherReport."""
        from minivess.ensemble.weightwatcher import WeightWatcherReport, analyze_model

        model = nn.Linear(10, 5)
        mock_watcher = _make_mock_ww()

        with patch("weightwatcher.WeightWatcher", return_value=mock_watcher):
            report = analyze_model(model)

        assert isinstance(report, WeightWatcherReport)

    def test_analyze_passes_threshold(self) -> None:
        """Alpha below threshold should pass gate."""
        from minivess.ensemble.weightwatcher import analyze_model

        model = nn.Linear(10, 5)
        mock_watcher = _make_mock_ww(alpha=2.0)

        with patch("weightwatcher.WeightWatcher", return_value=mock_watcher):
            report = analyze_model(model, alpha_threshold=5.0)

        assert report.passed_gate is True
        assert report.alpha_weighted == 2.0

    def test_analyze_fails_threshold(self) -> None:
        """Alpha above threshold should fail gate."""
        from minivess.ensemble.weightwatcher import analyze_model

        model = nn.Linear(10, 5)
        mock_watcher = _make_mock_ww(alpha=7.0)

        with patch("weightwatcher.WeightWatcher", return_value=mock_watcher):
            report = analyze_model(model, alpha_threshold=5.0)

        assert report.passed_gate is False

    def test_analyze_boundary_threshold(self) -> None:
        """Alpha exactly at threshold should pass gate (<=)."""
        from minivess.ensemble.weightwatcher import analyze_model

        model = nn.Linear(10, 5)
        mock_watcher = _make_mock_ww(alpha=5.0)

        with patch("weightwatcher.WeightWatcher", return_value=mock_watcher):
            report = analyze_model(model, alpha_threshold=5.0)

        assert report.passed_gate is True

    def test_analyze_custom_threshold(self) -> None:
        """Custom threshold should be respected."""
        from minivess.ensemble.weightwatcher import analyze_model

        model = nn.Linear(10, 5)
        mock_watcher = _make_mock_ww(alpha=3.0)

        with patch("weightwatcher.WeightWatcher", return_value=mock_watcher):
            report = analyze_model(model, alpha_threshold=2.0)

        assert report.passed_gate is False

    def test_report_num_layers(self) -> None:
        """Report should capture num_layers from summary."""
        from minivess.ensemble.weightwatcher import analyze_model

        model = nn.Linear(10, 5)
        mock_watcher = _make_mock_ww(num_layers=7)

        with patch("weightwatcher.WeightWatcher", return_value=mock_watcher):
            report = analyze_model(model)

        assert report.num_layers == 7
