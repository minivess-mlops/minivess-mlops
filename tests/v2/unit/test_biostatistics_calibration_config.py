"""Tests for calibration metrics in BiostatisticsConfig (Phase B3).

Validates that calibration metrics are included in defaults, ROPE values,
and the new calibration_co_primary_metrics field.
"""

from __future__ import annotations


class TestBiostatisticsConfigCalibrationMetrics:
    """BiostatisticsConfig should include calibration metrics."""

    def test_default_metrics_include_calibration(self) -> None:
        from minivess.config.biostatistics_config import BiostatisticsConfig

        cfg = BiostatisticsConfig()
        assert "cal_ece" in cfg.metrics
        assert "cal_mce" in cfg.metrics
        assert "cal_brier" in cfg.metrics
        assert "cal_nll" in cfg.metrics
        assert "cal_ace" in cfg.metrics
        assert "cal_ba_ece" in cfg.metrics

    def test_rope_includes_calibration(self) -> None:
        from minivess.config.biostatistics_config import BiostatisticsConfig

        cfg = BiostatisticsConfig()
        assert "cal_ece" in cfg.rope_values
        assert cfg.rope_values["cal_ece"] > 0
        assert "cal_brier" in cfg.rope_values
        assert cfg.rope_values["cal_brier"] > 0
        assert "cal_ba_ece" in cfg.rope_values
        assert cfg.rope_values["cal_ba_ece"] > 0

    def test_calibration_co_primary_metrics(self) -> None:
        from minivess.config.biostatistics_config import BiostatisticsConfig

        cfg = BiostatisticsConfig()
        assert hasattr(cfg, "calibration_co_primary_metrics")
        assert "cal_ece" in cfg.calibration_co_primary_metrics
        assert "cal_ba_ece" in cfg.calibration_co_primary_metrics

    def test_calibration_co_primary_metrics_are_subset_of_metrics(self) -> None:
        from minivess.config.biostatistics_config import BiostatisticsConfig

        cfg = BiostatisticsConfig()
        for m in cfg.calibration_co_primary_metrics:
            assert m in cfg.metrics, (
                f"calibration co-primary metric '{m}' not in default metrics list"
            )

    def test_calibration_rope_values_all_positive(self) -> None:
        from minivess.config.biostatistics_config import BiostatisticsConfig

        cfg = BiostatisticsConfig()
        cal_rope_keys = [k for k in cfg.rope_values if k.startswith("cal_")]
        assert len(cal_rope_keys) >= 3, "Expected at least 3 calibration ROPE values"
        for key in cal_rope_keys:
            assert cfg.rope_values[key] > 0, f"ROPE for {key} must be positive"

    def test_segmentation_metrics_still_present(self) -> None:
        """Adding calibration metrics must not remove segmentation metrics."""
        from minivess.config.biostatistics_config import BiostatisticsConfig

        cfg = BiostatisticsConfig()
        for m in (
            "dsc",
            "hd95",
            "assd",
            "nsd",
            "cldice",
            "be_0",
            "be_1",
            "junction_f1",
        ):
            assert m in cfg.metrics, f"Segmentation metric '{m}' missing from defaults"
