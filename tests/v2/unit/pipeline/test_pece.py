"""Tests for pECE (pixel-wise ECE) metric — Li et al. 2025, arXiv:2503.05107.

TDD RED phase: these tests define the expected behavior of compute_pece().
Reference implementation: github.com/EagleAdelaide/SDC-Loss (MIT license).
"""

from __future__ import annotations

import numpy as np
import pytest

from minivess.config.biostatistics_config import BiostatisticsConfig


class TestComputePeceExists:
    """Function must be importable and callable."""

    def test_importable(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_pece

        assert callable(compute_pece)

    def test_signature(self) -> None:
        import inspect

        from minivess.pipeline.calibration_metrics import compute_pece

        sig = inspect.signature(compute_pece)
        params = list(sig.parameters.keys())
        assert "probs" in params
        assert "labels" in params
        assert "fp_weight" in params
        assert "n_bins" in params


class TestComputePeceBasicBehavior:
    """Core behavior: pECE penalizes FP overconfidence."""

    def test_perfect_prediction_lower_than_random(self) -> None:
        """Perfect prediction should have lower pECE than random."""
        from minivess.pipeline.calibration_metrics import compute_pece

        # Perfect: FG at 0.9, BG at 0.1
        probs_perfect = np.concatenate([np.full(20, 0.9), np.full(80, 0.1)])
        labels = np.concatenate([np.ones(20), np.zeros(80)])
        pece_perfect = compute_pece(probs_perfect, labels, fp_weight=2.0, n_bins=10)

        # Random: uniform probs
        probs_random = np.random.default_rng(42).random(100)
        pece_random = compute_pece(probs_random, labels, fp_weight=2.0, n_bins=10)

        assert pece_perfect < pece_random, (
            f"Perfect should have lower pECE than random: {pece_perfect} vs {pece_random}"
        )

    def test_overconfident_fps_high_pece(self) -> None:
        """Overconfident FPs: high confidence on wrong voxels → high pECE."""
        from minivess.pipeline.calibration_metrics import compute_pece

        # 100 voxels: 20 FP at confidence 0.9 (wrong!), 80 TN at 0.1
        probs = np.concatenate([np.full(20, 0.9), np.full(80, 0.1)])
        labels = np.zeros(100)  # ALL background — the 20 high-conf are FPs
        result = compute_pece(probs, labels, fp_weight=2.0, n_bins=10)
        assert result > 0.3, f"Overconfident FPs should have high pECE, got {result}"

    def test_low_confidence_fps_lower_pece(self) -> None:
        """Low-confidence FPs: same spatial error but uncertain → lower pECE."""
        from minivess.pipeline.calibration_metrics import compute_pece

        # Same 20 FPs but at low confidence (0.3 instead of 0.9)
        probs_high = np.concatenate([np.full(20, 0.9), np.full(80, 0.1)])
        probs_low = np.concatenate([np.full(20, 0.3), np.full(80, 0.1)])
        labels = np.zeros(100)

        pece_high = compute_pece(probs_high, labels, fp_weight=2.0, n_bins=10)
        pece_low = compute_pece(probs_low, labels, fp_weight=2.0, n_bins=10)
        assert pece_low < pece_high, (
            f"Low-conf FPs should have lower pECE: {pece_low} vs {pece_high}"
        )

    def test_fp_weight_increases_penalty(self) -> None:
        """Higher fp_weight increases pECE for same FP configuration."""
        from minivess.pipeline.calibration_metrics import compute_pece

        probs = np.concatenate([np.full(20, 0.8), np.full(80, 0.1)])
        labels = np.zeros(100)  # All FPs

        pece_low_w = compute_pece(probs, labels, fp_weight=1.0, n_bins=10)
        pece_high_w = compute_pece(probs, labels, fp_weight=3.0, n_bins=10)
        assert pece_high_w > pece_low_w, "Higher fp_weight should increase pECE"


class TestComputePeceEdgeCases:
    """Edge cases: empty bins, all zeros, all ones."""

    def test_all_zero_labels(self) -> None:
        """All background — every prediction is a potential FP."""
        from minivess.pipeline.calibration_metrics import compute_pece

        probs = np.random.default_rng(42).random(100)
        labels = np.zeros(100)
        result = compute_pece(probs, labels, fp_weight=2.0, n_bins=10)
        assert np.isfinite(result)

    def test_all_one_labels(self) -> None:
        """All foreground — no FPs possible."""
        from minivess.pipeline.calibration_metrics import compute_pece

        probs = np.random.default_rng(42).random(100)
        labels = np.ones(100)
        result = compute_pece(probs, labels, fp_weight=2.0, n_bins=10)
        assert np.isfinite(result)

    def test_empty_input(self) -> None:
        """Empty arrays should return 0."""
        from minivess.pipeline.calibration_metrics import compute_pece

        result = compute_pece(np.array([]), np.array([]), fp_weight=2.0, n_bins=10)
        assert result == 0.0


class TestComputePeceInAllMetrics:
    """pECE must appear in compute_all_calibration_metrics(tier='comprehensive')."""

    def test_pece_in_comprehensive_tier(self) -> None:
        from minivess.pipeline.calibration_metrics import (
            compute_all_calibration_metrics,
        )

        rng = np.random.default_rng(42)
        probs = rng.random(1000)
        labels = (rng.random(1000) > 0.5).astype(float)

        result = compute_all_calibration_metrics(probs, labels, tier="comprehensive")
        assert "pece" in result, f"pECE missing from comprehensive tier. Keys: {list(result.keys())}"

    def test_pece_not_in_fast_tier(self) -> None:
        from minivess.pipeline.calibration_metrics import (
            compute_all_calibration_metrics,
        )

        rng = np.random.default_rng(42)
        probs = rng.random(1000)
        labels = (rng.random(1000) > 0.5).astype(float)

        result = compute_all_calibration_metrics(probs, labels, tier="fast")
        assert "pece" not in result, "pECE should NOT be in fast tier"


class TestComputePeceConfig:
    """fp_weight must come from config, not hardcoded (Rule #29)."""

    def test_config_has_pece_fp_weight(self) -> None:
        cfg = BiostatisticsConfig()
        assert hasattr(cfg, "pece_fp_weight")
        assert cfg.pece_fp_weight > 0

    def test_config_has_cal_pece_in_metrics(self) -> None:
        """default.yaml should include cal_pece in the full metric list."""
        # The default config may not include it until we update,
        # but the BiostatisticsConfig should accept it
        cfg = BiostatisticsConfig(metrics=["dsc", "cldice", "cal_pece"])
        assert "cal_pece" in cfg.metrics
