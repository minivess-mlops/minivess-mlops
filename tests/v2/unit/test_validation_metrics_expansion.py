from __future__ import annotations

import pytest

from minivess.pipeline.validation_metrics import (
    compute_compound_masd_cldice,
    normalize_masd,
)


class TestNormalizeMasd:
    """Tests for MASD normalization (distance → [0,1] score)."""

    def test_perfect_segmentation_gives_one(self):
        """MASD=0 (perfect) should normalize to 1.0."""
        assert normalize_masd(0.0) == pytest.approx(1.0)

    def test_max_masd_gives_zero(self):
        """MASD >= max_masd should normalize to 0.0."""
        assert normalize_masd(50.0) == pytest.approx(0.0)
        assert normalize_masd(100.0) == pytest.approx(0.0)

    def test_midrange(self):
        """MASD=25 with max=50 should give 0.5."""
        assert normalize_masd(25.0, max_masd=50.0) == pytest.approx(0.5)

    def test_custom_max_masd(self):
        """Custom max_masd changes the normalization range."""
        assert normalize_masd(10.0, max_masd=100.0) == pytest.approx(0.9)

    def test_negative_masd_clamped(self):
        """Negative MASD (shouldn't happen) gets clamped to 1.0."""
        assert normalize_masd(-1.0) == pytest.approx(1.0)

    def test_nan_returns_zero(self):
        """NaN MASD should return 0.0 (worst score)."""
        assert normalize_masd(float("nan")) == pytest.approx(0.0)


class TestCompoundMasdCldice:
    """Tests for compound metric: 0.5*(1-norm_masd) + 0.5*cldice."""

    def test_perfect_returns_one(self):
        """MASD=0, clDice=1.0 → compound = 1.0."""
        assert compute_compound_masd_cldice(masd=0.0, cldice=1.0) == pytest.approx(1.0)

    def test_worst_returns_zero(self):
        """MASD=max, clDice=0.0 → compound = 0.0."""
        assert compute_compound_masd_cldice(masd=50.0, cldice=0.0) == pytest.approx(0.0)

    def test_formula_correctness(self):
        """Verify the formula: 0.5 * normalize_masd(masd) + 0.5 * cldice."""
        masd, cldice = 10.0, 0.8
        expected = 0.5 * normalize_masd(masd) + 0.5 * cldice
        assert compute_compound_masd_cldice(masd=masd, cldice=cldice) == pytest.approx(
            expected
        )

    def test_custom_weights(self):
        """Custom weights change the compound value."""
        val_equal = compute_compound_masd_cldice(
            masd=25.0, cldice=0.6, w_masd=0.5, w_cldice=0.5
        )
        val_skewed = compute_compound_masd_cldice(
            masd=25.0, cldice=0.6, w_masd=0.8, w_cldice=0.2
        )
        assert val_equal != pytest.approx(val_skewed, abs=1e-3)

    def test_nan_cldice_returns_half_masd(self):
        """When clDice is NaN, compound should degrade gracefully."""
        result = compute_compound_masd_cldice(masd=0.0, cldice=float("nan"))
        # 0.5*1.0 + 0.5*nan → should return 0.0 (fail-safe)
        assert result == pytest.approx(0.0)

    def test_result_in_zero_one_range(self):
        """Compound must always be in [0, 1]."""
        for masd in [0.0, 5.0, 25.0, 50.0, 100.0]:
            for cldice in [0.0, 0.5, 1.0]:
                val = compute_compound_masd_cldice(masd=masd, cldice=cldice)
                assert 0.0 <= val <= 1.0, (
                    f"Out of range: masd={masd}, cldice={cldice} → {val}"
                )
