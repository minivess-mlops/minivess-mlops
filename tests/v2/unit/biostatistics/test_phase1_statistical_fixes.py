"""Tests for Phase 1 statistical fixes (Tasks 1.2-1.7).

1.2: BCa → percentile for small N
1.3: Mixed-effects ANOVA (fold as random effect)
1.4: Hierarchical gatekeeping for co-primary endpoints
1.5: Generic N-factor condition key parser
1.6: Seed propagation
1.7: Sensitivity concordance (parametric vs nonparametric)
"""

from __future__ import annotations

import numpy as np
import pytest

from minivess.config.biostatistics_config import BiostatisticsConfig

_CFG = BiostatisticsConfig()


# ── Task 1.2: BCa threshold ───────────────────────────────────────────


class TestBcaThreshold:
    """BCa only used when N >= bca_min_n; percentile otherwise."""

    def test_config_has_bca_min_n(self) -> None:
        assert hasattr(_CFG, "bca_min_n")
        assert _CFG.bca_min_n >= 10  # Reasonable minimum

    def test_percentile_for_small_n(self) -> None:
        from minivess.pipeline.biostatistics_statistics import bootstrap_ci

        rng = np.random.default_rng(42)
        data = rng.normal(0.8, 0.03, size=5)  # N=5, well below bca_min_n
        lo, hi, method = bootstrap_ci(data, n_bootstrap=200, seed=42)
        assert method == "percentile"
        assert lo < hi

    def test_bca_for_large_n(self) -> None:
        from minivess.pipeline.biostatistics_statistics import bootstrap_ci

        rng = np.random.default_rng(42)
        data = rng.normal(0.8, 0.03, size=50)  # N=50, above bca_min_n
        lo, hi, method = bootstrap_ci(data, n_bootstrap=200, seed=42)
        assert method == "bca"
        assert lo < hi


# ── Task 1.4: Hierarchical gatekeeping ────────────────────────────────


class TestHierarchicalGatekeeping:
    """Co-primaries at alpha/2, secondaries conditional BH-FDR."""

    def test_function_exists(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            apply_hierarchical_gatekeeping,
        )

        assert callable(apply_hierarchical_gatekeeping)

    def test_co_primary_bonferroni(self) -> None:
        """Co-primary p-values corrected at alpha/2."""
        from minivess.pipeline.biostatistics_statistics import (
            apply_hierarchical_gatekeeping,
        )

        result = apply_hierarchical_gatekeeping(
            co_primary_p={
                "cldice": 0.01,  # significant at alpha/2=0.025
                "masd": 0.03,  # not significant at alpha/2=0.025
            },
            secondary_p={"dsc": 0.04, "hd95": 0.06},
            alpha=_CFG.alpha,
        )
        assert result["cldice"]["significant"] is True
        assert result["masd"]["significant"] is False

    def test_secondary_conditional_on_co_primary(self) -> None:
        """Secondary tests proceed only if at least one co-primary rejects."""
        from minivess.pipeline.biostatistics_statistics import (
            apply_hierarchical_gatekeeping,
        )

        # One co-primary significant → gate opens
        result = apply_hierarchical_gatekeeping(
            co_primary_p={"cldice": 0.01, "masd": 0.10},
            secondary_p={"dsc": 0.03},
            alpha=_CFG.alpha,
        )
        assert result["dsc"]["tested"] is True

    def test_secondary_blocked_when_no_coprimary_rejects(self) -> None:
        """If no co-primary rejects, secondaries are NOT tested."""
        from minivess.pipeline.biostatistics_statistics import (
            apply_hierarchical_gatekeeping,
        )

        result = apply_hierarchical_gatekeeping(
            co_primary_p={"cldice": 0.10, "masd": 0.20},
            secondary_p={"dsc": 0.001},
            alpha=_CFG.alpha,
        )
        assert result["dsc"]["tested"] is False
        assert result["dsc"]["significant"] is False


# ── Task 1.5: Generic N-factor condition key parser ───────────────────


class TestConditionKeyParser:
    """Round-trip encode/decode for N-factor condition keys."""

    def test_encode_decode_roundtrip(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            decode_condition_key,
            encode_condition_key,
        )

        factors = {
            "model_family": "dynunet",
            "loss_name": "dice_ce",
            "with_aux_calib": "false",
            "post_training_method": "none",
            "recalibration": "none",
            "ensemble_strategy": "per_loss_single_best",
        }
        encoded = encode_condition_key(factors)
        decoded = decode_condition_key(encoded)
        assert decoded == factors

    def test_encode_rejects_double_underscore_in_values(self) -> None:
        from minivess.pipeline.biostatistics_statistics import encode_condition_key

        with pytest.raises(ValueError, match="must not contain"):
            encode_condition_key({"key": "bad__value"})

    def test_2_factor_roundtrip(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            decode_condition_key,
            encode_condition_key,
        )

        factors = {"model_family": "dynunet", "loss_name": "dice_ce"}
        encoded = encode_condition_key(factors)
        decoded = decode_condition_key(encoded)
        assert decoded == factors

    def test_key_separator_is_double_underscore(self) -> None:
        from minivess.pipeline.biostatistics_statistics import encode_condition_key

        encoded = encode_condition_key({"a": "1", "b": "2"})
        assert "__" in encoded


# ── Task 1.6: Seed propagation ────────────────────────────────────────


class TestSeedPropagation:
    """All random operations use config.seed for reproducibility."""

    def test_stratified_permutation_deterministic(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        rng = np.random.default_rng(100)
        a = {0: rng.normal(0.8, 0.03, 23), 1: rng.normal(0.8, 0.03, 23)}
        b = {0: rng.normal(0.75, 0.03, 23), 1: rng.normal(0.75, 0.03, 23)}

        r1 = stratified_permutation_test(a, b, n_permutations=99, seed=_CFG.seed)
        r2 = stratified_permutation_test(a, b, n_permutations=99, seed=_CFG.seed)
        assert r1.p_value == r2.p_value

    def test_bootstrap_ci_deterministic(self) -> None:
        from minivess.pipeline.biostatistics_statistics import bootstrap_ci

        data = np.random.default_rng(200).normal(0.8, 0.03, 30)
        lo1, hi1, _ = bootstrap_ci(data, n_bootstrap=100, seed=_CFG.seed)
        lo2, hi2, _ = bootstrap_ci(data, n_bootstrap=100, seed=_CFG.seed)
        assert lo1 == lo2
        assert hi1 == hi2


# ── Task 1.7: Sensitivity concordance ─────────────────────────────────


class TestSensitivityConcordance:
    """Parametric vs nonparametric agreement check."""

    def test_function_exists(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            sensitivity_concordance,
        )

        assert callable(sensitivity_concordance)

    def test_concordant_when_both_significant(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            sensitivity_concordance,
        )

        rng = np.random.default_rng(300)
        a = rng.normal(0.85, 0.02, 30)
        b = rng.normal(0.75, 0.02, 30)

        result = sensitivity_concordance(a, b, alpha=_CFG.alpha)
        assert result["concordant"] is True
        assert result["parametric_significant"] is True
        assert result["nonparametric_significant"] is True

    def test_concordant_when_both_nonsignificant(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            sensitivity_concordance,
        )

        rng = np.random.default_rng(400)
        a = rng.normal(0.80, 0.02, 30)
        b = rng.normal(0.80, 0.02, 30)

        result = sensitivity_concordance(a, b, alpha=_CFG.alpha)
        assert result["concordant"] is True
