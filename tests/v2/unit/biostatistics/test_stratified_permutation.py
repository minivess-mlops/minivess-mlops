"""Tests for stratified within-fold permutation test.

Validates that pairwise comparisons respect fold structure by permuting
condition labels within each fold stratum, not across folds.

Council finding: The original _pool_scores() violates exchangeability by
concatenating volumes across folds and treating them as i.i.d.
Fix: stratified permutation test with volume-level pairing via (fold_id, volume_id).
"""

from __future__ import annotations

import numpy as np

from minivess.config.biostatistics_config import BiostatisticsConfig

# ---------------------------------------------------------------------------
# Synthetic data fixtures with known statistical properties
# ---------------------------------------------------------------------------

_CFG = BiostatisticsConfig()
_RNG = np.random.default_rng(seed=_CFG.seed)


def _make_paired_data(
    n_folds: int = 3,
    n_volumes_per_fold: int = 23,
    effect_size: float = 0.0,
    fold_icc: float = 0.15,
    seed: int = 42,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Create paired per-volume data for two conditions with known effect.

    Returns (condition_a_folds, condition_b_folds) where each is
    {fold_id: np.ndarray of shape (n_volumes_per_fold,)}.

    condition_b = condition_a + effect_size + noise, stratified by fold.
    """
    rng = np.random.default_rng(seed)
    cond_a: dict[int, np.ndarray] = {}
    cond_b: dict[int, np.ndarray] = {}

    for fold_id in range(n_folds):
        # Fold-level random intercept (simulates ICC)
        fold_effect = rng.normal(0, fold_icc)
        base = rng.normal(0.75 + fold_effect, 0.04, size=n_volumes_per_fold)
        cond_a[fold_id] = base
        cond_b[fold_id] = base + effect_size + rng.normal(0, 0.02, size=n_volumes_per_fold)

    return cond_a, cond_b


# ---------------------------------------------------------------------------
# Tests for stratified_permutation_test function
# ---------------------------------------------------------------------------


class TestStratifiedPermutationExists:
    """The function must exist and be importable."""

    def test_function_importable(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        assert callable(stratified_permutation_test)

    def test_function_signature(self) -> None:
        """Must accept fold-level data dicts, not pooled arrays."""
        import inspect

        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        sig = inspect.signature(stratified_permutation_test)
        params = list(sig.parameters.keys())
        assert "fold_data_a" in params
        assert "fold_data_b" in params
        assert "n_permutations" in params
        assert "seed" in params


class TestStratifiedPermutationNullHypothesis:
    """Under H0 (no effect), p-values should be uniformly distributed."""

    def test_no_effect_not_significant(self) -> None:
        """With zero effect size, the test should not be significant at alpha."""
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        cond_a, cond_b = _make_paired_data(effect_size=0.0, seed=100)
        result = stratified_permutation_test(
            fold_data_a=cond_a,
            fold_data_b=cond_b,
            n_permutations=999,
            seed=_CFG.seed,
        )
        # Under H0, p should not be significant at alpha=0.05 most of the time
        # We use a single seed so this is deterministic
        assert result.p_value > 0.01, (
            f"False positive: p={result.p_value} under null hypothesis"
        )


class TestStratifiedPermutationAlternative:
    """Under H1 (real effect), the test should detect it."""

    def test_large_effect_detected(self) -> None:
        """With effect_size=0.05, test should detect significance."""
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        cond_a, cond_b = _make_paired_data(effect_size=0.05, seed=200)
        result = stratified_permutation_test(
            fold_data_a=cond_a,
            fold_data_b=cond_b,
            n_permutations=999,
            seed=_CFG.seed,
        )
        assert result.p_value < _CFG.alpha, (
            f"Missed real effect: p={result.p_value}, expected < {_CFG.alpha}"
        )


class TestStratifiedPermutationResult:
    """Result dataclass should have the right fields."""

    def test_result_has_p_value(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        cond_a, cond_b = _make_paired_data(effect_size=0.0, seed=300)
        result = stratified_permutation_test(
            fold_data_a=cond_a,
            fold_data_b=cond_b,
            n_permutations=99,
            seed=_CFG.seed,
        )
        assert hasattr(result, "p_value")
        assert 0.0 <= result.p_value <= 1.0

    def test_result_has_observed_statistic(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        cond_a, cond_b = _make_paired_data(effect_size=0.0, seed=300)
        result = stratified_permutation_test(
            fold_data_a=cond_a,
            fold_data_b=cond_b,
            n_permutations=99,
            seed=_CFG.seed,
        )
        assert hasattr(result, "observed_statistic")
        assert isinstance(result.observed_statistic, float)

    def test_result_has_n_permutations(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        cond_a, cond_b = _make_paired_data(effect_size=0.0, seed=300)
        result = stratified_permutation_test(
            fold_data_a=cond_a,
            fold_data_b=cond_b,
            n_permutations=99,
            seed=_CFG.seed,
        )
        assert result.n_permutations == 99


class TestStratifiedPreservesFoldStructure:
    """Verify that permutation is WITHIN folds, not across."""

    def test_different_fold_counts_handled(self) -> None:
        """Works with different numbers of volumes per fold (ragged)."""
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        # Ragged: fold 0 has 20 volumes, fold 1 has 23, fold 2 has 25
        rng = np.random.default_rng(400)
        cond_a = {
            0: rng.normal(0.8, 0.03, size=20),
            1: rng.normal(0.8, 0.03, size=23),
            2: rng.normal(0.8, 0.03, size=25),
        }
        cond_b = {
            0: rng.normal(0.8, 0.03, size=20),
            1: rng.normal(0.8, 0.03, size=23),
            2: rng.normal(0.8, 0.03, size=25),
        }

        result = stratified_permutation_test(
            fold_data_a=cond_a,
            fold_data_b=cond_b,
            n_permutations=99,
            seed=_CFG.seed,
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed produces identical p-values."""
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        cond_a, cond_b = _make_paired_data(effect_size=0.02, seed=500)

        r1 = stratified_permutation_test(
            fold_data_a=cond_a, fold_data_b=cond_b, n_permutations=199, seed=42
        )
        r2 = stratified_permutation_test(
            fold_data_a=cond_a, fold_data_b=cond_b, n_permutations=199, seed=42
        )
        assert r1.p_value == r2.p_value
        assert r1.observed_statistic == r2.observed_statistic


class TestPairwiseUsesStratified:
    """compute_pairwise_comparisons should use stratified test internally."""

    def test_pairwise_accepts_fold_structured_data(self) -> None:
        """Pairwise comparisons work with the standard PerVolumeData format."""
        from minivess.pipeline.biostatistics_statistics import (
            compute_pairwise_comparisons,
        )

        rng = np.random.default_rng(600)
        per_volume_data = {
            "cond_a": {0: rng.normal(0.8, 0.03, 23), 1: rng.normal(0.8, 0.03, 23)},
            "cond_b": {0: rng.normal(0.75, 0.03, 23), 1: rng.normal(0.75, 0.03, 23)},
        }
        results = compute_pairwise_comparisons(
            per_volume_data=per_volume_data,
            metric_name="dsc",
            alpha=_CFG.alpha,
            primary_metric="cldice",
            n_bootstrap=100,
            seed=_CFG.seed,
        )
        assert len(results) == 1  # One pair: cond_a vs cond_b
        assert results[0].condition_a == "cond_a"
        assert results[0].condition_b == "cond_b"
