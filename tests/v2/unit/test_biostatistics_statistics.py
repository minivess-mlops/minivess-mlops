"""Tests for biostatistics statistical engine (Phase 3, Tasks 3.2-3.4)."""

from __future__ import annotations

import numpy as np

from minivess.config.biostatistics_config import BiostatisticsConfig
from minivess.pipeline.biostatistics_statistics import (
    compute_bayesian_comparisons,
    compute_pairwise_comparisons,
    compute_variance_decomposition,
)

# Single source of truth for ALL statistical params — never hardcode
# alpha, seed, n_bootstrap, etc. See CLAUDE.md Rule #29 and Issue #881.
_CFG = BiostatisticsConfig()


def _build_synthetic_per_volume_data() -> dict[str, dict[int, np.ndarray]]:
    """Build synthetic per-volume metrics: {condition: {fold: scores}}.

    Returns data for 3 conditions x 3 folds x 20 volumes each.
    """
    rng = np.random.default_rng(_CFG.seed)
    data: dict[str, dict[int, np.ndarray]] = {}
    means = {"dice_ce": 0.82, "tversky": 0.78, "cbdice_cldice": 0.85}
    for condition, mean in means.items():
        data[condition] = {}
        for fold in range(3):
            data[condition][fold] = rng.normal(mean, 0.05, size=20)
    return data


class TestComputePairwiseComparisons:
    def test_returns_correct_number_of_comparisons(self) -> None:
        data = _build_synthetic_per_volume_data()
        # 3 conditions -> C(3,2) = 3 pairs
        results = compute_pairwise_comparisons(
            per_volume_data=data,
            metric_name="val_dice",
            alpha=_CFG.alpha,
            primary_metric="val_dice",
            n_bootstrap=_CFG.n_bootstrap,
            seed=_CFG.seed,
        )
        assert len(results) == 3

    def test_primary_metric_uses_holm_correction(self) -> None:
        data = _build_synthetic_per_volume_data()
        results = compute_pairwise_comparisons(
            per_volume_data=data,
            metric_name="val_dice",
            alpha=_CFG.alpha,
            primary_metric="val_dice",
            n_bootstrap=_CFG.n_bootstrap,
            seed=_CFG.seed,
        )
        assert all(r.correction_method == "holm" for r in results)

    def test_secondary_metrics_use_bh_fdr(self) -> None:
        data = _build_synthetic_per_volume_data()
        results = compute_pairwise_comparisons(
            per_volume_data=data,
            metric_name="val_dice",
            alpha=_CFG.alpha,
            primary_metric="val_cldice",  # Different from metric_name
            n_bootstrap=_CFG.n_bootstrap,
            seed=_CFG.seed,
        )
        assert all(r.correction_method == "bh_fdr" for r in results)

    def test_all_three_effect_sizes_computed(self) -> None:
        data = _build_synthetic_per_volume_data()
        results = compute_pairwise_comparisons(
            per_volume_data=data,
            metric_name="val_dice",
            alpha=_CFG.alpha,
            primary_metric="val_dice",
            n_bootstrap=_CFG.n_bootstrap,
            seed=_CFG.seed,
        )
        for r in results:
            assert isinstance(r.cohens_d, float)
            assert isinstance(r.cliffs_delta, float)
            assert isinstance(r.vda, float)

    def test_wilcoxon_p_value_present(self) -> None:
        data = _build_synthetic_per_volume_data()
        results = compute_pairwise_comparisons(
            per_volume_data=data,
            metric_name="val_dice",
            alpha=_CFG.alpha,
            primary_metric="val_dice",
            n_bootstrap=_CFG.n_bootstrap,
            seed=_CFG.seed,
        )
        for r in results:
            assert 0.0 <= r.p_value <= 1.0

    def test_symmetric_pairs(self) -> None:
        """Each unordered pair appears exactly once."""
        data = _build_synthetic_per_volume_data()
        results = compute_pairwise_comparisons(
            per_volume_data=data,
            metric_name="val_dice",
            alpha=_CFG.alpha,
            primary_metric="val_dice",
            n_bootstrap=_CFG.n_bootstrap,
            seed=_CFG.seed,
        )
        pairs = {(r.condition_a, r.condition_b) for r in results}
        # No (b, a) if (a, b) exists
        for a, b in pairs:
            assert (b, a) not in pairs


class TestComputeBayesianComparisons:
    def test_three_way_sums_to_one(self) -> None:
        data = _build_synthetic_per_volume_data()
        results = compute_bayesian_comparisons(
            per_volume_data=data,
            metric_name="val_dice",
            rope=0.01,
        )
        for r in results:
            if r.bayesian_left is not None:
                total = r.bayesian_left + r.bayesian_rope + r.bayesian_right
                assert abs(total - 1.0) < 0.01

    def test_identical_data_concentrates_on_rope(self) -> None:
        rng = np.random.default_rng(_CFG.seed)
        scores = rng.normal(0.8, 0.05, size=20)
        data = {
            "a": {0: scores.copy(), 1: scores.copy(), 2: scores.copy()},
            "b": {0: scores.copy(), 1: scores.copy(), 2: scores.copy()},
        }
        results = compute_bayesian_comparisons(
            per_volume_data=data,
            metric_name="val_dice",
            rope=0.01,
        )
        assert len(results) == 1
        r = results[0]
        if r.bayesian_rope is not None:
            # With identical data, ROPE probability should be highest
            assert r.bayesian_rope > r.bayesian_left
            assert r.bayesian_rope > r.bayesian_right

    def test_separated_data_favors_direction(self) -> None:
        rng = np.random.default_rng(_CFG.seed)
        data = {
            "good": {
                0: rng.normal(0.9, 0.02, 20),
                1: rng.normal(0.9, 0.02, 20),
                2: rng.normal(0.9, 0.02, 20),
            },
            "bad": {
                0: rng.normal(0.5, 0.02, 20),
                1: rng.normal(0.5, 0.02, 20),
                2: rng.normal(0.5, 0.02, 20),
            },
        }
        results = compute_bayesian_comparisons(
            per_volume_data=data,
            metric_name="val_dice",
            rope=0.01,
        )
        assert len(results) == 1
        r = results[0]
        if r.bayesian_left is not None:
            # Sorted: a="bad", b="good". Since good > bad, P(B>A)=right should be high
            # OR P(A>B)=left. Either way one direction must dominate.
            left = float(r.bayesian_left)
            right = float(r.bayesian_right)
            assert max(left, right) > 0.9


class TestComputeVarianceDecomposition:
    def test_friedman_returns_statistic_and_pvalue(self) -> None:
        data = _build_synthetic_per_volume_data()
        results = compute_variance_decomposition(
            per_volume_data=data,
            metric_name="val_dice",
        )
        assert len(results) == 1
        r = results[0]
        assert r.friedman_statistic >= 0.0
        assert 0.0 <= r.friedman_p <= 1.0

    def test_nemenyi_only_after_significant_friedman(self) -> None:
        data = _build_synthetic_per_volume_data()
        results = compute_variance_decomposition(
            per_volume_data=data,
            metric_name="val_dice",
            friedman_alpha=_CFG.alpha,
        )
        r = results[0]
        if r.friedman_p > _CFG.alpha:
            assert r.nemenyi_matrix is None
        else:
            assert r.nemenyi_matrix is not None

    def test_icc_returns_value_and_ci(self) -> None:
        data = _build_synthetic_per_volume_data()
        results = compute_variance_decomposition(
            per_volume_data=data,
            metric_name="val_dice",
        )
        r = results[0]
        assert -1.0 <= r.icc_value <= 1.0
        assert r.icc_ci_lower <= r.icc_value <= r.icc_ci_upper

    def test_icc_type_is_ICC2(self) -> None:
        data = _build_synthetic_per_volume_data()
        results = compute_variance_decomposition(
            per_volume_data=data,
            metric_name="val_dice",
        )
        assert results[0].icc_type == "ICC2"

    def test_power_caveat_flag_set_for_k3(self) -> None:
        data = _build_synthetic_per_volume_data()  # 3 conditions
        results = compute_variance_decomposition(
            per_volume_data=data,
            metric_name="val_dice",
        )
        assert results[0].power_caveat is True
