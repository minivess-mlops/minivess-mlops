"""Tests for rank stability analysis (Phase 4).

Validates Kendall's tau rank concordance between metrics, demonstrating
how metric choice affects model ranking. The clDice vs DSC rank inversion
IS a paper finding.

Reference: MetricsReloaded (Maier-Hein et al., 2024)
"""

from __future__ import annotations

import numpy as np


def _make_rank_stability_data(
    *,
    n_conditions: int = 4,
    n_folds: int = 3,
    n_volumes: int = 23,
    seed: int = 42,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Create synthetic data where DSC and clDice rankings disagree.

    Deliberately creates a rank inversion: model_0 is best on DSC but
    worst on clDice (simulates volume-filling vs topology-preserving).
    """
    rng = np.random.default_rng(seed)
    data: dict[str, dict[str, dict[int, np.ndarray]]] = {}

    conditions = [f"model_{i}__loss_a__calibTrue" for i in range(n_conditions)]

    # DSC: model_0 > model_1 > model_2 > model_3
    data["dsc"] = {}
    for i, cond in enumerate(conditions):
        data["dsc"][cond] = {}
        for fold_id in range(n_folds):
            data["dsc"][cond][fold_id] = rng.normal(
                0.9 - 0.05 * i, 0.02, size=n_volumes
            )

    # clDice: model_3 > model_2 > model_1 > model_0 (INVERTED ranking)
    data["cldice"] = {}
    for i, cond in enumerate(conditions):
        data["cldice"][cond] = {}
        for fold_id in range(n_folds):
            data["cldice"][cond][fold_id] = rng.normal(
                0.6 + 0.1 * i, 0.02, size=n_volumes
            )

    # MASD: model_3 > model_2 > model_1 > model_0 (agrees with clDice)
    # Lower is better for MASD
    data["masd"] = {}
    for i, cond in enumerate(conditions):
        data["masd"][cond] = {}
        for fold_id in range(n_folds):
            data["masd"][cond][fold_id] = rng.normal(3.0 - 0.5 * i, 0.1, size=n_volumes)

    return data


class TestRankStabilityReturnsResult:
    """Rank stability function returns the expected dataclass."""

    def test_returns_rank_concordance_result(self) -> None:
        from minivess.pipeline.biostatistics_rank_stability import (
            RankConcordanceResult,
            compute_rank_concordance,
        )

        data = _make_rank_stability_data()
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["dsc", "cldice", "masd"],
            higher_is_better={"dsc": True, "cldice": True, "masd": False},
        )
        assert isinstance(result, RankConcordanceResult)


class TestKendallTauMatrix:
    """Kendall's tau concordance matrix between metrics."""

    def test_tau_matrix_has_all_metric_pairs(self) -> None:
        from minivess.pipeline.biostatistics_rank_stability import (
            compute_rank_concordance,
        )

        data = _make_rank_stability_data()
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["dsc", "cldice"],
            higher_is_better={"dsc": True, "cldice": True},
        )
        # C(2,2) = 1 pair
        assert len(result.tau_matrix) == 1
        pair = result.tau_matrix[0]
        assert pair.metric_a == "cldice"  # sorted
        assert pair.metric_b == "dsc"

    def test_dsc_cldice_inversion_detected(self) -> None:
        """DSC vs clDice should show negative tau (rank inversion)."""
        from minivess.pipeline.biostatistics_rank_stability import (
            compute_rank_concordance,
        )

        data = _make_rank_stability_data()
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["dsc", "cldice"],
            higher_is_better={"dsc": True, "cldice": True},
        )
        pair = result.tau_matrix[0]
        assert pair.tau < 0, f"Expected negative tau for rank inversion, got {pair.tau}"

    def test_cldice_masd_agreement(self) -> None:
        """clDice and MASD should show positive tau (rank agreement)."""
        from minivess.pipeline.biostatistics_rank_stability import (
            compute_rank_concordance,
        )

        data = _make_rank_stability_data()
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["cldice", "masd"],
            higher_is_better={"cldice": True, "masd": False},
        )
        pair = result.tau_matrix[0]
        assert pair.tau > 0, f"Expected positive tau for agreement, got {pair.tau}"

    def test_tau_p_value_in_range(self) -> None:
        from minivess.pipeline.biostatistics_rank_stability import (
            compute_rank_concordance,
        )

        data = _make_rank_stability_data()
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["dsc", "cldice"],
            higher_is_better={"dsc": True, "cldice": True},
        )
        for pair in result.tau_matrix:
            assert 0.0 <= pair.p_value <= 1.0


class TestRankStabilityConditionRanks:
    """Model rankings per metric."""

    def test_condition_ranks_present(self) -> None:
        from minivess.pipeline.biostatistics_rank_stability import (
            compute_rank_concordance,
        )

        data = _make_rank_stability_data()
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["dsc", "cldice"],
            higher_is_better={"dsc": True, "cldice": True},
        )
        assert "dsc" in result.condition_ranks
        assert "cldice" in result.condition_ranks

    def test_all_conditions_ranked(self) -> None:
        from minivess.pipeline.biostatistics_rank_stability import (
            compute_rank_concordance,
        )

        data = _make_rank_stability_data(n_conditions=4)
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["dsc"],
            higher_is_better={"dsc": True},
        )
        assert len(result.condition_ranks["dsc"]) == 4


class TestRankInversionSummary:
    """Summary of rank inversions for the paper."""

    def test_rank_inversions_detected(self) -> None:
        from minivess.pipeline.biostatistics_rank_stability import (
            compute_rank_concordance,
        )

        data = _make_rank_stability_data()
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["dsc", "cldice", "masd"],
            higher_is_better={"dsc": True, "cldice": True, "masd": False},
        )
        # Should detect DSC vs clDice inversion
        assert result.n_inversions >= 1
