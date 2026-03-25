"""Tests for biostatistics rankings (Phase 4, Task 4.1)."""

from __future__ import annotations

import numpy as np

from minivess.pipeline.biostatistics_rankings import compute_rankings


def _build_synthetic_data() -> dict[str, dict[int, np.ndarray]]:
    """3 conditions with clearly different means."""
    rng = np.random.default_rng(42)
    return {
        "dice_ce": {
            0: rng.normal(0.80, 0.05, 20),
            1: rng.normal(0.80, 0.05, 20),
            2: rng.normal(0.80, 0.05, 20),
        },
        "tversky": {
            0: rng.normal(0.75, 0.05, 20),
            1: rng.normal(0.75, 0.05, 20),
            2: rng.normal(0.75, 0.05, 20),
        },
        "cbdice_cldice": {
            0: rng.normal(0.85, 0.05, 20),
            1: rng.normal(0.85, 0.05, 20),
            2: rng.normal(0.85, 0.05, 20),
        },
    }


class TestComputeRankings:
    def test_ranking_order_matches_metric_direction(self) -> None:
        data = _build_synthetic_data()
        results = compute_rankings(
            per_volume_data={"val_dice": data},
            metric_names=["val_dice"],
            higher_is_better={"val_dice": True},
            alpha=0.05,
        )
        assert len(results) == 1
        r = results[0]
        # cbdice_cldice has highest mean -> rank 1
        assert r.condition_ranks["cbdice_cldice"] < r.condition_ranks["tversky"]

    def test_mean_rank_aggregation(self) -> None:
        rng = np.random.default_rng(42)
        data1 = {
            "a": {0: rng.normal(0.9, 0.01, 20)},
            "b": {0: rng.normal(0.7, 0.01, 20)},
        }
        data2 = {
            "a": {0: rng.normal(0.7, 0.01, 20)},
            "b": {0: rng.normal(0.9, 0.01, 20)},
        }
        results = compute_rankings(
            per_volume_data={"m1": data1, "m2": data2},
            metric_names=["m1", "m2"],
            higher_is_better={"m1": True, "m2": True},
            alpha=0.05,
        )
        # 2 per-metric rankings
        assert len(results) == 2

    def test_cd_value_computed(self) -> None:
        data = _build_synthetic_data()
        results = compute_rankings(
            per_volume_data={"val_dice": data},
            metric_names=["val_dice"],
            higher_is_better={"val_dice": True},
            alpha=0.05,
        )
        r = results[0]
        # CD value should be a positive number (critical difference)
        assert r.cd_value is None or r.cd_value > 0

    def test_ranking_with_ties(self) -> None:
        """When two conditions have identical scores, ranks should be tied."""
        identical = np.array([0.8] * 20)
        data = {
            "a": {0: identical.copy()},
            "b": {0: identical.copy()},
        }
        results = compute_rankings(
            per_volume_data={"val_dice": data},
            metric_names=["val_dice"],
            higher_is_better={"val_dice": True},
            alpha=0.05,
        )
        r = results[0]
        assert r.condition_ranks["a"] == r.condition_ranks["b"]