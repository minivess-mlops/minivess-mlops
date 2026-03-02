"""Tests for champion selection model-agnostic integration (SAM-15).

Verifies that champion_tagger.py works correctly when model entries
come from different model families (DynUNet, SAM3 variants).
"""

from __future__ import annotations

from minivess.pipeline.champion_tagger import (
    rank_then_aggregate,
    select_champions,
)


def _make_cross_model_entries() -> list[dict[str, object]]:
    """Create analysis entries spanning multiple model families."""
    return [
        {
            "entry_type": "per_fold",
            "model_name": "dynunet_cbdice_cldice",
            "loss_function": "cbdice_cldice",
            "fold_id": 0,
            "primary_metric_value": 0.824,
        },
        {
            "entry_type": "per_fold",
            "model_name": "sam3_vanilla",
            "loss_function": "dice_ce",
            "fold_id": 0,
            "primary_metric_value": 0.45,
        },
        {
            "entry_type": "per_fold",
            "model_name": "sam3_topolora",
            "loss_function": "cbdice_cldice",
            "fold_id": 0,
            "primary_metric_value": 0.52,
        },
        {
            "entry_type": "cv_mean",
            "model_name": "dynunet_cbdice_cldice",
            "loss_function": "cbdice_cldice",
            "fold_id": -1,
            "primary_metric_value": 0.82,
        },
        {
            "entry_type": "cv_mean",
            "model_name": "sam3_topolora",
            "loss_function": "cbdice_cldice",
            "fold_id": -1,
            "primary_metric_value": 0.51,
        },
    ]


class TestChampionSelectionCrossModel:
    """select_champions works with entries from different model families."""

    def test_selects_best_single_fold_across_families(self) -> None:
        entries = _make_cross_model_entries()
        selection = select_champions(
            entries,
            primary_metric="dsc",
            maximize=True,
        )
        assert selection.best_single_fold is not None
        assert selection.best_single_fold.model_name == "dynunet_cbdice_cldice"
        assert selection.best_single_fold.metric_value == 0.824

    def test_selects_best_cv_mean_across_families(self) -> None:
        entries = _make_cross_model_entries()
        selection = select_champions(
            entries,
            primary_metric="dsc",
            maximize=True,
        )
        assert selection.best_cv_mean is not None
        assert selection.best_cv_mean.loss_function == "cbdice_cldice"

    def test_rank_then_aggregate_cross_model(self) -> None:
        entries = [
            {"model_id": "dynunet", "dsc": 0.824, "cldice": 0.906, "hd95": 3.2},
            {"model_id": "sam3_vanilla", "dsc": 0.45, "cldice": 0.38, "hd95": 12.5},
            {"model_id": "sam3_topolora", "dsc": 0.52, "cldice": 0.55, "hd95": 9.1},
            {"model_id": "sam3_hybrid", "dsc": 0.61, "cldice": 0.63, "hd95": 7.3},
        ]
        result = rank_then_aggregate(
            entries,
            maximize_metrics=["dsc", "cldice"],
            minimize_metrics=["hd95"],
        )
        # DynUNet should win balanced (best on all metrics)
        assert result["balanced"] == "dynunet"
        assert result["topology"] == "dynunet"  # best cldice
        assert result["overlap"] == "dynunet"  # best dsc
