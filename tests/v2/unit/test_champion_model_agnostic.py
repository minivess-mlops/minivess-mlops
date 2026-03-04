"""Tests for champion selection model-agnostic integration (SAM-15, #221).

Verifies that champion_tagger.py works correctly when model entries
come from different model families (DynUNet, SAM3 variants).
Includes filesystem tagging + discovery integration tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minivess.pipeline.champion_tagger import (
    clear_champion_tags_filesystem,
    rank_then_aggregate,
    select_champions,
    write_champion_tags_filesystem,
)
from minivess.pipeline.deploy_champion_discovery import discover_champions

if TYPE_CHECKING:
    from pathlib import Path


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


def _setup_mlruns(
    tmp_path: Path, experiment_id: str, runs: list[dict[str, str]]
) -> Path:
    """Create mock mlruns filesystem structure for multiple runs."""
    mlruns = tmp_path / "mlruns"
    for run in runs:
        tags_dir = mlruns / experiment_id / run["run_id"] / "tags"
        tags_dir.mkdir(parents=True)
        # Write model family tag (as MLflow would)
        (tags_dir / "model_family").write_text(run["model_family"], encoding="utf-8")
    return mlruns


class TestChampionFilesystemCrossModel:
    """Filesystem tagging + discovery works with SAM3 and DynUNet runs."""

    def test_write_tags_for_sam3_runs(self, tmp_path: Path) -> None:
        """Champion tags written for SAM3 runs identical to DynUNet runs."""
        runs = [
            {
                "run_id": "run_dynunet_0",
                "model_family": "dynunet",
                "loss_type": "dice_ce",
            },
            {
                "run_id": "run_sam3_vanilla_0",
                "model_family": "sam3_vanilla",
                "loss_type": "dice_ce",
            },
            {
                "run_id": "run_sam3_topolora_0",
                "model_family": "sam3_topolora",
                "loss_type": "cbdice_cldice",
            },
        ]
        mlruns = _setup_mlruns(tmp_path, "1", runs)

        # SAM3 vanilla wins on dice_ce
        entries = [
            {
                "entry_type": "per_fold",
                "model_name": "sam3_vanilla",
                "loss_function": "dice_ce",
                "fold_id": 0,
                "primary_metric_value": 0.55,
            },
            {
                "entry_type": "per_fold",
                "model_name": "dynunet",
                "loss_function": "dice_ce",
                "fold_id": 0,
                "primary_metric_value": 0.50,
            },
        ]
        selection = select_champions(entries, primary_metric="dsc", maximize=True)
        assert selection.best_single_fold is not None
        assert selection.best_single_fold.model_name == "sam3_vanilla"

        written = write_champion_tags_filesystem(
            mlruns,
            "1",
            selection,
            runs=runs,
        )
        # Both dice_ce runs get tagged (same loss function)
        assert written > 0

        # Verify champion tag files exist
        for run in runs:
            if run["loss_type"] == "dice_ce":
                tag_file = (
                    mlruns / "1" / run["run_id"] / "tags" / "champion_best_single_fold"
                )
                assert tag_file.exists()

    def test_discover_champions_mixed_families(self, tmp_path: Path) -> None:
        """discover_champions finds SAM3 champions alongside DynUNet."""
        runs = [
            {
                "run_id": "dynunet_run",
                "model_family": "dynunet",
                "loss_type": "dice_ce",
            },
            {
                "run_id": "sam3_run",
                "model_family": "sam3_hybrid",
                "loss_type": "cbdice_cldice",
            },
        ]
        mlruns = _setup_mlruns(tmp_path, "1", runs)

        # Manually write champion tags on the SAM3 run
        tags_dir = mlruns / "1" / "sam3_run" / "tags"
        (tags_dir / "champion_best_single_fold").write_text("true", encoding="utf-8")

        champions = discover_champions(mlruns, "1")
        assert len(champions) == 1
        assert champions[0].run_id == "sam3_run"
        assert champions[0].category == "overlap"  # best_single_fold → overlap

    def test_clear_and_retag_across_families(self, tmp_path: Path) -> None:
        """Clear old tags, retag with new SAM3 winner — no model family leakage."""
        runs = [
            {
                "run_id": "old_dynunet",
                "model_family": "dynunet",
                "loss_type": "dice_ce",
            },
            {
                "run_id": "new_sam3",
                "model_family": "sam3_topolora",
                "loss_type": "cbdice_cldice",
            },
        ]
        mlruns = _setup_mlruns(tmp_path, "1", runs)

        # Old DynUNet champion
        old_tag = mlruns / "1" / "old_dynunet" / "tags" / "champion_best_cv_mean"
        old_tag.write_text("true", encoding="utf-8")

        # Clear old tags
        cleared = clear_champion_tags_filesystem(mlruns, "1")
        assert cleared == 1
        assert not old_tag.exists()

        # New SAM3 winner
        entries = [
            {
                "entry_type": "cv_mean",
                "model_name": "sam3_topolora",
                "loss_function": "cbdice_cldice",
                "fold_id": -1,
                "primary_metric_value": 0.75,
            },
        ]
        selection = select_champions(entries, primary_metric="dsc", maximize=True)
        write_champion_tags_filesystem(mlruns, "1", selection, runs=runs)

        # Discover should find SAM3 champion, not old DynUNet
        champions = discover_champions(mlruns, "1")
        assert len(champions) == 1
        assert champions[0].run_id == "new_sam3"
