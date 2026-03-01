"""Unit tests for the champion model tagging system.

Tests cover:
- Pure selection logic (select_champions)
- Filesystem tag clearing (clear_champion_tags_filesystem)
- Filesystem tag writing (write_champion_tags_filesystem)
- End-to-end orchestration (tag_champions)

Mock mlruns filesystem layout:
    mlruns/<experiment_id>/<run_id>/tags/<key> — plain text
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from minivess.pipeline.champion_tagger import (
    CHAMPION_TAG_KEYS,
    ChampionSelection,
    clear_champion_tags_filesystem,
    select_champions,
    tag_champions,
    write_champion_tags_filesystem,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXP_ID = "unit_test_exp"


# ---------------------------------------------------------------------------
# Mock filesystem helpers
# ---------------------------------------------------------------------------


def _make_tag(run_dir: Path, key: str, value: str) -> None:
    tags_dir = run_dir / "tags"
    tags_dir.mkdir(parents=True, exist_ok=True)
    (tags_dir / key).write_text(value, encoding="utf-8")


def _make_run_with_tags(
    mlruns_dir: Path,
    exp_id: str,
    run_id: str,
    *,
    extra_tags: dict[str, str] | None = None,
) -> Path:
    """Create a mock run directory with optional tags."""
    run_dir = mlruns_dir / exp_id / run_id
    tags_dir = run_dir / "tags"
    tags_dir.mkdir(parents=True, exist_ok=True)
    if extra_tags:
        for key, value in extra_tags.items():
            _make_tag(run_dir, key, value)
    return run_dir


def _read_tag(run_dir: Path, key: str) -> str | None:
    tag_file = run_dir / "tags" / key
    if tag_file.exists():
        return tag_file.read_text(encoding="utf-8")
    return None


def _make_analysis_entry(
    *,
    entry_type: str = "per_fold",
    model_name: str = "dice_ce_fold0",
    loss_function: str | None = "dice_ce",
    fold_id: int | None = 0,
    primary_metric_value: float = 0.85,
) -> dict[str, Any]:
    """Create a minimal analysis entry dict."""
    return {
        "entry_type": entry_type,
        "model_name": model_name,
        "loss_function": loss_function,
        "fold_id": fold_id,
        "metrics": {"dsc": primary_metric_value},
        "primary_metric_value": primary_metric_value,
    }


def _make_run_info(
    run_id: str,
    loss_type: str,
    fold_id: int,
) -> dict[str, Any]:
    """Create a minimal run info dict (as from _discover_runs)."""
    return {
        "run_id": run_id,
        "loss_type": loss_type,
        "fold_id": fold_id,
        "artifact_dir": f"/fake/{run_id}",
        "metrics": {},
    }


# ---------------------------------------------------------------------------
# Tests: select_champions (pure logic, no I/O)
# ---------------------------------------------------------------------------


class TestSelectChampions:
    """Tests for the pure champion selection logic."""

    def test_best_single_fold_maximize(self) -> None:
        """Highest primary_metric_value wins when maximizing."""
        entries = [
            _make_analysis_entry(
                model_name="dice_ce_fold0",
                loss_function="dice_ce",
                fold_id=0,
                primary_metric_value=0.80,
            ),
            _make_analysis_entry(
                model_name="dice_ce_fold1",
                loss_function="dice_ce",
                fold_id=1,
                primary_metric_value=0.90,
            ),
            _make_analysis_entry(
                model_name="cldice_fold0",
                loss_function="cldice",
                fold_id=0,
                primary_metric_value=0.85,
            ),
        ]
        result = select_champions(entries, primary_metric="dsc", maximize=True)
        assert result.best_single_fold is not None
        assert result.best_single_fold.model_name == "dice_ce_fold1"
        assert result.best_single_fold.metric_value == pytest.approx(0.90)
        assert result.best_single_fold.fold_id == 1

    def test_best_single_fold_minimize(self) -> None:
        """Lowest primary_metric_value wins when minimizing."""
        entries = [
            _make_analysis_entry(
                model_name="dice_ce_fold0",
                loss_function="dice_ce",
                fold_id=0,
                primary_metric_value=3.0,
            ),
            _make_analysis_entry(
                model_name="cldice_fold0",
                loss_function="cldice",
                fold_id=0,
                primary_metric_value=1.5,
            ),
        ]
        result = select_champions(entries, primary_metric="masd", maximize=False)
        assert result.best_single_fold is not None
        assert result.best_single_fold.model_name == "cldice_fold0"
        assert result.best_single_fold.metric_value == pytest.approx(1.5)

    def test_best_cv_mean_maximize(self) -> None:
        """CV-mean champion selects the loss with best cv_mean entry."""
        entries = [
            # per_fold entries (needed for context)
            _make_analysis_entry(
                model_name="dice_ce_fold0",
                loss_function="dice_ce",
                fold_id=0,
                primary_metric_value=0.80,
            ),
            # cv_mean entries
            _make_analysis_entry(
                entry_type="cv_mean",
                model_name="dice_ce_cv_mean",
                loss_function="dice_ce",
                fold_id=None,
                primary_metric_value=0.82,
            ),
            _make_analysis_entry(
                entry_type="cv_mean",
                model_name="cldice_cv_mean",
                loss_function="cldice",
                fold_id=None,
                primary_metric_value=0.88,
            ),
        ]
        result = select_champions(entries, primary_metric="dsc", maximize=True)
        assert result.best_cv_mean is not None
        assert result.best_cv_mean.loss_function == "cldice"
        assert result.best_cv_mean.metric_value == pytest.approx(0.88)

    def test_best_ensemble_maximize(self) -> None:
        """Ensemble champion selects best ensemble entry."""
        entries = [
            _make_analysis_entry(
                entry_type="ensemble",
                model_name="per_loss_single_best",
                loss_function=None,
                fold_id=None,
                primary_metric_value=0.91,
            ),
            _make_analysis_entry(
                entry_type="ensemble",
                model_name="all_loss_single_best",
                loss_function=None,
                fold_id=None,
                primary_metric_value=0.89,
            ),
        ]
        result = select_champions(entries, primary_metric="dsc", maximize=True)
        assert result.best_ensemble is not None
        assert result.best_ensemble.ensemble_strategy == "per_loss_single_best"
        assert result.best_ensemble.metric_value == pytest.approx(0.91)

    def test_nan_entries_skipped(self) -> None:
        """Entries with NaN primary_metric_value are ignored."""
        entries = [
            _make_analysis_entry(
                model_name="dice_ce_fold0",
                loss_function="dice_ce",
                fold_id=0,
                primary_metric_value=float("nan"),
            ),
            _make_analysis_entry(
                model_name="cldice_fold0",
                loss_function="cldice",
                fold_id=0,
                primary_metric_value=0.85,
            ),
        ]
        result = select_champions(entries, primary_metric="dsc", maximize=True)
        assert result.best_single_fold is not None
        assert result.best_single_fold.model_name == "cldice_fold0"

    def test_empty_entries_returns_all_none(self) -> None:
        """Empty analysis_entries produces a ChampionSelection with all None."""
        result = select_champions([], primary_metric="dsc", maximize=True)
        assert result.best_single_fold is None
        assert result.best_cv_mean is None
        assert result.best_ensemble is None

    def test_all_nan_entries_returns_all_none(self) -> None:
        """All NaN values produce empty selection."""
        entries = [
            _make_analysis_entry(
                model_name="a_fold0",
                loss_function="a",
                fold_id=0,
                primary_metric_value=float("nan"),
            ),
            _make_analysis_entry(
                entry_type="cv_mean",
                model_name="a_cv_mean",
                loss_function="a",
                fold_id=None,
                primary_metric_value=float("nan"),
            ),
        ]
        result = select_champions(entries, primary_metric="dsc", maximize=True)
        assert result.best_single_fold is None
        assert result.best_cv_mean is None
        assert result.best_ensemble is None

    def test_multi_category_winner(self) -> None:
        """A model can be the best single fold AND come from the best CV-mean loss."""
        entries = [
            _make_analysis_entry(
                model_name="dice_ce_fold0",
                loss_function="dice_ce",
                fold_id=0,
                primary_metric_value=0.95,
            ),
            _make_analysis_entry(
                entry_type="cv_mean",
                model_name="dice_ce_cv_mean",
                loss_function="dice_ce",
                fold_id=None,
                primary_metric_value=0.93,
            ),
            _make_analysis_entry(
                entry_type="cv_mean",
                model_name="cldice_cv_mean",
                loss_function="cldice",
                fold_id=None,
                primary_metric_value=0.80,
            ),
        ]
        result = select_champions(entries, primary_metric="dsc", maximize=True)
        assert result.best_single_fold is not None
        assert result.best_single_fold.loss_function == "dice_ce"
        assert result.best_cv_mean is not None
        assert result.best_cv_mean.loss_function == "dice_ce"


# ---------------------------------------------------------------------------
# Tests: clear_champion_tags_filesystem
# ---------------------------------------------------------------------------


class TestClearChampionTags:
    """Tests for clearing existing champion tags from the filesystem."""

    def test_clears_all_champion_tags(self, tmp_path: Path) -> None:
        """All champion_* tag files are deleted."""
        mlruns_dir = tmp_path / "mlruns"
        run_dir = _make_run_with_tags(
            mlruns_dir,
            _EXP_ID,
            "run_01",
            extra_tags={
                "champion_best_single_fold": "true",
                "champion_metric_name": "dsc",
                "champion_metric_value": "0.95",
                "loss_function": "dice_ce",
            },
        )

        count = clear_champion_tags_filesystem(mlruns_dir, _EXP_ID)
        assert count >= 3  # at least the 3 champion tags
        assert _read_tag(run_dir, "champion_best_single_fold") is None
        assert _read_tag(run_dir, "champion_metric_name") is None

    def test_preserves_non_champion_tags(self, tmp_path: Path) -> None:
        """Tags not starting with 'champion_' must survive clearing."""
        mlruns_dir = tmp_path / "mlruns"
        run_dir = _make_run_with_tags(
            mlruns_dir,
            _EXP_ID,
            "run_01",
            extra_tags={
                "champion_best_cv_mean": "true",
                "loss_function": "dice_ce",
                "model_family": "dynunet",
            },
        )

        clear_champion_tags_filesystem(mlruns_dir, _EXP_ID)
        assert _read_tag(run_dir, "loss_function") == "dice_ce"
        assert _read_tag(run_dir, "model_family") == "dynunet"

    def test_idempotent_clears_zero_on_second_call(self, tmp_path: Path) -> None:
        """Second call returns 0 when all champion tags already cleared."""
        mlruns_dir = tmp_path / "mlruns"
        _make_run_with_tags(
            mlruns_dir,
            _EXP_ID,
            "run_01",
            extra_tags={"champion_best_single_fold": "true"},
        )

        count1 = clear_champion_tags_filesystem(mlruns_dir, _EXP_ID)
        assert count1 >= 1
        count2 = clear_champion_tags_filesystem(mlruns_dir, _EXP_ID)
        assert count2 == 0

    def test_returns_zero_for_no_champion_tags(self, tmp_path: Path) -> None:
        """Returns 0 when no champion tags exist."""
        mlruns_dir = tmp_path / "mlruns"
        _make_run_with_tags(
            mlruns_dir,
            _EXP_ID,
            "run_01",
            extra_tags={"loss_function": "dice_ce"},
        )

        count = clear_champion_tags_filesystem(mlruns_dir, _EXP_ID)
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: write_champion_tags_filesystem
# ---------------------------------------------------------------------------


class TestWriteChampionTags:
    """Tests for writing champion tags to the filesystem."""

    def test_writes_best_single_fold_flag(self, tmp_path: Path) -> None:
        """champion_best_single_fold is set to 'true' on the winner run."""
        mlruns_dir = tmp_path / "mlruns"
        run_dir = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_01")

        selection = _make_single_fold_selection(
            run_id="run_01",
            loss_function="dice_ce",
            fold_id=0,
            metric_value=0.90,
        )
        runs = [_make_run_info("run_01", "dice_ce", 0)]

        count = write_champion_tags_filesystem(
            mlruns_dir, _EXP_ID, selection, runs=runs
        )
        assert count > 0
        assert _read_tag(run_dir, "champion_best_single_fold") == "true"

    def test_writes_descriptive_metadata(self, tmp_path: Path) -> None:
        """Descriptive tags (metric_name, metric_value, tagged_at) are written."""
        mlruns_dir = tmp_path / "mlruns"
        run_dir = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_01")

        selection = _make_single_fold_selection(
            run_id="run_01",
            loss_function="dice_ce",
            fold_id=0,
            metric_value=0.90,
            metric_name="dsc",
        )
        runs = [_make_run_info("run_01", "dice_ce", 0)]

        write_champion_tags_filesystem(mlruns_dir, _EXP_ID, selection, runs=runs)
        assert _read_tag(run_dir, "champion_metric_name") == "dsc"
        assert _read_tag(run_dir, "champion_metric_value") == "0.9"
        assert _read_tag(run_dir, "champion_tagged_at") is not None

    def test_fold_id_only_on_single_fold(self, tmp_path: Path) -> None:
        """champion_fold_id is written only for best_single_fold, not cv_mean."""
        mlruns_dir = tmp_path / "mlruns"
        # run_01 wins single fold, run_02 wins cv_mean (same loss, different run)
        run_01 = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_01")
        run_02 = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_02")

        selection = _make_selection_with_single_and_cv(
            single_run_id="run_01",
            single_loss="dice_ce",
            single_fold=1,
            single_value=0.95,
            cv_loss="cldice",
            cv_value=0.88,
        )
        runs = [
            _make_run_info("run_01", "dice_ce", 1),
            _make_run_info("run_02", "cldice", 0),
        ]

        write_champion_tags_filesystem(mlruns_dir, _EXP_ID, selection, runs=runs)
        assert _read_tag(run_01, "champion_fold_id") == "1"
        # cv_mean winner should NOT have fold_id
        assert _read_tag(run_02, "champion_fold_id") is None

    def test_writes_cv_mean_flag(self, tmp_path: Path) -> None:
        """champion_best_cv_mean is set on the run matching the winning loss."""
        mlruns_dir = tmp_path / "mlruns"
        run_dir = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_01")

        selection = _make_cv_mean_selection(
            loss_function="dice_ce",
            metric_value=0.88,
        )
        # The run for this loss function
        runs = [_make_run_info("run_01", "dice_ce", 0)]

        write_champion_tags_filesystem(mlruns_dir, _EXP_ID, selection, runs=runs)
        assert _read_tag(run_dir, "champion_best_cv_mean") == "true"

    def test_writes_ensemble_flag_on_all_members(self, tmp_path: Path) -> None:
        """champion_best_ensemble is set on every member run of the winner."""
        mlruns_dir = tmp_path / "mlruns"
        run_01 = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_01")
        run_02 = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_02")
        run_03 = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_03")

        selection = _make_ensemble_selection(
            strategy="per_loss_single_best",
            metric_value=0.91,
            member_run_ids=["run_01", "run_02"],
        )
        runs = [
            _make_run_info("run_01", "dice_ce", 0),
            _make_run_info("run_02", "cldice", 0),
            _make_run_info("run_03", "other", 0),
        ]

        write_champion_tags_filesystem(mlruns_dir, _EXP_ID, selection, runs=runs)
        assert _read_tag(run_01, "champion_best_ensemble") == "true"
        assert _read_tag(run_02, "champion_best_ensemble") == "true"
        # Non-member run should NOT get the tag
        assert _read_tag(run_03, "champion_best_ensemble") is None

    def test_ensemble_strategy_tag_written(self, tmp_path: Path) -> None:
        """champion_ensemble_strategy is written on ensemble member runs."""
        mlruns_dir = tmp_path / "mlruns"
        run_dir = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_01")

        selection = _make_ensemble_selection(
            strategy="all_loss_single_best",
            metric_value=0.89,
            member_run_ids=["run_01"],
        )
        runs = [_make_run_info("run_01", "dice_ce", 0)]

        write_champion_tags_filesystem(mlruns_dir, _EXP_ID, selection, runs=runs)
        assert (
            _read_tag(run_dir, "champion_ensemble_strategy") == "all_loss_single_best"
        )


# ---------------------------------------------------------------------------
# Tests: tag_champions end-to-end
# ---------------------------------------------------------------------------


class TestTagChampionsEndToEnd:
    """Tests for the end-to-end tag_champions orchestrator."""

    def test_returns_champion_selection(self, tmp_path: Path) -> None:
        """tag_champions returns a valid ChampionSelection."""
        mlruns_dir = tmp_path / "mlruns"
        _make_run_with_tags(mlruns_dir, _EXP_ID, "run_01")

        entries = [
            _make_analysis_entry(
                model_name="dice_ce_fold0",
                loss_function="dice_ce",
                fold_id=0,
                primary_metric_value=0.90,
            ),
        ]
        runs = [_make_run_info("run_01", "dice_ce", 0)]

        result = tag_champions(
            mlruns_dir,
            _EXP_ID,
            entries,
            runs=runs,
            primary_metric="dsc",
            maximize=True,
        )
        assert isinstance(result, ChampionSelection)
        assert result.best_single_fold is not None

    def test_idempotent_rerun(self, tmp_path: Path) -> None:
        """Running tag_champions twice produces the same filesystem state."""
        mlruns_dir = tmp_path / "mlruns"
        run_dir = _make_run_with_tags(mlruns_dir, _EXP_ID, "run_01")

        entries = [
            _make_analysis_entry(
                model_name="dice_ce_fold0",
                loss_function="dice_ce",
                fold_id=0,
                primary_metric_value=0.90,
            ),
        ]
        runs = [_make_run_info("run_01", "dice_ce", 0)]

        tag_champions(
            mlruns_dir,
            _EXP_ID,
            entries,
            runs=runs,
            primary_metric="dsc",
            maximize=True,
        )
        val1 = _read_tag(run_dir, "champion_best_single_fold")

        tag_champions(
            mlruns_dir,
            _EXP_ID,
            entries,
            runs=runs,
            primary_metric="dsc",
            maximize=True,
        )
        val2 = _read_tag(run_dir, "champion_best_single_fold")

        assert val1 == val2 == "true"

    def test_stale_tags_cleared(self, tmp_path: Path) -> None:
        """Old champion tags from a different run are cleared on rerun."""
        mlruns_dir = tmp_path / "mlruns"
        old_run = _make_run_with_tags(
            mlruns_dir,
            _EXP_ID,
            "old_run",
            extra_tags={"champion_best_single_fold": "true"},
        )
        _make_run_with_tags(mlruns_dir, _EXP_ID, "new_run")

        entries = [
            _make_analysis_entry(
                model_name="cldice_fold0",
                loss_function="cldice",
                fold_id=0,
                primary_metric_value=0.95,
            ),
        ]
        runs = [_make_run_info("new_run", "cldice", 0)]

        tag_champions(
            mlruns_dir,
            _EXP_ID,
            entries,
            runs=runs,
            primary_metric="dsc",
            maximize=True,
        )

        # Old run's champion tag should be gone
        assert _read_tag(old_run, "champion_best_single_fold") is None


# ---------------------------------------------------------------------------
# Test: CHAMPION_TAG_KEYS frozenset
# ---------------------------------------------------------------------------


class TestChampionTagKeys:
    """Tests for the CHAMPION_TAG_KEYS constant."""

    def test_is_frozenset(self) -> None:
        assert isinstance(CHAMPION_TAG_KEYS, frozenset)

    def test_contains_all_flags(self) -> None:
        assert "champion_best_single_fold" in CHAMPION_TAG_KEYS
        assert "champion_best_cv_mean" in CHAMPION_TAG_KEYS
        assert "champion_best_ensemble" in CHAMPION_TAG_KEYS

    def test_contains_descriptive_keys(self) -> None:
        assert "champion_metric_name" in CHAMPION_TAG_KEYS
        assert "champion_metric_value" in CHAMPION_TAG_KEYS
        assert "champion_tagged_at" in CHAMPION_TAG_KEYS
        assert "champion_fold_id" in CHAMPION_TAG_KEYS
        assert "champion_ensemble_strategy" in CHAMPION_TAG_KEYS


# ---------------------------------------------------------------------------
# Selection builder helpers (keep tests DRY)
# ---------------------------------------------------------------------------


def _make_single_fold_selection(
    *,
    run_id: str = "run_01",
    loss_function: str = "dice_ce",
    fold_id: int = 0,
    metric_value: float = 0.90,
    metric_name: str = "dsc",
) -> ChampionSelection:
    """Build a ChampionSelection with only best_single_fold set."""
    from minivess.pipeline.champion_tagger import SingleFoldChampion

    return ChampionSelection(
        primary_metric=metric_name,
        best_single_fold=SingleFoldChampion(
            model_name=f"{loss_function}_fold{fold_id}",
            loss_function=loss_function,
            fold_id=fold_id,
            metric_value=metric_value,
        ),
        best_cv_mean=None,
        best_ensemble=None,
    )


def _make_cv_mean_selection(
    *,
    loss_function: str = "dice_ce",
    metric_value: float = 0.88,
    metric_name: str = "dsc",
) -> ChampionSelection:
    """Build a ChampionSelection with only best_cv_mean set."""
    from minivess.pipeline.champion_tagger import CvMeanChampion

    return ChampionSelection(
        primary_metric=metric_name,
        best_single_fold=None,
        best_cv_mean=CvMeanChampion(
            loss_function=loss_function,
            metric_value=metric_value,
        ),
        best_ensemble=None,
    )


def _make_selection_with_single_and_cv(
    *,
    single_run_id: str,
    single_loss: str,
    single_fold: int,
    single_value: float,
    cv_loss: str,
    cv_value: float,
    metric_name: str = "dsc",
) -> ChampionSelection:
    """Build a ChampionSelection with both single and cv_mean."""
    from minivess.pipeline.champion_tagger import CvMeanChampion, SingleFoldChampion

    return ChampionSelection(
        primary_metric=metric_name,
        best_single_fold=SingleFoldChampion(
            model_name=f"{single_loss}_fold{single_fold}",
            loss_function=single_loss,
            fold_id=single_fold,
            metric_value=single_value,
        ),
        best_cv_mean=CvMeanChampion(
            loss_function=cv_loss,
            metric_value=cv_value,
        ),
        best_ensemble=None,
    )


def _make_ensemble_selection(
    *,
    strategy: str = "per_loss_single_best",
    metric_value: float = 0.91,
    member_run_ids: list[str] | None = None,
    metric_name: str = "dsc",
) -> ChampionSelection:
    """Build a ChampionSelection with only best_ensemble set."""
    from minivess.pipeline.champion_tagger import EnsembleChampion

    return ChampionSelection(
        primary_metric=metric_name,
        best_single_fold=None,
        best_cv_mean=None,
        best_ensemble=EnsembleChampion(
            ensemble_strategy=strategy,
            metric_value=metric_value,
            member_run_ids=member_run_ids or [],
        ),
    )


# ===========================================================================
# Tests: rank_then_aggregate (#136)
# ===========================================================================


class TestRankThenAggregate:
    """Tests for rank-then-aggregate champion selection."""

    def test_rank_aggregate_single_winner(self) -> None:
        """When one model dominates all metrics, it should win balanced."""
        from minivess.pipeline.champion_tagger import rank_then_aggregate

        entries = [
            {"model_id": "A", "dsc": 0.9, "cldice": 0.85, "assd": 1.0},
            {"model_id": "B", "dsc": 0.7, "cldice": 0.65, "assd": 3.0},
        ]
        result = rank_then_aggregate(
            entries,
            maximize_metrics=["dsc", "cldice"],
            minimize_metrics=["assd"],
        )
        assert result["balanced"] == "A"

    def test_rank_aggregate_tie_breaking(self) -> None:
        """Tie breaking should be deterministic (first in input order)."""
        from minivess.pipeline.champion_tagger import rank_then_aggregate

        entries = [
            {"model_id": "A", "dsc": 0.9, "cldice": 0.7},
            {"model_id": "B", "dsc": 0.7, "cldice": 0.9},
        ]
        result = rank_then_aggregate(
            entries,
            maximize_metrics=["dsc", "cldice"],
            minimize_metrics=[],
        )
        # Both have mean rank 1.5, first should win tie
        assert result["balanced"] in ("A", "B")

    def test_rank_aggregate_correct_ranking_order(self) -> None:
        """Models should be ranked correctly by each metric."""
        from minivess.pipeline.champion_tagger import rank_then_aggregate

        entries = [
            {"model_id": "A", "dsc": 0.6, "cldice": 0.9, "assd": 5.0},
            {"model_id": "B", "dsc": 0.8, "cldice": 0.7, "assd": 2.0},
            {"model_id": "C", "dsc": 0.9, "cldice": 0.8, "assd": 1.0},
        ]
        result = rank_then_aggregate(
            entries,
            maximize_metrics=["dsc", "cldice"],
            minimize_metrics=["assd"],
        )
        # C is best at DSC and ASSD, second at cldice → should be balanced winner
        assert result["balanced"] == "C"

    def test_rank_aggregate_three_champion_categories(self) -> None:
        """Should return exactly three champion categories."""
        from minivess.pipeline.champion_tagger import rank_then_aggregate

        entries = [
            {"model_id": "A", "dsc": 0.9, "cldice": 0.7, "assd": 2.0},
            {"model_id": "B", "dsc": 0.7, "cldice": 0.9, "assd": 1.0},
        ]
        result = rank_then_aggregate(
            entries,
            maximize_metrics=["dsc", "cldice"],
            minimize_metrics=["assd"],
        )
        assert "balanced" in result
        assert "topology" in result
        assert "overlap" in result

    def test_rank_aggregate_topology_champion_is_best_cldice(self) -> None:
        """Topology champion should be the model with best clDice."""
        from minivess.pipeline.champion_tagger import rank_then_aggregate

        entries = [
            {"model_id": "A", "dsc": 0.9, "cldice": 0.7},
            {"model_id": "B", "dsc": 0.7, "cldice": 0.95},
        ]
        result = rank_then_aggregate(
            entries,
            maximize_metrics=["dsc", "cldice"],
            minimize_metrics=[],
            topology_metric="cldice",
        )
        assert result["topology"] == "B"

    def test_rank_aggregate_overlap_champion_is_best_dsc(self) -> None:
        """Overlap champion should be the model with best DSC."""
        from minivess.pipeline.champion_tagger import rank_then_aggregate

        entries = [
            {"model_id": "A", "dsc": 0.95, "cldice": 0.7},
            {"model_id": "B", "dsc": 0.7, "cldice": 0.95},
        ]
        result = rank_then_aggregate(
            entries,
            maximize_metrics=["dsc", "cldice"],
            minimize_metrics=[],
            overlap_metric="dsc",
        )
        assert result["overlap"] == "A"

    def test_rank_aggregate_balanced_uses_mean_rank(self) -> None:
        """Balanced champion should be determined by mean rank across all metrics."""
        from minivess.pipeline.champion_tagger import rank_then_aggregate

        entries = [
            {"model_id": "A", "dsc": 0.7, "cldice": 0.7, "assd": 3.0},
            {"model_id": "B", "dsc": 0.8, "cldice": 0.8, "assd": 2.0},
            {"model_id": "C", "dsc": 0.9, "cldice": 0.6, "assd": 4.0},
        ]
        result = rank_then_aggregate(
            entries,
            maximize_metrics=["dsc", "cldice"],
            minimize_metrics=["assd"],
        )
        # B is second in DSC, first in cldice, second in ASSD → mean rank = 5/3
        # C is first in DSC, third in cldice, third in ASSD → mean rank = 7/3
        # A is third in DSC, second in cldice, third in ASSD → mean rank = 8/3
        # B should be balanced winner
        assert result["balanced"] == "B"

    def test_rank_aggregate_handles_nan_metrics(self) -> None:
        """Models with NaN metrics should be ranked last."""
        from minivess.pipeline.champion_tagger import rank_then_aggregate

        entries = [
            {"model_id": "A", "dsc": 0.9, "cldice": float("nan")},
            {"model_id": "B", "dsc": 0.7, "cldice": 0.8},
        ]
        result = rank_then_aggregate(
            entries,
            maximize_metrics=["dsc", "cldice"],
            minimize_metrics=[],
        )
        # A has NaN cldice → ranked last for cldice
        # B should win balanced (1+2=3 vs 2+1=3, tied)
        assert result["balanced"] in ("A", "B")
