"""Unit tests for MLflow run cleanup module.

Tests cover incomplete run identification, run status classification,
dry-run vs actual cleanup, trash-based deletion (reversible), and
audit log generation.  All tests use tmp_path for an isolated mock
MLflow directory layout — no real mlruns/ is read or modified.

MLflow filesystem layout assumed:
    mlruns/<experiment_id>/<run_id>/
        tags/<key>          — plain text
        metrics/<key>       — lines "<timestamp> <value> <step>"
        params/<key>        — plain text
        artifacts/          — arbitrary nested artifacts
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from minivess.pipeline.mlruns_cleanup import (
    RunStatus,
    classify_run,
    cleanup_incomplete_runs,
    identify_incomplete_runs,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers — build a mock MLflow filesystem tree in tmp_path
# ---------------------------------------------------------------------------

_EXPERIMENT_ID = "test_exp_cleanup"


def _make_run(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
    *,
    run_name: str = "test_run",
    num_metric_entries: int = 300,
    num_folds: int = 3,
    extra_tags: dict[str, str] | None = None,
) -> Path:
    """Create a mock run directory with realistic MLflow layout.

    Args:
        mlruns_dir: Root mock mlruns directory.
        experiment_id: Experiment ID subdirectory.
        run_id: Run ID subdirectory.
        run_name: Value for mlflow.runName tag.
        num_metric_entries: Number of train_loss entries to create.
        num_folds: Number of folds (epoch resets to 1).
        extra_tags: Additional {key: value} pairs for tags/.

    Returns:
        Path to the created run directory.
    """
    run_dir = mlruns_dir / experiment_id / run_id
    tags_dir = run_dir / "tags"
    metrics_dir = run_dir / "metrics"
    params_dir = run_dir / "params"
    artifacts_dir = run_dir / "artifacts"

    tags_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    params_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    # Write run name tag
    (tags_dir / "mlflow.runName").write_text(run_name, encoding="utf-8")

    # Write train_loss metric entries
    if num_metric_entries > 0:
        lines = []
        epochs_per_fold = num_metric_entries // max(num_folds, 1)
        for i in range(num_metric_entries):
            fold_epoch = (i % epochs_per_fold) + 1
            timestamp = 1772000000000 + i * 1000
            loss_val = 0.5 - i * 0.001
            lines.append(f"{timestamp} {loss_val} {fold_epoch}")
        (metrics_dir / "train_loss").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    if extra_tags:
        for key, value in extra_tags.items():
            (tags_dir / key).write_text(value, encoding="utf-8")

    return run_dir


# ---------------------------------------------------------------------------
# TestClassifyRun
# ---------------------------------------------------------------------------


class TestClassifyRun:
    """Tests for classify_run() which determines a run's completeness."""

    def test_complete_run_300_entries(self, tmp_path: Path) -> None:
        """A run with 300 train_loss entries (3x100) is COMPLETE."""
        _make_run(tmp_path, _EXPERIMENT_ID, "run_complete", num_metric_entries=300)
        status = classify_run(
            tmp_path, _EXPERIMENT_ID, "run_complete", expected_entries=300
        )
        assert status.is_complete is True
        assert status.run_id == "run_complete"
        assert status.num_entries == 300

    def test_incomplete_run_2_entries(self, tmp_path: Path) -> None:
        """A run with only 2 entries is INCOMPLETE (early abort)."""
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "run_aborted",
            num_metric_entries=2,
            num_folds=1,
        )
        status = classify_run(
            tmp_path, _EXPERIMENT_ID, "run_aborted", expected_entries=300
        )
        assert status.is_complete is False
        assert status.num_entries == 2

    def test_incomplete_run_0_entries(self, tmp_path: Path) -> None:
        """A run with 0 entries (false start) is INCOMPLETE."""
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "run_empty",
            num_metric_entries=0,
            num_folds=0,
        )
        status = classify_run(
            tmp_path, _EXPERIMENT_ID, "run_empty", expected_entries=300
        )
        assert status.is_complete is False
        assert status.num_entries == 0

    def test_run_name_is_captured(self, tmp_path: Path) -> None:
        """classify_run captures the mlflow.runName tag."""
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "run_named",
            run_name="dice_ce_20260226_120703",
            num_metric_entries=2,
            num_folds=1,
        )
        status = classify_run(
            tmp_path, _EXPERIMENT_ID, "run_named", expected_entries=300
        )
        assert status.run_name == "dice_ce_20260226_120703"

    def test_missing_train_loss_is_incomplete(self, tmp_path: Path) -> None:
        """A run with no train_loss metric file at all is INCOMPLETE."""
        run_dir = tmp_path / _EXPERIMENT_ID / "run_no_metric"
        (run_dir / "tags").mkdir(parents=True)
        (run_dir / "tags" / "mlflow.runName").write_text("bad", encoding="utf-8")
        (run_dir / "metrics").mkdir(parents=True)
        # No train_loss file created

        status = classify_run(
            tmp_path, _EXPERIMENT_ID, "run_no_metric", expected_entries=300
        )
        assert status.is_complete is False
        assert status.num_entries == 0

    def test_nonexistent_run_raises(self, tmp_path: Path) -> None:
        """classify_run raises FileNotFoundError for non-existent run."""
        with pytest.raises(FileNotFoundError):
            classify_run(tmp_path, _EXPERIMENT_ID, "nonexistent", expected_entries=300)


# ---------------------------------------------------------------------------
# TestIdentifyIncompleteRuns
# ---------------------------------------------------------------------------


class TestIdentifyIncompleteRuns:
    """Tests for identify_incomplete_runs() which scans an experiment."""

    def test_finds_incomplete_among_complete(self, tmp_path: Path) -> None:
        """Correctly separates incomplete from complete runs."""
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_1", num_metric_entries=300)
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_2", num_metric_entries=300)
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "incomplete_1",
            num_metric_entries=2,
            num_folds=1,
        )
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "incomplete_2",
            num_metric_entries=0,
            num_folds=0,
        )

        result = identify_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300
        )
        incomplete_ids = {r.run_id for r in result}
        assert incomplete_ids == {"incomplete_1", "incomplete_2"}

    def test_all_complete_returns_empty(self, tmp_path: Path) -> None:
        """When all runs are complete, returns empty list."""
        _make_run(tmp_path, _EXPERIMENT_ID, "ok_1", num_metric_entries=300)
        _make_run(tmp_path, _EXPERIMENT_ID, "ok_2", num_metric_entries=300)

        result = identify_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300
        )
        assert result == []

    def test_all_incomplete_returns_all(self, tmp_path: Path) -> None:
        """When all runs are incomplete, returns all."""
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "bad_1",
            num_metric_entries=5,
            num_folds=1,
        )
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "bad_2",
            num_metric_entries=0,
            num_folds=0,
        )

        result = identify_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300
        )
        assert len(result) == 2

    def test_empty_experiment_returns_empty(self, tmp_path: Path) -> None:
        """Empty experiment directory returns empty list."""
        (tmp_path / _EXPERIMENT_ID).mkdir(parents=True)
        result = identify_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300
        )
        assert result == []

    def test_nonexistent_experiment_returns_empty(self, tmp_path: Path) -> None:
        """Non-existent experiment directory returns empty list."""
        result = identify_incomplete_runs(
            tmp_path, "nonexistent_exp", expected_entries=300
        )
        assert result == []

    def test_skips_meta_yaml(self, tmp_path: Path) -> None:
        """meta.yaml files in experiment dir are not treated as runs."""
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "complete",
            num_metric_entries=300,
        )
        # meta.yaml is a file, not a dir
        (tmp_path / _EXPERIMENT_ID / "meta.yaml").write_text(
            "name: test", encoding="utf-8"
        )

        result = identify_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300
        )
        assert result == []


# ---------------------------------------------------------------------------
# TestCleanupIncompleteRuns — dry-run mode
# ---------------------------------------------------------------------------


class TestCleanupDryRun:
    """Tests for cleanup_incomplete_runs() in dry-run mode."""

    def test_dry_run_does_not_delete(self, tmp_path: Path) -> None:
        """Dry run identifies but does not move/delete any runs."""
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_1", num_metric_entries=300)
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "incomplete_1",
            num_metric_entries=2,
            num_folds=1,
        )

        result = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=True
        )

        # Run still exists in original location
        assert (tmp_path / _EXPERIMENT_ID / "incomplete_1").is_dir()
        # Result reports it as identified
        assert len(result.identified) == 1
        assert result.identified[0].run_id == "incomplete_1"
        # But nothing was moved
        assert result.moved == 0

    def test_dry_run_preserves_complete_runs(self, tmp_path: Path) -> None:
        """Dry run never touches complete runs."""
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_1", num_metric_entries=300)
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "incomplete_1",
            num_metric_entries=2,
            num_folds=1,
        )

        cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=True
        )

        assert (tmp_path / _EXPERIMENT_ID / "complete_1").is_dir()


# ---------------------------------------------------------------------------
# TestCleanupIncompleteRuns — actual cleanup
# ---------------------------------------------------------------------------


class TestCleanupActual:
    """Tests for cleanup_incomplete_runs() with actual deletion."""

    def test_moves_incomplete_to_trash(self, tmp_path: Path) -> None:
        """Incomplete runs are moved to .trash/<experiment_id>/<run_id>."""
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_1", num_metric_entries=300)
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "incomplete_1",
            run_name="dice_ce_20260226_120703",
            num_metric_entries=2,
            num_folds=1,
        )

        result = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=False
        )

        # Original location should be gone
        assert not (tmp_path / _EXPERIMENT_ID / "incomplete_1").is_dir()
        # Trash location should exist
        trash_dir = tmp_path / ".trash" / _EXPERIMENT_ID / "incomplete_1"
        assert trash_dir.is_dir()
        # Tags should still be readable in trash
        run_name = (trash_dir / "tags" / "mlflow.runName").read_text(encoding="utf-8")
        assert run_name == "dice_ce_20260226_120703"
        # Result counts
        assert result.moved == 1

    def test_preserves_complete_runs(self, tmp_path: Path) -> None:
        """Complete runs are never touched during cleanup."""
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_1", num_metric_entries=300)
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_2", num_metric_entries=300)
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "incomplete_1",
            num_metric_entries=5,
            num_folds=1,
        )

        cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=False
        )

        # Complete runs untouched
        assert (tmp_path / _EXPERIMENT_ID / "complete_1").is_dir()
        assert (tmp_path / _EXPERIMENT_ID / "complete_2").is_dir()

    def test_moves_multiple_incomplete(self, tmp_path: Path) -> None:
        """All incomplete runs are moved in a single cleanup call."""
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_1", num_metric_entries=300)
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "bad_1",
            num_metric_entries=2,
            num_folds=1,
        )
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "bad_2",
            num_metric_entries=0,
            num_folds=0,
        )
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "bad_3",
            num_metric_entries=50,
            num_folds=1,
        )

        result = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=False
        )

        assert result.moved == 3
        for rid in ["bad_1", "bad_2", "bad_3"]:
            assert not (tmp_path / _EXPERIMENT_ID / rid).is_dir()
            assert (tmp_path / ".trash" / _EXPERIMENT_ID / rid).is_dir()

    def test_no_incomplete_means_no_moves(self, tmp_path: Path) -> None:
        """When all runs are complete, nothing is moved."""
        _make_run(tmp_path, _EXPERIMENT_ID, "ok_1", num_metric_entries=300)
        _make_run(tmp_path, _EXPERIMENT_ID, "ok_2", num_metric_entries=300)

        result = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=False
        )

        assert result.moved == 0
        assert not (tmp_path / ".trash").exists()

    def test_idempotent_second_cleanup(self, tmp_path: Path) -> None:
        """Running cleanup twice is safe — second run finds nothing."""
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_1", num_metric_entries=300)
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "incomplete_1",
            num_metric_entries=2,
            num_folds=1,
        )

        result1 = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=False
        )
        result2 = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=False
        )

        assert result1.moved == 1
        assert result2.moved == 0


# ---------------------------------------------------------------------------
# TestCleanupResult
# ---------------------------------------------------------------------------


class TestCleanupResult:
    """Tests for CleanupResult audit information."""

    def test_result_has_experiment_id(self, tmp_path: Path) -> None:
        """CleanupResult contains the experiment ID."""
        _make_run(tmp_path, _EXPERIMENT_ID, "complete_1", num_metric_entries=300)
        result = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=True
        )
        assert result.experiment_id == _EXPERIMENT_ID

    def test_result_has_total_runs(self, tmp_path: Path) -> None:
        """CleanupResult reports total run count (complete + incomplete)."""
        _make_run(tmp_path, _EXPERIMENT_ID, "c1", num_metric_entries=300)
        _make_run(tmp_path, _EXPERIMENT_ID, "c2", num_metric_entries=300)
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "i1",
            num_metric_entries=2,
            num_folds=1,
        )

        result = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=True
        )
        assert result.total_runs == 3
        assert result.complete_runs == 2

    def test_result_summary_string(self, tmp_path: Path) -> None:
        """CleanupResult.summary() returns a human-readable string."""
        _make_run(tmp_path, _EXPERIMENT_ID, "c1", num_metric_entries=300)
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "i1",
            run_name="dice_ce_20260226_120703",
            num_metric_entries=2,
            num_folds=1,
        )

        result = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=True
        )
        summary = result.summary()
        assert "dice_ce_20260226_120703" in summary
        assert "2 entries" in summary or "2/" in summary

    def test_result_identified_has_run_statuses(self, tmp_path: Path) -> None:
        """Result.identified contains RunStatus objects."""
        _make_run(
            tmp_path,
            _EXPERIMENT_ID,
            "i1",
            run_name="test_run",
            num_metric_entries=5,
            num_folds=1,
        )

        result = cleanup_incomplete_runs(
            tmp_path, _EXPERIMENT_ID, expected_entries=300, dry_run=True
        )
        assert len(result.identified) == 1
        status = result.identified[0]
        assert isinstance(status, RunStatus)
        assert status.run_id == "i1"
        assert status.run_name == "test_run"
        assert status.num_entries == 5
        assert status.is_complete is False
