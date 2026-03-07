"""Tests for biostatistics source run discovery (Phase 2, Tasks 2.2-2.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from minivess.pipeline.biostatistics_discovery import (
    BiostatisticsValidationError,
    discover_source_runs,
    validate_source_completeness,
)
from minivess.pipeline.biostatistics_types import SourceRunManifest

if TYPE_CHECKING:
    from pathlib import Path


def _create_mock_mlruns(
    tmp_path: Path,
    conditions: list[str],
    n_folds: int = 3,
    *,
    include_failed: bool = False,
) -> Path:
    """Create a minimal mlruns directory structure for testing."""
    mlruns = tmp_path / "mlruns"
    exp_dir = mlruns / "1"
    exp_dir.mkdir(parents=True)

    # Experiment meta
    (exp_dir / "meta.yaml").write_text(
        yaml.dump({"name": "test_experiment"}), encoding="utf-8"
    )

    run_idx = 0
    for loss in conditions:
        for fold in range(n_folds):
            run_id = f"run_{run_idx:03d}"
            run_dir = exp_dir / run_id
            run_dir.mkdir()
            (run_dir / "meta.yaml").write_text(
                yaml.dump({"status": "FINISHED"}), encoding="utf-8"
            )
            params_dir = run_dir / "params"
            params_dir.mkdir()
            (params_dir / "loss_name").write_text(loss, encoding="utf-8")
            (params_dir / "fold_id").write_text(str(fold), encoding="utf-8")
            run_idx += 1

    if include_failed:
        run_id = f"run_{run_idx:03d}"
        run_dir = exp_dir / run_id
        run_dir.mkdir()
        (run_dir / "meta.yaml").write_text(
            yaml.dump({"status": "FAILED"}), encoding="utf-8"
        )
        params_dir = run_dir / "params"
        params_dir.mkdir()
        (params_dir / "loss_name").write_text("dice_ce", encoding="utf-8")
        (params_dir / "fold_id").write_text("0", encoding="utf-8")

    return mlruns


class TestDiscoverSourceRuns:
    def test_discover_finds_finished_runs(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns(tmp_path, ["dice_ce", "tversky"], n_folds=3)
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        assert len(manifest.runs) == 6  # 2 conditions x 3 folds

    def test_discover_excludes_failed_runs(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns(
            tmp_path, ["dice_ce"], n_folds=2, include_failed=True
        )
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        # 2 FINISHED + 1 FAILED -> should find only 2
        assert len(manifest.runs) == 2
        assert all(r.status == "FINISHED" for r in manifest.runs)

    def test_discover_multi_experiment(self, tmp_path: Path) -> None:
        mlruns = tmp_path / "mlruns"
        # Create two experiments
        for exp_idx, exp_name in enumerate(["exp_a", "exp_b"], start=1):
            exp_dir = mlruns / str(exp_idx)
            exp_dir.mkdir(parents=True)
            (exp_dir / "meta.yaml").write_text(
                yaml.dump({"name": exp_name}), encoding="utf-8"
            )
            run_dir = exp_dir / "run_001"
            run_dir.mkdir()
            (run_dir / "meta.yaml").write_text(
                yaml.dump({"status": "FINISHED"}), encoding="utf-8"
            )
            params = run_dir / "params"
            params.mkdir()
            (params / "loss_name").write_text("dice_ce", encoding="utf-8")
            (params / "fold_id").write_text("0", encoding="utf-8")

        manifest = discover_source_runs(mlruns, ["exp_a", "exp_b"])
        assert len(manifest.runs) == 2

    def test_fingerprint_is_sha256(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns(tmp_path, ["dice_ce"], n_folds=1)
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        assert len(manifest.fingerprint) == 64  # SHA-256 hex digest

    def test_empty_experiment_returns_empty_manifest(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns(tmp_path, ["dice_ce"], n_folds=1)
        manifest = discover_source_runs(mlruns, ["nonexistent_experiment"])
        assert len(manifest.runs) == 0

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        manifest = discover_source_runs(tmp_path / "nope", ["test"])
        assert len(manifest.runs) == 0


class TestValidateSourceCompleteness:
    def test_validation_passes_for_complete_data(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns(tmp_path, ["dice_ce", "tversky"], n_folds=3)
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        result = validate_source_completeness(manifest, min_folds=3, min_conditions=2)
        assert result.valid is True
        assert result.n_conditions == 2
        assert result.n_folds_per_condition == 3

    def test_validation_fails_missing_folds(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns(tmp_path, ["dice_ce", "tversky"], n_folds=1)
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        with pytest.raises(BiostatisticsValidationError, match="fold"):
            validate_source_completeness(manifest, min_folds=3, min_conditions=2)

    def test_validation_fails_too_few_conditions(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns(tmp_path, ["dice_ce"], n_folds=3)
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        with pytest.raises(BiostatisticsValidationError, match="condition"):
            validate_source_completeness(manifest, min_folds=3, min_conditions=2)

    def test_validation_error_is_raised_not_silent(self, tmp_path: Path) -> None:
        """Validation must raise, never return valid=False silently."""
        manifest = SourceRunManifest.from_runs([])
        with pytest.raises(BiostatisticsValidationError):
            validate_source_completeness(manifest, min_folds=1, min_conditions=2)
