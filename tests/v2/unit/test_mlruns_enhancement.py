"""Unit tests for post-hoc MLflow run metadata enhancement.

Tests cover filesystem-level tag writing, production-run identification,
idempotency, and the batch enhancement helper.  All tests use tmp_path for
an isolated mock MLflow directory layout — no real mlruns/ is read.

MLflow filesystem layout assumed:
    mlruns/<experiment_id>/<run_id>/
        tags/<key>          — plain text
        metrics/<key>       — lines "<timestamp> <value> <step>"
        params/<key>        — plain text
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from minivess.pipeline.mlruns_enhancement import (
    backfill_architecture_params,
    enhance_all_production_runs,
    enhance_run_tags,
    get_git_commit,
    get_software_versions,
    identify_production_runs,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers — build a mock MLflow filesystem tree in tmp_path
# ---------------------------------------------------------------------------

_EXPERIMENT_ID = "test_exp_001"


def _make_run(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
    *,
    fold2_metric: bool = True,
    loss_function: str | None = "dice_ce",
    extra_tags: dict[str, str] | None = None,
    extra_params: dict[str, str] | None = None,
) -> Path:
    """Create a mock run directory with the expected MLflow layout.

    Args:
        mlruns_dir: Root mock mlruns directory.
        experiment_id: Experiment ID subdirectory.
        run_id: Run ID subdirectory.
        fold2_metric: When True, creates an ``eval_fold2_dsc`` metric file
            (marking this as a production/complete run).
        loss_function: Value to write to ``tags/loss_function``.  Pass
            ``None`` to omit the tag entirely.
        extra_tags: Additional ``{key: value}`` pairs written to ``tags/``.

    Returns:
        Path to the run directory.
    """
    run_dir = mlruns_dir / experiment_id / run_id
    tags_dir = run_dir / "tags"
    metrics_dir = run_dir / "metrics"
    tags_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if fold2_metric:
        (metrics_dir / "eval_fold2_dsc").write_text(
            "1700000000 0.85 99\n", encoding="utf-8"
        )
    else:
        # Only fold0/fold1 present (incomplete run)
        (metrics_dir / "eval_fold0_dsc").write_text(
            "1700000000 0.80 49\n", encoding="utf-8"
        )

    if loss_function is not None:
        (tags_dir / "loss_function").write_text(loss_function, encoding="utf-8")

    if extra_tags:
        for key, value in extra_tags.items():
            (tags_dir / key).write_text(value, encoding="utf-8")

    if extra_params:
        params_dir = run_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        for key, value in extra_params.items():
            (params_dir / key).write_text(value, encoding="utf-8")

    return run_dir


def _mlruns(tmp_path: Path) -> Path:
    """Return the root mlruns directory inside tmp_path."""
    d = tmp_path / "mlruns"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Tests: identify_production_runs
# ---------------------------------------------------------------------------


class TestIdentifyProductionRuns:
    """Tests for the production-run identification function."""

    def test_identify_production_runs_returns_correct_count(
        self, tmp_path: Path
    ) -> None:
        """Four production runs out of six total are returned."""
        mlruns_dir = _mlruns(tmp_path)
        # 4 complete runs
        for i in range(4):
            _make_run(mlruns_dir, _EXPERIMENT_ID, f"run_{i:02d}", fold2_metric=True)
        # 2 incomplete runs (no fold2)
        for i in range(4, 6):
            _make_run(mlruns_dir, _EXPERIMENT_ID, f"run_{i:02d}", fold2_metric=False)

        result = identify_production_runs(mlruns_dir, _EXPERIMENT_ID)

        assert len(result) == 4

    def test_identify_production_runs_excludes_incomplete(self, tmp_path: Path) -> None:
        """Runs without eval_fold2_* metrics are excluded from results."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(mlruns_dir, _EXPERIMENT_ID, "complete_run", fold2_metric=True)
        _make_run(mlruns_dir, _EXPERIMENT_ID, "incomplete_run", fold2_metric=False)

        result = identify_production_runs(mlruns_dir, _EXPERIMENT_ID)

        assert "complete_run" in result
        assert "incomplete_run" not in result

    def test_identify_production_runs_nonexistent_dir(self, tmp_path: Path) -> None:
        """Returns empty list when the experiment directory does not exist."""
        mlruns_dir = _mlruns(tmp_path)
        # No experiment directory created

        result = identify_production_runs(mlruns_dir, "nonexistent_exp")

        assert result == []

    def test_identify_production_runs_returns_sorted(self, tmp_path: Path) -> None:
        """Returned run IDs are sorted lexicographically."""
        mlruns_dir = _mlruns(tmp_path)
        for run_id in ["run_c", "run_a", "run_b"]:
            _make_run(mlruns_dir, _EXPERIMENT_ID, run_id, fold2_metric=True)

        result = identify_production_runs(mlruns_dir, _EXPERIMENT_ID)

        assert result == sorted(result)

    def test_identify_production_runs_skips_meta_yaml(self, tmp_path: Path) -> None:
        """The meta.yaml file at experiment level is not treated as a run."""
        mlruns_dir = _mlruns(tmp_path)
        exp_dir = mlruns_dir / _EXPERIMENT_ID
        exp_dir.mkdir(parents=True, exist_ok=True)
        # Create a file named meta.yaml (not a run directory)
        (exp_dir / "meta.yaml").write_text("name: test\n", encoding="utf-8")
        _make_run(mlruns_dir, _EXPERIMENT_ID, "real_run", fold2_metric=True)

        result = identify_production_runs(mlruns_dir, _EXPERIMENT_ID)

        assert "meta.yaml" not in result
        assert "real_run" in result


# ---------------------------------------------------------------------------
# Tests: get_software_versions
# ---------------------------------------------------------------------------


class TestGetSoftwareVersions:
    """Tests for the software version collector."""

    def test_get_software_versions_has_python(self) -> None:
        """python_version is always present in the returned dict."""
        versions = get_software_versions()

        assert "python_version" in versions
        assert len(versions["python_version"]) > 0

    def test_get_software_versions_has_pytorch(self) -> None:
        """pytorch_version is present when torch is installed."""
        pytest.importorskip("torch")

        versions = get_software_versions()

        assert "pytorch_version" in versions
        assert len(versions["pytorch_version"]) > 0

    def test_get_software_versions_has_cuda_key(self) -> None:
        """cuda_version is present (value may be 'N/A') when torch is available."""
        pytest.importorskip("torch")

        versions = get_software_versions()

        assert "cuda_version" in versions

    def test_get_software_versions_returns_strings(self) -> None:
        """All returned values are strings."""
        versions = get_software_versions()

        for key, value in versions.items():
            assert isinstance(value, str), f"Expected str for {key}, got {type(value)}"

    def test_get_software_versions_torch_import_error(self) -> None:
        """When torch is not importable, pytorch_version and cuda_version absent."""
        import builtins

        real_import = builtins.__import__

        def _blocked_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "torch":
                raise ImportError("torch not available")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocked_import):
            versions = get_software_versions()

        assert "python_version" in versions
        assert "pytorch_version" not in versions


# ---------------------------------------------------------------------------
# Tests: get_git_commit
# ---------------------------------------------------------------------------


class TestGetGitCommit:
    """Tests for the git commit hash retrieval function."""

    def test_get_git_commit_returns_hash(self) -> None:
        """Returns a non-empty string (SHA or 'unknown')."""
        commit = get_git_commit()

        assert isinstance(commit, str)
        assert len(commit) > 0

    def test_get_git_commit_returns_unknown_when_git_missing(self) -> None:
        """Returns 'unknown' when git binary is not found."""
        with patch(
            "minivess.pipeline.mlruns_enhancement.subprocess.run",
            side_effect=FileNotFoundError("git not found"),
        ):
            commit = get_git_commit()

        assert commit == "unknown"

    def test_get_git_commit_returns_unknown_on_error(self) -> None:
        """Returns 'unknown' when git command fails (e.g. not in a repo)."""
        import subprocess

        with patch(
            "minivess.pipeline.mlruns_enhancement.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            commit = get_git_commit()

        assert commit == "unknown"

    def test_get_git_commit_sha_length(self) -> None:
        """When git succeeds, commit hash is 40 hex characters."""
        fake_sha = "a" * 40
        with patch("minivess.pipeline.mlruns_enhancement.subprocess.run") as mock_run:
            mock_run.return_value.stdout = f"{fake_sha}\n"
            commit = get_git_commit()

        assert commit == fake_sha


# ---------------------------------------------------------------------------
# Tests: enhance_run_tags
# ---------------------------------------------------------------------------


class TestEnhanceRunTags:
    """Tests for the per-run tag enhancement function."""

    def test_enhance_run_adds_loss_type_alias(self, tmp_path: Path) -> None:
        """loss_type tag is created from loss_function when absent."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(mlruns_dir, _EXPERIMENT_ID, "run_00", loss_function="dice_ce")

        added = enhance_run_tags(mlruns_dir, _EXPERIMENT_ID, "run_00")

        assert "loss_type" in added
        assert added["loss_type"] == "dice_ce"
        # Verify file was actually written
        loss_type_path = mlruns_dir / _EXPERIMENT_ID / "run_00" / "tags" / "loss_type"
        assert loss_type_path.exists()
        assert loss_type_path.read_text(encoding="utf-8") == "dice_ce"

    def test_enhance_run_skips_existing_tags(self, tmp_path: Path) -> None:
        """Pre-existing tag files are not overwritten (idempotent)."""
        mlruns_dir = _mlruns(tmp_path)
        # Pre-populate loss_type with a different value
        _make_run(
            mlruns_dir,
            _EXPERIMENT_ID,
            "run_00",
            loss_function="dice_ce",
            extra_tags={"loss_type": "original_value"},
        )

        added = enhance_run_tags(mlruns_dir, _EXPERIMENT_ID, "run_00")

        # loss_type was already present — must not be overwritten
        assert "loss_type" not in added
        loss_type_path = mlruns_dir / _EXPERIMENT_ID / "run_00" / "tags" / "loss_type"
        assert loss_type_path.read_text(encoding="utf-8") == "original_value"

    def test_enhance_run_adds_software_versions(self, tmp_path: Path) -> None:
        """Software version tags are written as files."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(mlruns_dir, _EXPERIMENT_ID, "run_00")

        with patch(
            "minivess.pipeline.mlruns_enhancement.get_software_versions",
            return_value={
                "python_version": "3.12.0",
                "pytorch_version": "2.2.0",
            },
        ):
            added = enhance_run_tags(mlruns_dir, _EXPERIMENT_ID, "run_00")

        tags_dir = mlruns_dir / _EXPERIMENT_ID / "run_00" / "tags"
        assert "python_version" in added
        assert (tags_dir / "python_version").read_text(encoding="utf-8") == "3.12.0"
        assert "pytorch_version" in added
        assert (tags_dir / "pytorch_version").read_text(encoding="utf-8") == "2.2.0"

    def test_enhance_run_adds_hardware_spec(self, tmp_path: Path) -> None:
        """Hardware spec tags are written as files."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(mlruns_dir, _EXPERIMENT_ID, "run_00")

        with patch(
            "minivess.pipeline.mlruns_enhancement.get_hardware_spec",
            return_value={
                "total_ram_gb": "64.0",
                "gpu_model": "NVIDIA RTX 4090",
                "gpu_vram_mb": "24564",
            },
        ):
            added = enhance_run_tags(mlruns_dir, _EXPERIMENT_ID, "run_00")

        tags_dir = mlruns_dir / _EXPERIMENT_ID / "run_00" / "tags"
        assert "total_ram_gb" in added
        assert (tags_dir / "total_ram_gb").read_text(encoding="utf-8") == "64.0"
        assert "gpu_model" in added
        assert (tags_dir / "gpu_model").read_text(encoding="utf-8") == "NVIDIA RTX 4090"

    def test_enhance_run_adds_git_commit(self, tmp_path: Path) -> None:
        """git_commit tag is written when absent."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(mlruns_dir, _EXPERIMENT_ID, "run_00")
        fake_sha = "b" * 40

        with patch(
            "minivess.pipeline.mlruns_enhancement.get_git_commit",
            return_value=fake_sha,
        ):
            added = enhance_run_tags(mlruns_dir, _EXPERIMENT_ID, "run_00")

        assert "git_commit" in added
        git_path = mlruns_dir / _EXPERIMENT_ID / "run_00" / "tags" / "git_commit"
        assert git_path.read_text(encoding="utf-8") == fake_sha

    def test_enhance_run_returns_empty_when_no_tags_dir(self, tmp_path: Path) -> None:
        """Returns empty dict when the run's tags directory is missing."""
        mlruns_dir = _mlruns(tmp_path)
        # Create a run directory WITHOUT a tags/ subdirectory
        run_dir = mlruns_dir / _EXPERIMENT_ID / "no_tags_run"
        run_dir.mkdir(parents=True)

        result = enhance_run_tags(mlruns_dir, _EXPERIMENT_ID, "no_tags_run")

        assert result == {}

    def test_enhance_run_no_loss_type_when_no_loss_function(
        self, tmp_path: Path
    ) -> None:
        """When loss_function tag is absent, loss_type is not created."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(mlruns_dir, _EXPERIMENT_ID, "run_00", loss_function=None)

        added = enhance_run_tags(mlruns_dir, _EXPERIMENT_ID, "run_00")

        assert "loss_type" not in added
        loss_type_path = mlruns_dir / _EXPERIMENT_ID / "run_00" / "tags" / "loss_type"
        assert not loss_type_path.exists()

    def test_enhance_run_idempotent_on_second_call(self, tmp_path: Path) -> None:
        """Second call to enhance_run_tags adds nothing (all tags exist)."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(mlruns_dir, _EXPERIMENT_ID, "run_00", loss_function="dice_focal")

        # First call: tags are added
        first = enhance_run_tags(mlruns_dir, _EXPERIMENT_ID, "run_00")
        assert len(first) > 0

        # Second call: everything is already present
        second = enhance_run_tags(mlruns_dir, _EXPERIMENT_ID, "run_00")
        assert second == {}


# ---------------------------------------------------------------------------
# Tests: enhance_all_production_runs
# ---------------------------------------------------------------------------


class TestEnhanceAllProductionRuns:
    """Tests for the batch enhancement helper."""

    def test_enhance_all_production_runs(self, tmp_path: Path) -> None:
        """All production runs receive tags; incomplete runs are skipped."""
        mlruns_dir = _mlruns(tmp_path)
        # 2 production, 1 incomplete
        for i in range(2):
            _make_run(
                mlruns_dir,
                _EXPERIMENT_ID,
                f"prod_{i:02d}",
                fold2_metric=True,
                loss_function="dice_ce",
            )
        _make_run(
            mlruns_dir,
            _EXPERIMENT_ID,
            "incomplete_00",
            fold2_metric=False,
            loss_function="dice_ce",
        )

        with (
            patch(
                "minivess.pipeline.mlruns_enhancement.get_software_versions",
                return_value={"python_version": "3.12.0"},
            ),
            patch(
                "minivess.pipeline.mlruns_enhancement.get_hardware_spec",
                return_value={"total_ram_gb": "32.0"},
            ),
            patch(
                "minivess.pipeline.mlruns_enhancement.get_git_commit",
                return_value="c" * 40,
            ),
        ):
            results = enhance_all_production_runs(mlruns_dir, _EXPERIMENT_ID)

        # Exactly the 2 production runs are enhanced
        assert len(results) == 2
        assert "prod_00" in results
        assert "prod_01" in results
        assert "incomplete_00" not in results

        # Each enhanced run has expected keys
        for run_id in ["prod_00", "prod_01"]:
            assert "loss_type" in results[run_id]
            assert results[run_id]["loss_type"] == "dice_ce"

    def test_enhance_all_production_runs_empty_when_no_experiment(
        self, tmp_path: Path
    ) -> None:
        """Returns empty dict when experiment directory does not exist."""
        mlruns_dir = _mlruns(tmp_path)

        results = enhance_all_production_runs(mlruns_dir, "nonexistent_exp_id")

        assert results == {}

    def test_enhance_all_production_runs_skips_runs_already_enhanced(
        self, tmp_path: Path
    ) -> None:
        """Runs that already have all tags contribute nothing to the result."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(
            mlruns_dir,
            _EXPERIMENT_ID,
            "already_done",
            fold2_metric=True,
            loss_function="dice_ce",
            extra_tags={
                "loss_type": "dice_ce",
                "python_version": "3.12.0",
                "pytorch_version": "2.2.0",
                "monai_version": "1.3.0",
                "cuda_version": "12.1",
                "total_ram_gb": "64.0",
                "gpu_model": "RTX 4090",
                "gpu_vram_mb": "24564",
                "git_commit": "a" * 40,
            },
        )

        with (
            patch(
                "minivess.pipeline.mlruns_enhancement.get_software_versions",
                return_value={
                    "python_version": "3.12.0",
                    "pytorch_version": "2.2.0",
                    "monai_version": "1.3.0",
                    "cuda_version": "12.1",
                },
            ),
            patch(
                "minivess.pipeline.mlruns_enhancement.get_hardware_spec",
                return_value={
                    "total_ram_gb": "64.0",
                    "gpu_model": "RTX 4090",
                    "gpu_vram_mb": "24564",
                },
            ),
            patch(
                "minivess.pipeline.mlruns_enhancement.get_git_commit",
                return_value="a" * 40,
            ),
        ):
            results = enhance_all_production_runs(mlruns_dir, _EXPERIMENT_ID)

        # Nothing was added — run is omitted from results
        assert results == {}

    def test_enhance_all_production_runs_returns_added_tags(
        self, tmp_path: Path
    ) -> None:
        """Returned dict maps run_id to the dict of newly added tags."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(
            mlruns_dir,
            _EXPERIMENT_ID,
            "single_run",
            fold2_metric=True,
            loss_function="cldice",
        )

        with (
            patch(
                "minivess.pipeline.mlruns_enhancement.get_software_versions",
                return_value={"python_version": "3.12.0"},
            ),
            patch(
                "minivess.pipeline.mlruns_enhancement.get_hardware_spec",
                return_value={"total_ram_gb": "16.0"},
            ),
            patch(
                "minivess.pipeline.mlruns_enhancement.get_git_commit",
                return_value="d" * 40,
            ),
        ):
            results = enhance_all_production_runs(mlruns_dir, _EXPERIMENT_ID)

        assert "single_run" in results
        run_added = results["single_run"]
        assert run_added["loss_type"] == "cldice"
        assert run_added["python_version"] == "3.12.0"
        assert run_added["total_ram_gb"] == "16.0"
        assert run_added["git_commit"] == "d" * 40


# ---------------------------------------------------------------------------
# Tests: backfill_architecture_params
# ---------------------------------------------------------------------------


class TestBackfillArchitectureParams:
    """Tests for retroactive architecture param backfilling."""

    def test_backfill_adds_arch_filters_with_default(self, tmp_path: Path) -> None:
        """Runs missing arch_filters get backfilled with default [32,64,128,256]."""
        mlruns_dir = _mlruns(tmp_path)
        run_dir = _make_run(mlruns_dir, _EXPERIMENT_ID, "run_00")
        # Ensure params dir exists (real runs always have it)
        (run_dir / "params").mkdir(parents=True, exist_ok=True)

        results = backfill_architecture_params(mlruns_dir, _EXPERIMENT_ID)

        assert "run_00" in results
        assert "arch_filters" in results["run_00"]
        param_file = run_dir / "params" / "arch_filters"
        assert param_file.exists()
        assert param_file.read_text(encoding="utf-8") == "[32, 64, 128, 256]"

    def test_backfill_skips_runs_with_existing_arch_filters(
        self, tmp_path: Path
    ) -> None:
        """Runs that already have arch_filters are not modified."""
        mlruns_dir = _mlruns(tmp_path)
        _make_run(
            mlruns_dir,
            _EXPERIMENT_ID,
            "run_00",
            extra_params={"arch_filters": "[16, 32, 64, 128]"},
        )

        results = backfill_architecture_params(mlruns_dir, _EXPERIMENT_ID)

        assert results == {}
        # Original value preserved
        param_file = mlruns_dir / _EXPERIMENT_ID / "run_00" / "params" / "arch_filters"
        assert param_file.read_text(encoding="utf-8") == "[16, 32, 64, 128]"

    def test_backfill_custom_default_filters(self, tmp_path: Path) -> None:
        """Explicit default_filters are used when no checkpoint is present."""
        mlruns_dir = _mlruns(tmp_path)
        run_dir = _make_run(mlruns_dir, _EXPERIMENT_ID, "run_00")
        (run_dir / "params").mkdir(parents=True, exist_ok=True)

        backfill_architecture_params(
            mlruns_dir,
            _EXPERIMENT_ID,
            default_filters=[16, 32, 64, 128],
        )

        param_file = run_dir / "params" / "arch_filters"
        assert param_file.read_text(encoding="utf-8") == "[16, 32, 64, 128]"

    def test_backfill_nonexistent_experiment(self, tmp_path: Path) -> None:
        """Returns empty dict for nonexistent experiment."""
        mlruns_dir = _mlruns(tmp_path)

        results = backfill_architecture_params(mlruns_dir, "nonexistent")

        assert results == {}

    def test_backfill_idempotent(self, tmp_path: Path) -> None:
        """Second call adds nothing when first call already backfilled."""
        mlruns_dir = _mlruns(tmp_path)
        run_dir = _make_run(mlruns_dir, _EXPERIMENT_ID, "run_00")
        (run_dir / "params").mkdir(parents=True, exist_ok=True)

        first = backfill_architecture_params(mlruns_dir, _EXPERIMENT_ID)
        second = backfill_architecture_params(mlruns_dir, _EXPERIMENT_ID)

        assert len(first) == 1
        assert second == {}
