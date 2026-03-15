"""Tests for scripts/scan_experiments.py (T23).

RED phase: tests written before implementation.

Validates:
- Experiment data loading from a mock DuckDB / mlruns structure
- Champion selection logic (best metric per model family)
- Output YAML validity and schema
- No import re (CLAUDE.md Rule #16)
- Idempotency: running twice produces identical output
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import the scanner (will fail in RED phase)
# ---------------------------------------------------------------------------
from scripts.scan_experiments import (  # noqa: E402
    RunMetrics,
    find_champion,
    load_runs_from_parquet,
    write_experiments_yaml,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal mock run data
# ---------------------------------------------------------------------------

MOCK_RUNS = [
    RunMetrics(
        run_id="run001",
        experiment_name="dynunet_loss_variation_v2",
        model_family="dynunet",
        loss_function="cbdice_cldice",
        fold=0,
        dsc_mean=0.7716,
        dsc_std=0.0162,
        cldice_mean=0.9060,
        cldice_std=0.0075,
        epochs=100,
    ),
    RunMetrics(
        run_id="run002",
        experiment_name="dynunet_loss_variation_v2",
        model_family="dynunet",
        loss_function="dice_ce",
        fold=0,
        dsc_mean=0.8242,
        dsc_std=0.0136,
        cldice_mean=0.8317,
        cldice_std=0.0188,
        epochs=100,
    ),
    RunMetrics(
        run_id="run003",
        experiment_name="dynunet_loss_variation_v2",
        model_family="dynunet",
        loss_function="cbdice",
        fold=0,
        dsc_mean=0.7666,
        dsc_std=0.0232,
        cldice_mean=0.7992,
        cldice_std=0.0202,
        epochs=100,
    ),
]


# ---------------------------------------------------------------------------
# Unit tests: RunMetrics dataclass
# ---------------------------------------------------------------------------


class TestRunMetrics:
    def test_creates_from_fields(self) -> None:
        run = MOCK_RUNS[0]
        assert run.run_id == "run001"
        assert run.model_family == "dynunet"
        assert run.dsc_mean == pytest.approx(0.7716)

    def test_has_required_fields(self) -> None:
        run = MOCK_RUNS[0]
        required = [
            "run_id",
            "experiment_name",
            "model_family",
            "loss_function",
            "fold",
            "dsc_mean",
            "cldice_mean",
        ]
        for field in required:
            assert hasattr(run, field), f"RunMetrics missing field: {field}"


# ---------------------------------------------------------------------------
# Unit tests: find_champion
# ---------------------------------------------------------------------------


class TestFindChampion:
    def test_finds_best_cldice(self) -> None:
        champion = find_champion(MOCK_RUNS, metric="cldice_mean")
        assert champion is not None
        assert champion.loss_function == "cbdice_cldice"

    def test_finds_best_dsc(self) -> None:
        champion = find_champion(MOCK_RUNS, metric="dsc_mean")
        assert champion is not None
        assert champion.loss_function == "dice_ce"

    def test_returns_none_for_empty(self) -> None:
        champion = find_champion([], metric="dsc_mean")
        assert champion is None

    def test_returns_run_metrics_type(self) -> None:
        champion = find_champion(MOCK_RUNS, metric="cldice_mean")
        assert isinstance(champion, RunMetrics)


# ---------------------------------------------------------------------------
# Integration tests: write YAML
# ---------------------------------------------------------------------------


class TestWriteExperimentsYaml:
    def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        out_path = tmp_path / "dynunet_loss_variation_v2.yaml"
        write_experiments_yaml(
            runs=MOCK_RUNS,
            experiment_name="dynunet_loss_variation_v2",
            out_path=out_path,
        )
        assert out_path.exists()
        data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert "runs" in data
        assert "_meta" in data

    def test_yaml_has_champion_section(self, tmp_path: Path) -> None:
        out_path = tmp_path / "out.yaml"
        write_experiments_yaml(
            runs=MOCK_RUNS,
            experiment_name="dynunet_loss_variation_v2",
            out_path=out_path,
        )
        data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert "champion" in data

    def test_champion_is_correct(self, tmp_path: Path) -> None:
        out_path = tmp_path / "out.yaml"
        write_experiments_yaml(
            runs=MOCK_RUNS,
            experiment_name="dynunet_loss_variation_v2",
            out_path=out_path,
        )
        data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        # cbdice_cldice has best cldice
        assert data["champion"]["by_cldice"]["loss_function"] == "cbdice_cldice"
        # dice_ce has best dsc
        assert data["champion"]["by_dsc"]["loss_function"] == "dice_ce"

    def test_generated_by_field(self, tmp_path: Path) -> None:
        out_path = tmp_path / "out.yaml"
        write_experiments_yaml(
            runs=MOCK_RUNS,
            experiment_name="dynunet_loss_variation_v2",
            out_path=out_path,
        )
        data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert data["_meta"]["generated_by"] == "scripts/scan_experiments.py"

    def test_idempotent(self, tmp_path: Path) -> None:
        out_path = tmp_path / "out.yaml"
        write_experiments_yaml(MOCK_RUNS, "dynunet_loss_variation_v2", out_path)
        content_1 = out_path.read_text(encoding="utf-8")
        write_experiments_yaml(MOCK_RUNS, "dynunet_loss_variation_v2", out_path)
        content_2 = out_path.read_text(encoding="utf-8")
        assert content_1 == content_2


# ---------------------------------------------------------------------------
# Smoke test: load_runs_from_parquet with empty dir (no actual DuckDB needed)
# ---------------------------------------------------------------------------


class TestLoadRunsFromParquet:
    def test_returns_empty_list_for_missing_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        runs = load_runs_from_parquet(missing)
        assert runs == []

    def test_returns_empty_list_for_empty_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "parquet"
        empty_dir.mkdir()
        runs = load_runs_from_parquet(empty_dir)
        assert runs == []


# ---------------------------------------------------------------------------
# Static analysis: verify no `import re` in script
# ---------------------------------------------------------------------------


class TestNoBannedImports:
    def test_no_import_re(self) -> None:
        """CLAUDE.md Rule #16: import re is BANNED."""
        script_path = Path("scripts/scan_experiments.py")
        if not script_path.exists():
            pytest.skip("Script not yet implemented (RED phase)")
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "re", (
                        "import re is BANNED (CLAUDE.md Rule #16)"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "re", "from re import ... is BANNED"

    def test_uses_pathlib(self) -> None:
        script_path = Path("scripts/scan_experiments.py")
        if not script_path.exists():
            pytest.skip("Script not yet implemented (RED phase)")
        source = script_path.read_text(encoding="utf-8")
        assert "pathlib" in source, "Script must use pathlib.Path"
