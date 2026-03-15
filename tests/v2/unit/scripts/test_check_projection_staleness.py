"""Tests for scripts/check_projection_staleness.py (T24).

RED phase: tests written before implementation.

Validates:
- Staleness detection: KG file newer than .tex output → stale
- Fresh detection: .tex output newer than KG file → fresh
- Missing .tex files reported in 'missing_tex' category
- JSON report schema: { stale, fresh, missing_tex }
- Exit code semantics: 0 = all fresh, 1 = ≥1 stale
- No import re (CLAUDE.md Rule #16)
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the staleness checker (will fail in RED phase)
# ---------------------------------------------------------------------------
from scripts.check_projection_staleness import (  # noqa: E402
    ProjectionStatus,
    StalenessReport,
    check_all_projections,
    check_projection,
    load_projections,
    write_staleness_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_PROJECTIONS_YAML = """\
projections:
  - id: test_proj_1
    output: methods/methods-03.tex
    depends_on:
      code_structure: [flows.yaml]
    status: scaffold_only

  - id: test_proj_2
    output: results/results-01.tex
    depends_on:
      experiments: [dynunet_loss_variation_v2.yaml]
    status: scaffold_only
"""


@pytest.fixture
def projections_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "projections.yaml"
    p.write_text(MINIMAL_PROJECTIONS_YAML, encoding="utf-8")
    return p


@pytest.fixture
def kg_dir(tmp_path: Path) -> Path:
    d = tmp_path / "knowledge-graph"
    d.mkdir()
    # Create a fresh code_structure source file
    (d / "flows.yaml").write_text(
        "_meta:\n  last_updated: '2026-03-15'\n", encoding="utf-8"
    )
    (d / "dynunet_loss_variation_v2.yaml").write_text(
        "_meta:\n  last_updated: '2026-03-15'\n", encoding="utf-8"
    )
    return d


# ---------------------------------------------------------------------------
# Unit tests: load_projections
# ---------------------------------------------------------------------------


class TestLoadProjections:
    def test_loads_valid_yaml(self, projections_yaml: Path) -> None:
        projections = load_projections(projections_yaml)
        assert len(projections) == 2

    def test_projection_has_id_and_output(self, projections_yaml: Path) -> None:
        projections = load_projections(projections_yaml)
        for p in projections:
            assert "id" in p
            assert "output" in p

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_projections(missing)


# ---------------------------------------------------------------------------
# Unit tests: check_projection
# ---------------------------------------------------------------------------


class TestCheckProjection:
    def test_missing_output_is_missing(self, tmp_path: Path, kg_dir: Path) -> None:
        projection = {
            "id": "proj_missing",
            "output": str(tmp_path / "missing.tex"),
            "depends_on": {"code_structure": ["flows.yaml"]},
        }
        status = check_projection(projection, kg_root=kg_dir, repo_root=tmp_path)
        assert status == ProjectionStatus.MISSING

    def test_stale_when_kg_newer(self, tmp_path: Path, kg_dir: Path) -> None:
        # Create old .tex file
        tex_file = tmp_path / "methods-03.tex"
        tex_file.write_text("old content", encoding="utf-8")
        # Make kg file newer than tex
        time.sleep(0.01)
        (kg_dir / "flows.yaml").write_text("updated", encoding="utf-8")

        projection = {
            "id": "proj_stale",
            "output": str(tex_file),
            "depends_on": {"code_structure": ["flows.yaml"]},
        }
        status = check_projection(projection, kg_root=kg_dir, repo_root=tmp_path)
        assert status == ProjectionStatus.STALE

    def test_fresh_when_tex_newer(self, tmp_path: Path, kg_dir: Path) -> None:
        # Create kg file first
        kg_file = kg_dir / "flows.yaml"
        kg_file.write_text("old kg", encoding="utf-8")
        # Create tex file afterwards (newer)
        time.sleep(0.01)
        tex_file = tmp_path / "methods-03.tex"
        tex_file.write_text("fresh content", encoding="utf-8")

        projection = {
            "id": "proj_fresh",
            "output": str(tex_file),
            "depends_on": {"code_structure": ["flows.yaml"]},
        }
        status = check_projection(projection, kg_root=kg_dir, repo_root=tmp_path)
        assert status == ProjectionStatus.FRESH

    def test_fresh_when_no_depends_on(self, tmp_path: Path, kg_dir: Path) -> None:
        tex_file = tmp_path / "methods-03.tex"
        tex_file.write_text("content", encoding="utf-8")
        projection = {
            "id": "proj_no_deps",
            "output": str(tex_file),
            "depends_on": {},
        }
        status = check_projection(projection, kg_root=kg_dir, repo_root=tmp_path)
        assert status == ProjectionStatus.FRESH


# ---------------------------------------------------------------------------
# Unit tests: check_all_projections
# ---------------------------------------------------------------------------


class TestCheckAllProjections:
    def test_returns_staleness_report(self, tmp_path: Path, kg_dir: Path) -> None:
        projections = [
            {
                "id": "p1",
                "output": str(tmp_path / "missing.tex"),
                "depends_on": {"code_structure": ["flows.yaml"]},
            }
        ]
        report = check_all_projections(projections, kg_root=kg_dir, repo_root=tmp_path)
        assert isinstance(report, StalenessReport)

    def test_missing_tex_categorised(self, tmp_path: Path, kg_dir: Path) -> None:
        projections = [
            {
                "id": "proj_missing",
                "output": str(tmp_path / "nonexistent.tex"),
                "depends_on": {},
            }
        ]
        report = check_all_projections(projections, kg_root=kg_dir, repo_root=tmp_path)
        assert "proj_missing" in report.missing_tex

    def test_all_fresh_returns_empty_stale(self, tmp_path: Path, kg_dir: Path) -> None:
        tex = tmp_path / "fresh.tex"
        (kg_dir / "flows.yaml").write_text("old", encoding="utf-8")
        time.sleep(0.01)
        tex.write_text("fresh", encoding="utf-8")

        projections = [
            {
                "id": "fresh_proj",
                "output": str(tex),
                "depends_on": {"code_structure": ["flows.yaml"]},
            }
        ]
        report = check_all_projections(projections, kg_root=kg_dir, repo_root=tmp_path)
        assert report.stale == []


# ---------------------------------------------------------------------------
# Integration: write_staleness_report
# ---------------------------------------------------------------------------


class TestWriteStalenessReport:
    def test_writes_json(self, tmp_path: Path) -> None:
        import json

        report = StalenessReport(stale=["p1"], fresh=["p2"], missing_tex=["p3"])
        out_path = tmp_path / "staleness.json"
        write_staleness_report(report, out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "stale" in data
        assert "fresh" in data
        assert "missing_tex" in data

    def test_exit_code_nonzero_when_stale(self) -> None:
        report = StalenessReport(stale=["p1"], fresh=[], missing_tex=[])
        assert report.has_issues()

    def test_exit_code_zero_when_all_fresh(self) -> None:
        report = StalenessReport(stale=[], fresh=["p1"], missing_tex=[])
        assert not report.has_issues()

    def test_missing_tex_counts_as_issue(self) -> None:
        report = StalenessReport(stale=[], fresh=[], missing_tex=["p1"])
        assert report.has_issues()


# ---------------------------------------------------------------------------
# Static analysis: no import re
# ---------------------------------------------------------------------------


class TestNoBannedImports:
    def test_no_import_re(self) -> None:
        script_path = Path("scripts/check_projection_staleness.py")
        if not script_path.exists():
            pytest.skip("Not yet implemented (RED phase)")
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "re", "import re BANNED (Rule #16)"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "re", "from re import BANNED"

    def test_uses_pathlib(self) -> None:
        script_path = Path("scripts/check_projection_staleness.py")
        if not script_path.exists():
            pytest.skip("Not yet implemented (RED phase)")
        assert "pathlib" in script_path.read_text(encoding="utf-8")
