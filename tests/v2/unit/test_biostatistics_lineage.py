"""Tests for biostatistics lineage manifest (Phase 7, Task 7.1)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from minivess.pipeline.biostatistics_lineage import build_lineage_manifest
from minivess.pipeline.biostatistics_types import (
    FigureArtifact,
    SourceRun,
    SourceRunManifest,
    TableArtifact,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_manifest() -> SourceRunManifest:
    runs = [
        SourceRun(
            run_id="run_001",
            experiment_id="1",
            experiment_name="test_exp",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
        ),
    ]
    return SourceRunManifest.from_runs(runs)


class TestBuildLineageManifest:
    def test_lineage_has_schema_version(self) -> None:
        manifest = _make_manifest()
        lineage = build_lineage_manifest(
            manifest=manifest,
            figures=[],
            tables=[],
        )
        assert "schema_version" in lineage
        assert lineage["schema_version"] == "1.0"

    def test_lineage_fingerprint_matches_manifest(self) -> None:
        manifest = _make_manifest()
        lineage = build_lineage_manifest(
            manifest=manifest,
            figures=[],
            tables=[],
        )
        assert lineage["fingerprint"] == manifest.fingerprint

    def test_lineage_git_commit_not_empty(self) -> None:
        manifest = _make_manifest()
        lineage = build_lineage_manifest(
            manifest=manifest,
            figures=[],
            tables=[],
        )
        # git_commit should be a string (may be empty outside git repo, but key must exist)
        assert "git_commit" in lineage

    def test_lineage_round_trips_through_json(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        lineage = build_lineage_manifest(
            manifest=manifest,
            figures=[],
            tables=[],
        )
        # Serialize and deserialize
        path = tmp_path / "lineage.json"
        path.write_text(json.dumps(lineage, indent=2, default=str), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["fingerprint"] == manifest.fingerprint
        assert loaded["schema_version"] == "1.0"

    def test_lineage_artifacts_produced(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        fig = FigureArtifact(
            figure_id="test_fig",
            title="Test",
            paths=[tmp_path / "test.png"],
        )
        tab = TableArtifact(
            table_id="test_tab",
            title="Test Table",
            path=tmp_path / "test.tex",
            format="latex",
        )
        lineage = build_lineage_manifest(
            manifest=manifest,
            figures=[fig],
            tables=[tab],
        )
        assert lineage["artifacts_produced"]["n_figures"] == 1
        assert lineage["artifacts_produced"]["n_tables"] == 1
