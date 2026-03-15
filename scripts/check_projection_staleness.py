"""Check staleness of projections.yaml downstream .tex files.

Reads knowledge-graph/manuscript/projections.yaml and compares each projection's
depends_on KG file modification times against the downstream .tex file mtime.

Reports:
  stale       — KG source newer than .tex output (needs regeneration)
  fresh       — .tex output newer than all KG sources
  missing_tex — .tex output does not exist yet

Exit code: 0 = all fresh, 1 = ≥1 stale or missing

Usage:
    uv run python scripts/check_projection_staleness.py
    uv run python scripts/check_projection_staleness.py --projections knowledge-graph/manuscript/projections.yaml

CLAUDE.md Rule #16: import re is BANNED.
CLAUDE.md Rule #6:  Use pathlib.Path throughout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_PROJECTIONS = REPO_ROOT / "knowledge-graph" / "manuscript" / "projections.yaml"
DEFAULT_KG_ROOT = REPO_ROOT / "knowledge-graph"


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


class ProjectionStatus(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"


@dataclass
class StalenessReport:
    stale: list[str] = field(default_factory=list)
    fresh: list[str] = field(default_factory=list)
    missing_tex: list[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        """Return True if any projections are stale or missing."""
        return bool(self.stale or self.missing_tex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stale": self.stale,
            "fresh": self.fresh,
            "missing_tex": self.missing_tex,
            "has_issues": self.has_issues(),
            "summary": {
                "n_stale": len(self.stale),
                "n_fresh": len(self.fresh),
                "n_missing": len(self.missing_tex),
            },
        }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_projections(projections_path: Path) -> list[dict[str, Any]]:
    """Load projection entries from a projections.yaml file.

    Args:
        projections_path: Path to projections.yaml

    Returns:
        List of projection dicts (id, output, depends_on, ...).

    Raises:
        FileNotFoundError: if projections_path does not exist.
    """
    if not projections_path.exists():
        raise FileNotFoundError(f"Projections file not found: {projections_path}")
    data = yaml.safe_load(projections_path.read_text(encoding="utf-8"))
    return data.get("projections", [])


def _collect_kg_source_paths(depends_on: dict[str, Any], kg_root: Path) -> list[Path]:
    """Collect the actual file Paths for all KG source references in depends_on.

    Args:
        depends_on: The depends_on dict from a projection entry.
        kg_root: Root of the knowledge-graph/ directory.

    Returns:
        List of existing Path objects. Non-existent paths are silently skipped.
    """
    paths: list[Path] = []
    for _category, refs in depends_on.items():
        if not isinstance(refs, list):
            continue
        for ref in refs:
            # refs may be bare filenames ("flows.yaml") or full paths
            candidate = kg_root / str(ref)
            if candidate.exists():
                paths.append(candidate)
    return paths


def check_projection(
    projection: dict[str, Any],
    kg_root: Path,
    repo_root: Path,
) -> ProjectionStatus:
    """Check whether a single projection is fresh, stale, or missing.

    Args:
        projection: A single projection dict (with 'output', 'depends_on').
        kg_root: Root of the knowledge-graph/ directory.
        repo_root: Root of the repository.

    Returns:
        ProjectionStatus enum value.
    """
    output_str: str = str(projection.get("output", ""))
    if not output_str:
        return ProjectionStatus.MISSING

    # Resolve output path — may be repo-relative or absolute
    output_path = Path(output_str)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    if not output_path.exists():
        return ProjectionStatus.MISSING

    tex_mtime = output_path.stat().st_mtime

    depends_on: dict[str, Any] = projection.get("depends_on", {})
    if not depends_on:
        return ProjectionStatus.FRESH

    kg_sources = _collect_kg_source_paths(depends_on, kg_root)
    if not kg_sources:
        return ProjectionStatus.FRESH

    # Stale if any KG source is newer than the .tex output
    latest_kg_mtime = max(p.stat().st_mtime for p in kg_sources)
    if latest_kg_mtime > tex_mtime:
        return ProjectionStatus.STALE

    return ProjectionStatus.FRESH


def check_all_projections(
    projections: list[dict[str, Any]],
    kg_root: Path,
    repo_root: Path,
) -> StalenessReport:
    """Check all projections and return a consolidated StalenessReport.

    Args:
        projections: List of projection dicts from load_projections().
        kg_root: Root of the knowledge-graph/ directory.
        repo_root: Root of the repository.

    Returns:
        StalenessReport with stale, fresh, missing_tex lists.
    """
    report = StalenessReport()
    for projection in projections:
        proj_id: str = str(projection.get("id", "unknown"))
        status = check_projection(projection, kg_root=kg_root, repo_root=repo_root)
        if status == ProjectionStatus.STALE:
            report.stale.append(proj_id)
        elif status == ProjectionStatus.MISSING:
            report.missing_tex.append(proj_id)
        else:
            report.fresh.append(proj_id)
    return report


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------


def write_staleness_report(report: StalenessReport, out_path: Path) -> None:
    """Write the staleness report as JSON.

    Args:
        report: StalenessReport to serialise.
        out_path: Destination JSON file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )


def _print_report(report: StalenessReport) -> None:
    if report.stale:
        print(f"STALE ({len(report.stale)}): {', '.join(report.stale)}")
    if report.missing_tex:
        print(f"MISSING ({len(report.missing_tex)}): {', '.join(report.missing_tex)}")
    if report.fresh:
        print(f"FRESH  ({len(report.fresh)}): {', '.join(report.fresh)}")
    if not report.has_issues():
        print("✓ All projections are up-to-date")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check projections.yaml staleness — exit 1 if any stale/missing"
    )
    parser.add_argument(
        "--projections",
        type=Path,
        default=DEFAULT_PROJECTIONS,
    )
    parser.add_argument(
        "--kg-root",
        type=Path,
        default=DEFAULT_KG_ROOT,
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional: write JSON report to this path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    projections = load_projections(args.projections)
    print(f"Checking {len(projections)} projections from {args.projections}")

    report = check_all_projections(
        projections,
        kg_root=args.kg_root,
        repo_root=REPO_ROOT,
    )
    _print_report(report)

    if args.report_out:
        write_staleness_report(report, args.report_out)
        print(f"Report written: {args.report_out}")

    return 1 if report.has_issues() else 0


if __name__ == "__main__":
    raise SystemExit(main())
