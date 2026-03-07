"""Lineage manifest generation for the biostatistics flow.

Builds a JSON-serializable lineage manifest with provenance info,
source fingerprint, git commit, and artifact inventory.

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from minivess.pipeline.biostatistics_types import (
        FigureArtifact,
        SourceRunManifest,
        TableArtifact,
    )

logger = logging.getLogger(__name__)


def build_lineage_manifest(
    manifest: SourceRunManifest,
    figures: list[FigureArtifact],
    tables: list[TableArtifact],
) -> dict[str, Any]:
    """Build a lineage manifest for the biostatistics flow run.

    Parameters
    ----------
    manifest:
        Source run manifest with fingerprint.
    figures:
        Generated figures.
    tables:
        Generated tables.

    Returns
    -------
    JSON-serializable dict with lineage information.
    """
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "fingerprint": manifest.fingerprint,
        "git_commit": _get_git_commit(),
        "n_source_runs": len(manifest.runs),
        "source_experiments": _get_source_experiments(manifest),
        "artifacts_produced": {
            "n_figures": len(figures),
            "n_tables": len(tables),
            "figure_ids": [f.figure_id for f in figures],
            "table_ids": [t.table_id for t in tables],
        },
        "statistical_methods": [
            "Wilcoxon signed-rank (scipy.stats.wilcoxon)",
            "Holm-Bonferroni correction (primary metric)",
            "Benjamini-Hochberg FDR (secondary metrics)",
            "Cohen's d (paired, pooled SD)",
            "Cliff's delta (non-parametric)",
            "Vargha-Delaney A (common language effect size)",
            "Bayesian signed-rank with ROPE (baycomp)",
            "Friedman test (scipy.stats.friedmanchisquare)",
            "Nemenyi post-hoc (scikit_posthocs)",
            "ICC(2,1) (pingouin.intraclass_corr)",
        ],
    }


def _get_git_commit() -> str:
    """Get current git commit hash, or empty string if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _get_source_experiments(manifest: SourceRunManifest) -> list[str]:
    """Extract unique experiment names from the manifest."""
    return sorted({r.experiment_name for r in manifest.runs})
