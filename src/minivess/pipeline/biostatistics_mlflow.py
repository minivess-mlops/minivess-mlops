"""MLflow run logging for the biostatistics flow.

Creates a single MLflow run in the 'minivess_biostatistics' experiment
with all biostatistics artifacts, tags, and lineage.

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import mlflow

if TYPE_CHECKING:
    from minivess.pipeline.biostatistics_types import (
        FigureArtifact,
        SourceRunManifest,
        TableArtifact,
    )

logger = logging.getLogger(__name__)


def log_biostatistics_run(
    manifest: SourceRunManifest,
    lineage: dict[str, Any],
    figures: list[FigureArtifact],
    tables: list[TableArtifact],
    db_path: Any | None = None,
) -> str:
    """Log a biostatistics run to MLflow.

    Creates ONE run in 'minivess_biostatistics' experiment with:
    - Tags: upstream_fingerprint, n_source_runs, n_conditions, n_figures, n_tables
    - Artifacts: lineage JSON, figures, tables, DuckDB

    Parameters
    ----------
    manifest:
        Source run manifest.
    lineage:
        Lineage manifest dict.
    figures:
        Generated figures.
    tables:
        Generated tables.
    db_path:
        Optional path to DuckDB file to log as artifact.

    Returns
    -------
    MLflow run ID.
    """
    mlflow.set_experiment("minivess_biostatistics")

    with mlflow.start_run() as run:
        run_id: str = str(run.info.run_id)

        # Tags
        mlflow.set_tag("upstream_fingerprint", manifest.fingerprint)
        mlflow.set_tag("n_source_runs", str(len(manifest.runs)))

        conditions = {r.loss_function for r in manifest.runs}
        mlflow.set_tag("n_conditions", str(len(conditions)))
        mlflow.set_tag("n_figures", str(len(figures)))
        mlflow.set_tag("n_tables", str(len(tables)))

        # Log lineage as JSON param (truncated if needed)
        lineage_str = json.dumps(lineage, default=str)
        if len(lineage_str) <= 500:
            mlflow.log_param("lineage_summary", lineage_str)

        # Log figure artifacts
        for fig in figures:
            for path in fig.paths:
                if path.exists():
                    mlflow.log_artifact(str(path), "figures")
            if fig.sidecar_path is not None and fig.sidecar_path.exists():
                mlflow.log_artifact(str(fig.sidecar_path), "sidecars")

        # Log table artifacts
        for tab in tables:
            if tab.path.exists():
                mlflow.log_artifact(str(tab.path), "tables")

        # Log DuckDB
        if db_path is not None:
            from pathlib import Path

            db_file = Path(str(db_path))
            if db_file.exists():
                mlflow.log_artifact(str(db_file), "duckdb")

        logger.info("Logged biostatistics run %s to MLflow", run_id)
        return run_id
