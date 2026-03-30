"""JSON sidecar models for biostatistics figures and tables.

Every figure and table produced by the biostatistics pipeline has a
JSON sidecar containing the exact data, parameters, and provenance
needed to reproduce it. Pydantic models enforce schema.

Phase 2 council decision: sidecars are the Python-R bridge contract.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class FigureSidecar(BaseModel):
    """JSON sidecar for a biostatistics figure."""

    figure_id: str = Field(..., description="Unique figure identifier (e.g., 'F4_forest_plot')")
    title: str = Field(..., description="Human-readable title")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp of generation",
    )
    duckdb_sha256: str = Field(default="", description="SHA-256 hash of input DuckDB file")
    git_sha: str = Field(default="", description="Git commit SHA at generation time")
    config_hash: str = Field(default="", description="Hash of BiostatisticsConfig used")
    data: dict[str, Any] = Field(default_factory=dict, description="Figure-specific data payload")
    output_files: list[str] = Field(
        default_factory=list,
        description="Paths to generated files (PDF, PNG) relative to output_dir",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (R session info, library versions, etc.)",
    )


class TableSidecar(BaseModel):
    """JSON sidecar for a biostatistics LaTeX table."""

    table_id: str = Field(..., description="Unique table identifier (e.g., 'T1_comparison')")
    title: str = Field(..., description="Human-readable title")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp of generation",
    )
    duckdb_sha256: str = Field(default="", description="SHA-256 hash of input DuckDB file")
    git_sha: str = Field(default="", description="Git commit SHA at generation time")
    data: dict[str, Any] = Field(default_factory=dict, description="Underlying table data")
    output_files: list[str] = Field(
        default_factory=list,
        description="Paths to generated files (.tex) relative to output_dir",
    )


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sidecar(sidecar: FigureSidecar | TableSidecar, output_path: Path) -> Path:
    """Write a sidecar to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sidecar.model_dump(), f, indent=2, default=str)
    return output_path


def load_sidecar(path: Path) -> dict[str, Any]:
    """Load a sidecar JSON file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)
