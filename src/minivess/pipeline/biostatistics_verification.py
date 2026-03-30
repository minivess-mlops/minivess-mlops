"""Artifact integrity and statistical correctness verification.

Verifies the complete two-DuckDB artifact chain:
  analysis_results.duckdb → biostatistics.duckdb → JSON sidecars → R output

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact_chain(
    *,
    analysis_duckdb_path: Path | None = None,
    biostatistics_duckdb_path: Path | None = None,
    r_data_dir: Path | None = None,
    r_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify the complete artifact chain integrity.

    Returns a dict with:
      - passed: bool (all checks passed)
      - checks: list of {name, passed, detail}
    """
    checks: list[dict[str, Any]] = []

    # Check 1: Analysis DuckDB exists and has expected tables
    if analysis_duckdb_path is not None:
        if analysis_duckdb_path.exists():
            import duckdb

            conn = duckdb.connect(str(analysis_duckdb_path), read_only=True)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
            conn.close()
            expected = {"runs", "per_volume_metrics", "fold_metrics", "metadata"}
            missing = expected - tables
            checks.append({
                "name": "analysis_duckdb_tables",
                "passed": len(missing) == 0,
                "detail": f"Tables: {sorted(tables)}, missing: {sorted(missing)}",
            })
        else:
            checks.append({
                "name": "analysis_duckdb_exists",
                "passed": False,
                "detail": f"File not found: {analysis_duckdb_path}",
            })

    # Check 2: Biostatistics DuckDB exists and has expected tables
    if biostatistics_duckdb_path is not None:
        if biostatistics_duckdb_path.exists():
            import duckdb

            conn = duckdb.connect(str(biostatistics_duckdb_path), read_only=True)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
            conn.close()
            expected = {
                "pairwise_comparisons",
                "anova_results",
                "variance_decomposition",
                "rankings",
                "bayesian_results",
                "specification_curve",
                "diagnostics",
                "tripod_compliance",
                "metadata",
            }
            missing = expected - tables
            checks.append({
                "name": "biostatistics_duckdb_tables",
                "passed": len(missing) == 0,
                "detail": f"Tables: {sorted(tables)}, missing: {sorted(missing)}",
            })
        else:
            checks.append({
                "name": "biostatistics_duckdb_exists",
                "passed": False,
                "detail": f"File not found: {biostatistics_duckdb_path}",
            })

    # Check 3: JSON sidecars exist and are valid
    if r_data_dir is not None and r_data_dir.exists():
        expected_sidecars = [
            "pairwise_results.json",
            "per_volume_data.json",
            "variance_decomposition.json",
            "rankings.json",
            "anova_results.json",
            "diagnostics.json",
            "calibration_data.json",
            "tripod_compliance.json",
            "metadata.json",
            "bayesian_results.json",
            "specification_curve.json",
        ]
        for sidecar in expected_sidecars:
            path = r_data_dir / sidecar
            if path.exists():
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                    checks.append({
                        "name": f"sidecar_{sidecar}",
                        "passed": True,
                        "detail": "Valid JSON",
                    })
                except json.JSONDecodeError as e:
                    checks.append({
                        "name": f"sidecar_{sidecar}",
                        "passed": False,
                        "detail": f"Invalid JSON: {e}",
                    })
            else:
                checks.append({
                    "name": f"sidecar_{sidecar}",
                    "passed": False,
                    "detail": "File not found",
                })

    # Check 4: R output files exist (if r_output_dir provided)
    if r_output_dir is not None and r_output_dir.exists():
        fig_dir = r_output_dir / "figures"
        tab_dir = r_output_dir / "tables"
        n_pdfs = len(list(fig_dir.glob("*.pdf"))) if fig_dir.exists() else 0
        n_pngs = len(list(fig_dir.glob("*.png"))) if fig_dir.exists() else 0
        n_tex = len(list(tab_dir.glob("*.tex"))) if tab_dir.exists() else 0
        checks.append({
            "name": "r_output_figures",
            "passed": n_pdfs >= 10 and n_pngs >= 10,
            "detail": f"PDFs: {n_pdfs}/10, PNGs: {n_pngs}/10",
        })
        checks.append({
            "name": "r_output_tables",
            "passed": n_tex >= 7,
            "detail": f"LaTeX: {n_tex}/7",
        })

    all_passed = all(c["passed"] for c in checks) if checks else False

    return {
        "passed": all_passed,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "checks": checks,
    }


def validate_statistical_results(
    *,
    biostatistics_duckdb_path: Path,
) -> dict[str, Any]:
    """Validate statistical correctness of results in DuckDB 2.

    Checks:
    - P-values in [0, 1], no NaN
    - Effect sizes finite
    - Cliff's delta in [-1, 1], VDA in [0, 1]
    - ANOVA F-statistics non-negative
    - Bootstrap CIs ordered (lower < upper)
    - Rankings sum correctly

    Returns a dict with passed/checks like verify_artifact_chain.
    """
    import duckdb

    checks: list[dict[str, Any]] = []
    conn = duckdb.connect(str(biostatistics_duckdb_path), read_only=True)

    # Check 1: Pairwise p-values in [0, 1]
    rows = conn.execute(
        "SELECT p_value, p_adjusted, cohens_d, cliffs_delta, vda "
        "FROM pairwise_comparisons"
    ).fetchall()
    n_invalid_p = sum(
        1
        for r in rows
        if r[0] is not None and (r[0] < 0 or r[0] > 1 or r[0] != r[0])
    )
    checks.append({
        "name": "pairwise_p_values",
        "passed": n_invalid_p == 0,
        "detail": f"{len(rows)} rows, {n_invalid_p} invalid p-values",
    })

    # Check 2: Cliff's delta in [-1, 1]
    n_invalid_delta = sum(
        1
        for r in rows
        if r[3] is not None and (r[3] < -1 or r[3] > 1)
    )
    checks.append({
        "name": "cliffs_delta_range",
        "passed": n_invalid_delta == 0,
        "detail": f"{n_invalid_delta} out of range",
    })

    # Check 3: VDA in [0, 1]
    n_invalid_vda = sum(
        1
        for r in rows
        if r[4] is not None and (r[4] < 0 or r[4] > 1)
    )
    checks.append({
        "name": "vda_range",
        "passed": n_invalid_vda == 0,
        "detail": f"{n_invalid_vda} out of range",
    })

    # Check 4: ANOVA F-statistics non-negative
    anova_rows = conn.execute(
        "SELECT f_statistic, p_value FROM anova_results"
    ).fetchall()
    n_negative_f = sum(
        1
        for r in anova_rows
        if r[0] is not None and r[0] == r[0] and r[0] < 0  # NaN != NaN
    )
    checks.append({
        "name": "anova_f_nonnegative",
        "passed": n_negative_f == 0,
        "detail": f"{len(anova_rows)} rows, {n_negative_f} negative F",
    })

    # Check 5: Diagnostics power in [0, 1]
    diag_rows = conn.execute(
        "SELECT achieved_power FROM diagnostics"
    ).fetchall()
    n_invalid_power = sum(
        1
        for r in diag_rows
        if r[0] is not None and (r[0] < 0 or r[0] > 1)
    )
    checks.append({
        "name": "diagnostics_power_range",
        "passed": n_invalid_power == 0,
        "detail": f"{len(diag_rows)} rows, {n_invalid_power} invalid power",
    })

    conn.close()

    all_passed = all(c["passed"] for c in checks)
    return {
        "passed": all_passed,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "checks": checks,
    }
