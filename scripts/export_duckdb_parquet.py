"""Export MLflow experiment data to DuckDB + Parquet for analysis.

Creates a DuckDB database from MLflow filesystem data and exports
tables as Parquet files for downstream use in dashboards, notebooks,
and paper generation.

Run:
    uv run python scripts/export_duckdb_parquet.py
    uv run python scripts/export_duckdb_parquet.py --output-dir outputs/duckdb
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

EXPERIMENTS = {
    "dynunet_loss_variation_v2": "843896622863223169",
    "dynunet_half_width_v1": "859817033030295110",
    "minivess_evaluation": "309410943942924473",
}


def main() -> int:
    """Export experiment data to DuckDB and Parquet."""
    import argparse

    parser = argparse.ArgumentParser(description="Export to DuckDB + Parquet")
    parser.add_argument(
        "--output-dir", default="outputs/duckdb", help="Output directory"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DUCKDB + PARQUET EXPORT")
    logger.info("=" * 70)

    from minivess.pipeline.duckdb_extraction import extract_runs_to_duckdb

    parquet_dir = output_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    total_tables = 0
    total_rows = 0

    for exp_name, exp_id in EXPERIMENTS.items():
        exp_dir = MLRUNS_DIR / exp_id
        if not exp_dir.exists():
            logger.warning("  Skipping %s — directory not found", exp_name)
            continue

        logger.info("\n[%s] (ID: %s)", exp_name, exp_id)

        # Create DuckDB file
        db_path = output_dir / f"{exp_name}.duckdb"
        try:
            db = extract_runs_to_duckdb(MLRUNS_DIR, exp_id, db_path=db_path)
        except Exception:
            logger.exception("  Failed to extract %s", exp_name)
            continue

        logger.info("  DuckDB: %s", db_path)

        # Export each table as Parquet
        tables = ["runs", "params", "eval_metrics", "training_metrics", "champion_tags"]
        for table_name in tables:
            try:
                result = db.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                count = result[0] if result else 0

                if count > 0:
                    pq_path = parquet_dir / f"{exp_name}_{table_name}.parquet"
                    db.execute(f"COPY {table_name} TO '{pq_path}' (FORMAT PARQUET)")
                    logger.info(
                        "    %s: %d rows → %s",
                        table_name,
                        count,
                        pq_path.name,
                    )
                    total_tables += 1
                    total_rows += count
                else:
                    logger.info("    %s: 0 rows (skipped)", table_name)
            except Exception:
                logger.exception("    Failed to export %s", table_name)

        # Print sample queries
        try:
            logger.info("\n  Sample query — eval metrics summary:")
            result = db.execute("""
                SELECT metric_name,
                       AVG(point_estimate) as mean,
                       COUNT(*) as n
                FROM eval_metrics
                GROUP BY metric_name
                ORDER BY metric_name
            """).fetchall()
            for row in result:
                logger.info("    %s: mean=%.4f (n=%d)", row[0], row[1], row[2])
        except Exception:
            logger.debug("  Sample query failed (empty tables)")

        db.close()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    all_files = list(output_dir.rglob("*"))
    artifact_files = [f for f in all_files if f.is_file()]
    logger.info("  Total files:  %d", len(artifact_files))
    logger.info("  Total tables: %d", total_tables)
    logger.info("  Total rows:   %d", total_rows)
    for f in sorted(artifact_files):
        size_kb = f.stat().st_size / 1024
        logger.info("    %s (%.1f KB)", f.relative_to(output_dir), size_kb)

    logger.info("\nDUCKDB + PARQUET EXPORT COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
