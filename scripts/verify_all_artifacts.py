"""Verify all pipeline artifacts exist and are valid.

Checks every artifact from all phases (P0-P7) of the quasi-E2E
debugging pipeline. Validates file sizes, loadability, and content.

Run:
    uv run python scripts/verify_all_artifacts.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Minimum file sizes (bytes) for validation
MIN_JSON_SIZE = 50
MIN_MD_SIZE = 100
MIN_TEX_SIZE = 50
MIN_PNG_SIZE = 1000
MIN_SVG_SIZE = 500
MIN_PARQUET_SIZE = 200
MIN_DUCKDB_SIZE = 1000
MIN_ONNX_SIZE = 10000


def check_file(
    path: Path,
    *,
    min_size: int = 0,
    category: str = "unknown",
    root: Path | None = None,
) -> tuple[bool, str]:
    """Check a single file exists and meets size threshold."""
    display_root = root or OUTPUTS_DIR
    if not path.exists():
        return False, f"MISSING: {path}"
    size = path.stat().st_size
    if size < min_size:
        return False, f"TOO SMALL ({size}B < {min_size}B): {path}"
    try:
        rel = path.relative_to(display_root)
    except ValueError:
        rel = path
    return True, f"OK ({size:,}B): {rel}"


def validate_json(path: Path) -> tuple[bool, str]:
    """Validate JSON file is parseable."""
    if not path.exists():
        return False, f"MISSING: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            return False, f"EMPTY JSON: {path}"
        return True, f"Valid JSON ({len(str(data)):,} chars)"
    except json.JSONDecodeError as e:
        return False, f"INVALID JSON: {e}"


def validate_parquet(path: Path) -> tuple[bool, str]:
    """Validate Parquet file is loadable."""
    if not path.exists():
        return False, f"MISSING: {path}"
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return True, f"Valid Parquet ({table.num_rows} rows, {table.num_columns} cols)"
    except Exception as e:
        return False, f"INVALID Parquet: {e}"


def validate_duckdb(path: Path) -> tuple[bool, str]:
    """Validate DuckDB file has tables."""
    if not path.exists():
        return False, f"MISSING: {path}"
    try:
        import duckdb

        conn = duckdb.connect(str(path), read_only=True)
        tables = conn.execute("SHOW TABLES").fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        return (
            True,
            f"Valid DuckDB ({len(table_names)} tables: {', '.join(table_names)})",
        )
    except Exception as e:
        return False, f"INVALID DuckDB: {e}"


def validate_onnx(path: Path) -> tuple[bool, str]:
    """Validate ONNX model file."""
    if not path.exists():
        return False, f"MISSING: {path}"
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        return True, (
            f"Valid ONNX (inputs={[i.name for i in inputs]}, "
            f"outputs={[o.name for o in outputs]})"
        )
    except Exception as e:
        return False, f"INVALID ONNX: {e}"


def validate_png(path: Path) -> tuple[bool, str]:
    """Validate PNG file header."""
    if not path.exists():
        return False, f"MISSING: {path}"
    try:
        data = path.read_bytes()
        # PNG magic bytes
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return False, f"INVALID PNG header: {path}"
        return True, f"Valid PNG ({len(data):,} bytes)"
    except Exception as e:
        return False, f"INVALID PNG: {e}"


def main() -> int:
    """Run full artifact verification."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify all pipeline artifacts")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root output directory",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()

    logger.info("=" * 70)
    logger.info("ARTIFACT VERIFICATION")
    logger.info("=" * 70)
    logger.info("  Root: %s", output_root)

    results: dict[str, list[tuple[bool, str]]] = {}
    total_ok = 0
    total_fail = 0

    # ── Data Flow artifacts ──
    data_checks: list[tuple[bool, str]] = []
    data_dir = output_root / "data_flow"
    for name in [
        "data_flow_summary.json",
        "data_provenance.json",
        "splits_summary.json",
    ]:
        ok, msg = check_file(data_dir / name, min_size=MIN_JSON_SIZE, category="data")
        data_checks.append((ok, msg))
        if ok:
            jok, jmsg = validate_json(data_dir / name)
            data_checks.append((jok, f"  {jmsg}"))
    results["Data Flow"] = data_checks

    # ── Analysis Flow artifacts ──
    analysis_checks: list[tuple[bool, str]] = []
    analysis_dir = output_root / "analysis"

    # Figures
    figures_dir = analysis_dir / "figures"
    for fig_name in [
        "loss_comparison",
        "fold_heatmap",
        "metric_correlation",
        "sensitivity_heatmap",
    ]:
        for ext in ["png", "svg"]:
            fpath = figures_dir / f"{fig_name}.{ext}"
            min_size = MIN_PNG_SIZE if ext == "png" else MIN_SVG_SIZE
            ok, msg = check_file(fpath, min_size=min_size, category="analysis")
            analysis_checks.append((ok, msg))
            if ok and ext == "png":
                pok, pmsg = validate_png(fpath)
                analysis_checks.append((pok, f"  {pmsg}"))

    # Tables
    for table_name in ["comparison_table.md", "comparison_table.tex"]:
        ok, msg = check_file(
            analysis_dir / table_name, min_size=MIN_TEX_SIZE, category="analysis"
        )
        analysis_checks.append((ok, msg))

    # Half-width tables
    for table_name in ["comparison_half_width.md", "comparison_half_width.tex"]:
        ok, msg = check_file(
            analysis_dir / table_name, min_size=MIN_TEX_SIZE, category="analysis"
        )
        analysis_checks.append((ok, msg))

    results["Analysis Flow"] = analysis_checks

    # ── Cross-experiment comparison ──
    comparison_checks: list[tuple[bool, str]] = []
    comp_dir = output_root / "comparison"
    for name in [
        "cross_experiment_comparison.png",
        "cross_experiment_comparison.svg",
        "cross_experiment_delta.md",
    ]:
        fpath = comp_dir / name
        min_size = (
            MIN_PNG_SIZE
            if name.endswith(".png")
            else MIN_SVG_SIZE
            if name.endswith(".svg")
            else MIN_MD_SIZE
        )
        ok, msg = check_file(fpath, min_size=min_size, category="comparison")
        comparison_checks.append((ok, msg))
    results["Cross-Experiment Comparison"] = comparison_checks

    # ── DuckDB + Parquet artifacts ──
    duckdb_checks: list[tuple[bool, str]] = []
    duckdb_dir = output_root / "duckdb"
    parquet_dir = duckdb_dir / "parquet"

    for db_name in [
        "dynunet_loss_variation_v2.duckdb",
        "dynunet_half_width_v1.duckdb",
        "minivess_evaluation.duckdb",
    ]:
        fpath = duckdb_dir / db_name
        ok, msg = check_file(fpath, min_size=MIN_DUCKDB_SIZE, category="duckdb")
        duckdb_checks.append((ok, msg))
        if ok:
            dok, dmsg = validate_duckdb(fpath)
            duckdb_checks.append((dok, f"  {dmsg}"))

    # Parquet files
    if parquet_dir.exists():
        for pq_file in sorted(parquet_dir.glob("*.parquet")):
            ok, msg = check_file(pq_file, min_size=MIN_PARQUET_SIZE, category="parquet")
            duckdb_checks.append((ok, msg))
            if ok:
                pok, pmsg = validate_parquet(pq_file)
                duckdb_checks.append((pok, f"  {pmsg}"))
    results["DuckDB + Parquet"] = duckdb_checks

    # ── Deploy Flow artifacts ──
    deploy_checks: list[tuple[bool, str]] = []
    deploy_dir = output_root / "deploy"
    onnx_dir = deploy_dir / "onnx"
    if onnx_dir.exists():
        onnx_files = sorted(onnx_dir.glob("*.onnx"))
        for onnx_file in onnx_files:
            ok, msg = check_file(onnx_file, min_size=MIN_ONNX_SIZE, category="deploy")
            deploy_checks.append((ok, msg))
            if ok:
                ook, omsg = validate_onnx(onnx_file)
                deploy_checks.append((ook, f"  {omsg}"))
    else:
        deploy_checks.append((False, f"MISSING directory: {onnx_dir}"))
    results["Deploy Flow"] = deploy_checks

    # ── Dashboard Flow artifacts ──
    dashboard_checks: list[tuple[bool, str]] = []
    dashboard_dir = output_root / "dashboard"
    for name in [
        "everything_dashboard_report.md",
        "everything_dashboard_metadata.json",
    ]:
        fpath = dashboard_dir / name
        min_size = MIN_MD_SIZE if name.endswith(".md") else MIN_JSON_SIZE
        ok, msg = check_file(fpath, min_size=min_size, category="dashboard")
        dashboard_checks.append((ok, msg))
        if ok and name.endswith(".json"):
            jok, jmsg = validate_json(fpath)
            dashboard_checks.append((jok, f"  {jmsg}"))
    results["Dashboard Flow"] = dashboard_checks

    # ── Pipeline artifacts (P7) ──
    pipeline_checks: list[tuple[bool, str]] = []
    pipeline_dir = output_root / "pipeline"
    chain_path = pipeline_dir / "trigger_chain_results.json"
    ok, msg = check_file(chain_path, min_size=MIN_JSON_SIZE, category="pipeline")
    pipeline_checks.append((ok, msg))
    if ok:
        jok, jmsg = validate_json(chain_path)
        pipeline_checks.append((jok, f"  {jmsg}"))
        # Also verify chain status
        chain_data = json.loads(chain_path.read_text(encoding="utf-8"))
        all_success = chain_data.get("all_success", False)
        pipeline_checks.append((all_success, f"  Chain all_success={all_success}"))
    results["Pipeline Trigger Chain"] = pipeline_checks

    # ── Paper artifacts (P7-T3) ──
    paper_checks: list[tuple[bool, str]] = []
    paper_dir = output_root / "paper_artifacts"
    if paper_dir.exists():
        readme = paper_dir / "README.md"
        ok, msg = check_file(readme, min_size=MIN_MD_SIZE, category="paper")
        paper_checks.append((ok, msg))

        paper_fig_dir = paper_dir / "figures"
        if paper_fig_dir.exists():
            for f in sorted(paper_fig_dir.iterdir()):
                ok, msg = check_file(f, min_size=100, category="paper")
                paper_checks.append((ok, msg))

        paper_tbl_dir = paper_dir / "tables"
        if paper_tbl_dir.exists():
            for f in sorted(paper_tbl_dir.iterdir()):
                ok, msg = check_file(f, min_size=50, category="paper")
                paper_checks.append((ok, msg))
    else:
        paper_checks.append((False, f"MISSING directory: {paper_dir}"))
    results["Paper Artifacts"] = paper_checks

    # ── Print results ──
    logger.info("")
    for section, checks in results.items():
        section_ok = all(ok for ok, _ in checks)
        icon = "PASS" if section_ok else "FAIL"
        logger.info("[%4s] %s", icon, section)
        for ok, msg in checks:
            prefix = "  OK " if ok else "  FAIL"
            logger.info("  %s %s", prefix, msg)
            if ok:
                total_ok += 1
            else:
                total_fail += 1
        logger.info("")

    # ── Summary ──
    logger.info("=" * 70)
    logger.info("ARTIFACT INVENTORY")
    logger.info("=" * 70)

    # Count actual files per directory
    categories = {
        "Data Flow": output_root / "data_flow",
        "Analysis (figures)": output_root / "analysis" / "figures",
        "Analysis (tables)": output_root / "analysis",
        "Comparison": output_root / "comparison",
        "DuckDB databases": output_root / "duckdb",
        "Parquet files": output_root / "duckdb" / "parquet",
        "Deploy (ONNX)": output_root / "deploy" / "onnx",
        "Dashboard": output_root / "dashboard",
        "Pipeline": output_root / "pipeline",
        "Paper artifacts": output_root / "paper_artifacts",
    }

    total_files = 0
    for cat_name, cat_dir in categories.items():
        if cat_dir.exists():
            files = [f for f in cat_dir.iterdir() if f.is_file()]
            count = len(files)
            total_files += count
            exts: dict[str, int] = {}
            for f in files:
                ext = f.suffix or "(no ext)"
                exts[ext] = exts.get(ext, 0) + 1
            ext_str = ", ".join(f"{v}{k}" for k, v in sorted(exts.items()))
            logger.info("  %-25s %3d files  (%s)", cat_name, count, ext_str)
        else:
            logger.info("  %-25s   - (not found)", cat_name)

    logger.info("  %s", "-" * 55)
    logger.info("  %-25s %3d files total", "TOTAL", total_files)

    logger.info("\n  Checks passed: %d", total_ok)
    logger.info("  Checks failed: %d", total_fail)
    logger.info(
        "\n  STATUS: %s", "ALL VERIFIED" if total_fail == 0 else "SOME FAILURES"
    )

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
