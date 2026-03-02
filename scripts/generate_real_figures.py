"""Generate paper-quality figures and LaTeX tables from real experiment data.

Reads eval metrics from mlruns filesystem, builds a ComparisonTable,
and generates all registered figures + LaTeX comparison table.

Run:
    uv run python scripts/generate_real_figures.py
    uv run python scripts/generate_real_figures.py --output-dir outputs/figures
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Experiment IDs
EXPERIMENTS = {
    "dynunet_loss_variation_v2": "843896622863223169",
    "dynunet_half_width_v1": "859817033030295110",
}

N_FOLDS = 3
EVAL_METRICS = ["dsc", "centreline_dsc", "measured_masd"]


def read_metric(exp_id: str, run_id: str, metric_name: str) -> float:
    """Read the last metric value from MLflow filesystem."""
    metric_file = MLRUNS_DIR / exp_id / run_id / "metrics" / metric_name
    if not metric_file.exists():
        return float("nan")
    lines = metric_file.read_text(encoding="utf-8").strip().split("\n")
    if lines:
        parts = lines[-1].split()
        if len(parts) >= 2:
            return float(parts[1])
    return float("nan")


def read_tag(exp_id: str, run_id: str, tag_key: str) -> str:
    """Read a tag value from MLflow filesystem."""
    tag_file = MLRUNS_DIR / exp_id / run_id / "tags" / tag_key
    if not tag_file.exists():
        return ""
    return tag_file.read_text(encoding="utf-8").strip()


def find_complete_runs(exp_id: str) -> list[str]:
    """Find all run directories with complete 3-fold evaluation."""
    exp_dir = MLRUNS_DIR / exp_id
    if not exp_dir.exists():
        return []
    complete: list[str] = []
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "meta.yaml").exists():
            continue
        metrics_dir = run_dir / "metrics"
        if metrics_dir.exists():
            has_fold2 = any(
                f.name.startswith("eval_fold2") for f in metrics_dir.iterdir()
            )
            if has_fold2:
                complete.append(run_dir.name)
    return complete


def build_comparison_table_from_mlruns(
    exp_id: str,
    experiment_name: str,
) -> tuple[object, dict[str, dict[str, list[float]]]]:
    """Build a ComparisonTable from real MLflow metrics.

    Returns (ComparisonTable, raw_data_dict).
    """
    from minivess.pipeline.comparison import ComparisonTable, LossResult, MetricSummary

    runs = find_complete_runs(exp_id)
    logger.info("  Found %d complete runs in %s", len(runs), experiment_name)

    losses: list[LossResult] = []
    raw_data: dict[str, dict[str, list[float]]] = {}

    for run_id in runs:
        loss_name = read_tag(exp_id, run_id, "loss_function")
        if not loss_name:
            continue

        # Read per-fold metrics
        fold_data: dict[str, list[float]] = {}
        for metric in EVAL_METRICS:
            values = []
            for fold in range(N_FOLDS):
                val = read_metric(exp_id, run_id, f"eval_fold{fold}_{metric}")
                values.append(val)
            fold_data[metric] = values

        raw_data[loss_name] = fold_data

        # Build MetricSummary per metric
        metrics: dict[str, MetricSummary] = {}
        for metric in EVAL_METRICS:
            values = fold_data[metric]
            valid = [v for v in values if not math.isnan(v)]
            if not valid:
                continue
            arr = np.array(valid)
            mean_val = float(arr.mean())
            std_val = float(arr.std()) if len(arr) > 1 else 0.0
            metrics[metric] = MetricSummary(
                mean=mean_val,
                std=std_val,
                ci_lower=mean_val - 1.96 * std_val,
                ci_upper=mean_val + 1.96 * std_val,
                per_fold=values,
            )

        losses.append(
            LossResult(
                loss_name=loss_name,
                num_folds=N_FOLDS,
                metrics=metrics,
            )
        )

    table = ComparisonTable(
        losses=losses,
        metric_names=EVAL_METRICS,
    )
    return table, raw_data


def main() -> int:
    """Generate figures and tables from real experiment data."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate real figures + LaTeX")
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("FIGURE + TABLE GENERATION FROM REAL DATA")
    logger.info("=" * 70)

    # 1. Build ComparisonTable from primary experiment
    logger.info("\n[1/4] Building ComparisonTable from dynunet_loss_variation_v2")
    table, raw_data = build_comparison_table_from_mlruns(
        EXPERIMENTS["dynunet_loss_variation_v2"],
        "dynunet_loss_variation_v2",
    )

    # Print raw data summary
    for loss_name, metrics in raw_data.items():
        logger.info("\n  %s:", loss_name)
        for metric_name, values in metrics.items():
            logger.info(
                "    %s: %s (mean=%.4f)",
                metric_name,
                [f"{v:.4f}" for v in values],
                np.nanmean(values),
            )

    # 2. Generate markdown comparison table
    logger.info("\n[2/4] Generating comparison tables")
    from minivess.pipeline.comparison import (
        format_comparison_latex,
        format_comparison_markdown,
    )

    md = format_comparison_markdown(table)
    md_path = output_dir / "comparison_table.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("  Markdown: %s", md_path)

    latex = format_comparison_latex(table)
    tex_path = output_dir / "comparison_table.tex"
    tex_path.write_text(latex, encoding="utf-8")
    logger.info("  LaTeX:    %s", tex_path)

    # Print both for verification
    logger.info("\n  --- Markdown Preview ---")
    for line in md.split("\n")[:20]:
        logger.info("  %s", line)

    logger.info("\n  --- LaTeX Preview ---")
    for line in latex.split("\n")[:20]:
        logger.info("  %s", line)

    # 3. Generate all figures
    logger.info("\n[3/4] Generating figures")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    from minivess.pipeline.viz.generate_all_figures import generate_all_figures

    summary = generate_all_figures(
        output_dir=figures_dir,
        comparison_table=table,
        formats=["png", "svg"],
    )

    logger.info("  Succeeded: %s", summary["succeeded"])
    logger.info("  Failed:    %s", summary["failed"])

    # Verify files exist
    for fmt in ["png", "svg"]:
        files = list(figures_dir.glob(f"*.{fmt}"))
        logger.info("  %s files: %d", fmt.upper(), len(files))
        for f in files:
            logger.info("    %s (%.1f KB)", f.name, f.stat().st_size / 1024)

    # 4. Also build half-width comparison if available
    logger.info("\n[4/4] Checking half-width experiment")
    hw_exp_id = EXPERIMENTS.get("dynunet_half_width_v1")
    if hw_exp_id:
        hw_runs = find_complete_runs(hw_exp_id)
        if hw_runs:
            hw_table, hw_raw = build_comparison_table_from_mlruns(
                hw_exp_id,
                "dynunet_half_width_v1",
            )
            hw_md = format_comparison_markdown(hw_table)
            hw_md_path = output_dir / "comparison_half_width.md"
            hw_md_path.write_text(hw_md, encoding="utf-8")
            logger.info("  Half-width table: %s", hw_md_path)

            hw_latex = format_comparison_latex(hw_table)
            hw_tex_path = output_dir / "comparison_half_width.tex"
            hw_tex_path.write_text(hw_latex, encoding="utf-8")
            logger.info("  Half-width LaTeX: %s", hw_tex_path)
        else:
            logger.info("  No complete half-width runs found")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    all_files = list(output_dir.rglob("*"))
    artifact_files = [f for f in all_files if f.is_file()]
    logger.info("  Total artifacts: %d files", len(artifact_files))
    for f in sorted(artifact_files):
        logger.info("    %s", f.relative_to(output_dir))

    logger.info("\nFIGURE GENERATION COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
