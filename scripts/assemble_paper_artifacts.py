"""Assemble paper-ready artifacts into outputs/paper_artifacts/.

Copies all figures (PNG+SVG), tables (TEX), and data exports
into a single directory with a README containing LaTeX inclusion
commands for academic submission.

Run:
    uv run python scripts/assemble_paper_artifacts.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main() -> int:
    """Assemble paper artifacts."""
    import argparse

    parser = argparse.ArgumentParser(description="Assemble paper artifacts")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_artifacts",
        help="Paper artifacts directory",
    )
    args = parser.parse_args()

    paper_dir = Path(args.output_dir)
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    data_dir = paper_dir / "data"

    for d in [paper_dir, figures_dir, tables_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ASSEMBLING PAPER ARTIFACTS")
    logger.info("=" * 70)

    copied_files: list[str] = []

    # ── Figures ──
    logger.info("\n[1/4] Copying figures")
    figure_sources = {
        # Analysis figures
        "fig1_loss_comparison": OUTPUTS_DIR
        / "analysis"
        / "figures"
        / "loss_comparison",
        "fig2_fold_heatmap": OUTPUTS_DIR / "analysis" / "figures" / "fold_heatmap",
        "fig3_metric_correlation": OUTPUTS_DIR
        / "analysis"
        / "figures"
        / "metric_correlation",
        "fig4_sensitivity_heatmap": OUTPUTS_DIR
        / "analysis"
        / "figures"
        / "sensitivity_heatmap",
        # Cross-experiment comparison
        "fig5_cross_experiment": OUTPUTS_DIR
        / "comparison"
        / "cross_experiment_comparison",
    }

    for fig_name, source_base in figure_sources.items():
        for ext in ["png", "svg"]:
            source = source_base.with_suffix(f".{ext}")
            if source.exists():
                dest = figures_dir / f"{fig_name}.{ext}"
                shutil.copy2(source, dest)
                copied_files.append(str(dest.relative_to(paper_dir)))
                logger.info("  %s", dest.relative_to(paper_dir))

    # ── Tables ──
    logger.info("\n[2/4] Copying tables")
    table_sources = {
        "table1_main_comparison.tex": OUTPUTS_DIR / "analysis" / "comparison_table.tex",
        "table1_main_comparison.md": OUTPUTS_DIR / "analysis" / "comparison_table.md",
        "table2_half_width.tex": OUTPUTS_DIR / "analysis" / "comparison_half_width.tex",
        "table2_half_width.md": OUTPUTS_DIR / "analysis" / "comparison_half_width.md",
        "table3_cross_experiment_delta.md": OUTPUTS_DIR
        / "comparison"
        / "cross_experiment_delta.md",
    }

    for table_name, source in table_sources.items():
        if source.exists():
            dest = tables_dir / table_name
            shutil.copy2(source, dest)
            copied_files.append(str(dest.relative_to(paper_dir)))
            logger.info("  %s", dest.relative_to(paper_dir))

    # ── Supplementary data ──
    logger.info("\n[3/4] Copying supplementary data")
    parquet_dir = OUTPUTS_DIR / "duckdb" / "parquet"
    if parquet_dir.exists():
        for pq_file in sorted(parquet_dir.glob("*.parquet")):
            dest = data_dir / pq_file.name
            shutil.copy2(pq_file, dest)
            copied_files.append(str(dest.relative_to(paper_dir)))
            logger.info("  %s", dest.relative_to(paper_dir))

    # ── README ──
    logger.info("\n[4/4] Generating README.md")
    readme_content = _generate_readme(figures_dir, tables_dir, data_dir)
    readme_path = paper_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    copied_files.append("README.md")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PAPER ARTIFACTS ASSEMBLED")
    logger.info("=" * 70)
    logger.info("  Directory: %s", paper_dir)
    logger.info("  Total files: %d", len(copied_files))
    logger.info(
        "  Figures: %d",
        len([f for f in copied_files if f.startswith("figures/")]),
    )
    logger.info(
        "  Tables: %d",
        len([f for f in copied_files if f.startswith("tables/")]),
    )
    logger.info(
        "  Data: %d",
        len([f for f in copied_files if f.startswith("data/")]),
    )

    return 0


def _generate_readme(
    figures_dir: Path,
    tables_dir: Path,
    data_dir: Path,
) -> str:
    """Generate README.md with LaTeX inclusion commands."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Paper Artifacts — MiniVess MLOps",
        "",
        f"Generated: {timestamp}",
        "",
        "## Figures",
        "",
        "| Figure | Description | Formats |",
        "| --- | --- | --- |",
        "| fig1_loss_comparison | Cross-loss performance comparison (4 losses, 3 folds, 100 epochs) | PNG, SVG |",
        "| fig2_fold_heatmap | Per-fold metric heatmap showing fold stability | PNG, SVG |",
        "| fig3_metric_correlation | Metric correlation matrix (DSC vs clDSC vs MASD) | PNG, SVG |",
        "| fig4_sensitivity_heatmap | Loss sensitivity analysis across metrics | PNG, SVG |",
        "| fig5_cross_experiment | Full-width vs half-width DynUNet comparison | PNG, SVG |",
        "",
        "### LaTeX Figure Inclusion",
        "",
        "```latex",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=\\textwidth]{figures/fig1_loss_comparison.png}",
        "  \\caption{Cross-loss comparison of 4 compound loss functions on MiniVess.",
        "    Mean and standard deviation across 3 folds, 100 epochs.}",
        "  \\label{fig:loss-comparison}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.8\\textwidth]{figures/fig2_fold_heatmap.png}",
        "  \\caption{Per-fold metric heatmap showing cross-validation stability.}",
        "  \\label{fig:fold-heatmap}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.7\\textwidth]{figures/fig3_metric_correlation.png}",
        "  \\caption{Pairwise metric correlation: DSC, centreline DSC, and MASD.}",
        "  \\label{fig:metric-correlation}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.8\\textwidth]{figures/fig4_sensitivity_heatmap.png}",
        "  \\caption{Loss function sensitivity across evaluation metrics.}",
        "  \\label{fig:sensitivity}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=\\textwidth]{figures/fig5_cross_experiment.png}",
        "  \\caption{Full-width (128 filters) vs half-width (64 filters) DynUNet.}",
        "  \\label{fig:cross-experiment}",
        "\\end{figure}",
        "```",
        "",
        "## Tables",
        "",
        "| Table | Description | Formats |",
        "| --- | --- | --- |",
        "| table1_main_comparison | Main loss comparison (dynunet_loss_variation_v2) | TEX, MD |",
        "| table2_half_width | Half-width ablation results (dynunet_half_width_v1) | TEX, MD |",
        "| table3_cross_experiment_delta | Full vs half width metric deltas | MD |",
        "",
        "### LaTeX Table Inclusion",
        "",
        "```latex",
        "\\input{tables/table1_main_comparison.tex}",
        "\\input{tables/table2_half_width.tex}",
        "```",
        "",
        "## Supplementary Data",
        "",
        "Parquet files for reproducibility and extended analysis:",
        "",
    ]

    if data_dir.exists():
        for pq_file in sorted(data_dir.glob("*.parquet")):
            lines.append(f"- `{pq_file.name}`")

    lines.extend(
        [
            "",
            "### Loading Parquet Data",
            "",
            "```python",
            "import pandas as pd",
            "",
            "# Load evaluation metrics",
            'df = pd.read_parquet("data/dynunet_loss_variation_v2_eval_metrics.parquet")',
            "print(df.describe())",
            "```",
            "",
            "## Source",
            "",
            "Generated by `scripts/assemble_paper_artifacts.py` from the",
            "quasi-E2E debugging pipeline (`test/quasi-e2e-debugging` branch).",
            "All data from real MiniVess experiments (70 volumes, 4 losses,",
            "3 folds, 100 epochs).",
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
