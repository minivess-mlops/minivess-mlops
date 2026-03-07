"""LaTeX table generation for the biostatistics flow.

Generates publication-quality LaTeX tables with booktabs formatting.
All output to a config-driven output_dir (Docker volume-mounted).

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from minivess.pipeline.biostatistics_types import (
    PairwiseResult,
    RankingResult,
    TableArtifact,
    VarianceDecompositionResult,
)

logger = logging.getLogger(__name__)


def generate_tables(
    pairwise: list[PairwiseResult],
    variance: list[VarianceDecompositionResult],
    rankings: list[RankingResult],
    output_dir: Path,
) -> list[TableArtifact]:
    """Generate all biostatistics LaTeX tables.

    Parameters
    ----------
    pairwise:
        Pairwise comparison results.
    variance:
        Variance decomposition results.
    rankings:
        Ranking results.
    output_dir:
        Directory for table outputs.

    Returns
    -------
    List of TableArtifact references.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: list[TableArtifact] = []

    # T1: Main comparison table
    if pairwise:
        t = _generate_comparison_table(pairwise, output_dir)
        tables.append(t)

    # T3: Effect sizes table
    if pairwise:
        t = _generate_effect_size_table(pairwise, output_dir)
        tables.append(t)

    # T4: Variance decomposition table
    if variance:
        t = _generate_variance_table(variance, output_dir)
        tables.append(t)

    # T5: Ranking summary table
    if rankings:
        t = _generate_ranking_table(rankings, output_dir)
        tables.append(t)

    logger.info("Generated %d LaTeX tables in %s", len(tables), output_dir)
    return tables


# ---------------------------------------------------------------------------
# Internal table generators
# ---------------------------------------------------------------------------


def _generate_comparison_table(
    pairwise: list[PairwiseResult],
    output_dir: Path,
) -> TableArtifact:
    """Generate main pairwise comparison table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pairwise Statistical Comparisons}",
        r"\label{tab:biostat_comparison}",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Condition A & Condition B & $p$ & $p_{\text{adj}}$ & Cohen's $d$ & Sig. \\",
        r"\midrule",
    ]

    for r in pairwise:
        sig_marker = "$^{*}$" if r.significant else ""
        lines.append(
            f"{r.condition_a} & {r.condition_b} & "
            f"{r.p_value:.4f} & {r.p_adjusted:.4f} & "
            f"{r.cohens_d:.3f} & {sig_marker} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    path = output_dir / "comparison_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")

    return TableArtifact(
        table_id="comparison_table",
        title="Pairwise Statistical Comparisons",
        path=path,
        format="latex",
    )


def _generate_effect_size_table(
    pairwise: list[PairwiseResult],
    output_dir: Path,
) -> TableArtifact:
    """Generate effect size comparison table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect Sizes}",
        r"\label{tab:biostat_effect_sizes}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Condition A & Condition B & Cohen's $d$ & Cliff's $\delta$ & VDA \\",
        r"\midrule",
    ]

    # Find max absolute values for bolding
    max_d = max(abs(r.cohens_d) for r in pairwise) if pairwise else 0
    max_cd = max(abs(r.cliffs_delta) for r in pairwise) if pairwise else 0

    for r in pairwise:
        d_str = (
            f"\\textbf{{{r.cohens_d:.3f}}}"
            if abs(r.cohens_d) == max_d
            else f"{r.cohens_d:.3f}"
        )
        cd_str = (
            f"\\textbf{{{r.cliffs_delta:.3f}}}"
            if abs(r.cliffs_delta) == max_cd
            else f"{r.cliffs_delta:.3f}"
        )
        lines.append(
            f"{r.condition_a} & {r.condition_b} & {d_str} & {cd_str} & {r.vda:.3f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    path = output_dir / "effect_size_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")

    return TableArtifact(
        table_id="effect_size_table",
        title="Effect Sizes",
        path=path,
        format="latex",
    )


def _generate_variance_table(
    variance: list[VarianceDecompositionResult],
    output_dir: Path,
) -> TableArtifact:
    """Generate variance decomposition table (Friedman + ICC)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Variance Decomposition}",
        r"\label{tab:biostat_variance}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & $\chi^2_F$ & $p_F$ & ICC(2,1) & 95\% CI \\",
        r"\midrule",
    ]

    for v in variance:
        lines.append(
            f"{v.metric} & {v.friedman_statistic:.2f} & "
            f"{v.friedman_p:.4f} & {v.icc_value:.3f} & "
            f"[{v.icc_ci_lower:.3f}, {v.icc_ci_upper:.3f}] \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    path = output_dir / "variance_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")

    return TableArtifact(
        table_id="variance_table",
        title="Variance Decomposition",
        path=path,
        format="latex",
    )


def _generate_ranking_table(
    rankings: list[RankingResult],
    output_dir: Path,
) -> TableArtifact:
    """Generate ranking summary table."""
    # Collect all conditions across metrics
    all_conditions: set[str] = set()
    for r in rankings:
        all_conditions.update(r.condition_ranks.keys())
    conditions = sorted(all_conditions)

    header_cols = " & ".join(f"\\textbf{{{c}}}" for c in conditions)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Multi-Metric Rankings}",
        r"\label{tab:biostat_rankings}",
        f"\\begin{{tabular}}{{l{'r' * len(conditions)}}}",
        r"\toprule",
        f"Metric & {header_cols} \\\\",
        r"\midrule",
    ]

    for r in rankings:
        ranks = [f"{r.condition_ranks.get(c, '-'):.1f}" for c in conditions]
        lines.append(f"{r.metric} & {' & '.join(ranks)} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    path = output_dir / "ranking_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")

    return TableArtifact(
        table_id="ranking_table",
        title="Multi-Metric Rankings",
        path=path,
        format="latex",
    )
