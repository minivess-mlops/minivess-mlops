"""Multi-metric ranking for the biostatistics flow.

Computes per-metric rankings and mean ranks across metrics
using the Demsar (2006) rank-then-aggregate approach.

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from minivess.pipeline.biostatistics_types import RankingResult

logger = logging.getLogger(__name__)

# Type alias: metric_name -> {condition -> {fold -> scores}}
MultiMetricData = dict[str, dict[str, dict[int, np.ndarray]]]


def compute_rankings(
    per_volume_data: MultiMetricData,
    metric_names: list[str],
    higher_is_better: dict[str, bool],
) -> list[RankingResult]:
    """Compute per-metric rankings using mean scores.

    For each metric, ranks conditions by mean score (rank 1 = best).
    Lower rank is better. Ties receive average rank.

    Parameters
    ----------
    per_volume_data:
        {metric_name: {condition: {fold: scores_array}}}.
    metric_names:
        List of metrics to rank.
    higher_is_better:
        {metric_name: bool} — True if higher values are better.

    Returns
    -------
    List of RankingResult, one per metric.
    """
    results: list[RankingResult] = []

    for metric in metric_names:
        if metric not in per_volume_data:
            continue

        metric_data = per_volume_data[metric]
        conditions = sorted(metric_data.keys())

        # Compute mean score per condition
        mean_scores: dict[str, float] = {}
        for cond in conditions:
            fold_data = metric_data[cond]
            all_scores = np.concatenate(
                [fold_data[k] for k in sorted(fold_data.keys())]
            )
            mean_scores[cond] = float(np.mean(all_scores))

        # Rank (higher is better => negate for ascending rank)
        direction = higher_is_better.get(metric, True)
        values = [mean_scores[c] for c in conditions]
        if direction:
            # Higher is better: negate so that highest gets rank 1
            ranks = stats.rankdata([-v for v in values], method="average")
        else:
            # Lower is better: rank directly
            ranks = stats.rankdata(values, method="average")

        condition_ranks = {c: float(r) for c, r in zip(conditions, ranks, strict=True)}
        mean_ranks = condition_ranks.copy()  # For single metric, mean_ranks == ranks

        # Critical difference (Demsar 2006)
        cd_value = _critical_difference(len(conditions), len(values))

        results.append(
            RankingResult(
                metric=metric,
                condition_ranks=condition_ranks,
                mean_ranks=mean_ranks,
                cd_value=cd_value,
            )
        )

    return results


def _critical_difference(k: int, n: int) -> float | None:
    """Compute Nemenyi critical difference for k treatments and n subjects.

    CD = q_alpha * sqrt(k*(k+1) / (6*n))

    Uses q_alpha values from Demsar (2006) Table 5 for alpha=0.05.
    """
    # q_alpha values for alpha=0.05 (Nemenyi)
    q_values = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }

    if k < 2 or k not in q_values:
        return None

    q_alpha = q_values[k]
    return float(q_alpha * np.sqrt(k * (k + 1) / (6 * n)))
