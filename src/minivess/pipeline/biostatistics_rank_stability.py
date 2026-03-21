"""Rank stability analysis for the biostatistics flow.

Computes Kendall's tau rank concordance between metrics to detect rank
inversions — a key finding for tubular structure segmentation where DSC
and clDice often disagree on model ranking.

The clDice vs DSC rank inversion IS a paper finding (demonstrates why
MetricsReloaded-aligned metric selection matters).

Pure functions — no Prefect, no Docker dependency.

References
----------
- Maier-Hein et al. (2024). "Metrics Reloaded." *Nature Methods*.
- Kendall (1938). "A new measure of rank correlation." *Biometrika*.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TauPair:
    """Kendall's tau for one pair of metrics."""

    metric_a: str
    metric_b: str
    tau: float
    p_value: float
    concordant: bool  # True if tau > 0 (rank agreement)


@dataclass
class RankConcordanceResult:
    """Complete rank concordance analysis."""

    tau_matrix: list[TauPair]
    condition_ranks: dict[str, dict[str, float]]  # {metric: {condition: rank}}
    n_inversions: int  # Number of metric pairs with negative tau
    n_pairs: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_rank_concordance(
    per_volume_data: dict[str, dict[str, dict[int, np.ndarray]]],
    metric_names: list[str],
    higher_is_better: dict[str, bool],
) -> RankConcordanceResult:
    """Compute Kendall's tau rank concordance between all metric pairs.

    Parameters
    ----------
    per_volume_data:
        ``{metric: {condition_key: {fold_id: scores}}}``
    metric_names:
        Metrics to include.
    higher_is_better:
        Direction for each metric.

    Returns
    -------
    RankConcordanceResult with tau matrix and per-metric rankings.
    """
    # Step 1: Compute mean score per condition per metric
    condition_means: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        if metric not in per_volume_data:
            continue
        condition_means[metric] = {}
        for cond, folds in per_volume_data[metric].items():
            all_scores = np.concatenate([folds[k] for k in sorted(folds.keys())])
            mean = float(np.mean(all_scores))
            # Negate for "lower is better" so higher rank = better
            if not higher_is_better.get(metric, True):
                mean = -mean
            condition_means[metric][cond] = mean

    # Step 2: Rank conditions per metric (rank 1 = best)
    condition_ranks: dict[str, dict[str, float]] = {}
    for metric, means in condition_means.items():
        conditions = sorted(means.keys())
        values = [means[c] for c in conditions]
        # rankdata: 1=smallest, so negate for "higher is better"
        ranks = stats.rankdata([-v for v in values], method="average")
        condition_ranks[metric] = dict(
            zip(conditions, [float(r) for r in ranks], strict=True)
        )

    # Step 3: Compute Kendall's tau for all metric pairs
    tau_pairs: list[TauPair] = []
    available_metrics = sorted(condition_means.keys())
    pairs = list(itertools.combinations(available_metrics, 2))

    for metric_a, metric_b in pairs:
        # Get ranks for shared conditions
        shared_conditions = sorted(
            set(condition_ranks[metric_a].keys())
            & set(condition_ranks[metric_b].keys())
        )
        if len(shared_conditions) < 2:
            continue

        ranks_a = [condition_ranks[metric_a][c] for c in shared_conditions]
        ranks_b = [condition_ranks[metric_b][c] for c in shared_conditions]

        tau, p_value = stats.kendalltau(ranks_a, ranks_b)
        tau_pairs.append(
            TauPair(
                metric_a=metric_a,
                metric_b=metric_b,
                tau=float(tau),
                p_value=float(p_value),
                concordant=float(tau) > 0,
            )
        )

    n_inversions = sum(1 for tp in tau_pairs if tp.tau < 0)

    return RankConcordanceResult(
        tau_matrix=tau_pairs,
        condition_ranks=condition_ranks,
        n_inversions=n_inversions,
        n_pairs=len(tau_pairs),
    )
