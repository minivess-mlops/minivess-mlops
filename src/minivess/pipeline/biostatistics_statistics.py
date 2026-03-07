"""Statistical engine for the biostatistics flow.

Core statistical computations: pairwise comparisons, Bayesian signed-rank,
Friedman + ICC variance decomposition. All functions are PURE — they read
data dicts and return result dataclasses. Zero Prefect dependency.

References
----------
- Vargha and Delaney (2000) — Cliff's delta, VDA
- Benavoli et al. (2017) — Bayesian signed-rank with ROPE
- Demsar (2006) — Friedman + Nemenyi
- Koo and Li (2016) — ICC(2,1)
- Maier-Hein et al. (2024) — Two-tier MCC
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
from scipy import stats

from minivess.pipeline.biostatistics_types import (
    PairwiseResult,
    VarianceDecompositionResult,
)
from minivess.pipeline.comparison import (
    cliffs_delta,
    cohens_d,
    holm_bonferroni_correction,
    vargha_delaney_a,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type alias: condition -> fold -> per-volume scores
# ---------------------------------------------------------------------------
PerVolumeData = dict[str, dict[int, np.ndarray]]


# ---------------------------------------------------------------------------
# Task 3.2: Pairwise comparisons
# ---------------------------------------------------------------------------


def compute_pairwise_comparisons(
    per_volume_data: PerVolumeData,
    metric_name: str,
    alpha: float = 0.05,
    primary_metric: str = "val_dice",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> list[PairwiseResult]:
    """Compute pairwise statistical comparisons between conditions.

    For each pair of conditions:
    1. Pool per-volume scores across folds
    2. Wilcoxon signed-rank test
    3. Cohen's d, Cliff's delta, VDA
    4. Two-tier MCC: Holm for primary metric, BH-FDR for secondary

    Parameters
    ----------
    per_volume_data:
        {condition: {fold: per_volume_scores_array}}.
    metric_name:
        Name of the metric being compared.
    alpha:
        Significance level.
    primary_metric:
        Primary metric name (gets Holm correction; others get BH-FDR).
    n_bootstrap:
        Bootstrap resamples (unused here — reserved for bootstrap CI).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of PairwiseResult, one per unordered pair.
    """
    conditions = sorted(per_volume_data.keys())
    pairs = list(itertools.combinations(conditions, 2))

    is_primary = metric_name == primary_metric
    correction_method = "holm" if is_primary else "bh_fdr"

    # Compute raw p-values and effect sizes
    raw_results: list[dict[str, Any]] = []
    raw_p_values: list[float] = []

    for cond_a, cond_b in pairs:
        scores_a = _pool_scores(per_volume_data[cond_a])
        scores_b = _pool_scores(per_volume_data[cond_b])

        # Align lengths (use minimum)
        n = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:n]
        scores_b = scores_b[:n]

        # Wilcoxon signed-rank test
        try:
            stat_result = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
            p_value = float(stat_result.pvalue)
        except ValueError:
            # All differences are zero
            p_value = 1.0

        # Effect sizes
        d = cohens_d(scores_a, scores_b)
        cd = cliffs_delta(scores_a, scores_b)
        vda = vargha_delaney_a(scores_a, scores_b)

        raw_results.append(
            {
                "condition_a": cond_a,
                "condition_b": cond_b,
                "p_value": p_value,
                "cohens_d": d,
                "cliffs_delta": cd,
                "vda": vda,
            }
        )
        raw_p_values.append(p_value)

    # Apply multiple comparison correction
    if is_primary:
        adjusted = holm_bonferroni_correction(raw_p_values, alpha)
    else:
        adjusted = _bh_fdr_correction(raw_p_values, alpha)

    results: list[PairwiseResult] = []
    for i, raw in enumerate(raw_results):
        p_adj, significant = adjusted[i]
        results.append(
            PairwiseResult(
                condition_a=raw["condition_a"],
                condition_b=raw["condition_b"],
                metric=metric_name,
                p_value=raw["p_value"],
                p_adjusted=p_adj,
                correction_method=correction_method,
                significant=significant,
                cohens_d=raw["cohens_d"],
                cliffs_delta=raw["cliffs_delta"],
                vda=raw["vda"],
            )
        )

    return results


# ---------------------------------------------------------------------------
# Task 3.3: Bayesian signed-rank with ROPE
# ---------------------------------------------------------------------------


def compute_bayesian_comparisons(
    per_volume_data: PerVolumeData,
    metric_name: str,
    rope: float = 0.01,
) -> list[PairwiseResult]:
    """Compute Bayesian signed-rank comparisons with ROPE.

    Uses baycomp.SignedRankTest.probs() for P(A>B), P(rope), P(B>A).
    Gracefully skips if baycomp is not installed.

    Parameters
    ----------
    per_volume_data:
        {condition: {fold: per_volume_scores_array}}.
    metric_name:
        Name of the metric.
    rope:
        Region of practical equivalence width.

    Returns
    -------
    List of PairwiseResult with bayesian fields populated.
    """
    conditions = sorted(per_volume_data.keys())
    pairs = list(itertools.combinations(conditions, 2))

    results: list[PairwiseResult] = []
    for cond_a, cond_b in pairs:
        scores_a = _pool_scores(per_volume_data[cond_a])
        scores_b = _pool_scores(per_volume_data[cond_b])

        n = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:n]
        scores_b = scores_b[:n]

        b_left, b_rope, b_right = _bayesian_signed_rank(scores_a, scores_b, rope)

        results.append(
            PairwiseResult(
                condition_a=cond_a,
                condition_b=cond_b,
                metric=metric_name,
                p_value=0.0,
                p_adjusted=0.0,
                correction_method="bayesian",
                significant=False,
                cohens_d=0.0,
                cliffs_delta=0.0,
                vda=0.5,
                bayesian_left=b_left,
                bayesian_rope=b_rope,
                bayesian_right=b_right,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Task 3.4: Variance decomposition (Friedman + ICC)
# ---------------------------------------------------------------------------


def compute_variance_decomposition(
    per_volume_data: PerVolumeData,
    metric_name: str,
    friedman_alpha: float = 0.05,
) -> list[VarianceDecompositionResult]:
    """Compute Friedman test + Nemenyi post-hoc + ICC(2,1).

    Parameters
    ----------
    per_volume_data:
        {condition: {fold: per_volume_scores_array}}.
    metric_name:
        Name of the metric.
    friedman_alpha:
        Significance level for Friedman test (Nemenyi post-hoc only if significant).

    Returns
    -------
    List with one VarianceDecompositionResult.
    """
    conditions = sorted(per_volume_data.keys())

    # Pool scores per condition (across folds)
    pooled = {c: _pool_scores(per_volume_data[c]) for c in conditions}

    # Align to minimum length
    min_n = min(len(v) for v in pooled.values())
    aligned = [pooled[c][:min_n] for c in conditions]

    # Friedman test
    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(*aligned)
        friedman_stat = float(friedman_stat)
        friedman_p = float(friedman_p)
    except ValueError:
        friedman_stat = 0.0
        friedman_p = 1.0

    # Nemenyi post-hoc (only if Friedman is significant)
    nemenyi_matrix = None
    if friedman_p < friedman_alpha:
        nemenyi_matrix = _compute_nemenyi(aligned, conditions)

    # ICC(2,1)
    icc_value, icc_ci_lower, icc_ci_upper = _compute_icc(aligned, conditions)

    power_caveat = len(conditions) <= 5

    return [
        VarianceDecompositionResult(
            metric=metric_name,
            friedman_statistic=friedman_stat,
            friedman_p=friedman_p,
            nemenyi_matrix=nemenyi_matrix,
            icc_value=icc_value,
            icc_ci_lower=icc_ci_lower,
            icc_ci_upper=icc_ci_upper,
            icc_type="ICC2",
            power_caveat=power_caveat,
        )
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pool_scores(fold_data: dict[int, np.ndarray]) -> np.ndarray:
    """Pool per-volume scores across all folds into a single array."""
    arrays = [fold_data[k] for k in sorted(fold_data.keys())]
    return np.concatenate(arrays)


def _bh_fdr_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Benjamini-Hochberg FDR correction.

    Uses statsmodels.stats.multitest.multipletests for battle-tested reliability.
    """
    if not p_values:
        return []

    from statsmodels.stats.multitest import multipletests

    reject, pvals_adj, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return list(zip(pvals_adj.tolist(), reject.tolist(), strict=True))


def _bayesian_signed_rank(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    rope: float,
) -> tuple[float | None, float | None, float | None]:
    """Bayesian signed-rank test via baycomp.

    Returns (P(A>B), P(rope), P(B>A)), or (None, None, None) if baycomp unavailable.
    """
    try:
        from baycomp import SignedRankTest

        probs = SignedRankTest.probs(scores_a, scores_b, rope=rope)
        # baycomp returns (p_left, p_rope, p_right) = (P(A>B), P(rope), P(B>A))
        return float(probs[0]), float(probs[1]), float(probs[2])
    except ImportError:
        logger.warning("baycomp not installed — skipping Bayesian signed-rank test")
        return None, None, None
    except Exception:
        logger.warning("Bayesian signed-rank failed", exc_info=True)
        return None, None, None


def _compute_nemenyi(
    aligned: list[np.ndarray],
    conditions: list[str],
) -> dict[str, dict[str, float]]:
    """Compute Nemenyi post-hoc test p-value matrix."""
    try:
        import scikit_posthocs as sp

        # scikit-posthocs expects a list of arrays for posthoc_nemenyi_friedman
        result = sp.posthoc_nemenyi_friedman(np.column_stack(aligned))
        matrix: dict[str, dict[str, float]] = {}
        for i, cond_i in enumerate(conditions):
            matrix[cond_i] = {}
            for j, cond_j in enumerate(conditions):
                matrix[cond_i][cond_j] = float(result.iloc[i, j])
        return matrix
    except ImportError:
        logger.warning("scikit-posthocs not installed — skipping Nemenyi")
        return {}
    except Exception:
        logger.warning("Nemenyi post-hoc failed", exc_info=True)
        return {}


def _compute_icc(
    aligned: list[np.ndarray],
    conditions: list[str],
) -> tuple[float, float, float]:
    """Compute ICC(2,1) using pingouin.

    Returns (icc_value, ci_lower, ci_upper).
    """
    try:
        import pandas as pd
        import pingouin as pg

        # Build long-format DataFrame for pingouin
        rows = []
        n_subjects = len(aligned[0])
        for rater_idx, (_cond, scores) in enumerate(
            zip(conditions, aligned, strict=True)
        ):
            for subj_idx in range(n_subjects):
                rows.append(
                    {
                        "Subject": subj_idx,
                        "Rater": rater_idx,
                        "Score": float(scores[subj_idx]),
                    }
                )
        df = pd.DataFrame(rows)

        icc_df = pg.intraclass_corr(
            data=df, targets="Subject", raters="Rater", ratings="Score"
        )
        # ICC2 = "ICC2" type in pingouin
        icc2_row = icc_df[icc_df["Type"] == "ICC2"]
        if icc2_row.empty:
            return 0.0, 0.0, 0.0

        icc_val = float(icc2_row["ICC"].iloc[0])
        ci = icc2_row["CI95%"].iloc[0]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
        return icc_val, ci_lower, ci_upper
    except ImportError:
        logger.warning("pingouin not installed — skipping ICC")
        return 0.0, 0.0, 0.0
    except Exception:
        logger.warning("ICC computation failed", exc_info=True)
        return 0.0, 0.0, 0.0
