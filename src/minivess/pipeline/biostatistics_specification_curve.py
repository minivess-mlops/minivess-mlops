"""Specification curve analysis for the biostatistics flow.

Systematically varies ALL researcher degrees of freedom and demonstrates
how conclusions change across analytical choices. This is a key component
of the preregistration-equivalent analysis for the Nature Protocols paper.

Degrees of freedom explored:
- Metric choice (clDice, MASD, DSC, HD95, ...)
- Aggregation method (mean, median)
- Condition pairs (all pairwise comparisons)

The output is a sorted curve of effect sizes across all specifications,
with an indicator panel showing which choices were made for each.

References
----------
- Simonsohn et al. (2020). "Specification Curve Analysis." *Psychological Methods*.
- Simonsohn et al. (2015). "Specification curve: Descriptive and inferential
  statistics on all reasonable specifications."

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Specification:
    """A single analytical specification (one point on the curve).

    Each specification represents one combination of analytical choices:
    metric × aggregation × condition_pair.
    """

    metric: str
    aggregation: str  # "mean" or "median"
    condition_a: str
    condition_b: str
    effect_size: float  # Cohen's d (standardised mean difference)
    p_value: float  # Wilcoxon signed-rank p-value
    mean_diff: float  # Raw mean difference (A - B)
    significant: bool  # p < alpha after correction


@dataclass
class SpecificationCurveResult:
    """Complete specification curve analysis result."""

    specifications: list[Specification]
    median_effect: float
    fraction_significant: float
    permutation_p: float | None = None
    n_permutations: int = 0
    degrees_of_freedom: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_specification_curve(
    per_volume_data: dict[str, dict[str, dict[int, np.ndarray]]],
    factor_names: list[str],
    metric_names: list[str],
    higher_is_better: dict[str, bool],
    aggregation_methods: list[str] | None = None,
    alpha: float = 0.05,
    n_permutations: int = 0,
    seed: int = 42,
) -> SpecificationCurveResult:
    """Compute specification curve across all researcher degrees of freedom.

    Parameters
    ----------
    per_volume_data:
        ``{metric: {condition_key: {fold_id: scores}}}``
    factor_names:
        Factor names (for documentation/indicator panel).
    metric_names:
        Metrics to include in the curve.
    higher_is_better:
        Direction for each metric.
    aggregation_methods:
        Aggregation methods to vary. Default: ``["mean", "median"]``.
    alpha:
        Significance level for BH-FDR correction.
    n_permutations:
        Number of permutations for the joint test. 0 = skip.
    seed:
        Random seed.

    Returns
    -------
    SpecificationCurveResult sorted by effect size.
    """
    if aggregation_methods is None:
        aggregation_methods = ["mean", "median"]

    # Generate all specifications
    specs: list[Specification] = []

    for metric in metric_names:
        if metric not in per_volume_data:
            continue

        conditions = sorted(per_volume_data[metric].keys())
        pairs = list(itertools.combinations(conditions, 2))

        for agg in aggregation_methods:
            for cond_a, cond_b in pairs:
                spec = _compute_single_specification(
                    per_volume_data[metric],
                    metric=metric,
                    aggregation=agg,
                    condition_a=cond_a,
                    condition_b=cond_b,
                    higher_is_better=higher_is_better.get(metric, True),
                )
                specs.append(spec)

    # Apply BH-FDR correction across all specifications
    if specs:
        p_values = [s.p_value for s in specs]
        corrected = _bh_fdr_correction(p_values, alpha)
        corrected_specs: list[Specification] = []
        for spec, (_p_adj, significant) in zip(specs, corrected, strict=True):
            corrected_specs.append(
                Specification(
                    metric=spec.metric,
                    aggregation=spec.aggregation,
                    condition_a=spec.condition_a,
                    condition_b=spec.condition_b,
                    effect_size=spec.effect_size,
                    p_value=spec.p_value,
                    mean_diff=spec.mean_diff,
                    significant=significant,
                )
            )
        specs = corrected_specs

    # Sort by effect size
    specs.sort(key=lambda s: s.effect_size)

    # Summary statistics
    effects = [s.effect_size for s in specs]
    median_effect = float(np.median(effects)) if effects else 0.0
    n_sig = sum(1 for s in specs if s.significant)
    frac_sig = n_sig / len(specs) if specs else 0.0

    # Permutation test for joint inference
    perm_p: float | None = None
    if n_permutations > 0 and specs:
        perm_p = _permutation_test(
            per_volume_data=per_volume_data,
            metric_names=metric_names,
            higher_is_better=higher_is_better,
            aggregation_methods=aggregation_methods,
            observed_median=median_effect,
            n_permutations=n_permutations,
            seed=seed,
        )

    # Document degrees of freedom
    dof: dict[str, list[str]] = {
        "metrics": metric_names,
        "aggregation": aggregation_methods,
        "factors": factor_names,
    }

    return SpecificationCurveResult(
        specifications=specs,
        median_effect=median_effect,
        fraction_significant=frac_sig,
        permutation_p=perm_p,
        n_permutations=n_permutations,
        degrees_of_freedom=dof,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_single_specification(
    condition_data: dict[str, dict[int, np.ndarray]],
    *,
    metric: str,
    aggregation: str,
    condition_a: str,
    condition_b: str,
    higher_is_better: bool,
) -> Specification:
    """Compute one specification: one metric × one aggregation × one pair."""
    scores_a = _pool_and_aggregate(condition_data[condition_a], aggregation)
    scores_b = _pool_and_aggregate(condition_data[condition_b], aggregation)

    # Align lengths
    n = min(len(scores_a), len(scores_b))
    scores_a = scores_a[:n]
    scores_b = scores_b[:n]

    # Direction: for "lower is better" metrics, flip sign
    if not higher_is_better:
        scores_a = -scores_a
        scores_b = -scores_b

    # Effect size: Cohen's d
    diff = scores_a - scores_b
    pooled_std = np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2)
    cohens_d = float(np.mean(diff) / pooled_std) if pooled_std > 0 else 0.0

    # Statistical test: Wilcoxon signed-rank
    try:
        stat_result = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
        p_value = float(stat_result.pvalue)
    except ValueError:
        p_value = 1.0

    return Specification(
        metric=metric,
        aggregation=aggregation,
        condition_a=condition_a,
        condition_b=condition_b,
        effect_size=cohens_d,
        p_value=p_value,
        mean_diff=float(np.mean(diff)),
        significant=False,  # Will be updated after FDR correction
    )


def _pool_and_aggregate(
    fold_data: dict[int, np.ndarray],
    aggregation: str,
) -> np.ndarray:
    """Pool per-volume scores across folds using the specified aggregation.

    For "mean": concatenate all volumes across folds.
    For "median": compute per-volume medians across folds (if aligned),
    otherwise concatenate.
    """
    arrays = [fold_data[k] for k in sorted(fold_data.keys())]

    if aggregation == "median" and len(arrays) > 1:
        # If all arrays have the same length, compute element-wise median
        lengths = [len(a) for a in arrays]
        if len(set(lengths)) == 1:
            stacked = np.stack(arrays)
            return np.asarray(np.median(stacked, axis=0))

    # Default: concatenate
    return np.concatenate(arrays)


def _bh_fdr_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Benjamini-Hochberg FDR correction."""
    if not p_values:
        return []

    from statsmodels.stats.multitest import multipletests

    reject, pvals_adj, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return list(zip(pvals_adj.tolist(), reject.tolist(), strict=True))


def _permutation_test(
    per_volume_data: dict[str, dict[str, dict[int, np.ndarray]]],
    metric_names: list[str],
    higher_is_better: dict[str, bool],
    aggregation_methods: list[str],
    observed_median: float,
    n_permutations: int,
    seed: int,
) -> float:
    """Joint permutation test for the specification curve.

    Shuffles **observations** across conditions (not just labels) to create
    a null distribution where conditions are exchangeable. The p-value is
    the fraction of permuted median absolute effects >= observed.

    Memory-efficient: pre-allocates shuffle buffer, reuses arrays, and runs
    periodic gc.collect() to prevent accumulation. See metalearning:
    .claude/metalearning/2026-03-21-ram-crash-biostatistics-test.md

    Reference: Simonsohn et al. (2020), Section 4.
    """
    rng = np.random.default_rng(seed)
    abs_observed = abs(observed_median)
    n_extreme = 0

    # Pre-compute structure once — reuse across all permutations
    metric_structures: dict[str, tuple[np.ndarray, list[tuple[str, int, int]]]] = {}
    for metric in metric_names:
        if metric not in per_volume_data:
            continue
        conditions = sorted(per_volume_data[metric].keys())
        all_scores: list[float] = []
        structure: list[tuple[str, int, int]] = []
        for cond in conditions:
            for fold_id in sorted(per_volume_data[metric][cond].keys()):
                arr = per_volume_data[metric][cond][fold_id]
                all_scores.extend(arr.tolist())
                structure.append((cond, fold_id, len(arr)))
        # Pre-allocate buffer array — shuffle in-place each iteration
        metric_structures[metric] = (np.array(all_scores), structure)

    # MemoryMonitor: process-level guardrails (ported from foundation-PLR)
    from minivess.observability.memory_monitor import MemoryMonitor

    monitor = MemoryMonitor(warning_threshold_gb=4.0, critical_threshold_gb=6.0)
    _gc_interval = 50

    for i in range(n_permutations):
        # Build permuted data by shuffling pre-allocated buffers in-place
        permuted_data: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        for metric, (buffer, structure) in metric_structures.items():
            rng.shuffle(buffer)  # In-place shuffle — no new allocation
            permuted_data[metric] = {}
            idx = 0
            for cond, fold_id, n_obs in structure:
                if cond not in permuted_data[metric]:
                    permuted_data[metric][cond] = {}
                permuted_data[metric][cond][fold_id] = buffer[idx : idx + n_obs]
                idx += n_obs

        # Compute specification curve on permuted data
        perm_effects: list[float] = []
        for metric in metric_names:
            if metric not in permuted_data:
                continue
            conditions = sorted(permuted_data[metric].keys())
            pairs = list(itertools.combinations(conditions, 2))
            for agg in aggregation_methods:
                for cond_a, cond_b in pairs:
                    spec = _compute_single_specification(
                        permuted_data[metric],
                        metric=metric,
                        aggregation=agg,
                        condition_a=cond_a,
                        condition_b=cond_b,
                        higher_is_better=higher_is_better.get(metric, True),
                    )
                    perm_effects.append(spec.effect_size)

        perm_median = float(np.median(perm_effects)) if perm_effects else 0.0
        if abs(perm_median) >= abs_observed:
            n_extreme += 1

        # Explicit cleanup + periodic GC via MemoryMonitor
        del permuted_data
        del perm_effects
        if (i + 1) % _gc_interval == 0:
            monitor.enforce()
            if n_permutations >= 100:
                logger.debug(
                    "Permutation %d/%d complete (RSS=%.1f GB)",
                    i + 1,
                    n_permutations,
                    monitor.get_usage_gb(),
                )

    return (n_extreme + 1) / (n_permutations + 1)  # +1 for observed itself
