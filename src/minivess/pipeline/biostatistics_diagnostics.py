"""Power analysis and diagnostics for the biostatistics flow.

Simulation-based power estimation for the stratified permutation test.
Reports achieved power, effective N, and recommended additional folds.

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_power_diagnostics(
    *,
    per_volume_data: dict[str, dict[int, np.ndarray]],
    metric_name: str,
    alpha: float,
    seed: int,
    n_simulations: int = 200,
    effect_sizes: tuple[float, ...] = (0.2, 0.5, 0.8),
) -> list[dict[str, Any]]:
    """Compute simulation-based power for the stratified permutation test.

    For each assumed Cohen's d, generates synthetic datasets with the observed
    fold structure and known effect size, runs stratified_permutation_test(),
    and counts the rejection rate as achieved power.

    Parameters
    ----------
    per_volume_data:
        Single metric: ``{condition_key: {fold_id: scores_array}}``.
    metric_name:
        Name of the metric (for labeling).
    alpha:
        Significance level (from BiostatisticsConfig — Rule 29).
    seed:
        Random seed (from BiostatisticsConfig — Rule 29).
    n_simulations:
        Number of synthetic datasets per effect size.
    effect_sizes:
        Cohen's d values to test power at.

    Returns
    -------
    List of diagnostic records, one per effect size.
    """
    from minivess.pipeline.biostatistics_statistics import (
        stratified_permutation_test,
    )

    # Determine observed data structure
    conditions = list(per_volume_data.keys())
    if len(conditions) < 2:
        logger.warning("Power diagnostics need >= 2 conditions, got %d", len(conditions))
        return []

    # Use first condition as reference for structure
    ref_folds = per_volume_data[conditions[0]]
    n_folds = len(ref_folds)
    n_volumes_per_fold = [len(ref_folds[f]) for f in sorted(ref_folds.keys())]

    # Compute ICC within fold (intra-class correlation)
    icc_within_fold = _estimate_icc(per_volume_data)

    # Compute design effect
    mean_cluster_size = float(np.mean(n_volumes_per_fold)) if n_volumes_per_fold else 1.0
    design_effect = 1.0 + (mean_cluster_size - 1.0) * max(icc_within_fold, 0.0)
    effective_n = int(
        sum(n_volumes_per_fold) / design_effect
    ) if design_effect > 0 else sum(n_volumes_per_fold)

    results: list[dict[str, Any]] = []

    for d in effect_sizes:
        rejections = 0

        for sim_idx in range(n_simulations):
            sim_seed = seed + sim_idx * 1000 + int(d * 100)
            sim_rng = np.random.default_rng(sim_seed)

            # Generate synthetic paired data with known effect size d
            fold_a: dict[int, np.ndarray] = {}
            fold_b: dict[int, np.ndarray] = {}

            for fold_id, n_vol in enumerate(n_volumes_per_fold):
                # Condition A: standard normal
                a = sim_rng.standard_normal(n_vol)
                # Condition B: shifted by d (Cohen's d = mean_diff / pooled_sd)
                b = a + d + sim_rng.standard_normal(n_vol) * 0.1  # Small noise
                fold_a[fold_id] = a
                fold_b[fold_id] = b

            result = stratified_permutation_test(
                fold_a, fold_b,
                n_permutations=99,  # Fast for simulation
                seed=sim_seed,
            )
            if result.p_value < alpha:
                rejections += 1

        achieved_power = rejections / n_simulations

        # Recommend additional folds needed for 80% power
        recommended_folds = _recommend_folds(
            achieved_power=achieved_power,
            current_folds=n_folds,
            target_power=0.80,
        )

        results.append({
            "metric": metric_name,
            "test_type": "stratified_permutation",
            "alpha_used": alpha,
            "effect_size_assumed": d,
            "achieved_power": achieved_power,
            "effective_n": effective_n,
            "design_effect": round(design_effect, 3),
            "icc_within_fold": round(icc_within_fold, 4),
            "min_detectable_effect": d,
            "recommended_additional_folds": recommended_folds,
            "recommendation": _make_recommendation(
                d, achieved_power, n_folds, recommended_folds
            ),
        })

    logger.info(
        "Power diagnostics for %s: n_folds=%d, effective_n=%d, ICC=%.4f",
        metric_name,
        n_folds,
        effective_n,
        icc_within_fold,
    )

    return results


def _estimate_icc(
    per_volume_data: dict[str, dict[int, np.ndarray]],
) -> float:
    """Estimate ICC within fold from observed data.

    Uses one-way random effects ICC(1,1) approximation.
    Returns 0.0 if computation fails.
    """
    # Collect all scores grouped by fold across conditions
    fold_scores: dict[int, list[float]] = {}
    for _cond, folds in per_volume_data.items():
        for fold_id, scores in folds.items():
            fold_scores.setdefault(fold_id, []).extend(scores.tolist())

    if len(fold_scores) < 2:
        return 0.0

    # One-way ANOVA to estimate ICC(1,1)
    groups = list(fold_scores.values())
    grand_mean = float(np.mean([s for g in groups for s in g]))
    n_groups = len(groups)
    n_per_group = [len(g) for g in groups]

    # Between-group sum of squares
    ss_between = sum(
        n * (float(np.mean(g)) - grand_mean) ** 2
        for n, g in zip(n_per_group, groups, strict=True)
    )
    # Within-group sum of squares
    ss_within = sum(
        sum((x - float(np.mean(g))) ** 2 for x in g)
        for g in groups
    )

    df_between = n_groups - 1
    df_within = sum(n_per_group) - n_groups

    if df_between <= 0 or df_within <= 0:
        return 0.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # ICC(1,1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
    k = float(np.mean(n_per_group))
    denom = ms_between + (k - 1) * ms_within
    if denom <= 0:
        return 0.0

    icc = (ms_between - ms_within) / denom
    return max(min(float(icc), 1.0), -1.0)  # Clamp to [-1, 1]


def _recommend_folds(
    *,
    achieved_power: float,
    current_folds: int,
    target_power: float,
) -> int:
    """Estimate additional folds needed to reach target power.

    Simple heuristic: power scales roughly with sqrt(n_folds).
    """
    if achieved_power >= target_power:
        return 0

    if achieved_power <= 0.0:
        # Very underpowered — suggest substantial increase
        return max(10 - current_folds, 3)

    # sqrt scaling: target_folds ≈ current_folds * (target_power / achieved_power)^2
    ratio = (target_power / achieved_power) ** 2
    estimated_total = int(np.ceil(current_folds * ratio))
    additional = max(estimated_total - current_folds, 1)
    return min(additional, 20)  # Cap at 20 additional folds


def _make_recommendation(
    d: float,
    achieved_power: float,
    n_folds: int,
    recommended_additional: int,
) -> str:
    """Generate a human-readable power recommendation."""
    if achieved_power >= 0.80:
        return (
            f"Adequate power ({achieved_power:.0%}) for d={d} with {n_folds} folds."
        )
    if recommended_additional == 0:
        return f"Power={achieved_power:.0%} at d={d}, already at target."

    return (
        f"Underpowered ({achieved_power:.0%}) for d={d} with {n_folds} folds. "
        f"Recommend {recommended_additional} additional folds to approach 80% power."
    )
