"""Export biostatistics results as JSON/CSV for R/ggplot2 consumption.

Foundation-PLR pattern: Python exports data files, R scripts read and render.
No rpy2, no subprocess — just file I/O.

Output directory: {output_dir}/r_data/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from minivess.pipeline.biostatistics_sidecar import FigureSidecar, write_sidecar
from minivess.pipeline.biostatistics_types import (
    PairwiseResult,
    RankingResult,
    VarianceDecompositionResult,
)

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def export_pairwise_results(
    results: list[PairwiseResult],
    output_dir: Path,
) -> Path:
    """Export pairwise comparison results as JSON for R forest plot."""
    data = [
        {
            "condition_a": r.condition_a,
            "condition_b": r.condition_b,
            "metric": r.metric,
            "p_value": r.p_value,
            "p_adjusted": r.p_adjusted,
            "correction_method": r.correction_method,
            "significant": r.significant,
            "cohens_d": r.cohens_d,
            "cliffs_delta": r.cliffs_delta,
            "vda": r.vda,
            "bayesian_left": r.bayesian_left,
            "bayesian_rope": r.bayesian_rope,
            "bayesian_right": r.bayesian_right,
        }
        for r in results
    ]
    path = output_dir / "pairwise_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    logger.info("Exported %d pairwise results to %s", len(data), path)
    return path


def export_per_volume_data(
    per_volume_data: dict[str, dict[str, dict[int, Any]]],
    output_dir: Path,
) -> Path:
    """Export per-volume data as JSON for R distribution plots."""
    # Flatten to a list of records for easy R ingestion
    records: list[dict[str, Any]] = []
    for metric, conditions in per_volume_data.items():
        for condition, folds in conditions.items():
            for fold_id, scores in folds.items():
                arr = np.asarray(scores)
                for i, val in enumerate(arr):
                    records.append({
                        "metric": metric,
                        "condition": condition,
                        "fold_id": int(fold_id),
                        "volume_idx": i,
                        "value": float(val),
                    })

    path = output_dir / "per_volume_data.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, cls=_NumpyEncoder)
    logger.info("Exported %d per-volume records to %s", len(records), path)
    return path


def export_variance_results(
    results: list[VarianceDecompositionResult],
    output_dir: Path,
) -> Path:
    """Export variance decomposition results as JSON for R plots."""
    data = [
        {
            "metric": r.metric,
            "friedman_statistic": r.friedman_statistic,
            "friedman_p": r.friedman_p,
            "icc_value": r.icc_value,
            "icc_ci_lower": r.icc_ci_lower,
            "icc_ci_upper": r.icc_ci_upper,
            "icc_type": r.icc_type,
            "power_caveat": r.power_caveat,
        }
        for r in results
    ]
    path = output_dir / "variance_decomposition.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    return path


def export_rankings(
    results: list[RankingResult],
    output_dir: Path,
) -> Path:
    """Export ranking results as JSON for R CD diagram."""
    data = [
        {
            "metric": r.metric,
            "condition_ranks": r.condition_ranks,
            "mean_ranks": r.mean_ranks,
            "cd_value": r.cd_value,
        }
        for r in results
    ]
    path = output_dir / "rankings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    return path


def export_all_r_data(
    pairwise: list[PairwiseResult],
    variance: list[VarianceDecompositionResult],
    rankings: list[RankingResult],
    per_volume_data: dict[str, dict[str, dict[int, Any]]],
    output_dir: Path,
) -> list[Path]:
    """Export all biostatistics results for R consumption."""
    r_data_dir = output_dir / "r_data"
    r_data_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        export_pairwise_results(pairwise, r_data_dir),
        export_per_volume_data(per_volume_data, r_data_dir),
        export_variance_results(variance, r_data_dir),
        export_rankings(rankings, r_data_dir),
    ]
    logger.info("Exported all R data to %s (%d files)", r_data_dir, len(paths))
    return paths
