"""Tests for Riley bootstrap instability analysis.

Validates compute_riley_instability() and _generate_instability_plot() which
assess model ranking stability via bootstrap resampling.

References
----------
Riley et al. (2021). "Minimum sample size for developing a multivariable
prediction model." Stat Med.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _make_per_volume_data_with_clear_winner(
    seed: int = 42,
    n_volumes: int = 23,
    n_folds: int = 3,
) -> dict[str, dict[int, np.ndarray]]:
    """Create synthetic per-volume data where model_A is clearly the best.

    Returns dict: {condition_key: {fold_id: np.ndarray}}
    """
    rng = np.random.default_rng(seed)

    data: dict[str, dict[int, np.ndarray]] = {}
    # model_A is clearly the best (mean ~0.90)
    data["model_A"] = {}
    for fold_id in range(n_folds):
        data["model_A"][fold_id] = rng.normal(0.90, 0.02, size=n_volumes)

    # model_B is second best (mean ~0.80)
    data["model_B"] = {}
    for fold_id in range(n_folds):
        data["model_B"][fold_id] = rng.normal(0.80, 0.03, size=n_volumes)

    # model_C is worst (mean ~0.65)
    data["model_C"] = {}
    for fold_id in range(n_folds):
        data["model_C"][fold_id] = rng.normal(0.65, 0.04, size=n_volumes)

    return data


class TestRileyInstabilityRankShape:
    """T5: Rank matrix has shape (n_bootstrap, n_models)."""

    def test_riley_instability_rank_shape(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            compute_riley_instability,
        )

        per_volume_data = _make_per_volume_data_with_clear_winner()
        n_bootstrap = 100

        result = compute_riley_instability(
            per_volume_data=per_volume_data,
            metric_name="cldice",
            n_bootstrap=n_bootstrap,
            seed=42,
        )

        rank_matrix = result["rank_matrix"]
        assert rank_matrix.shape == (n_bootstrap, 3)  # 3 models


class TestRileyInstabilityDeterministicSeed:
    """T5: Same seed produces identical results."""

    def test_riley_instability_deterministic_seed(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            compute_riley_instability,
        )

        per_volume_data = _make_per_volume_data_with_clear_winner()

        result1 = compute_riley_instability(
            per_volume_data=per_volume_data,
            metric_name="cldice",
            n_bootstrap=50,
            seed=42,
        )
        result2 = compute_riley_instability(
            per_volume_data=per_volume_data,
            metric_name="cldice",
            n_bootstrap=50,
            seed=42,
        )

        np.testing.assert_array_equal(result1["rank_matrix"], result2["rank_matrix"])


class TestRileyInstabilityStableModelHighFraction:
    """T5: Clearly best model has high rank stability fraction."""

    def test_riley_instability_stable_model_high_fraction(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            compute_riley_instability,
        )

        per_volume_data = _make_per_volume_data_with_clear_winner()

        result = compute_riley_instability(
            per_volume_data=per_volume_data,
            metric_name="cldice",
            n_bootstrap=500,
            seed=42,
        )

        stability = result["stability_fractions"]
        # model_A (the best) should have high stability (rank=1 most of the time)
        assert stability["model_A"] > 0.8, (
            f"Best model stability should be >0.8, got {stability['model_A']}"
        )


class TestRileyInstabilityPlotGeneratesFile:
    """T5: Instability plot creates PNG file."""

    def test_riley_instability_plot_generates_file(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from minivess.pipeline.biostatistics_figures import _generate_instability_plot
        from minivess.pipeline.biostatistics_statistics import (
            compute_riley_instability,
        )

        per_volume_data = _make_per_volume_data_with_clear_winner()
        instability_result = compute_riley_instability(
            per_volume_data=per_volume_data,
            metric_name="cldice",
            n_bootstrap=100,
            seed=42,
        )

        artifact = _generate_instability_plot(
            instability_result=instability_result,
            output_dir=tmp_path,
        )

        assert artifact is not None
        png_paths = [p for p in artifact.paths if str(p).endswith(".png")]
        assert len(png_paths) >= 1
        assert png_paths[0].exists()


class TestRileyInstabilityPlotSidecar:
    """T5: JSON sidecar exists alongside instability plot."""

    def test_riley_instability_plot_sidecar(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from minivess.pipeline.biostatistics_figures import _generate_instability_plot
        from minivess.pipeline.biostatistics_statistics import (
            compute_riley_instability,
        )

        per_volume_data = _make_per_volume_data_with_clear_winner()
        instability_result = compute_riley_instability(
            per_volume_data=per_volume_data,
            metric_name="cldice",
            n_bootstrap=100,
            seed=42,
        )

        artifact = _generate_instability_plot(
            instability_result=instability_result,
            output_dir=tmp_path,
        )

        assert artifact is not None
        assert artifact.sidecar_path is not None
        assert artifact.sidecar_path.exists()

        sidecar = json.loads(artifact.sidecar_path.read_text(encoding="utf-8"))
        assert "figure_id" in sidecar
        assert "generated_at" in sidecar
        assert "data" in sidecar
