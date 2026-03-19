"""Tests for factorial ANOVA figure generation (interaction plots + variance lollipop).

Validates _generate_interaction_plot() and _generate_variance_lollipop() which produce
publication-quality figures for Model x Loss factorial analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minivess.pipeline.biostatistics_types import FactorialAnovaResult


def _make_mock_anova_result() -> FactorialAnovaResult:
    """Create a mock FactorialAnovaResult for testing."""
    return FactorialAnovaResult(
        metric="cldice",
        n_models=4,
        n_losses=3,
        f_values={
            "Model": 12.5,
            "Loss": 3.2,
            "Model:Loss": 1.8,
        },
        p_values={
            "Model": 0.0001,
            "Loss": 0.04,
            "Model:Loss": 0.12,
        },
        eta_squared_partial={
            "Model": 0.35,
            "Loss": 0.08,
            "Model:Loss": 0.04,
            "Fold": 0.02,
            "Residual": 0.51,
        },
        omega_squared={
            "Model": 0.30,
            "Loss": 0.06,
            "Model:Loss": 0.02,
            "Fold": 0.01,
            "Residual": 0.50,
        },
    )


def _make_mock_per_volume_data(
    seed: int = 42,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Create synthetic per-volume data for interaction plot testing.

    Returns dict: {metric: {condition_key: {fold_id: np.ndarray}}}
    """
    rng = np.random.default_rng(seed)
    models = ["dynunet", "mambavesselnet", "sam3_vanilla", "vesselfm"]
    losses = ["dice_ce", "cbdice_cldice", "dice_ce_cldice"]

    data: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for metric in ("cldice", "masd"):
        data[metric] = {}
        for i, model in enumerate(models):
            for j, loss in enumerate(losses):
                condition_key = f"{model}__{loss}"
                data[metric][condition_key] = {}
                for fold_id in range(3):
                    base = 0.7 + 0.05 * i + 0.02 * j
                    if metric == "masd":
                        base = 2.0 - 0.3 * i - 0.1 * j
                    data[metric][condition_key][fold_id] = rng.normal(
                        base, 0.05, size=23
                    )
    return data


class TestInteractionPlotGeneratesFile:
    """T2: _generate_interaction_plot creates a PNG file."""

    def test_interaction_plot_generates_file(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from minivess.pipeline.biostatistics_figures import _generate_interaction_plot

        anova_result = _make_mock_anova_result()
        per_volume_data = _make_mock_per_volume_data()

        artifact = _generate_interaction_plot(
            anova_result=anova_result,
            per_volume_data=per_volume_data,
            output_dir=tmp_path,
        )

        assert artifact is not None
        png_paths = [p for p in artifact.paths if str(p).endswith(".png")]
        assert len(png_paths) >= 1
        assert png_paths[0].exists()


class TestInteractionPlotTwoPanels:
    """T2: Interaction plot has 2 axes (one per metric)."""

    def test_interaction_plot_two_panels(self, tmp_path: Path) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        from minivess.pipeline.biostatistics_figures import _generate_interaction_plot

        anova_result = _make_mock_anova_result()
        per_volume_data = _make_mock_per_volume_data()

        artifact = _generate_interaction_plot(
            anova_result=anova_result,
            per_volume_data=per_volume_data,
            output_dir=tmp_path,
            metrics=["cldice", "masd"],
        )

        assert artifact is not None
        # Read the figure back to check axes count
        # The figure should have been saved; verify via sidecar metadata
        sidecar_path = artifact.sidecar_path
        assert sidecar_path is not None
        sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar_data["data"]["n_panels"] == 2

        plt.close("all")


class TestInteractionPlotSidecarJson:
    """T2: JSON sidecar file exists alongside the plot."""

    def test_interaction_plot_sidecar_json(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from minivess.pipeline.biostatistics_figures import _generate_interaction_plot

        anova_result = _make_mock_anova_result()
        per_volume_data = _make_mock_per_volume_data()

        artifact = _generate_interaction_plot(
            anova_result=anova_result,
            per_volume_data=per_volume_data,
            output_dir=tmp_path,
        )

        assert artifact is not None
        assert artifact.sidecar_path is not None
        assert artifact.sidecar_path.exists()

        sidecar = json.loads(artifact.sidecar_path.read_text(encoding="utf-8"))
        assert "figure_id" in sidecar
        assert "generated_at" in sidecar
        assert "data" in sidecar


class TestInteractionPlotOkabeItoColors:
    """T2: Verify Okabe-Ito palette is used for model lines."""

    def test_interaction_plot_okabe_ito_colors(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from minivess.pipeline.biostatistics_figures import (
            OKABE_ITO,
            _generate_interaction_plot,
        )

        anova_result = _make_mock_anova_result()
        per_volume_data = _make_mock_per_volume_data()

        artifact = _generate_interaction_plot(
            anova_result=anova_result,
            per_volume_data=per_volume_data,
            output_dir=tmp_path,
        )

        assert artifact is not None
        assert artifact.sidecar_path is not None
        # Verify sidecar records the palette used
        sidecar = json.loads(artifact.sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["data"]["palette"] == "okabe_ito"

        # Verify the OKABE_ITO palette has at least 4 colors (for 4 models)
        assert len(OKABE_ITO) >= 4


class TestVarianceLollipopGeneratesFile:
    """T2: _generate_variance_lollipop creates a PNG file."""

    def test_variance_lollipop_generates_file(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from minivess.pipeline.biostatistics_figures import _generate_variance_lollipop

        anova_result = _make_mock_anova_result()

        artifact = _generate_variance_lollipop(
            anova_result=anova_result,
            output_dir=tmp_path,
        )

        assert artifact is not None
        png_paths = [p for p in artifact.paths if str(p).endswith(".png")]
        assert len(png_paths) >= 1
        assert png_paths[0].exists()


class TestVarianceLollipopAllFactors:
    """T2: Variance lollipop shows all 5 factors."""

    def test_variance_lollipop_all_factors(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from minivess.pipeline.biostatistics_figures import _generate_variance_lollipop

        anova_result = _make_mock_anova_result()

        artifact = _generate_variance_lollipop(
            anova_result=anova_result,
            output_dir=tmp_path,
        )

        assert artifact is not None
        assert artifact.sidecar_path is not None
        sidecar = json.loads(artifact.sidecar_path.read_text(encoding="utf-8"))
        factors = sidecar["data"]["factors"]
        assert len(factors) == 5
        expected_factors = {"Model", "Loss", "Model:Loss", "Fold", "Residual"}
        assert set(factors) == expected_factors


class TestVarianceLollipopSidecarJson:
    """T2: JSON sidecar exists for variance lollipop."""

    def test_variance_lollipop_sidecar_json(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from minivess.pipeline.biostatistics_figures import _generate_variance_lollipop

        anova_result = _make_mock_anova_result()

        artifact = _generate_variance_lollipop(
            anova_result=anova_result,
            output_dir=tmp_path,
        )

        assert artifact is not None
        assert artifact.sidecar_path is not None
        assert artifact.sidecar_path.exists()

        sidecar = json.loads(artifact.sidecar_path.read_text(encoding="utf-8"))
        assert "figure_id" in sidecar
        assert sidecar["figure_id"] == "variance_lollipop"
