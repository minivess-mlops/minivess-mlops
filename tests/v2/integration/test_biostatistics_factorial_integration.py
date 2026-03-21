"""Integration test: full biostatistics pipeline with synthetic factorial data.

PR-A T8 (Issue #815): Wires all biostatistics components together —
ANOVA, figures, tables, lineage — with synthetic data that has
planted model and loss effects.

Must run in <60s, no GPU, no Docker, no MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

_MODELS = ["dynunet", "vesselfm", "sam3_vanilla", "mambavesselnet"]
_LOSSES = ["cbdice_cldice", "dice_ce", "focal"]
_N_FOLDS = 3
_N_VOLUMES = 23
_SEED = 42


def _generate_synthetic_per_volume_data() -> dict[
    str, dict[str, dict[int, np.ndarray]]
]:
    """Generate synthetic per-volume metric data with planted effects.

    Effects:
    - Model: dynunet > mambavesselnet (mean delta ~0.15 in clDice)
    - Loss: cbdice_cldice > focal (mean delta ~0.05)
    - Interaction: vesselfm benefits more from cbdice_cldice than dynunet

    Returns dict: {metric: {condition_key: {fold_id: np.ndarray of shape (N_VOLUMES,)}}}
    condition_key format: "model__loss" (double underscore)
    """
    rng = np.random.default_rng(_SEED)

    # Model base effects (clDice)
    model_effects = {
        "dynunet": 0.80,
        "vesselfm": 0.75,
        "sam3_vanilla": 0.72,
        "mambavesselnet": 0.65,
    }

    # Loss effects (additive, clDice)
    loss_effects = {
        "cbdice_cldice": 0.05,
        "dice_ce": 0.0,
        "focal": -0.03,
    }

    # Interaction: vesselfm + cbdice_cldice gets extra boost
    interaction = {("vesselfm", "cbdice_cldice"): 0.08}

    per_volume_data: dict[str, dict[str, dict[int, np.ndarray]]] = {
        "cldice": {},
        "masd": {},
        "dsc": {},
    }

    for model in _MODELS:
        for loss in _LOSSES:
            condition_key = f"{model}__{loss}"
            per_volume_data["cldice"][condition_key] = {}
            per_volume_data["masd"][condition_key] = {}
            per_volume_data["dsc"][condition_key] = {}

            base = model_effects[model] + loss_effects[loss]
            bonus = interaction.get((model, loss), 0.0)

            for fold_id in range(_N_FOLDS):
                # clDice: base + interaction + noise
                cldice_vals = np.clip(
                    base + bonus + rng.normal(0, 0.03, _N_VOLUMES),
                    0.0,
                    1.0,
                )
                per_volume_data["cldice"][condition_key][fold_id] = cldice_vals

                # MASD: inversely related to clDice (lower is better)
                masd_vals = np.clip(
                    (1.0 - cldice_vals) * 5.0 + rng.normal(0, 0.3, _N_VOLUMES),
                    0.1,
                    5.0,
                )
                per_volume_data["masd"][condition_key][fold_id] = masd_vals

                # DSC: correlated with clDice
                dsc_vals = np.clip(
                    cldice_vals + rng.normal(0.03, 0.02, _N_VOLUMES),
                    0.0,
                    1.0,
                )
                per_volume_data["dsc"][condition_key][fold_id] = dsc_vals

    return per_volume_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFactorialIntegrationSyntheticData:
    """Pipeline runs end-to-end without error."""

    @pytest.fixture()
    def synthetic_data(self) -> dict[str, dict[str, dict[int, np.ndarray]]]:
        return _generate_synthetic_per_volume_data()

    @pytest.fixture()
    def output_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "biostatistics_output"
        d.mkdir()
        return d

    def test_factorial_integration_synthetic_data(
        self, synthetic_data: dict, output_dir: Path
    ) -> None:
        """Full pipeline runs without error on synthetic data."""
        from minivess.pipeline.biostatistics_statistics import (
            compute_factorial_anova,
        )

        # Run ANOVA for each metric
        for metric_name in ["cldice", "masd", "dsc"]:
            result = compute_factorial_anova(synthetic_data, metric_name)
            assert result is not None
            assert result.metric == metric_name


class TestFactorialIntegrationAnovaOutput:
    """ANOVA detects planted effects."""

    def test_factorial_integration_anova_output(self) -> None:
        """ANOVA detects planted model effect (p < alpha from config)."""
        from minivess.config.biostatistics_config import BiostatisticsConfig
        from minivess.pipeline.biostatistics_statistics import (
            compute_factorial_anova,
        )

        cfg = BiostatisticsConfig()
        data = _generate_synthetic_per_volume_data()
        result = compute_factorial_anova(data, "cldice")

        # Model effect should be significant (planted ~0.15 difference)
        # pingouin uses lowercase factor names from the DataFrame columns
        model_key = "Model" if "Model" in result.p_values else "model"
        assert model_key in result.p_values
        assert result.p_values[model_key] < cfg.alpha

        # F-values should be positive
        assert result.f_values[model_key] > 0

        # Eta-squared partial should be positive for Model
        assert result.eta_squared_partial[model_key] > 0


class TestFactorialIntegrationFiguresGenerated:
    """All figure artifacts are created."""

    def test_factorial_integration_figures_generated(self, tmp_path: Path) -> None:
        """Interaction plot and other figures generate without error."""
        from minivess.pipeline.biostatistics_figures import (
            generate_cost_breakdown_figure,
        )

        cost_summary: dict[str, Any] = {
            "total_spot_cost_usd": 5.60,
            "total_ondemand_cost_usd": 8.40,
            "savings_pct": 33.33,
            "total_gpu_hours": 7.0,
            "cost_by_model": {"dynunet": 2.80, "vesselfm": 2.80},
            "cost_by_phase": {"training": 5.20, "debug": 0.40},
        }

        artifact = generate_cost_breakdown_figure(cost_summary, tmp_path)
        assert artifact.figure_id == "cost_breakdown"
        assert any(p.exists() for p in artifact.paths)


class TestFactorialIntegrationTablesGenerated:
    """Table artifacts are created from ANOVA results."""

    def test_factorial_integration_tables_generated(self, tmp_path: Path) -> None:
        """ANOVA table generates from factorial results."""
        from minivess.pipeline.biostatistics_statistics import (
            compute_factorial_anova,
        )
        from minivess.pipeline.biostatistics_tables import _generate_anova_table

        data = _generate_synthetic_per_volume_data()
        anova_result = compute_factorial_anova(data, "cldice")

        # Generate ANOVA table directly
        table = _generate_anova_table(anova_result, tmp_path)

        assert "anova" in table.table_id
        assert table.path.exists()
        assert table.format == "latex"

        # Verify LaTeX content
        content = table.path.read_text(encoding="utf-8")
        assert "\\begin{table}" in content
        assert "Model" in content


class TestFactorialIntegrationLineageComplete:
    """Lineage manifest has all required fields."""

    def test_factorial_integration_lineage_complete(self) -> None:
        """Lineage manifest includes cost_summary and artifact counts."""
        from minivess.pipeline.biostatistics_lineage import (
            build_lineage_manifest_with_cost,
        )
        from minivess.pipeline.biostatistics_types import (
            SourceRun,
            SourceRunManifest,
        )

        runs = [
            SourceRun(
                run_id=f"run_{i}",
                experiment_id="1",
                experiment_name="factorial_v1",
                loss_function="dice_ce",
                fold_id=i % 3,
                status="FINISHED",
            )
            for i in range(12)
        ]
        manifest = SourceRunManifest.from_runs(runs)

        cost_summary = {
            "total_spot_cost_usd": 12.50,
            "savings_pct": 33.33,
        }

        lineage = build_lineage_manifest_with_cost(
            manifest=manifest,
            figures=[],
            tables=[],
            cost_summary=cost_summary,
        )

        assert "schema_version" in lineage
        assert "fingerprint" in lineage
        assert "cost_summary" in lineage
        assert lineage["cost_summary"]["total_spot_cost_usd"] == 12.50
        assert lineage["n_source_runs"] == 12


class TestFactorialIntegrationDataShape:
    """Synthetic data has correct factorial structure."""

    def test_factorial_data_shape(self) -> None:
        """4 models x 3 losses = 12 conditions, 3 folds, 23 volumes each."""
        data = _generate_synthetic_per_volume_data()

        assert set(data.keys()) == {"cldice", "masd", "dsc"}

        for metric_name, conditions in data.items():
            # 4 models x 3 losses = 12 conditions
            assert len(conditions) == 12, f"{metric_name}: expected 12 conditions"

            for cond_key, folds in conditions.items():
                # Double underscore separator
                assert "__" in cond_key, f"Missing __ in {cond_key}"
                # 3 folds
                assert len(folds) == _N_FOLDS, f"{cond_key}: expected {_N_FOLDS} folds"

                for _fold_id, values in folds.items():
                    # 23 volumes each
                    assert len(values) == _N_VOLUMES


# ---------------------------------------------------------------------------
# Phase 6: New integration tests for spec curve + rank stability
# ---------------------------------------------------------------------------


class TestFactorialIntegrationSpecCurve:
    """Specification curve works on full synthetic factorial data."""

    def test_spec_curve_on_factorial_data(self) -> None:
        """Spec curve produces valid output on 12-condition synthetic data."""
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _generate_synthetic_per_volume_data()
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss"],
            metric_names=["cldice", "dsc"],
            higher_is_better={"cldice": True, "dsc": True},
            n_permutations=50,
            seed=42,
        )
        # 12 conditions → C(12,2) = 66 pairs × 2 metrics × 2 aggregations = 264
        assert len(result.specifications) >= 66
        assert result.permutation_p is not None
        assert 0.0 <= result.fraction_significant <= 1.0

    def test_spec_curve_median_effect_reasonable(self) -> None:
        """Median effect size should be non-zero with planted effects."""
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _generate_synthetic_per_volume_data()
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss"],
            metric_names=["cldice"],
            higher_is_better={"cldice": True},
        )
        # With planted model effects, median should be non-zero
        assert result.median_effect != 0.0


class TestFactorialIntegrationRankStability:
    """Rank stability analysis on full synthetic factorial data."""

    def test_rank_concordance_on_factorial_data(self) -> None:
        """Rank concordance computes for all metric pairs."""
        from minivess.pipeline.biostatistics_rank_stability import (
            compute_rank_concordance,
        )

        data = _generate_synthetic_per_volume_data()
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["cldice", "masd", "dsc"],
            higher_is_better={"cldice": True, "masd": False, "dsc": True},
        )
        # C(3,2) = 3 metric pairs
        assert len(result.tau_matrix) == 3
        # All 3 metrics should have rankings
        assert len(result.condition_ranks) == 3

    def test_cldice_dsc_concordance_positive(self) -> None:
        """clDice and DSC should agree (positive tau) in this synthetic data.

        Note: In REAL data from the factorial experiment, we expect rank
        INVERSION between DSC and clDice for tubular structures — that IS
        a paper finding. This synthetic data is correlated, not inverted.
        """
        from minivess.pipeline.biostatistics_rank_stability import (
            compute_rank_concordance,
        )

        data = _generate_synthetic_per_volume_data()
        result = compute_rank_concordance(
            per_volume_data=data,
            metric_names=["cldice", "dsc"],
            higher_is_better={"cldice": True, "dsc": True},
        )
        # In this synthetic data, clDice and DSC are correlated
        assert result.tau_matrix[0].tau > 0


class TestFactorialIntegrationK1Fallback:
    """K=1 fallback ANOVA works on synthetic data."""

    def test_k1_anova_on_factorial_data(self) -> None:
        """ANOVA with K=1 fold uses per-volume replication."""
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        # Modify data to have only 1 fold
        data = _generate_synthetic_per_volume_data()
        k1_data: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        for metric, conditions in data.items():
            k1_data[metric] = {}
            for cond, folds in conditions.items():
                k1_data[metric][cond] = {0: folds[0]}  # Keep only fold 0

        result = compute_factorial_anova(k1_data, "cldice")
        assert result.n_folds == 1
        assert result.replication_method == "per_volume"
        # Should still detect the planted model effect
        model_p = result.p_values.get("Model") or result.p_values.get("model")
        assert model_p is not None
        from minivess.config.biostatistics_config import BiostatisticsConfig

        assert model_p < BiostatisticsConfig().alpha
