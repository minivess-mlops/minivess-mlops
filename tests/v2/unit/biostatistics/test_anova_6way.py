"""Tests for 5-way and 6-way factorial ANOVA + condition key composition.

Phases 2 + 2.5 of the final QA plan:
- Validate compute_factorial_anova works with 5 and 6 factors
- Test _build_per_volume_data composes multi-factor condition keys
- Test biostatistics_flow imports and uses compute_factorial_anova
"""

from __future__ import annotations

import itertools

import numpy as np


def _make_synthetic_data(
    factor_names: list[str],
    levels_per_factor: list[list[str]],
    n_volumes: int = 10,
    seed: int = 42,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Create synthetic per-volume data for N-way ANOVA testing.

    Returns {metric: {condition_key: {fold_id: scores_array}}}.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, dict[str, dict[int, np.ndarray]]] = {}

    metric_data: dict[str, dict[int, np.ndarray]] = {}
    for combo in itertools.product(*levels_per_factor):
        condition_key = "__".join(combo)
        # Add a model-dependent effect so ANOVA finds significance
        effect = sum(hash(v) % 10 for v in combo) / 100.0
        scores = rng.normal(0.7 + effect, 0.1, size=n_volumes)
        metric_data[condition_key] = {0: scores}

    data["cldice"] = metric_data
    return data


class TestAnovaFiveWay:
    """Validate compute_factorial_anova with 5 factors (Layers A+B)."""

    def test_5way_anova_produces_5_main_effects(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            compute_factorial_anova,
        )

        factor_names = [
            "model_family",
            "loss_name",
            "aux_calibration",
            "post_training_method",
            "recalibration",
        ]
        levels = [
            ["dynunet", "sam3"],
            ["dice_ce", "cbdice"],
            ["false", "true"],
            ["none", "swa"],
            ["none", "temp_scaling"],
        ]
        data = _make_synthetic_data(factor_names, levels)

        result = compute_factorial_anova(
            per_volume_data=data,
            metric_name="cldice",
            factor_names=factor_names,
        )

        assert result.metric == "cldice"
        assert len(result.factor_names) == 5
        # At least 5 main effects should be present in p_values
        assert len(result.p_values) >= 5


class TestAnovaSixWay:
    """Validate compute_factorial_anova with 6 factors (all layers)."""

    def test_6way_anova_produces_6_main_effects(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            compute_factorial_anova,
        )

        factor_names = [
            "model_family",
            "loss_name",
            "aux_calibration",
            "post_training_method",
            "recalibration",
            "ensemble_strategy",
        ]
        levels = [
            ["dynunet", "sam3"],
            ["dice_ce", "cbdice"],
            ["false", "true"],
            ["none", "swa"],
            ["none", "temp_scaling"],
            ["per_loss_single_best", "all_loss_all_best"],
        ]
        data = _make_synthetic_data(factor_names, levels)

        result = compute_factorial_anova(
            per_volume_data=data,
            metric_name="cldice",
            factor_names=factor_names,
        )

        assert result.metric == "cldice"
        assert len(result.factor_names) == 6
        assert len(result.p_values) >= 6


class TestLayeredAnovaProgression:
    """Test the 3-step layered ANOVA: 3-way → 5-way → 6-way."""

    def test_layered_progression_factor_names_match(self) -> None:
        from minivess.pipeline.biostatistics_statistics import (
            compute_factorial_anova,
        )

        all_factor_names = [
            "model_family",
            "loss_name",
            "aux_calibration",
            "post_training_method",
            "recalibration",
            "ensemble_strategy",
        ]
        all_levels = [
            ["dynunet", "sam3"],
            ["dice_ce", "cbdice"],
            ["false", "true"],
            ["none", "swa"],
            ["none", "temp_scaling"],
            ["per_loss", "all_loss"],
        ]

        # Generate data with 6-factor keys
        data = _make_synthetic_data(all_factor_names, all_levels)

        # 3-way (Layer A only) — use first 3 parts of condition keys
        data_3way: dict[str, dict[int, np.ndarray]] = {}
        for key, folds in data["cldice"].items():
            parts = key.split("__")
            key_3 = "__".join(parts[:3])
            if key_3 not in data_3way:
                data_3way[key_3] = {}
            for fold_id, scores in folds.items():
                existing = data_3way[key_3].get(fold_id, np.array([]))
                data_3way[key_3][fold_id] = np.concatenate([existing, scores])

        result_3 = compute_factorial_anova(
            per_volume_data={"cldice": data_3way},
            metric_name="cldice",
            factor_names=all_factor_names[:3],
        )
        assert result_3.factor_names == all_factor_names[:3]

        # 6-way (all layers)
        result_6 = compute_factorial_anova(
            per_volume_data=data,
            metric_name="cldice",
            factor_names=all_factor_names,
        )
        assert result_6.factor_names == all_factor_names


# ---------------------------------------------------------------------------
# Phase 2.5: Condition key composition
# ---------------------------------------------------------------------------


class TestBuildPerVolumeDataConditionKeys:
    """_build_per_volume_data must compose 6-factor condition keys."""

    def test_condition_key_has_6_parts(self) -> None:
        """Verify condition keys are composed from all 6 SourceRun fields."""
        from minivess.orchestration.flows.biostatistics_flow import (
            _compose_condition_key,
        )
        from minivess.pipeline.biostatistics_types import SourceRun

        run = SourceRun(
            run_id="r1",
            experiment_id="e1",
            experiment_name="test",
            loss_function="dice_ce",
            fold_id=0,
            status="FINISHED",
            model_family="dynunet",
            with_aux_calib=False,
            post_training_method="swa",
            recalibration="none",
            ensemble_strategy="all_loss_all_best",
        )
        key = _compose_condition_key(run)
        parts = key.split("__")
        assert len(parts) == 6, f"Expected 6 parts, got {len(parts)}: {parts}"
        assert parts[0] == "dynunet"
        assert parts[1] == "dice_ce"
        assert parts[3] == "swa"
        assert parts[5] == "all_loss_all_best"


class TestBiostatisticsFlowAnovaWiring:
    """Biostatistics flow must call compute_factorial_anova."""

    def test_flow_imports_factorial_anova(self) -> None:
        import inspect

        from minivess.orchestration.flows import biostatistics_flow

        source = inspect.getsource(biostatistics_flow)
        assert "compute_factorial_anova" in source or "factorial_anova" in source, (
            "biostatistics_flow must import and call compute_factorial_anova"
        )
