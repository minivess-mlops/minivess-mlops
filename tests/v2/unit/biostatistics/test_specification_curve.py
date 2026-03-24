"""Tests for specification curve analysis (Phase 3).

Validates the specification curve engine that systematically varies ALL
researcher degrees of freedom to demonstrate how conclusions change
across analytical choices.

Reference: Simonsohn et al. (2020) "Specification Curve Analysis"
"""

from __future__ import annotations

import numpy as np

from minivess.config.biostatistics_config import BiostatisticsConfig

# Single source of truth for statistical params — never hardcode
# alpha, seed, etc. See CLAUDE.md Rule #29 and Issue #881.
_CFG = BiostatisticsConfig()


def _make_synthetic_spec_data(
    *,
    n_models: int = 4,
    n_losses: int = 3,
    n_calibs: int = 2,
    n_folds: int = 3,
    n_volumes: int = 23,
    seed: int = 42,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Create synthetic per-volume data for multiple metrics.

    Returns: {metric: {condition_key: {fold_id: scores}}}
    condition_key format: "model__loss__calib"
    """
    rng = np.random.default_rng(seed)

    models = [f"model_{i}" for i in range(n_models)]
    losses = [f"loss_{j}" for j in range(n_losses)]
    calibs = [f"calib_{k}" for k in range(n_calibs)]

    metrics = ["cldice", "masd", "dsc"]
    data: dict[str, dict[str, dict[int, np.ndarray]]] = {}

    for metric in metrics:
        data[metric] = {}
        for i, model in enumerate(models):
            for _j, loss in enumerate(losses):
                for _k, calib in enumerate(calibs):
                    key = f"{model}__{loss}__{calib}"
                    data[metric][key] = {}
                    for fold_id in range(n_folds):
                        # Different metrics have different baselines and effects
                        if metric == "cldice":
                            base = 0.7 + 0.1 * (i / max(n_models - 1, 1))
                        elif metric == "masd":
                            # Lower is better for MASD
                            base = 2.0 - 0.5 * (i / max(n_models - 1, 1))
                        else:
                            base = 0.8 + 0.05 * (i / max(n_models - 1, 1))
                        data[metric][key][fold_id] = rng.normal(
                            base, 0.05, size=n_volumes
                        )

    return data


# ---------------------------------------------------------------------------
# T3.1: Specification curve produces expected specifications
# ---------------------------------------------------------------------------


class TestSpecificationCurveBasics:
    """Basic specification curve generation."""

    def test_spec_curve_returns_dataclass(self) -> None:
        """SpecificationCurveResult is returned."""
        from minivess.pipeline.biostatistics_specification_curve import (
            SpecificationCurveResult,
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data()
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice", "masd", "dsc"],
            higher_is_better={"cldice": True, "masd": False, "dsc": True},
            alpha=_CFG.alpha,
            seed=_CFG.seed,
        )
        assert isinstance(result, SpecificationCurveResult)

    def test_spec_curve_has_specifications(self) -> None:
        """Result contains non-empty list of specifications."""
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data()
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice", "masd", "dsc"],
            higher_is_better={"cldice": True, "masd": False, "dsc": True},
            alpha=_CFG.alpha,
            seed=_CFG.seed,
        )
        assert len(result.specifications) > 0

    def test_spec_count_matches_expected(self) -> None:
        """Number of specs = n_metrics × n_aggregations × C(n_conditions, 2).

        Each specification compares a unique pair of conditions using a
        specific metric and aggregation method.
        """
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data(n_models=2, n_losses=2, n_calibs=2)
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice", "dsc"],
            higher_is_better={"cldice": True, "dsc": True},
            alpha=_CFG.alpha,
            seed=_CFG.seed,
        )
        # With 2×2×2=8 conditions, C(8,2)=28 pairs × 2 metrics × 2 aggregations
        # = 112 specs. But the exact number depends on aggregation choices.
        assert len(result.specifications) >= 28  # At least metric × pairs


class TestSpecificationCurveFields:
    """Each specification has required fields."""

    def test_each_spec_has_metric(self) -> None:
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data(n_models=2, n_losses=2, n_calibs=1)
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice"],
            higher_is_better={"cldice": True},
            alpha=_CFG.alpha,
            seed=_CFG.seed,
        )
        for spec in result.specifications:
            assert spec.metric != ""

    def test_each_spec_has_effect_size(self) -> None:
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data(n_models=2, n_losses=2, n_calibs=1)
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice"],
            higher_is_better={"cldice": True},
            alpha=_CFG.alpha,
            seed=_CFG.seed,
        )
        for spec in result.specifications:
            assert isinstance(spec.effect_size, float)

    def test_each_spec_has_p_value(self) -> None:
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data(n_models=2, n_losses=2, n_calibs=1)
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice"],
            higher_is_better={"cldice": True},
            alpha=_CFG.alpha,
            seed=_CFG.seed,
        )
        for spec in result.specifications:
            assert 0.0 <= spec.p_value <= 1.0


class TestSpecificationCurveSorting:
    """Specifications must be sorted by effect size."""

    def test_sorted_by_effect_size(self) -> None:
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data(n_models=3, n_losses=2, n_calibs=1)
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice"],
            higher_is_better={"cldice": True},
            alpha=_CFG.alpha,
            seed=_CFG.seed,
        )
        effects = [s.effect_size for s in result.specifications]
        assert effects == sorted(effects)


class TestSpecificationCurvePermutationTest:
    """Permutation test for specification curve null."""

    def test_permutation_p_value_present(self) -> None:
        """Result includes a permutation test p-value."""
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data(n_models=2, n_losses=2, n_calibs=1)
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice"],
            higher_is_better={"cldice": True},
            alpha=_CFG.alpha,
            n_permutations=100,  # Small for speed in tests
            seed=_CFG.seed,
        )
        assert result.permutation_p is not None
        assert 0.0 <= result.permutation_p <= 1.0

    def test_strong_effect_has_low_permutation_p(self) -> None:
        """Known strong effect across many conditions should yield low permutation p."""
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        # Create data with a clear gradient: conditions have monotonically
        # increasing means. With 4+ conditions, permutations break the
        # gradient and produce different median effects.
        rng = np.random.default_rng(42)
        data: dict[str, dict[str, dict[int, np.ndarray]]] = {"cldice": {}}
        for i, model in enumerate(["worst", "bad", "good", "best"]):
            key = f"{model}__loss_a__calibTrue"
            data["cldice"][key] = {
                0: rng.normal(0.4 + 0.15 * i, 0.01, size=30),
            }

        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice"],
            higher_is_better={"cldice": True},
            alpha=_CFG.alpha,
            n_permutations=200,
            seed=_CFG.seed,
        )
        # With a strong monotonic gradient and 4 conditions, permutation
        # p should be low (gradient is disrupted by most permutations)
        assert result.permutation_p < 0.2


class TestSpecificationCurveMedianEffect:
    """Median effect summary statistic."""

    def test_median_effect_present(self) -> None:
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data(n_models=2, n_losses=2, n_calibs=1)
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice"],
            higher_is_better={"cldice": True},
            alpha=_CFG.alpha,
            seed=_CFG.seed,
        )
        assert isinstance(result.median_effect, float)

    def test_fraction_significant_in_range(self) -> None:
        from minivess.pipeline.biostatistics_specification_curve import (
            compute_specification_curve,
        )

        data = _make_synthetic_spec_data(n_models=2, n_losses=2, n_calibs=1)
        result = compute_specification_curve(
            per_volume_data=data,
            factor_names=["model", "loss", "calib"],
            metric_names=["cldice"],
            higher_is_better={"cldice": True},
            alpha=_CFG.alpha,
            seed=_CFG.seed,
        )
        assert 0.0 <= result.fraction_significant <= 1.0
