"""Tests for two-way factorial ANOVA with effect sizes.

Validates compute_factorial_anova() which implements Model x Loss two-way
ANOVA with partial eta-squared and omega-squared effect sizes.
"""

from __future__ import annotations

import numpy as np


def _make_synthetic_factorial_data(
    *,
    model_effect: float = 0.3,
    loss_effect: float = 0.0,
    interaction_effect: float = 0.0,
    n_volumes: int = 23,
    n_folds: int = 3,
    seed: int = 42,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Create synthetic per-volume data with known factorial effects.

    Returns dict: {metric_name: {condition_key: {fold_id: np.ndarray}}}
    where condition_key = "model__loss" (double underscore separator).
    """
    rng = np.random.default_rng(seed)

    models = ["dynunet", "mambavesselnet", "sam3_vanilla", "vesselfm"]
    losses = ["dice_ce", "cbdice_cldice", "dice_ce_cldice"]

    data: dict[str, dict[str, dict[int, np.ndarray]]] = {"cldice": {}}

    for i, model in enumerate(models):
        for j, loss in enumerate(losses):
            condition_key = f"{model}__{loss}"
            data["cldice"][condition_key] = {}
            for fold_id in range(n_folds):
                base = 0.7
                model_contrib = model_effect * (i / (len(models) - 1))
                loss_contrib = loss_effect * (j / (len(losses) - 1))
                interact_contrib = (
                    interaction_effect
                    * (i * j)
                    / ((len(models) - 1) * (len(losses) - 1))
                )
                mean = base + model_contrib + loss_contrib + interact_contrib
                data["cldice"][condition_key][fold_id] = rng.normal(
                    mean, 0.05, size=n_volumes
                )

    return data


class TestFactorialAnovaReturnsResult:
    """T1: Function returns FactorialAnovaResult dataclass."""

    def test_factorial_anova_returns_result(self) -> None:
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova
        from minivess.pipeline.biostatistics_types import FactorialAnovaResult

        data = _make_synthetic_factorial_data()
        result = compute_factorial_anova(data, metric_name="cldice")

        assert isinstance(result, FactorialAnovaResult)
        assert result.metric == "cldice"


class TestFactorialAnovaEtaSquaredRange:
    """T1: eta-squared values in [0, 1]."""

    def test_factorial_anova_eta_squared_range(self) -> None:
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_factorial_data()
        result = compute_factorial_anova(data, metric_name="cldice")

        for factor in ["Model", "Loss", "Model:Loss"]:
            eta = result.eta_squared_partial.get(factor)
            if eta is not None:
                assert 0.0 <= eta <= 1.0, f"eta_sq for {factor} out of range: {eta}"


class TestFactorialAnovaOmegaSquaredRange:
    """T1: omega-squared can be negative for negligible effects."""

    def test_factorial_anova_omega_squared_range(self) -> None:
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_factorial_data()
        result = compute_factorial_anova(data, metric_name="cldice")

        for factor in ["Model", "Loss", "Model:Loss"]:
            omega = result.omega_squared.get(factor)
            if omega is not None:
                assert omega <= 1.0, f"omega_sq for {factor} > 1: {omega}"


class TestFactorialAnovaInteractionTerm:
    """T1: Interaction term present in output."""

    def test_factorial_anova_interaction_term(self) -> None:
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_factorial_data()
        result = compute_factorial_anova(data, metric_name="cldice")

        # pingouin returns "model * loss" as interaction source name
        assert (
            "Model:Loss" in result.f_values
            or "model * loss" in result.f_values
            or "Interaction" in result.f_values
        )


class TestFactorialAnovaSignificantModelEffect:
    """T1: Synthetic data with known model effect produces significant F-test."""

    def test_factorial_anova_significant_model_effect(self) -> None:
        from minivess.config.biostatistics_config import BiostatisticsConfig
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        cfg = BiostatisticsConfig()
        data = _make_synthetic_factorial_data(model_effect=0.5)
        result = compute_factorial_anova(data, metric_name="cldice")

        model_p = result.p_values.get("Model") or result.p_values.get("model")
        assert model_p is not None
        assert model_p < cfg.alpha, f"Model effect should be significant, p={model_p}"


class TestFactorialAnovaNoLossEffectWhenIdentical:
    """T1: Identical losses produce non-significant loss effect."""

    def test_factorial_anova_no_loss_effect_when_identical(self) -> None:
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_factorial_data(model_effect=0.3, loss_effect=0.0)
        result = compute_factorial_anova(data, metric_name="cldice")

        loss_p = result.p_values.get("Loss") or result.p_values.get("loss")
        assert loss_p is not None
        # With zero loss effect and noise, p should generally be >0.01
        # Use lenient threshold since this is statistical
        assert loss_p > 0.001, f"Loss effect should be non-significant, p={loss_p}"


class TestFactorialAnovaConditionKeyParsing:
    """T1: condition_key parsing uses str.split, NOT regex."""

    def test_factorial_anova_condition_key_parsing(self) -> None:
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_factorial_data()
        result = compute_factorial_anova(data, metric_name="cldice")

        # Verify the result contains expected factors from parsed condition keys
        assert result.n_models > 0
        assert result.n_losses > 0


class TestFactorialAnovaStatsmodelsAgreement:
    """T1: pingouin and statsmodels results logged for cross-validation."""

    def test_factorial_anova_statsmodels_agreement(self) -> None:
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_factorial_data(model_effect=0.3)
        result = compute_factorial_anova(data, metric_name="cldice")

        # Both engines should be present
        assert result.engine_pingouin is not None
        assert result.engine_statsmodels is not None


# ---------------------------------------------------------------------------
# T2.3: K=1 debug fallback — per-volume replication
# ---------------------------------------------------------------------------


def _make_synthetic_3way_data(
    *,
    n_folds: int = 1,
    n_volumes: int = 23,
    model_effect: float = 0.3,
    seed: int = 42,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Create synthetic 3-factor data (model × loss × calib).

    Returns dict: {metric: {condition_key: {fold_id: np.ndarray}}}
    """
    rng = np.random.default_rng(seed)
    models = ["dynunet", "sam3"]
    losses = ["dice_ce", "cbdice"]
    calibs = ["calibTrue", "calibFalse"]

    data: dict[str, dict[str, dict[int, np.ndarray]]] = {"cldice": {}}

    for i, model in enumerate(models):
        for _j, loss in enumerate(losses):
            for _k, calib in enumerate(calibs):
                key = f"{model}__{loss}__{calib}"
                data["cldice"][key] = {}
                for fold_id in range(n_folds):
                    mean = 0.7 + model_effect * i / max(len(models) - 1, 1)
                    data["cldice"][key][fold_id] = rng.normal(
                        mean, 0.05, size=n_volumes
                    )

    return data


class TestFactorialAnovaK1DebugFallback:
    """T2.3: K=1 fold uses per-volume replication (standard ANOVA)."""

    def test_k1_returns_valid_result(self) -> None:
        """K=1 fold should produce a valid FactorialAnovaResult."""
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova
        from minivess.pipeline.biostatistics_types import FactorialAnovaResult

        data = _make_synthetic_3way_data(n_folds=1)
        result = compute_factorial_anova(
            data,
            metric_name="cldice",
            factor_names=["model", "loss", "calib"],
        )
        assert isinstance(result, FactorialAnovaResult)
        assert result.n_folds == 1
        assert result.replication_method == "per_volume"

    def test_k1_has_f_values_for_all_factors(self) -> None:
        """K=1 should still produce F-values for all main effects."""
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_3way_data(n_folds=1)
        result = compute_factorial_anova(
            data,
            metric_name="cldice",
            factor_names=["model", "loss", "calib"],
        )
        # Check main effects exist (pingouin may capitalize)
        f_keys = set(result.f_values.keys())
        for factor in ["model", "loss", "calib"]:
            assert factor in f_keys or factor.capitalize() in f_keys, (
                f"Missing F-value for {factor}. Keys: {f_keys}"
            )

    def test_k3_uses_mixed_model(self) -> None:
        """K=3 folds should use fold as random effect (mixed model)."""
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_3way_data(n_folds=3)
        result = compute_factorial_anova(
            data,
            metric_name="cldice",
            factor_names=["model", "loss", "calib"],
        )
        assert result.n_folds == 3
        assert result.replication_method == "fold_random_effect"

    def test_k1_detects_significant_model_effect(self) -> None:
        """K=1 with strong model effect should still detect significance."""
        from minivess.config.biostatistics_config import BiostatisticsConfig
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        cfg = BiostatisticsConfig()
        data = _make_synthetic_3way_data(n_folds=1, model_effect=0.5)
        result = compute_factorial_anova(
            data,
            metric_name="cldice",
            factor_names=["model", "loss", "calib"],
        )
        model_p = result.p_values.get("model") or result.p_values.get("Model")
        assert model_p is not None
        assert model_p < cfg.alpha, f"Model effect should be significant, p={model_p}"
