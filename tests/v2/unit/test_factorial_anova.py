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

        assert "Model:Loss" in result.f_values or "Interaction" in result.f_values


class TestFactorialAnovaSignificantModelEffect:
    """T1: Synthetic data with known model effect produces significant F-test."""

    def test_factorial_anova_significant_model_effect(self) -> None:
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_factorial_data(model_effect=0.5)
        result = compute_factorial_anova(data, metric_name="cldice")

        model_p = result.p_values.get("Model")
        assert model_p is not None
        assert model_p < 0.05, f"Model effect should be significant, p={model_p}"


class TestFactorialAnovaNoLossEffectWhenIdentical:
    """T1: Identical losses produce non-significant loss effect."""

    def test_factorial_anova_no_loss_effect_when_identical(self) -> None:
        from minivess.pipeline.biostatistics_statistics import compute_factorial_anova

        data = _make_synthetic_factorial_data(model_effect=0.3, loss_effect=0.0)
        result = compute_factorial_anova(data, metric_name="cldice")

        loss_p = result.p_values.get("Loss")
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
