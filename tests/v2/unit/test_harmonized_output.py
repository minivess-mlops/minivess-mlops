"""Tests for harmonized serving output schema (Phase 1, #333).

Verifies that HarmonizedSegmentationOutput works with Optional[T]=None
pattern for deterministic, probabilistic, ensemble, and conformal models.
"""

from __future__ import annotations

import numpy as np

from minivess.serving.harmonized_output import (
    HarmonizedSegmentationOutput,
    validate_output,
)


def _make_deterministic_output() -> HarmonizedSegmentationOutput:
    """Minimal deterministic model output (no UQ)."""
    shape = (8, 32, 32)
    return HarmonizedSegmentationOutput(
        binary_mask=np.ones(shape, dtype=np.uint8),
        probabilities=np.full(shape, 0.9, dtype=np.float32),
        volume_id="mv01",
        model_name="dynunet_cbdice_cldice",
    )


def _make_probabilistic_output() -> HarmonizedSegmentationOutput:
    """Probabilistic model output with UQ fields populated."""
    shape = (8, 32, 32)
    return HarmonizedSegmentationOutput(
        binary_mask=np.ones(shape, dtype=np.uint8),
        probabilities=np.full(shape, 0.9, dtype=np.float32),
        volume_id="mv01",
        model_name="dynunet_ensemble_mean",
        uncertainty_map=np.full(shape, 0.1, dtype=np.float32),
        aleatoric_uncertainty=np.full(shape, 0.05, dtype=np.float32),
        epistemic_uncertainty=np.full(shape, 0.05, dtype=np.float32),
        mutual_information=np.full(shape, 0.02, dtype=np.float32),
    )


class TestDeterministicOutput:
    """Deterministic models: binary_mask + probabilities, UQ = None."""

    def test_valid(self) -> None:
        output = _make_deterministic_output()
        errors = validate_output(output)
        assert errors == []

    def test_uq_fields_are_none(self) -> None:
        output = _make_deterministic_output()
        assert output.uncertainty_map is None
        assert output.aleatoric_uncertainty is None
        assert output.epistemic_uncertainty is None


class TestProbabilisticOutput:
    """Probabilistic models: all UQ fields populated."""

    def test_valid(self) -> None:
        output = _make_probabilistic_output()
        errors = validate_output(output)
        assert errors == []

    def test_uq_fields_populated(self) -> None:
        output = _make_probabilistic_output()
        assert output.uncertainty_map is not None
        assert output.aleatoric_uncertainty is not None


class TestEnsembleOutput:
    """Ensemble models: n_ensemble_members populated."""

    def test_valid(self) -> None:
        shape = (8, 32, 32)
        output = HarmonizedSegmentationOutput(
            binary_mask=np.ones(shape, dtype=np.uint8),
            probabilities=np.full(shape, 0.85, dtype=np.float32),
            volume_id="mv01",
            model_name="ensemble_mean_4",
            n_ensemble_members=4,
            ensemble_strategy="mean",
            member_model_names=["m1", "m2", "m3", "m4"],
        )
        errors = validate_output(output)
        assert errors == []
        assert output.n_ensemble_members == 4


class TestConformalOutput:
    """Conformal models: prediction_set + coverage."""

    def test_valid(self) -> None:
        shape = (8, 32, 32)
        output = HarmonizedSegmentationOutput(
            binary_mask=np.ones(shape, dtype=np.uint8),
            probabilities=np.full(shape, 0.85, dtype=np.float32),
            volume_id="mv01",
            model_name="conformal_morpho",
            prediction_set=np.ones(shape, dtype=np.uint8),
            coverage_guarantee=0.9,
            conformal_alpha=0.1,
        )
        errors = validate_output(output)
        assert errors == []


class TestValidateOutputErrors:
    """validate_output catches schema violations."""

    def test_catches_wrong_dtype(self) -> None:
        shape = (8, 32, 32)
        output = HarmonizedSegmentationOutput(
            binary_mask=np.ones(shape, dtype=np.float32),  # WRONG: should be uint8
            probabilities=np.full(shape, 0.9, dtype=np.float32),
            volume_id="mv01",
            model_name="test",
        )
        errors = validate_output(output)
        assert any("uint8" in e for e in errors)

    def test_catches_shape_mismatch(self) -> None:
        output = HarmonizedSegmentationOutput(
            binary_mask=np.ones((8, 32, 32), dtype=np.uint8),
            probabilities=np.full((8, 32, 16), 0.9, dtype=np.float32),  # WRONG shape
            volume_id="mv01",
            model_name="test",
        )
        errors = validate_output(output)
        assert any("shape mismatch" in e for e in errors)

    def test_catches_probability_out_of_range(self) -> None:
        shape = (8, 32, 32)
        output = HarmonizedSegmentationOutput(
            binary_mask=np.ones(shape, dtype=np.uint8),
            probabilities=np.full(shape, 1.5, dtype=np.float32),  # WRONG: > 1
            volume_id="mv01",
            model_name="test",
        )
        errors = validate_output(output)
        assert any("outside [0, 1]" in e for e in errors)

    def test_catches_conformal_inconsistency(self) -> None:
        shape = (8, 32, 32)
        output = HarmonizedSegmentationOutput(
            binary_mask=np.ones(shape, dtype=np.uint8),
            probabilities=np.full(shape, 0.9, dtype=np.float32),
            volume_id="mv01",
            model_name="test",
            prediction_set=np.ones(shape, dtype=np.uint8),
            # coverage_guarantee missing!
        )
        errors = validate_output(output)
        assert any("coverage_guarantee" in e for e in errors)
