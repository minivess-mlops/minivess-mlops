"""Tests for deep ensemble uncertainty decomposition (Lakshminarayanan et al., 2017).

Verifies that DeepEnsemblePredictor computes:
- Total predictive uncertainty: entropy of mean softmax
- Aleatoric uncertainty: mean of individual entropies
- Epistemic uncertainty: mutual information (total - aleatoric)

GitHub Issue #88.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor
from minivess.ensemble.mc_dropout import UncertaintyOutput

# ---------------------------------------------------------------------------
# Mock model helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockOutput:
    """Mimics ModelAdapter output with a .prediction attribute."""

    prediction: Tensor


class _MockModel(nn.Module):
    """Simple model returning constant soft predictions for testing."""

    def __init__(self, prob: float = 0.8) -> None:
        super().__init__()
        self._prob = prob

    def forward(self, x: Tensor) -> _MockOutput:
        B, _C_in, D, H, W = x.shape
        fg = torch.full((B, 1, D, H, W), self._prob, device=x.device, dtype=x.dtype)
        bg = 1.0 - fg
        pred = torch.cat([bg, fg], dim=1)  # (B, 2, D, H, W)
        return _MockOutput(prediction=pred)


class _NoisyMockModel(nn.Module):
    """Model returning predictions with random noise for diversity testing."""

    def __init__(self, base_prob: float = 0.7, noise_scale: float = 0.15) -> None:
        super().__init__()
        self._base_prob = base_prob
        self._noise_scale = noise_scale

    def forward(self, x: Tensor) -> _MockOutput:
        B, _C_in, D, H, W = x.shape
        noise = torch.randn(B, 1, D, H, W, device=x.device, dtype=x.dtype)
        fg = (self._base_prob + self._noise_scale * noise).clamp(0.01, 0.99)
        bg = 1.0 - fg
        pred = torch.cat([bg, fg], dim=1)
        return _MockOutput(prediction=pred)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 2
IN_CH = 1
DEPTH = 8
HEIGHT = 8
WIDTH = 8


@pytest.fixture()
def images() -> Tensor:
    """Random input tensor (B, C_in, D, H, W)."""
    return torch.randn(BATCH, IN_CH, DEPTH, HEIGHT, WIDTH)


@pytest.fixture()
def uniform_ensemble() -> DeepEnsemblePredictor:
    """Ensemble of 3 identical models (same constant prob)."""
    models = [_MockModel(prob=0.8) for _ in range(3)]
    return DeepEnsemblePredictor(models)


@pytest.fixture()
def diverse_ensemble() -> DeepEnsemblePredictor:
    """Ensemble of 3 models with very different probabilities."""
    models = [_MockModel(prob=p) for p in [0.2, 0.5, 0.9]]
    return DeepEnsemblePredictor(models)


@pytest.fixture()
def single_model_ensemble() -> DeepEnsemblePredictor:
    """Ensemble with exactly 1 member."""
    return DeepEnsemblePredictor([_MockModel(prob=0.7)])


@pytest.fixture()
def noisy_ensemble() -> DeepEnsemblePredictor:
    """Ensemble of noisy models producing genuinely diverse outputs."""
    models = [_NoisyMockModel(base_prob=p, noise_scale=0.2) for p in [0.3, 0.5, 0.8]]
    return DeepEnsemblePredictor(models)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeepEnsembleUncertaintyDecomposition:
    """Lakshminarayanan et al. (2017) uncertainty decomposition tests."""

    def test_predict_returns_uncertainty_output(
        self, uniform_ensemble: DeepEnsemblePredictor, images: Tensor
    ) -> None:
        """predict() must return an UncertaintyOutput dataclass."""
        result = uniform_ensemble.predict(images)
        assert isinstance(result, UncertaintyOutput)
        assert result.method == "deep_ensemble"

    def test_total_uncertainty_shape(
        self, uniform_ensemble: DeepEnsemblePredictor, images: Tensor
    ) -> None:
        """Total uncertainty (uncertainty_map) must be (B, 1, D, H, W)."""
        result = uniform_ensemble.predict(images)
        expected_shape = (BATCH, 1, DEPTH, HEIGHT, WIDTH)
        assert result.uncertainty_map.shape == expected_shape

    def test_aleatoric_uncertainty_shape(
        self, uniform_ensemble: DeepEnsemblePredictor, images: Tensor
    ) -> None:
        """Aleatoric uncertainty must be (B, 1, D, H, W)."""
        result = uniform_ensemble.predict(images)
        aleatoric = result.metadata["aleatoric_uncertainty"]
        expected_shape = (BATCH, 1, DEPTH, HEIGHT, WIDTH)
        assert aleatoric.shape == expected_shape

    def test_epistemic_uncertainty_shape(
        self, uniform_ensemble: DeepEnsemblePredictor, images: Tensor
    ) -> None:
        """Epistemic uncertainty must be (B, 1, D, H, W)."""
        result = uniform_ensemble.predict(images)
        epistemic = result.metadata["epistemic_uncertainty"]
        expected_shape = (BATCH, 1, DEPTH, HEIGHT, WIDTH)
        assert epistemic.shape == expected_shape

    def test_epistemic_equals_total_minus_aleatoric(
        self, diverse_ensemble: DeepEnsemblePredictor, images: Tensor
    ) -> None:
        """Epistemic = total - aleatoric (mutual information identity)."""
        result = diverse_ensemble.predict(images)
        total = result.metadata["total_uncertainty"]
        aleatoric = result.metadata["aleatoric_uncertainty"]
        epistemic = result.metadata["epistemic_uncertainty"]
        expected_epistemic = total - aleatoric
        assert torch.allclose(epistemic, expected_epistemic, atol=1e-6)

    def test_single_model_epistemic_near_zero(
        self, single_model_ensemble: DeepEnsemblePredictor, images: Tensor
    ) -> None:
        """With 1 member, epistemic uncertainty should be ~0 (no disagreement)."""
        result = single_model_ensemble.predict(images)
        epistemic = result.metadata["epistemic_uncertainty"]
        # Single model: mean == individual, so total == aleatoric, epistemic ~ 0
        assert torch.allclose(epistemic, torch.zeros_like(epistemic), atol=1e-6)

    def test_diverse_models_epistemic_positive(
        self, noisy_ensemble: DeepEnsemblePredictor, images: Tensor
    ) -> None:
        """Diverse ensemble members should produce positive epistemic uncertainty."""
        result = noisy_ensemble.predict(images)
        epistemic = result.metadata["epistemic_uncertainty"]
        # Mean epistemic should be strictly positive for diverse members
        assert epistemic.mean().item() > 0.0

    def test_metadata_keys_present(
        self, uniform_ensemble: DeepEnsemblePredictor, images: Tensor
    ) -> None:
        """All required metadata keys must be present."""
        result = uniform_ensemble.predict(images)
        required_keys = {
            "total_uncertainty",
            "aleatoric_uncertainty",
            "epistemic_uncertainty",
            "n_members",
            "mean_variance",
        }
        assert required_keys.issubset(result.metadata.keys())

    def test_prediction_shape_unchanged(
        self, uniform_ensemble: DeepEnsemblePredictor, images: Tensor
    ) -> None:
        """Mean prediction shape should remain (B, C, D, H, W)."""
        result = uniform_ensemble.predict(images)
        # C=2 for binary segmentation (bg + fg)
        expected_shape = (BATCH, 2, DEPTH, HEIGHT, WIDTH)
        assert result.prediction.shape == expected_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_cpu_and_gpu_compatible(self) -> None:
        """Uncertainty decomposition should work on both CPU and CUDA tensors."""
        device = torch.device("cuda")
        models = [_MockModel(prob=0.6).to(device) for _ in range(2)]
        predictor = DeepEnsemblePredictor(models)
        x = torch.randn(1, IN_CH, 4, 4, 4, device=device)
        result = predictor.predict(x)
        assert result.uncertainty_map.device.type == "cuda"
        assert result.metadata["epistemic_uncertainty"].device.type == "cuda"
        assert result.metadata["aleatoric_uncertainty"].device.type == "cuda"
