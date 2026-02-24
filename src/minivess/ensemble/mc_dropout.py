"""MC Dropout uncertainty estimation for segmentation models.

Enables dropout at inference time and runs N stochastic forward passes
to estimate predictive uncertainty (Gal & Ghahramani, 2016).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyOutput:
    """Standardized output for uncertainty estimation methods."""

    prediction: Tensor  # (B, C, D, H, W) mean probabilities
    uncertainty_map: Tensor  # (B, 1, D, H, W) voxel-wise uncertainty
    method: str  # "mc_dropout" | "deep_ensemble" | "conformal"
    metadata: dict[str, Any] = field(default_factory=dict)


def _enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers while keeping everything else in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


class MCDropoutPredictor:
    """MC Dropout uncertainty estimation.

    Runs N stochastic forward passes with dropout enabled at inference,
    then computes the mean prediction and voxel-wise predictive entropy
    as the uncertainty map.

    Parameters
    ----------
    model:
        A ModelAdapter with dropout layers.
    n_samples:
        Number of stochastic forward passes.
    """

    def __init__(self, model: nn.Module, n_samples: int = 10) -> None:
        self.model = model
        self.n_samples = n_samples

    @torch.no_grad()
    def predict(self, images: Tensor) -> UncertaintyOutput:
        """Run MC Dropout inference.

        Parameters
        ----------
        images:
            Input tensor (B, C_in, D, H, W).

        Returns
        -------
        UncertaintyOutput with mean prediction and entropy-based uncertainty.
        """
        _enable_dropout(self.model)

        samples: list[Tensor] = []
        for _ in range(self.n_samples):
            output = self.model(images)
            samples.append(output.prediction)

        # (N, B, C, D, H, W)
        stacked = torch.stack(samples, dim=0)
        mean_pred = stacked.mean(dim=0)  # (B, C, D, H, W)

        # Predictive entropy: -sum(p * log(p)) across classes
        eps = 1e-8
        entropy = -(mean_pred * torch.log(mean_pred + eps)).sum(dim=1, keepdim=True)

        # Restore eval mode
        self.model.eval()

        return UncertaintyOutput(
            prediction=mean_pred,
            uncertainty_map=entropy,
            method="mc_dropout",
            metadata={"n_samples": self.n_samples},
        )
