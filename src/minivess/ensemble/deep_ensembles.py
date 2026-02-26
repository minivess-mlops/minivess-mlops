"""Deep ensemble uncertainty estimation for segmentation models.

Implements full Lakshminarayanan et al. (2017) uncertainty decomposition:
- Total predictive uncertainty: entropy of mean softmax (predictive entropy)
- Aleatoric uncertainty: mean of individual member entropies
- Epistemic uncertainty: mutual information (total - aleatoric)
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from minivess.ensemble.mc_dropout import UncertaintyOutput

logger = logging.getLogger(__name__)

# Small epsilon to avoid log(0)
_EPS = 1e-8


def _entropy(probs: Tensor) -> Tensor:
    """Compute per-voxel entropy across the class dimension.

    Parameters
    ----------
    probs:
        Probability tensor of shape (..., C, D, H, W) where C is the
        number of classes.

    Returns
    -------
    Entropy tensor of shape (..., 1, D, H, W).
    """
    # -sum(p * log(p)) across the class dimension (dim=-4 covers both
    # (B, C, D, H, W) and (M, B, C, D, H, W) when applied correctly)
    return -(probs * torch.log(probs + _EPS)).sum(dim=-4, keepdim=True)


class DeepEnsemblePredictor:
    """Deep ensemble with full uncertainty decomposition.

    Runs inference on each ensemble member and computes the
    Lakshminarayanan et al. (2017) uncertainty decomposition:

    - **Total** (predictive entropy): ``H[p_bar]`` where ``p_bar = mean(p_m)``
    - **Aleatoric**: ``mean(H[p_m])`` (expected entropy of individual members)
    - **Epistemic** (mutual information): ``total - aleatoric``

    Parameters
    ----------
    models:
        List of ModelAdapter instances (independently trained).
    """

    def __init__(self, models: list[nn.Module]) -> None:
        if not models:
            msg = "DeepEnsemblePredictor requires at least one model"
            raise ValueError(msg)
        self.models = models

    @torch.no_grad()
    def predict(self, images: Tensor) -> UncertaintyOutput:
        """Run deep ensemble inference with uncertainty decomposition.

        Parameters
        ----------
        images:
            Input tensor (B, C_in, D, H, W).

        Returns
        -------
        UncertaintyOutput with:
            - prediction: mean softmax probabilities (B, C, D, H, W)
            - uncertainty_map: total predictive uncertainty (B, 1, D, H, W)
            - metadata containing total, aleatoric, epistemic maps plus
              backward-compatible mean_variance and n_members.
        """
        predictions: list[Tensor] = []
        for model in self.models:
            model.eval()
            output = model(images)
            predictions.append(output.prediction)

        # (M, B, C, D, H, W)
        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0)  # (B, C, D, H, W)

        # --- Lakshminarayanan et al. (2017) decomposition ---

        # Total predictive uncertainty: entropy of mean softmax
        # H[p_bar] = -sum(p_bar * log(p_bar)) across classes
        # Shape: (B, 1, D, H, W)
        total_uncertainty = _entropy(mean_pred)

        # Aleatoric uncertainty: mean of individual member entropies
        # (1/M) * sum_m H[p_m]
        # _entropy on stacked (M, B, C, D, H, W) -> (M, B, 1, D, H, W)
        individual_entropies = _entropy(stacked)
        aleatoric_uncertainty = individual_entropies.mean(dim=0)  # (B, 1, D, H, W)

        # Epistemic uncertainty: mutual information = total - aleatoric
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

        # Backward-compatible: per-voxel variance across members, mean across classes
        mean_variance = stacked.var(dim=0, correction=0).mean(dim=1, keepdim=True)

        return UncertaintyOutput(
            prediction=mean_pred,
            uncertainty_map=total_uncertainty,
            method="deep_ensemble",
            metadata={
                "n_members": len(self.models),
                "total_uncertainty": total_uncertainty,
                "aleatoric_uncertainty": aleatoric_uncertainty,
                "epistemic_uncertainty": epistemic_uncertainty,
                "mean_variance": mean_variance,
            },
        )
