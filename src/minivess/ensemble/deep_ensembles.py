"""Deep ensemble uncertainty estimation for segmentation models.

Uses disagreement (variance) across independently trained models
as the uncertainty measure (Lakshminarayanan et al., 2017).
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from minivess.ensemble.mc_dropout import UncertaintyOutput

logger = logging.getLogger(__name__)


class DeepEnsemblePredictor:
    """Deep ensemble uncertainty from model disagreement.

    Runs inference on each ensemble member and computes
    the per-voxel variance as the uncertainty map.

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
        """Run deep ensemble inference.

        Parameters
        ----------
        images:
            Input tensor (B, C_in, D, H, W).

        Returns
        -------
        UncertaintyOutput with mean prediction and variance-based uncertainty.
        """
        predictions: list[Tensor] = []
        for model in self.models:
            model.eval()
            output = model(images)
            predictions.append(output.prediction)

        # (M, B, C, D, H, W)
        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0)  # (B, C, D, H, W)

        # Per-voxel variance across members, mean across classes
        # Use correction=0 (population variance) to handle single-model case
        variance = stacked.var(dim=0, correction=0).mean(dim=1, keepdim=True)

        return UncertaintyOutput(
            prediction=mean_pred,
            uncertainty_map=variance,
            method="deep_ensemble",
            metadata={"n_members": len(self.models)},
        )
