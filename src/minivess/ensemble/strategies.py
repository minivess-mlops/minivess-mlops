from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from minivess.adapters.base import ModelAdapter
    from minivess.config.models import EnsembleConfig

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Combine predictions from multiple segmentation models.

    Supports multiple strategies: mean, majority voting, weighted mean,
    and greedy soup (weight averaging).
    """

    def __init__(
        self,
        models: list[ModelAdapter],
        config: EnsembleConfig,
    ) -> None:
        self.models = models
        self.config = config

    @torch.no_grad()
    def predict(self, images: Tensor, **kwargs: object) -> Tensor:
        """Run ensemble prediction with the configured strategy."""
        predictions: list[Tensor] = []
        for model in self.models:
            model.eval()
            output = model(images, **kwargs)
            predictions.append(output.prediction)

        stacked = torch.stack(predictions, dim=0)  # (M, B, C, D, H, W)

        strategy = self.config.strategy.value
        if strategy == "mean":
            return self._mean_ensemble(stacked)
        if strategy == "majority_vote":
            return self._majority_voting(stacked)
        if strategy == "weighted":
            return self._weighted_mean(stacked)
        msg = f"Unsupported ensemble strategy: {strategy}"
        raise ValueError(msg)

    def _mean_ensemble(self, stacked: Tensor) -> Tensor:
        """Average probabilities across ensemble members."""
        return stacked.mean(dim=0)

    def _majority_voting(self, stacked: Tensor) -> Tensor:
        """Hard majority voting on argmax predictions."""
        # (M, B, C, D, H, W) -> argmax -> (M, B, D, H, W)
        votes = stacked.argmax(dim=2)
        # Mode across ensemble members
        result, _ = torch.mode(votes, dim=0)
        # Convert back to one-hot probabilities
        num_classes = stacked.shape[2]
        one_hot = torch.nn.functional.one_hot(result, num_classes)
        # Rearrange to (B, C, D, H, W)
        return one_hot.permute(0, -1, *range(1, one_hot.ndim - 1)).float()

    def _weighted_mean(self, stacked: Tensor) -> Tensor:
        """Weighted average with temperature scaling."""
        temperature = self.config.temperature
        # Apply temperature to each member's predictions
        scaled = stacked / temperature
        # Normalize across classes
        scaled = torch.softmax(scaled, dim=2)
        return scaled.mean(dim=0)


def greedy_soup(
    models: list[ModelAdapter],
    val_metric_fn: object,
    val_loader: object,
) -> dict[str, Tensor]:
    """Greedy model soup: iteratively average weights, keep if metric improves.

    Based on Wortsman et al., "Model soups: averaging weights of multiple
    fine-tuned models improves accuracy without increasing inference time."

    Parameters
    ----------
    models:
        List of fine-tuned model adapters (sorted by val metric).
    val_metric_fn:
        Callable(model, val_loader) -> float metric (higher is better).
    val_loader:
        Validation data loader.

    Returns
    -------
    State dict of the best greedy soup.
    """
    if not models:
        msg = "Need at least one model for greedy soup"
        raise ValueError(msg)

    # Start with the best single model
    best_state = {k: v.clone() for k, v in models[0].state_dict().items()}
    best_metric: float = val_metric_fn(models[0], val_loader)
    soup_members: list[int] = [0]

    logger.info("Greedy soup: starting with model 0, metric=%.4f", best_metric)

    for i in range(1, len(models)):
        # Try adding this model to the soup
        candidate_state: dict[str, Tensor] = {}
        n = len(soup_members) + 1
        for key in best_state:
            candidate_state[key] = (
                best_state[key] * (n - 1) / n + models[i].state_dict()[key] / n
            )

        # Evaluate candidate soup
        models[0].load_state_dict(candidate_state)
        candidate_metric: float = val_metric_fn(models[0], val_loader)

        if candidate_metric > best_metric:
            best_state = candidate_state
            best_metric = candidate_metric
            soup_members.append(i)
            logger.info(
                "Greedy soup: added model %d, metric=%.4f (%d members)",
                i,
                candidate_metric,
                len(soup_members),
            )
        else:
            logger.info(
                "Greedy soup: rejected model %d, metric=%.4f < %.4f",
                i,
                candidate_metric,
                best_metric,
            )

    # Restore best soup
    models[0].load_state_dict(best_state)
    return best_state
