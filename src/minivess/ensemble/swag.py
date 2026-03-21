"""SWAG posterior approximation (Maddox et al. 2019).

Wraps a base model with low-rank-plus-diagonal Gaussian posterior tracking.
Uses torch.optim.swa_utils.AveragedModel for running mean computation,
plus custom tracking for second moments and low-rank deviations.

Reference: Maddox et al. (2019), "A Simple Baseline for Bayesian Inference
in Deep Learning" (https://arxiv.org/abs/1902.02476)
Reference implementation: github.com/wjmaddox/swa_gaussian
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class SWAGModel:
    """SWAG posterior approximation over neural network weights.

    Maintains:
    - Running mean of weights (first moment, theta_SWA)
    - Running mean of squared weights (for diagonal covariance)
    - Low-rank deviation matrix D (K most recent weight deviations from mean)

    The posterior is: w ~ N(theta_SWA, 0.5 * (Sigma_diag + D @ D.T / K))
    where Sigma_diag = diag(sq_mean - mean^2) and D collects deviations.
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_rank: int = 20,
    ) -> None:
        self.base_model = base_model
        self.max_rank = max_rank
        self.n_models_collected = 0

        # Flatten parameter names for consistent ordering
        self._param_names = [name for name, _ in base_model.named_parameters()]

        # Initialize moment accumulators
        self._mean: dict[str, Tensor] = {}
        self._sq_mean: dict[str, Tensor] = {}
        self._deviations: list[
            dict[str, Tensor]
        ] = []  # Low-rank deviations (columns of D)

        for name, param in base_model.named_parameters():
            self._mean[name] = torch.zeros_like(param.data)
            self._sq_mean[name] = torch.zeros_like(param.data)

    def collect_model(self, model: nn.Module) -> None:
        """Collect weight snapshot for SWAG statistics.

        Should be called after each SWA epoch. Updates running mean,
        squared mean, and low-rank deviation matrix.
        """
        self.n_models_collected += 1
        n = self.n_models_collected

        # Compute current deviation from running mean BEFORE updating mean
        deviation: dict[str, Tensor] = {}

        for name, param in model.named_parameters():
            if name not in self._mean:
                continue

            new_data = param.data.detach()

            # Deviation from current mean (before update)
            if n > 1:
                deviation[name] = (new_data - self._mean[name]).clone()

            # Online update of mean: mean = mean + (x - mean) / n
            self._mean[name] = self._mean[name] + (new_data - self._mean[name]) / n

            # Online update of squared mean
            self._sq_mean[name] = (
                self._sq_mean[name] + (new_data.square() - self._sq_mean[name]) / n
            )

        # Store deviation (keep only last max_rank)
        if n > 1 and deviation:
            self._deviations.append(deviation)
            if len(self._deviations) > self.max_rank:
                self._deviations.pop(0)

        logger.debug(
            "SWAG collected model %d (deviations stored: %d/%d)",
            n,
            len(self._deviations),
            self.max_rank,
        )

    @property
    def has_covariance(self) -> bool:
        """Whether enough models have been collected for covariance estimation."""
        return self.n_models_collected >= 2

    def sample(self, scale: float = 1.0, seed: int | None = None) -> None:
        """Sample weights from the SWAG posterior and load into base_model.

        w = mean + (scale/sqrt(2)) * diag_sample + (scale/sqrt(2K)) * low_rank_sample

        Parameters
        ----------
        scale:
            Scale factor for the posterior (1.0 = full posterior, 0.0 = MAP).
        seed:
            Optional random seed for reproducibility.
        """
        if not self.has_covariance:
            # Fall back to mean weights if not enough samples collected
            self._load_mean()
            return

        if seed is not None:
            torch.manual_seed(seed)

        k = len(self._deviations)

        for name, param in self.base_model.named_parameters():
            if name not in self._mean:
                continue

            mean = self._mean[name]
            # Diagonal variance: var = sq_mean - mean^2, clamped to 0
            var = (self._sq_mean[name] - mean.square()).clamp(min=1e-30)
            std = var.sqrt()

            # Diagonal component: z1 ~ N(0, I) * std
            z1 = torch.randn_like(mean)
            diag_sample = z1 * std

            # Low-rank component: z2 ~ N(0, I_K), sum z2_i * d_i
            low_rank_sample = torch.zeros_like(mean)
            if k > 0:
                z2 = torch.randn(k, device=mean.device)
                for i, dev in enumerate(self._deviations):
                    if name in dev:
                        low_rank_sample = low_rank_sample + z2[i] * dev[name]

            # Combine: w = mean + scale * (diag/sqrt(2) + low_rank/sqrt(2K))
            sqrt2 = (2.0) ** 0.5
            sqrt2k = (2.0 * k) ** 0.5 if k > 0 else 1.0
            sampled = mean + scale * (diag_sample / sqrt2 + low_rank_sample / sqrt2k)

            param.data.copy_(sampled)

    def _load_mean(self) -> None:
        """Load the running mean weights into base_model."""
        for name, param in self.base_model.named_parameters():
            if name in self._mean:
                param.data.copy_(self._mean[name])

    def save(self, path: Path) -> None:
        """Save SWAG state (mean, sq_mean, deviations, metadata) to disk."""
        state = {
            "mean": self._mean,
            "sq_mean": self._sq_mean,
            "deviations": self._deviations,
            "n_models_collected": self.n_models_collected,
            "max_rank": self.max_rank,
            "param_names": self._param_names,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        logger.info(
            "SWAG model saved to %s (%d deviations)", path, len(self._deviations)
        )

    @classmethod
    def load(cls, path: Path, base_model: nn.Module) -> SWAGModel:
        """Load SWAG state from disk and attach to base_model."""
        state = torch.load(path, weights_only=False)
        swag = cls(base_model, max_rank=state["max_rank"])
        swag._mean = state["mean"]
        swag._sq_mean = state["sq_mean"]
        swag._deviations = state["deviations"]
        swag.n_models_collected = state["n_models_collected"]
        logger.info(
            "SWAG model loaded from %s (%d models, %d deviations)",
            path,
            swag.n_models_collected,
            len(swag._deviations),
        )
        return swag

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the base model with current weights."""
        result: Tensor = self.base_model(x)
        return result
