"""Federated learning support (NVIDIA FLARE-compatible).

Provides federated averaging, differential privacy configuration,
and multi-site training simulation for privacy-preserving model
training across clinical sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np


class FLStrategy(StrEnum):
    """Federated learning aggregation strategies."""

    FED_AVG = "fed_avg"
    FED_PROX = "fed_prox"
    SCAFFOLD = "scaffold"


@dataclass
class FLClientConfig:
    """Federated learning client configuration.

    Parameters
    ----------
    client_id:
        Unique client/site identifier.
    data_size:
        Number of training samples at this client.
    local_epochs:
        Number of local training epochs per round.
    """

    client_id: str
    data_size: int = 0
    local_epochs: int = 1


@dataclass
class FLServerConfig:
    """Federated learning server configuration.

    Parameters
    ----------
    num_rounds:
        Number of FL communication rounds.
    min_clients:
        Minimum number of clients to start a round.
    strategy:
        Aggregation strategy.
    """

    num_rounds: int = 5
    min_clients: int = 2
    strategy: str = "fed_avg"


@dataclass
class DPConfig:
    """Differential privacy configuration (DP-SGD).

    Parameters
    ----------
    epsilon:
        Privacy budget (lower = stronger privacy).
    delta:
        Probability of privacy breach.
    max_grad_norm:
        Maximum gradient norm for clipping.
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0


@dataclass
class FLRoundResult:
    """Result of a single federated learning round.

    Parameters
    ----------
    round_num:
        Round number.
    client_metrics:
        Per-client training metrics.
    aggregated_loss:
        Loss after global aggregation.
    """

    round_num: int
    client_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    aggregated_loss: float | None = None


class FederatedAveraging:
    """Federated Averaging (FedAvg) implementation.

    Computes weighted average of model parameters from multiple
    clients, proportional to their dataset sizes.
    """

    def compute_client_weights(
        self,
        data_sizes: list[int],
    ) -> list[float]:
        """Compute proportional weights for each client.

        Parameters
        ----------
        data_sizes:
            Number of samples at each client.
        """
        total = sum(data_sizes)
        if total == 0:
            n = len(data_sizes)
            return [1.0 / n] * n
        return [s / total for s in data_sizes]

    def aggregate_weights(
        self,
        client_weights: list[dict[str, Any]],
        data_sizes: list[int],
    ) -> dict[str, Any]:
        """Aggregate model weights via weighted averaging.

        Parameters
        ----------
        client_weights:
            List of state dictionaries from each client.
        data_sizes:
            Dataset size per client for proportional weighting.
        """
        proportions = self.compute_client_weights(data_sizes)
        aggregated: dict[str, Any] = {}

        # Get all parameter keys from the first client
        keys = client_weights[0].keys()

        for key in keys:
            weighted_sum = sum(
                proportions[i] * client_weights[i][key]
                for i in range(len(client_weights))
            )
            aggregated[key] = weighted_sum

        return aggregated


class FLSimulator:
    """Simulates multi-site federated training.

    Parameters
    ----------
    config:
        Server configuration.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        config: FLServerConfig,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.clients: list[FLClientConfig] = []
        self.round_results: list[FLRoundResult] = []
        self._rng = np.random.default_rng(seed)
        self._aggregator = FederatedAveraging()

    def add_client(
        self,
        client_id: str,
        data_size: int = 0,
        local_epochs: int = 1,
    ) -> None:
        """Register a client for federated training.

        Parameters
        ----------
        client_id:
            Client identifier.
        data_size:
            Number of training samples.
        local_epochs:
            Local epochs per round.
        """
        self.clients.append(
            FLClientConfig(
                client_id=client_id,
                data_size=data_size,
                local_epochs=local_epochs,
            ),
        )

    def simulate_round(self, round_num: int) -> FLRoundResult:
        """Simulate one federated learning round.

        Each client trains locally (simulated via random metrics),
        then weights are aggregated via FedAvg.

        Parameters
        ----------
        round_num:
            Round number.
        """
        client_metrics: dict[str, dict[str, float]] = {}

        for client in self.clients:
            # Simulate local training (loss decreases with rounds)
            base_loss = self._rng.uniform(0.3, 0.8)
            decay = 0.95**round_num
            loss = base_loss * decay
            dice = 1.0 - loss * self._rng.uniform(0.8, 1.2)

            client_metrics[client.client_id] = {
                "loss": float(loss),
                "dice": float(np.clip(dice, 0.0, 1.0)),
                "local_epochs": client.local_epochs,
            }

        # Aggregate (simulated loss)
        data_sizes = [c.data_size for c in self.clients]
        proportions = self._aggregator.compute_client_weights(data_sizes)
        agg_loss = sum(
            proportions[i] * client_metrics[c.client_id]["loss"]
            for i, c in enumerate(self.clients)
        )

        result = FLRoundResult(
            round_num=round_num,
            client_metrics=client_metrics,
            aggregated_loss=float(agg_loss),
        )
        self.round_results.append(result)
        return result

    def to_markdown(self) -> str:
        """Generate a federated training report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# Federated Learning Training Report",
            "",
            f"**Generated:** {now}",
            f"**Strategy:** {self.config.strategy}",
            f"**Rounds:** {len(self.round_results)}/{self.config.num_rounds}",
            f"**Clients:** {len(self.clients)}",
            "",
        ]

        # Client overview
        sections.extend(
            [
                "## Clients",
                "",
                "| Client | Data Size | Local Epochs |",
                "|--------|-----------|-------------|",
            ]
        )
        for c in self.clients:
            sections.append(f"| {c.client_id} | {c.data_size} | {c.local_epochs} |")

        # Round results
        if self.round_results:
            sections.extend(
                [
                    "",
                    "## Round Results",
                    "",
                    "| Round | Aggregated Loss | Client Losses |",
                    "|-------|----------------|---------------|",
                ]
            )
            for r in self.round_results:
                client_losses = ", ".join(
                    f"{cid}: {m['loss']:.4f}" for cid, m in r.client_metrics.items()
                )
                sections.append(
                    f"| {r.round_num} | {r.aggregated_loss:.4f} | {client_losses} |"
                )

        sections.append("")
        return "\n".join(sections)
