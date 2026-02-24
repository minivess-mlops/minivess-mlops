"""Tests for federated learning support (Issue #48)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T1: FLStrategy enum
# ---------------------------------------------------------------------------


class TestFLStrategy:
    """Test federated learning strategy enum."""

    def test_enum_values(self) -> None:
        """FLStrategy should have three strategies."""
        from minivess.pipeline.federated import FLStrategy

        assert FLStrategy.FED_AVG == "fed_avg"
        assert FLStrategy.FED_PROX == "fed_prox"
        assert FLStrategy.SCAFFOLD == "scaffold"


# ---------------------------------------------------------------------------
# T2: Configs
# ---------------------------------------------------------------------------


class TestFLClientConfig:
    """Test FL client configuration."""

    def test_construction(self) -> None:
        """FLClientConfig should capture client settings."""
        from minivess.pipeline.federated import FLClientConfig

        config = FLClientConfig(
            client_id="site_a",
            data_size=100,
            local_epochs=5,
        )
        assert config.client_id == "site_a"
        assert config.data_size == 100
        assert config.local_epochs == 5

    def test_defaults(self) -> None:
        """FLClientConfig should have sensible defaults."""
        from minivess.pipeline.federated import FLClientConfig

        config = FLClientConfig(client_id="site_a")
        assert config.local_epochs == 1
        assert config.data_size == 0


class TestFLServerConfig:
    """Test FL server configuration."""

    def test_construction(self) -> None:
        """FLServerConfig should capture server settings."""
        from minivess.pipeline.federated import FLServerConfig

        config = FLServerConfig(
            num_rounds=10,
            min_clients=2,
            strategy="fed_avg",
        )
        assert config.num_rounds == 10
        assert config.min_clients == 2

    def test_defaults(self) -> None:
        """FLServerConfig should have sensible defaults."""
        from minivess.pipeline.federated import FLServerConfig

        config = FLServerConfig()
        assert config.num_rounds == 5
        assert config.min_clients == 2
        assert config.strategy == "fed_avg"


class TestDPConfig:
    """Test differential privacy configuration."""

    def test_construction(self) -> None:
        """DPConfig should capture privacy settings."""
        from minivess.pipeline.federated import DPConfig

        config = DPConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        assert config.epsilon == 1.0
        assert config.delta == 1e-5

    def test_privacy_budget_ratio(self) -> None:
        """Lower epsilon should indicate stronger privacy."""
        from minivess.pipeline.federated import DPConfig

        strong = DPConfig(epsilon=0.1, delta=1e-5)
        weak = DPConfig(epsilon=10.0, delta=1e-5)
        assert strong.epsilon < weak.epsilon


# ---------------------------------------------------------------------------
# T3: FederatedAveraging
# ---------------------------------------------------------------------------


class TestFederatedAveraging:
    """Test federated averaging implementation."""

    def test_aggregate_weights(self) -> None:
        """aggregate_weights should produce weighted average."""
        from minivess.pipeline.federated import FederatedAveraging

        fa = FederatedAveraging()
        # Two clients with simple weight arrays
        client_weights = [
            {"layer1": np.array([1.0, 2.0, 3.0])},
            {"layer1": np.array([3.0, 4.0, 5.0])},
        ]
        data_sizes = [100, 100]  # Equal weight
        result = fa.aggregate_weights(client_weights, data_sizes)
        np.testing.assert_array_almost_equal(
            result["layer1"], [2.0, 3.0, 4.0]
        )

    def test_aggregate_weighted_unequal(self) -> None:
        """Aggregation with unequal sizes should weight proportionally."""
        from minivess.pipeline.federated import FederatedAveraging

        fa = FederatedAveraging()
        client_weights = [
            {"layer1": np.array([0.0])},
            {"layer1": np.array([10.0])},
        ]
        data_sizes = [900, 100]  # 90% weight to first client
        result = fa.aggregate_weights(client_weights, data_sizes)
        assert result["layer1"][0] < 2.0  # Closer to 0 than 10

    def test_compute_client_weight(self) -> None:
        """compute_client_weight should be proportional to data size."""
        from minivess.pipeline.federated import FederatedAveraging

        fa = FederatedAveraging()
        weights = fa.compute_client_weights([100, 200, 300])
        np.testing.assert_almost_equal(sum(weights), 1.0)
        assert weights[2] > weights[0]


# ---------------------------------------------------------------------------
# T4: FLSimulator
# ---------------------------------------------------------------------------


class TestFLSimulator:
    """Test FL training simulator."""

    def test_add_client(self) -> None:
        """add_client should register a client."""
        from minivess.pipeline.federated import FLServerConfig, FLSimulator

        sim = FLSimulator(FLServerConfig())
        sim.add_client(client_id="site_a", data_size=50)
        assert len(sim.clients) == 1

    def test_simulate_round(self) -> None:
        """simulate_round should produce round results."""
        from minivess.pipeline.federated import FLServerConfig, FLSimulator

        sim = FLSimulator(FLServerConfig(num_rounds=1), seed=42)
        sim.add_client("site_a", data_size=50)
        sim.add_client("site_b", data_size=80)
        result = sim.simulate_round(round_num=1)
        assert result.round_num == 1
        assert len(result.client_metrics) == 2
        assert result.aggregated_loss is not None

    def test_to_markdown(self) -> None:
        """to_markdown should produce a training report."""
        from minivess.pipeline.federated import FLServerConfig, FLSimulator

        sim = FLSimulator(FLServerConfig(num_rounds=1), seed=42)
        sim.add_client("site_a", data_size=50)
        sim.add_client("site_b", data_size=80)
        sim.simulate_round(1)
        md = sim.to_markdown()
        assert "Federated" in md
        assert "site_a" in md
        assert "site_b" in md
