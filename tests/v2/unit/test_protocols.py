"""Tests for Protocol types used for cross-package decoupling (Code Review R3.2).

Validates that concrete implementations satisfy the structural Protocol
contracts, enabling type-safe decoupling without import dependencies.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: Predictor protocol
# ---------------------------------------------------------------------------


class TestPredictorProtocol:
    """Test that adapters satisfy the Predictor protocol."""

    def test_dynunet_is_predictor(self) -> None:
        """DynUNetAdapter should satisfy the Predictor protocol."""
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily
        from minivess.utils.protocols import Predictor

        model = DynUNetAdapter(
            ModelConfig(
                family=ModelFamily.MONAI_DYNUNET,
                name="test",
                in_channels=1,
                out_channels=2,
            )
        )
        assert isinstance(model, Predictor)


# ---------------------------------------------------------------------------
# T2: Checkpointable protocol
# ---------------------------------------------------------------------------


class TestCheckpointableProtocol:
    """Test that adapters satisfy the Checkpointable protocol."""

    def test_dynunet_is_checkpointable(self) -> None:
        """DynUNetAdapter should be Checkpointable."""
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily
        from minivess.utils.protocols import Checkpointable

        model = DynUNetAdapter(
            ModelConfig(
                family=ModelFamily.MONAI_DYNUNET,
                name="test",
                in_channels=1,
                out_channels=2,
            )
        )
        assert isinstance(model, Checkpointable)

    def test_checkpointable_has_save_load(self) -> None:
        """Checkpointable should require save_checkpoint and load_checkpoint."""
        from minivess.utils.protocols import Checkpointable

        assert hasattr(Checkpointable, "save_checkpoint")
        assert hasattr(Checkpointable, "load_checkpoint")


# ---------------------------------------------------------------------------
# T3: MetricComputer protocol
# ---------------------------------------------------------------------------


class TestMetricComputerProtocol:
    """Test that metrics satisfy the MetricComputer protocol."""

    def test_segmentation_metrics_is_metric_computer(self) -> None:
        """SegmentationMetrics should satisfy MetricComputer."""
        from minivess.pipeline.metrics import SegmentationMetrics
        from minivess.utils.protocols import MetricComputer

        metrics = SegmentationMetrics(num_classes=2)
        assert isinstance(metrics, MetricComputer)

    def test_metric_computer_has_update_compute_reset(self) -> None:
        """MetricComputer should require update, compute, reset."""
        from minivess.utils.protocols import MetricComputer

        for method in ("update", "compute", "reset"):
            assert hasattr(MetricComputer, method)
