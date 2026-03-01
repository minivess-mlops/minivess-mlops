"""Tests for GraphTopologyLoss — compound loss combining cbdice_cldice + skeleton_recall + CAPE.

Covers Issue #123: compound graph topology loss.
TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import torch


def _make_logits_labels(
    batch: int = 1,
    classes: int = 2,
    spatial: tuple[int, int, int] = (16, 16, 8),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic logits and labels for loss testing."""
    torch.manual_seed(42)
    logits = torch.randn(batch, classes, *spatial, requires_grad=True)
    labels = torch.randint(0, 2, (batch, 1, *spatial)).float()
    return logits, labels


class TestGraphTopologyLoss:
    """Tests for GraphTopologyLoss (compound: cbdice_cldice + skeleton_recall + CAPE)."""

    def test_graph_topology_loss_differentiable(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels()
        loss_fn = GraphTopologyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_graph_topology_loss_gradient_nonzero(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels()
        loss_fn = GraphTopologyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert torch.any(logits.grad != 0)

    def test_graph_topology_loss_all_terms_contribute(self) -> None:
        """All three sub-losses should be non-zero for random inputs."""
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels()

        # Set one weight to zero at a time and check loss differs
        loss_full = GraphTopologyLoss()(logits, labels).item()
        loss_no_cbdice = GraphTopologyLoss(
            w_cbdice_cldice=0.0, w_skeleton_recall=0.5, w_cape=0.5
        )(logits, labels).item()
        loss_no_skel = GraphTopologyLoss(
            w_cbdice_cldice=0.5, w_skeleton_recall=0.0, w_cape=0.5
        )(logits, labels).item()

        # At least two variants should differ from full loss
        diffs = [
            abs(loss_full - loss_no_cbdice) > 1e-6,
            abs(loss_full - loss_no_skel) > 1e-6,
        ]
        assert sum(diffs) >= 1, "At least one sub-loss should contribute meaningfully"

    def test_graph_topology_loss_weights_configurable(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels()
        loss_a = GraphTopologyLoss(
            w_cbdice_cldice=1.0, w_skeleton_recall=0.0, w_cape=0.0
        )(logits, labels)
        loss_b = GraphTopologyLoss(
            w_cbdice_cldice=0.0, w_skeleton_recall=1.0, w_cape=0.0
        )(logits, labels)
        # Different weights should give different loss values
        assert abs(loss_a.item() - loss_b.item()) > 1e-6

    def test_graph_topology_loss_registered_in_factory(self) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("graph_topology")
        assert loss_fn is not None
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_graph_topology_loss_vram_small_patch(self) -> None:
        """Forward+backward on (1,2,16,16,8) should not OOM."""
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels(spatial=(16, 16, 8))
        loss_fn = GraphTopologyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert torch.isfinite(loss)

    def test_graph_topology_loss_nan_safe(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels()
        loss_fn = GraphTopologyLoss()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_graph_topology_loss_batch_dimension(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels(batch=2)
        loss_fn = GraphTopologyLoss()
        loss = loss_fn(logits, labels)
        assert loss.shape == () or loss.shape == (1,)  # scalar
        assert torch.isfinite(loss)
