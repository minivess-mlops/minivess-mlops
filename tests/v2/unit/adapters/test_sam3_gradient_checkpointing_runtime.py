"""Runtime tests for Sam3Backbone gradient checkpointing — T9 doughnut-hole test.

GAP: All existing GC tests are AST/string-search. NONE actually verify the
HuggingFace gradient_checkpointing_enable() method was called with correct args.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from torch import nn


def _make_mock_encoder_with_gc() -> MagicMock:
    """Create a mock encoder that has gradient_checkpointing_enable."""
    encoder = MagicMock(spec=nn.Module)
    encoder.gradient_checkpointing_enable = MagicMock()
    # Make it work as an nn.Module for parameters()
    encoder.parameters = MagicMock(return_value=iter([]))
    return encoder


def _make_mock_encoder_without_gc() -> MagicMock:
    """Create a mock encoder WITHOUT gradient_checkpointing_enable."""
    encoder = MagicMock(spec=nn.Module)
    del encoder.gradient_checkpointing_enable  # Remove the method
    encoder.parameters = MagicMock(return_value=iter([]))
    return encoder


class TestBackboneGradientCheckpointingRuntime:
    """T9: Runtime test that gradient_checkpointing_enable() is actually called."""

    def test_backbone_gc_enable_called_with_mock_encoder(self):
        """When gc=True and freeze=False, enable() must be called with non-reentrant."""
        from minivess.adapters.sam3_backbone import Sam3Backbone

        mock_encoder = _make_mock_encoder_with_gc()
        mock_fpn = nn.Identity()

        mock_config = MagicMock()
        mock_config.architecture_params = {}

        with patch.object(
            Sam3Backbone, "_load_sam3_encoder",
            return_value=(mock_encoder, mock_fpn),
        ):
            Sam3Backbone(
                config=mock_config,
                freeze=False,
                gradient_checkpointing=True,
            )

        mock_encoder.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def test_backbone_gc_not_called_when_false(self):
        """When gc=False, enable() must NOT be called."""
        from minivess.adapters.sam3_backbone import Sam3Backbone

        mock_encoder = _make_mock_encoder_with_gc()
        mock_fpn = nn.Identity()

        mock_config = MagicMock()
        mock_config.architecture_params = {}

        with patch.object(
            Sam3Backbone, "_load_sam3_encoder",
            return_value=(mock_encoder, mock_fpn),
        ):
            Sam3Backbone(
                config=mock_config,
                freeze=False,
                gradient_checkpointing=False,
            )

        mock_encoder.gradient_checkpointing_enable.assert_not_called()

    def test_backbone_gc_skipped_when_frozen(self):
        """When gc=True but freeze=True, enable() must NOT be called."""
        from minivess.adapters.sam3_backbone import Sam3Backbone

        mock_encoder = _make_mock_encoder_with_gc()
        mock_fpn = nn.Identity()

        mock_config = MagicMock()
        mock_config.architecture_params = {}

        with patch.object(
            Sam3Backbone, "_load_sam3_encoder",
            return_value=(mock_encoder, mock_fpn),
        ):
            Sam3Backbone(
                config=mock_config,
                freeze=True,
                gradient_checkpointing=True,
            )

        mock_encoder.gradient_checkpointing_enable.assert_not_called()

    def test_backbone_gc_warns_when_encoder_lacks_method(self, caplog):
        """When encoder lacks enable(), a warning must be logged."""
        from minivess.adapters.sam3_backbone import Sam3Backbone

        mock_encoder = _make_mock_encoder_without_gc()
        mock_fpn = nn.Identity()

        mock_config = MagicMock()
        mock_config.architecture_params = {}

        with patch.object(
            Sam3Backbone, "_load_sam3_encoder",
            return_value=(mock_encoder, mock_fpn),
        ):
            Sam3Backbone(
                config=mock_config,
                freeze=False,
                gradient_checkpointing=True,
            )

        assert any(
            "gradient_checkpointing" in record.message.lower()
            for record in caplog.records
        ), "Should warn when encoder lacks gradient_checkpointing_enable"
