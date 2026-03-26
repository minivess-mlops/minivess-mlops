"""Tests for resolve_patch_size — T20 regression test.

Bug: The _is_sam3 / _is_vesselfm / else logic appeared at two locations in
train_flow.py. If defaults change in one place, the other diverges.
"""

from __future__ import annotations

from pathlib import Path

TRAIN_FLOW = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "minivess"
    / "orchestration"
    / "flows"
    / "train_flow.py"
)


class TestResolvePatchSize:
    """T20: resolve_patch_size must be called from both locations."""

    def test_resolve_patch_size_sam3(self):
        """SAM3 variants should get (64, 64, 3)."""
        from minivess.orchestration.flows.train_flow import resolve_patch_size

        assert resolve_patch_size("sam3_vanilla") == (64, 64, 3)
        assert resolve_patch_size("sam3_topolora") == (64, 64, 3)
        assert resolve_patch_size("sam3_hybrid") == (64, 64, 3)

    def test_resolve_patch_size_vesselfm(self):
        """VesselFM should get (64, 64, 32)."""
        from minivess.orchestration.flows.train_flow import resolve_patch_size

        assert resolve_patch_size("vesselfm") == (64, 64, 32)

    def test_resolve_patch_size_default(self):
        """DynUNet and others should get (64, 64, 16)."""
        from minivess.orchestration.flows.train_flow import resolve_patch_size

        assert resolve_patch_size("dynunet") == (64, 64, 16)
        assert resolve_patch_size("mambavesselnet") == (64, 64, 16)

    def test_resolve_patch_size_config_override(self):
        """Config patch_size should override the default."""
        from minivess.orchestration.flows.train_flow import resolve_patch_size

        result = resolve_patch_size("sam3_vanilla", {"patch_size": [128, 128, 5]})
        assert result == (128, 128, 5)

    def test_resolve_patch_size_called_in_both_locations(self):
        """Both train_one_fold_task and post_training_subflow must call resolve_patch_size."""
        source = TRAIN_FLOW.read_text(encoding="utf-8")
        # Count calls to resolve_patch_size
        count = source.count("resolve_patch_size(")
        # At minimum: function definition + 2 call sites
        assert count >= 3, (
            f"resolve_patch_size should be called from at least 2 locations, "
            f"found {count} occurrences (including definition)"
        )
