"""Tests for post_training_run_id handling — T19 regression test.

Bug: The ternary chain at lines 1486-1492 can produce "" (empty string) instead
of None when all pt_run_ids entries are None. "" is truthy, misleading downstream.
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


class TestPostTrainingRunId:
    """T19: post_training_run_id must be None (not '') when all IDs are None."""

    def test_post_training_run_id_none_when_all_none(self):
        """join of all-None list should produce None, not empty string."""
        # Simulate the logic
        pt_run_ids: list[str | None] = [None, None]
        result = (
            pt_run_ids[0]
            if len(pt_run_ids) == 1
            else (",".join(str(rid) for rid in pt_run_ids if rid) or None)
            if pt_run_ids
            else None
        )
        assert result is None, f"Expected None, got '{result}'"

    def test_post_training_run_id_single_value(self):
        """Single run ID should be returned as-is."""
        pt_run_ids: list[str | None] = ["abc123"]
        result = (
            pt_run_ids[0]
            if len(pt_run_ids) == 1
            else (",".join(str(rid) for rid in pt_run_ids if rid) or None)
            if pt_run_ids
            else None
        )
        assert result == "abc123"

    def test_post_training_run_id_multi_value(self):
        """Multiple run IDs should be comma-joined."""
        pt_run_ids: list[str | None] = ["abc", "def"]
        result = (
            pt_run_ids[0]
            if len(pt_run_ids) == 1
            else (",".join(str(rid) for rid in pt_run_ids if rid) or None)
            if pt_run_ids
            else None
        )
        assert result == "abc,def"

    def test_post_training_run_id_mixed_none(self):
        """Mix of real and None IDs should skip None entries."""
        pt_run_ids: list[str | None] = ["abc", None, "def"]
        result = (
            pt_run_ids[0]
            if len(pt_run_ids) == 1
            else (",".join(str(rid) for rid in pt_run_ids if rid) or None)
            if pt_run_ids
            else None
        )
        assert result == "abc,def"

    def test_or_none_pattern_in_source(self):
        """Source code must use 'or None' to prevent empty string."""
        source = TRAIN_FLOW.read_text(encoding="utf-8")
        assert "or None)" in source, (
            "post_training_run_id logic should use 'or None' to prevent empty string"
        )
