"""Tests for dead code cleanup — T24 regression test.

Verifies dead code items identified in the review are cleaned up.
"""

from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3] / "src" / "minivess"


class TestDeadCodeCleanup:
    """T24: Dead code identified in review must be cleaned up."""

    def test_pre_training_checks_docstring_matches_reality(self):
        """Docstring must match actual number of checks (4, not 6)."""
        source = (SRC_DIR / "diagnostics" / "pre_training_checks.py").read_text(
            encoding="utf-8"
        )
        # The docstring should say "Four checks" not "Six checks"
        assert "Four checks" in source or "4 checks" in source or "four checks" in source, (
            "pre_training_checks.py docstring claims wrong number of checks. "
            "There are 4 checks: output_shape, gradient_flow, loss_sanity, nan_inf"
        )
        assert "Six checks" not in source, (
            "pre_training_checks.py docstring incorrectly claims 6 checks"
        )
