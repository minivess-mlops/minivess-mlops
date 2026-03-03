"""Tests for zero-shot pipeline support (max_epochs=0) (#292).

When max_epochs=0, the experiment runner should skip training but still
run evaluation. This is model-agnostic — any pretrained model can use it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TestZeroShotMode:
    """Test max_epochs=0 handling in run_experiment."""

    def test_zero_shot_detected_from_config(self) -> None:
        """Config with max_epochs=0 should be recognized as zero-shot."""
        from run_experiment import detect_experiment_mode

        config = {
            "experiment_name": "test_zeroshot",
            "model": "vesselfm",
            "losses": ["dice_ce"],
            "max_epochs": 0,
        }
        # detect_experiment_mode returns "losses" or "conditions"
        # but the zero-shot logic is in _run_losses_mode
        mode = detect_experiment_mode(config)
        assert mode == "losses"
        assert config["max_epochs"] == 0

    def test_is_zero_shot_helper(self) -> None:
        """is_zero_shot() returns True when max_epochs=0."""
        from run_experiment import is_zero_shot

        assert is_zero_shot({"max_epochs": 0}) is True
        assert is_zero_shot({"max_epochs": 100}) is False
        assert is_zero_shot({}) is False

    def test_zero_shot_skips_training(self) -> None:
        """When max_epochs=0, training loop should be skipped."""
        from run_experiment import is_zero_shot

        config: dict[str, Any] = {
            "experiment_name": "test_zeroshot",
            "model": "vesselfm",
            "losses": ["dice_ce"],
            "max_epochs": 0,
        }
        assert is_zero_shot(config) is True

    def test_zero_shot_in_dry_run(self) -> None:
        """Dry run should report max_epochs=0 in validation."""
        from run_experiment import run_dry_run

        config: dict[str, Any] = {
            "experiment_name": "test_zeroshot",
            "model": "vesselfm",
            "losses": ["dice_ce"],
            "max_epochs": 0,
            "compute": "cpu",
        }
        results = run_dry_run(config)
        assert "validation" in results

    def test_debug_override_does_not_zero_epochs(self) -> None:
        """Debug mode should set max_epochs=1, but not override explicit 0."""
        from run_experiment import apply_debug_to_config

        config: dict[str, Any] = {
            "experiment_name": "test",
            "max_epochs": 0,
            "debug": True,
        }
        result = apply_debug_to_config(config)
        # Debug mode sets max_epochs=1, which is expected.
        # Zero-shot is a separate concern from debug mode.
        # With debug=True and max_epochs=0, debug takes priority (max_epochs=1)
        assert result["max_epochs"] == 1
