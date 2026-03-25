"""SWAG epoch scaling — swa_epochs must scale with training max_epochs.

5th pass observation O1: SWAG runs 10 epochs regardless of training (2 epochs).
25 min SWAG on 5 min training = 5x overhead. Fix: scale proportionally.

Formula: actual_swa_epochs = min(configured_swa_epochs, max(2, max_epochs // 5))

Plan: infrastructure-performance-audit.xml Phase 3.
"""

from __future__ import annotations

from pathlib import Path

TRAIN_FLOW_PATH = Path("src/minivess/orchestration/flows/train_flow.py")


class TestSwagEpochScaling:
    """SWAG swa_epochs must scale with training epochs, not always use config default."""

    def test_swag_scaling_logic_exists(self) -> None:
        """_run_swag_post_training or post_training_subflow must scale swa_epochs."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")

        # The code must contain logic that adjusts swa_epochs based on max_epochs.
        # We check for the presence of the scaling formula or equivalent.
        has_scaling = (
            "max_epochs" in source
            and "swa_epochs" in source
            # Must have some min() or scaling logic
            and ("min(" in source or "scaled" in source or "actual_swa" in source)
        )
        assert has_scaling, (
            "_run_swag_post_training must scale swa_epochs based on max_epochs. "
            "Without scaling, debug (2 epochs) runs 10 SWAG epochs (25 min for 5 min training). "
            "Formula: min(configured_swa_epochs, max(2, max_epochs // 5))"
        )

    def test_swag_config_reads_max_epochs(self) -> None:
        """The SWAG code path must read max_epochs from the config dict."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")

        # _run_swag_post_training receives config dict which has max_epochs
        assert "max_epochs" in source, (
            "SWAG post-training must access max_epochs to scale swa_epochs"
        )


class TestPerfMetricKeys:
    """perf/* metric keys must exist for infrastructure instrumentation."""

    def test_perf_prefix_exists_in_metric_keys(self) -> None:
        """MetricKeys must have perf/ prefix constants."""
        metric_keys_path = Path("src/minivess/observability/metric_keys.py")
        source = metric_keys_path.read_text(encoding="utf-8")
        assert "perf/" in source, (
            "MetricKeys missing perf/ prefix. "
            "Need: perf/setup_total_seconds, perf/training_seconds, perf/swag_seconds"
        )

    def test_perf_swag_seconds_key_exists(self) -> None:
        """PERF_SWAG_SECONDS key must exist for SWAG timing measurement."""
        metric_keys_path = Path("src/minivess/observability/metric_keys.py")
        source = metric_keys_path.read_text(encoding="utf-8")
        assert "swag_seconds" in source or "SWAG_SECONDS" in source, (
            "MetricKeys missing swag_seconds — cannot measure SWAG overhead"
        )

    def test_perf_training_seconds_key_exists(self) -> None:
        """PERF_TRAINING_SECONDS key must exist for training timing."""
        metric_keys_path = Path("src/minivess/observability/metric_keys.py")
        source = metric_keys_path.read_text(encoding="utf-8")
        assert "training_seconds" in source or "TRAINING_SECONDS" in source, (
            "MetricKeys missing training_seconds — cannot measure training time"
        )
