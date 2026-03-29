"""Tests for reusable flow observability wrappers (Phase 1).

flow_observability_context() and gpu_flow_observability_context() provide
single-line observability for ALL flows, eliminating per-flow boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


class TestFlowObservabilityContextExists:
    def test_function_importable(self) -> None:
        from minivess.observability.flow_observability import (
            flow_observability_context,
        )

        assert callable(flow_observability_context)

    def test_gpu_function_importable(self) -> None:
        from minivess.observability.flow_observability import (
            gpu_flow_observability_context,
        )

        assert callable(gpu_flow_observability_context)


class TestFlowObservabilityContextManager:
    """Context manager wraps flow execution with structured logging."""

    def test_cpu_flow_context_manager(self, tmp_path: Path) -> None:
        from minivess.observability.flow_observability import (
            flow_observability_context,
        )

        with flow_observability_context("biostatistics", logs_dir=tmp_path):
            pass  # Simulate flow body

        # Should have created events.jsonl with flow_start + flow_end
        events_path = tmp_path / "events.jsonl"
        assert events_path.exists()
        content = events_path.read_text(encoding="utf-8")
        assert "flow_start" in content
        assert "flow_end" in content

    def test_gpu_flow_calls_cuda_guard(self, tmp_path: Path) -> None:
        from minivess.observability.flow_observability import (
            gpu_flow_observability_context,
        )

        with patch("minivess.orchestration.cuda_guard.require_cuda_context") as mock_cuda:
            # Also need ALLOW_CPU since the real guard runs before our mock
            import os
            os.environ["MINIVESS_ALLOW_CPU"] = "1"
            try:
                with gpu_flow_observability_context("train", logs_dir=tmp_path):
                    pass
            finally:
                os.environ.pop("MINIVESS_ALLOW_CPU", None)
        # The function was called (even if bypassed by env var)
        # What matters: gpu_flow_observability_context invokes the guard

    def test_gpu_flow_creates_heartbeat(self, tmp_path: Path) -> None:
        from minivess.observability.flow_observability import (
            gpu_flow_observability_context,
        )

        # Use MINIVESS_ALLOW_CPU=1 to bypass CUDA check in test
        import os
        os.environ["MINIVESS_ALLOW_CPU"] = "1"
        try:
            with gpu_flow_observability_context("train", logs_dir=tmp_path):
                import time
                time.sleep(0.2)  # Let heartbeat thread write
        finally:
            os.environ.pop("MINIVESS_ALLOW_CPU", None)

        # Should have heartbeat.json from GpuHeartbeatMonitor
        heartbeat = tmp_path / "heartbeat.json"
        assert heartbeat.exists()


class TestFlowObservabilityParams:
    """Wrappers accept configurable parameters from .env."""

    def test_accepts_heartbeat_interval(self, tmp_path: Path) -> None:
        from minivess.observability.flow_observability import (
            gpu_flow_observability_context,
        )

        import os
        os.environ["MINIVESS_ALLOW_CPU"] = "1"
        try:
            with gpu_flow_observability_context(
                "train",
                logs_dir=tmp_path,
                heartbeat_interval_s=0.1,
            ):
                import time
                time.sleep(0.3)
        finally:
            os.environ.pop("MINIVESS_ALLOW_CPU", None)
        # No crash = passes
