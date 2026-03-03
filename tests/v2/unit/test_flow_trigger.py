"""Tests for flow trigger chain mechanism.

Covers Task 5.1 of data-engineering-improvement-plan.xml.
Closes #180.
"""

from __future__ import annotations

from typing import Any


class TestFlowTriggerConfig:
    """FlowTriggerConfig dataclass."""

    def test_config_has_skip_flows(self) -> None:
        from minivess.orchestration.trigger import FlowTriggerConfig

        config = FlowTriggerConfig(skip_flows=["deploy"])
        assert "deploy" in config.skip_flows

    def test_config_defaults(self) -> None:
        from minivess.orchestration.trigger import FlowTriggerConfig

        config = FlowTriggerConfig()
        assert config.skip_flows == []
        assert config.dashboard_always is True
        assert config.dry_run is False


class TestFlowTriggerResult:
    """FlowTriggerResult dataclass."""

    def test_result_tracks_status(self) -> None:
        from minivess.orchestration.trigger import FlowTriggerResult

        result = FlowTriggerResult(
            flow_name="data",
            status="success",
            duration_s=1.5,
            error=None,
        )
        assert result.flow_name == "data"
        assert result.status == "success"

    def test_result_tracks_error(self) -> None:
        from minivess.orchestration.trigger import FlowTriggerResult

        result = FlowTriggerResult(
            flow_name="train",
            status="failed",
            duration_s=0.1,
            error="RuntimeError: OOM",
        )
        assert result.error is not None


class TestPipelineTriggerChain:
    """PipelineTriggerChain orchestrates flow execution."""

    def test_chain_has_6_flows(self) -> None:
        from minivess.orchestration.trigger import PipelineTriggerChain

        chain = PipelineTriggerChain()
        assert len(chain.flow_names) == 6

    def test_chain_order(self) -> None:
        from minivess.orchestration.trigger import PipelineTriggerChain

        chain = PipelineTriggerChain()
        expected = ["data", "train", "analyze", "deploy", "dashboard", "qa"]
        assert chain.flow_names == expected

    def test_chain_skips_disabled_flows(self) -> None:
        from minivess.orchestration.trigger import (
            FlowTriggerConfig,
            PipelineTriggerChain,
        )

        chain = PipelineTriggerChain()
        config = FlowTriggerConfig(skip_flows=["deploy"])
        results = chain.run_chain(trigger_source="manual", config=config)
        flow_names_executed = [r.flow_name for r in results]
        assert "deploy" not in flow_names_executed

    def test_chain_stops_on_core_failure(self) -> None:
        from minivess.orchestration.trigger import (
            FlowTriggerConfig,
            PipelineTriggerChain,
        )

        def failing_flow(**kwargs: Any) -> None:
            msg = "Simulated failure"
            raise RuntimeError(msg)

        chain = PipelineTriggerChain()
        chain.register_flow("data", failing_flow, is_core=True)

        config = FlowTriggerConfig()
        results = chain.run_chain(trigger_source="manual", config=config)
        # Data should fail
        data_result = results[0]
        assert data_result.status == "failed"
        # Subsequent core flows should be skipped
        for r in results[1:]:
            if r.flow_name not in ("dashboard", "qa"):
                assert r.status == "skipped", f"{r.flow_name} should be skipped"

    def test_chain_dashboard_runs_despite_core_failure(self) -> None:
        """Dashboard is best-effort — runs even when core flows fail."""
        from minivess.orchestration.trigger import (
            FlowTriggerConfig,
            PipelineTriggerChain,
        )

        def failing_flow(**kwargs: Any) -> None:
            msg = "Simulated failure"
            raise RuntimeError(msg)

        chain = PipelineTriggerChain()
        chain.register_flow("data", failing_flow, is_core=True)

        config = FlowTriggerConfig(dashboard_always=True)
        results = chain.run_chain(trigger_source="manual", config=config)
        flow_names = [r.flow_name for r in results]
        assert "dashboard" in flow_names

    def test_dry_run_does_not_execute(self) -> None:
        from minivess.orchestration.trigger import (
            FlowTriggerConfig,
            PipelineTriggerChain,
        )

        chain = PipelineTriggerChain()
        config = FlowTriggerConfig(dry_run=True)
        results = chain.run_chain(trigger_source="manual", config=config)
        for r in results:
            assert r.status == "skipped"
