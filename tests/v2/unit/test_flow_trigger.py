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

    def test_chain_has_9_flows(self) -> None:
        from minivess.orchestration.trigger import PipelineTriggerChain

        chain = PipelineTriggerChain()
        assert len(chain.flow_names) == 9

    def test_chain_order(self) -> None:
        from minivess.orchestration.trigger import PipelineTriggerChain

        chain = PipelineTriggerChain()
        expected = [
            "acquisition",
            "data",
            "train",
            "post_training",
            "analyze",
            "biostatistics",
            "deploy",
            "dashboard",
            "qa",
        ]
        assert chain.flow_names == expected

    def test_post_training_position(self) -> None:
        """post_training should be after train, before analyze."""
        from minivess.orchestration.trigger import PipelineTriggerChain

        chain = PipelineTriggerChain()
        names = chain.flow_names
        assert names.index("post_training") == names.index("train") + 1
        assert names.index("post_training") == names.index("analyze") - 1

    def test_biostatistics_position_and_best_effort(self) -> None:
        """biostatistics should be after analyze, best-effort."""
        from minivess.orchestration.trigger import PipelineTriggerChain

        chain = PipelineTriggerChain()
        names = chain.flow_names
        assert names.index("biostatistics") == names.index("analyze") + 1
        # Verify it's best-effort (not core)
        assert chain._flows["biostatistics"].is_core is False

    def test_post_training_is_best_effort(self) -> None:
        """post_training failure should NOT stop downstream core flows."""
        from minivess.orchestration.trigger import PipelineTriggerChain

        def failing_post_training(**kwargs: Any) -> None:
            msg = "Post-training plugin error"
            raise RuntimeError(msg)

        chain = PipelineTriggerChain()
        chain.register_flow("post_training", failing_post_training, is_core=False)

        results = chain.run_chain(trigger_source="test")
        pt_result = [r for r in results if r.flow_name == "post_training"][0]
        assert pt_result.status == "failed"

        # Analyze should still run (not skipped)
        analyze_result = [r for r in results if r.flow_name == "analyze"][0]
        assert analyze_result.status == "success"

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
        # Find the data flow result (may not be index 0 due to acquisition flow)
        results_by_name = {r.flow_name: r for r in results}
        assert results_by_name["data"].status == "failed"
        # Subsequent core flows should be skipped; best-effort flows still run
        best_effort = {
            "acquisition",
            "post_training",
            "biostatistics",
            "dashboard",
            "qa",
        }
        for r in results:
            if r.flow_name not in best_effort and r.flow_name != "data":
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
