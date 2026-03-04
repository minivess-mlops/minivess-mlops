"""Tests for PostTrainingPlugin protocol and dataclasses.

Phase 1 of post-training plugin architecture (#315).
"""

from __future__ import annotations

import pytest


class TestPluginProtocol:
    """PostTrainingPlugin should be a runtime-checkable Protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        from minivess.pipeline.post_training_plugin import PostTrainingPlugin

        assert hasattr(PostTrainingPlugin, "__protocol_attrs__") or hasattr(
            PostTrainingPlugin, "__abstractmethods__"
        )
        # It should be usable with isinstance
        assert isinstance(PostTrainingPlugin, type)

    def test_conforming_class_passes_isinstance(self) -> None:
        from minivess.pipeline.post_training_plugin import (
            PluginInput,
            PluginOutput,
            PostTrainingPlugin,
        )

        class _DummyPlugin:
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def requires_calibration_data(self) -> bool:
                return False

            def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
                return []

            def execute(self, plugin_input: PluginInput) -> PluginOutput:
                return PluginOutput(artifacts={}, metrics={}, model_paths=[])

        assert isinstance(_DummyPlugin(), PostTrainingPlugin)

    def test_non_conforming_class_fails_isinstance(self) -> None:
        from minivess.pipeline.post_training_plugin import PostTrainingPlugin

        class _NotAPlugin:
            pass

        assert not isinstance(_NotAPlugin(), PostTrainingPlugin)


class TestPluginDataclasses:
    """PluginInput and PluginOutput should serialize correctly."""

    def test_plugin_input_construction(self) -> None:
        from pathlib import Path

        from minivess.pipeline.post_training_plugin import PluginInput

        pi = PluginInput(
            checkpoint_paths=[Path("/tmp/ckpt1.pt")],
            config={"key": "value"},
        )
        assert len(pi.checkpoint_paths) == 1
        assert pi.config == {"key": "value"}
        assert pi.calibration_data is None

    def test_plugin_input_with_calibration(self) -> None:
        from minivess.pipeline.post_training_plugin import PluginInput

        pi = PluginInput(
            checkpoint_paths=[],
            config={},
            calibration_data={"scores": [0.5, 0.6]},
        )
        assert pi.calibration_data is not None

    def test_plugin_output_construction(self) -> None:
        from pathlib import Path

        from minivess.pipeline.post_training_plugin import PluginOutput

        po = PluginOutput(
            artifacts={"heatmap": "/tmp/heat.png"},
            metrics={"ece": 0.05},
            model_paths=[Path("/tmp/swa.pt")],
        )
        assert po.artifacts["heatmap"] == "/tmp/heat.png"
        assert po.metrics["ece"] == pytest.approx(0.05)
        assert len(po.model_paths) == 1


class TestPluginRegistry:
    """Plugin registry should map names to implementations."""

    def test_registry_lookup(self) -> None:
        from minivess.pipeline.post_training_plugin import (
            PluginInput,
            PluginOutput,
            PluginRegistry,
        )

        class _FakePlugin:
            @property
            def name(self) -> str:
                return "fake"

            @property
            def requires_calibration_data(self) -> bool:
                return False

            def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
                return []

            def execute(self, plugin_input: PluginInput) -> PluginOutput:
                return PluginOutput(artifacts={}, metrics={}, model_paths=[])

        registry = PluginRegistry()
        plugin = _FakePlugin()
        registry.register(plugin)
        assert registry.get("fake") is plugin

    def test_registry_unknown_raises(self) -> None:
        from minivess.pipeline.post_training_plugin import PluginRegistry

        registry = PluginRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent")

    def test_registry_all_names(self) -> None:
        from minivess.pipeline.post_training_plugin import (
            PluginInput,
            PluginOutput,
            PluginRegistry,
        )

        class _P1:
            @property
            def name(self) -> str:
                return "alpha"

            @property
            def requires_calibration_data(self) -> bool:
                return False

            def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
                return []

            def execute(self, plugin_input: PluginInput) -> PluginOutput:
                return PluginOutput(artifacts={}, metrics={}, model_paths=[])

        class _P2:
            @property
            def name(self) -> str:
                return "beta"

            @property
            def requires_calibration_data(self) -> bool:
                return True

            def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
                return []

            def execute(self, plugin_input: PluginInput) -> PluginOutput:
                return PluginOutput(artifacts={}, metrics={}, model_paths=[])

        registry = PluginRegistry()
        registry.register(_P1())
        registry.register(_P2())
        assert sorted(registry.all_names()) == ["alpha", "beta"]
