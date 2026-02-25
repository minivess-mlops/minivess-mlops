"""Tests for centralized configuration defaults (R5.16).

Verifies that ``minivess.config.defaults`` exports all scattered defaults
and that the original module-level names remain importable as re-exports.
"""

from __future__ import annotations


class TestCentralizedDefaults:
    """The ``defaults`` module exports all expected constant values."""

    def test_default_tracking_uri(self) -> None:
        from minivess.config.defaults import DEFAULT_TRACKING_URI

        assert DEFAULT_TRACKING_URI == "mlruns"

    def test_bento_model_tag(self) -> None:
        from minivess.config.defaults import BENTO_MODEL_TAG

        assert BENTO_MODEL_TAG == "minivess-segmentor"

    def test_default_batch_size(self) -> None:
        from minivess.config.defaults import DEFAULT_BATCH_SIZE

        assert DEFAULT_BATCH_SIZE == 2

    def test_default_llm_model(self) -> None:
        from minivess.config.defaults import DEFAULT_LLM_MODEL

        assert DEFAULT_LLM_MODEL == "anthropic:claude-sonnet-4-6"


class TestBackwardsCompatibility:
    """Original module-level names remain importable."""

    def test_tracking_uri_importable_from_tracking(self) -> None:
        from minivess.observability.tracking import _DEFAULT_TRACKING_URI

        assert _DEFAULT_TRACKING_URI == "mlruns"

    def test_bento_tag_importable_from_bento_service(self) -> None:
        from minivess.serving.bento_service import BENTO_MODEL_TAG

        assert BENTO_MODEL_TAG == "minivess-segmentor"

    def test_batch_size_importable_from_loader(self) -> None:
        from minivess.data.loader import _DEFAULT_BATCH_SIZE

        assert _DEFAULT_BATCH_SIZE == 2

    def test_default_model_importable_from_llm(self) -> None:
        from minivess.agents.llm import DEFAULT_MODEL

        assert DEFAULT_MODEL == "anthropic:claude-sonnet-4-6"
