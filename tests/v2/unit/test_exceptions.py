"""Tests for custom exception hierarchy (Code Review R3.3).

Validates the MinivessError hierarchy for domain-specific error handling.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# T1: Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Test the MinivessError exception hierarchy."""

    def test_base_exception_inherits_from_exception(self) -> None:
        """MinivessError should inherit from Exception."""
        from minivess.exceptions import MinivessError

        assert issubclass(MinivessError, Exception)

    def test_checkpoint_error_inherits_from_base(self) -> None:
        """CheckpointError should inherit from MinivessError."""
        from minivess.exceptions import CheckpointError, MinivessError

        assert issubclass(CheckpointError, MinivessError)

    def test_config_error_inherits_from_base(self) -> None:
        """ConfigError should inherit from MinivessError."""
        from minivess.exceptions import ConfigError, MinivessError

        assert issubclass(ConfigError, MinivessError)

    def test_pipeline_error_inherits_from_base(self) -> None:
        """PipelineError should inherit from MinivessError."""
        from minivess.exceptions import MinivessError, PipelineError

        assert issubclass(PipelineError, MinivessError)

    def test_serving_error_inherits_from_base(self) -> None:
        """ServingError should inherit from MinivessError."""
        from minivess.exceptions import MinivessError, ServingError

        assert issubclass(ServingError, MinivessError)

    def test_validation_error_inherits_from_base(self) -> None:
        """DataValidationError should inherit from MinivessError."""
        from minivess.exceptions import DataValidationError, MinivessError

        assert issubclass(DataValidationError, MinivessError)


# ---------------------------------------------------------------------------
# T2: Exception usage
# ---------------------------------------------------------------------------


class TestExceptionUsage:
    """Test that exceptions can be raised and caught properly."""

    def test_catch_specific_exception(self) -> None:
        """Specific exceptions should be catchable by their type."""
        from minivess.exceptions import CheckpointError

        with pytest.raises(CheckpointError, match="corrupt"):
            raise CheckpointError("corrupt checkpoint file")

    def test_catch_base_exception(self) -> None:
        """All domain exceptions should be catchable via MinivessError."""
        from minivess.exceptions import MinivessError, PipelineError

        with pytest.raises(MinivessError):
            raise PipelineError("training failed")

    def test_exception_preserves_message(self) -> None:
        """Exception should preserve the error message."""
        from minivess.exceptions import ConfigError

        try:
            raise ConfigError("invalid model family")
        except ConfigError as e:
            assert "invalid model family" in str(e)
