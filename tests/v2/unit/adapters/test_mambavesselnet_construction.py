"""Tests for mamba-ssm availability check (Glitch #10).

Verifies that _require_mamba() raises a clear error when mamba-ssm is
not installed, rather than silently failing or producing import errors
deep in the training loop.
"""

from __future__ import annotations

import pytest

from minivess.adapters.model_builder import _mamba_available, _require_mamba


@pytest.mark.model_construction
class TestMambaAvailability:
    """mamba-ssm availability detection and error handling."""

    def test_mamba_available_returns_bool(self) -> None:
        """_mamba_available() returns bool, not raises."""
        result = _mamba_available()
        assert isinstance(result, bool)

    def test_require_mamba_raises_when_unavailable(self) -> None:
        """_require_mamba() raises RuntimeError when mamba-ssm not installed."""
        if _mamba_available():
            pytest.skip("mamba-ssm IS installed — cannot test error path")
        with pytest.raises(RuntimeError, match="mamba-ssm not installed"):
            _require_mamba()

    def test_require_mamba_passes_when_available(self) -> None:
        """_require_mamba() does not raise when mamba-ssm is installed."""
        if not _mamba_available():
            pytest.skip("mamba-ssm not installed (needs nvcc for CUDA compilation)")
        _require_mamba()  # Should not raise
