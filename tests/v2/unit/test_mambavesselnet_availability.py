"""T01 — RED phase: ModelFamily.MAMBAVESSELNET enum + _mamba_available() gate.

Tests run in staging tier (CPU, no model loading, no mamba-ssm required).
"""

from __future__ import annotations

import sys


class TestModelFamilyEnum:
    """ModelFamily.MAMBAVESSELNET must exist with correct value."""

    def test_mambavesselnet_in_enum(self) -> None:
        from minivess.config.models import ModelFamily

        assert hasattr(ModelFamily, "MAMBAVESSELNET")

    def test_mambavesselnet_value(self) -> None:
        from minivess.config.models import ModelFamily

        assert ModelFamily.MAMBAVESSELNET == "mambavesselnet"

    def test_no_duplicate_values(self) -> None:
        from minivess.config.models import ModelFamily

        values = [m.value for m in ModelFamily]
        assert len(values) == len(set(values)), "Duplicate ModelFamily values detected"


class TestMambaAvailability:
    """_mamba_available() must return bool without raising exceptions."""

    def test_returns_bool(self) -> None:
        from minivess.adapters.model_builder import _mamba_available

        result = _mamba_available()
        assert isinstance(result, bool)

    def test_graceful_on_missing(self, monkeypatch: object) -> None:
        """When mamba_ssm is not in sys.modules, must return False."""

        import minivess.adapters.model_builder as mb

        # Force mamba_ssm import to fail
        monkeypatch.setitem(sys.modules, "mamba_ssm", None)  # type: ignore[arg-type]
        # Re-call — should return False gracefully
        result = mb._mamba_available()
        assert result is False

    def test_import_path(self) -> None:
        """_mamba_available must be importable from mambavesselnet.py."""
        from minivess.adapters.mambavesselnet import _mamba_available  # noqa: F401


class TestModelConfigWithMamba:
    """ModelConfig must accept ModelFamily.MAMBAVESSELNET without validation error."""

    def test_parses(self) -> None:
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.MAMBAVESSELNET,
            name="mambavesselnet_test",
            in_channels=1,
            out_channels=2,
        )
        assert config.family == ModelFamily.MAMBAVESSELNET
        assert config.in_channels == 1
        assert config.out_channels == 2
