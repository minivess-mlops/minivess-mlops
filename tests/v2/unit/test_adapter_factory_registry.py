"""Tests for model adapter factory registry refactor (T-02.1).

Verifies that build_adapter() uses a registry dict rather than if/elif chains,
and that all registered families are introspectable and buildable.

Closes: #474 (MONAI audit), #343 (ModelAdapter audit)
"""

from __future__ import annotations

import pytest

from minivess.config.models import ModelConfig, ModelFamily


class TestAdapterFactoryRegistry:
    """Registry dispatch must be data-driven (not if/elif)."""

    def test_registry_contains_dynunet(self) -> None:
        """ModelFamily.MONAI_DYNUNET must be in _MODEL_REGISTRY."""
        from minivess.adapters.model_builder import _MODEL_REGISTRY

        assert ModelFamily.MONAI_DYNUNET in _MODEL_REGISTRY

    def test_build_adapter_dynunet(self) -> None:
        """build_adapter with MONAI_DYNUNET config returns DynUNetAdapter."""
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test-dynunet",
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config)
        assert isinstance(adapter, DynUNetAdapter)

    def test_unknown_family_raises_value_error_with_available_list(self) -> None:
        """build_adapter with unregistered family raises ValueError listing available."""
        from minivess.adapters.model_builder import build_adapter

        config = ModelConfig(
            family=ModelFamily.CUSTOM,
            name="unknown",
            in_channels=1,
            out_channels=2,
        )
        with pytest.raises(ValueError, match=r"[Aa]vailable|[Ss]upported"):
            build_adapter(config)

    def test_registry_introspectable(self) -> None:
        """Can list all registered families at runtime as ModelFamily members."""
        from minivess.adapters.model_builder import _MODEL_REGISTRY

        families = list(_MODEL_REGISTRY.keys())
        assert len(families) > 0
        assert all(isinstance(f, ModelFamily) for f in families)

    def test_all_registered_monai_families_buildable(self) -> None:
        """Every registered non-SAM3 family can be instantiated without errors."""
        from minivess.adapters.model_builder import _MODEL_REGISTRY, build_adapter

        # SAM3 families require real weights + GPU; skip them here
        # VesselFM and MambaVesselNet require special deps; skip them too
        skip_prefixes = {"sam3", "vessel", "mamba"}

        monai_families = [
            f
            for f in _MODEL_REGISTRY
            if not any(f.value.startswith(p) for p in skip_prefixes)
            and f != ModelFamily.CUSTOM
        ]
        assert len(monai_families) >= 1, "At least MONAI_DYNUNET must be registered"

        for family in monai_families:
            config = ModelConfig(
                family=family,
                name=f"test-{family.value}",
                in_channels=1,
                out_channels=2,
            )
            adapter = build_adapter(config)
            assert adapter is not None, f"{family} returned None adapter"
