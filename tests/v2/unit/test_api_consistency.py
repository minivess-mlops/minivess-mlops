"""Tests for API naming consistency and __all__ exports (R5.26 + R5.27).

Validates that factory functions use build_* naming and that
all public modules define __all__.
"""

from __future__ import annotations

import importlib

# ---------------------------------------------------------------------------
# T1: Factory functions use build_* naming
# ---------------------------------------------------------------------------


class TestFactoryNaming:
    """Test that factory/builder functions use consistent build_* prefix."""

    def test_data_loader_uses_build_prefix(self) -> None:
        """Data loader factories should use build_* not create_*."""
        from minivess.data import loader

        assert hasattr(loader, "build_train_loader")
        assert hasattr(loader, "build_val_loader")

    def test_hpo_uses_build_prefix(self) -> None:
        """HPO factories should use build_* not create_*/make_*."""
        from minivess.pipeline import hpo

        assert hasattr(hpo, "build_study")
        assert hasattr(hpo, "build_objective")

    def test_gradio_uses_build_prefix(self) -> None:
        """Gradio demo factory should use build_* not create_*."""
        from minivess.serving import gradio_demo

        assert hasattr(gradio_demo, "build_demo")


# ---------------------------------------------------------------------------
# T2: exceptions.py has __all__
# ---------------------------------------------------------------------------


class TestExceptionsExports:
    """Test that exceptions.py defines __all__."""

    def test_exceptions_has_all(self) -> None:
        """exceptions.py should define __all__."""
        from minivess import exceptions

        assert hasattr(exceptions, "__all__")

    def test_exceptions_all_contains_base(self) -> None:
        """__all__ should contain MinivessError."""
        from minivess import exceptions

        assert "MinivessError" in exceptions.__all__

    def test_exceptions_all_count(self) -> None:
        """__all__ should contain all 5 exception classes."""
        from minivess import exceptions

        assert len(exceptions.__all__) >= 5


# ---------------------------------------------------------------------------
# T3: All public modules define __all__
# ---------------------------------------------------------------------------


class TestAllModulesHaveExports:
    """Test that all public package __init__.py files define __all__."""

    def test_all_packages_define_all(self) -> None:
        """Every subpackage should define __all__ in __init__.py."""
        packages = [
            "minivess.adapters",
            "minivess.agents",
            "minivess.compliance",
            "minivess.config",
            "minivess.data",
            "minivess.ensemble",
            "minivess.observability",
            "minivess.pipeline",
            "minivess.serving",
            "minivess.utils",
            "minivess.validation",
        ]
        for pkg_name in packages:
            mod = importlib.import_module(pkg_name)
            assert hasattr(mod, "__all__"), f"{pkg_name} missing __all__"
