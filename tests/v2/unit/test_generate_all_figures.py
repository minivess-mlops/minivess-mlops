"""Tests for the master figure generation orchestrator.

RED phase: Tests written before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

if TYPE_CHECKING:
    from pathlib import Path

matplotlib.use("Agg")


class TestFigureRegistry:
    """Tests for the figure registry and generation system."""

    def test_registry_not_empty(self) -> None:
        """FIGURE_REGISTRY has registered generators."""
        from minivess.pipeline.viz.generate_all_figures import FIGURE_REGISTRY

        assert len(FIGURE_REGISTRY) > 0

    def test_all_entries_have_required_keys(self) -> None:
        """Each registry entry has name, generator, category."""
        from minivess.pipeline.viz.generate_all_figures import FIGURE_REGISTRY

        for entry in FIGURE_REGISTRY:
            assert "name" in entry, f"Missing 'name' in {entry}"
            assert "generator" in entry, f"Missing 'generator' in {entry}"
            assert "category" in entry, f"Missing 'category' in {entry}"

    def test_list_figures(self) -> None:
        """list_figures returns non-empty list of figure names."""
        from minivess.pipeline.viz.generate_all_figures import list_figures

        names = list_figures()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_generate_single_figure(self, tmp_path: Path) -> None:
        """Can generate a specific registered figure."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.generate_all_figures import (
            FIGURE_REGISTRY,
            generate_figure,
        )

        # Use the first registered figure
        name = FIGURE_REGISTRY[0]["name"]
        result = generate_figure(name, output_dir=tmp_path)
        assert result is not None
        plt.close("all")

    def test_generate_unknown_figure_returns_none(self, tmp_path: Path) -> None:
        """Generating an unknown figure returns None."""
        from minivess.pipeline.viz.generate_all_figures import generate_figure

        result = generate_figure("nonexistent_figure_xyz", output_dir=tmp_path)
        assert result is None

    def test_generate_all(self, tmp_path: Path) -> None:
        """generate_all_figures produces a summary dict."""
        import matplotlib.pyplot as plt

        from minivess.pipeline.viz.generate_all_figures import generate_all_figures

        summary = generate_all_figures(output_dir=tmp_path)
        assert isinstance(summary, dict)
        assert "succeeded" in summary
        assert "failed" in summary
        assert isinstance(summary["succeeded"], list)
        assert isinstance(summary["failed"], list)
        plt.close("all")
