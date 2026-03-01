"""Tests for visualization plot_config, figure_dimensions, and figure_export.

Foundation tests for the visualization system (Phase 3).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib
import pytest

from minivess.pipeline.viz.figure_dimensions import FIGURE_DIMENSIONS, get_figsize
from minivess.pipeline.viz.figure_export import save_figure
from minivess.pipeline.viz.plot_config import (
    COLORS,
    LOSS_LABELS,
    setup_style,
)

if TYPE_CHECKING:
    from pathlib import Path

# Use non-interactive backend for tests
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# TestColors
# ---------------------------------------------------------------------------


class TestColors:
    """Tests for the Paul Tol colorblind-safe palette."""

    def test_four_loss_colors_defined(self) -> None:
        """All 4 loss functions have assigned colors."""
        for loss in ("dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice"):
            assert loss in COLORS, f"Missing color for {loss}"

    def test_colors_are_hex(self) -> None:
        """All colors are valid hex strings (#RRGGBB)."""
        for name, color in COLORS.items():
            assert color.startswith("#"), f"{name}: {color} not hex"
            assert len(color) == 7, f"{name}: {color} not #RRGGBB"

    def test_colors_unique(self) -> None:
        """No two loss functions share the same color."""
        loss_colors = [
            COLORS[k] for k in ("dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice")
        ]
        assert len(set(loss_colors)) == 4


# ---------------------------------------------------------------------------
# TestLossLabels
# ---------------------------------------------------------------------------


class TestLossLabels:
    """Tests for display label mapping."""

    def test_four_loss_labels_defined(self) -> None:
        """All 4 loss functions have display labels."""
        for loss in ("dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice"):
            assert loss in LOSS_LABELS, f"Missing label for {loss}"

    def test_labels_nonempty(self) -> None:
        """All labels are non-empty strings."""
        for name, label in LOSS_LABELS.items():
            assert isinstance(label, str), f"{name}: not a string"
            assert len(label) > 0, f"{name}: empty label"


# ---------------------------------------------------------------------------
# TestSetupStyle
# ---------------------------------------------------------------------------


class TestSetupStyle:
    """Tests for setup_style()."""

    def test_setup_paper_context(self) -> None:
        """Calling setup_style('paper') does not raise."""
        setup_style(context="paper")

    def test_setup_talk_context(self) -> None:
        """Calling setup_style('talk') does not raise."""
        setup_style(context="talk")

    def test_is_callable_side_effect(self) -> None:
        """setup_style is a callable side-effect function."""
        setup_style(context="paper")
        # Verify it's callable (side-effect only, returns None)
        assert callable(setup_style)


# ---------------------------------------------------------------------------
# TestFigureDimensions
# ---------------------------------------------------------------------------


class TestFigureDimensions:
    """Tests for figure dimension presets."""

    def test_presets_exist(self) -> None:
        """Standard presets are defined."""
        for preset in ("single", "double", "triple", "matrix", "forest"):
            assert preset in FIGURE_DIMENSIONS, f"Missing preset: {preset}"

    def test_presets_are_2tuples(self) -> None:
        """All presets are (width, height) tuples."""
        for name, dims in FIGURE_DIMENSIONS.items():
            assert len(dims) == 2, f"{name}: len={len(dims)}"
            assert dims[0] > 0, f"{name}: non-positive width"
            assert dims[1] > 0, f"{name}: non-positive height"

    def test_get_figsize_known_preset(self) -> None:
        """get_figsize returns correct dimensions for known preset."""
        dims = get_figsize("single")
        assert dims == FIGURE_DIMENSIONS["single"]

    def test_get_figsize_unknown_preset(self) -> None:
        """get_figsize raises KeyError for unknown preset."""
        with pytest.raises(KeyError):
            get_figsize("nonexistent_preset")


# ---------------------------------------------------------------------------
# TestSaveFigure
# ---------------------------------------------------------------------------


class TestSaveFigure:
    """Tests for multi-format figure export."""

    def test_saves_png(self, tmp_path: Path) -> None:
        """Saves PNG file at 300 DPI."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        path = save_figure(fig, "test_fig", output_dir=tmp_path, formats=["png"])
        assert (tmp_path / "test_fig.png").is_file()
        plt.close(fig)
        assert path is not None

    def test_saves_svg(self, tmp_path: Path) -> None:
        """Saves SVG file."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        save_figure(fig, "test_fig", output_dir=tmp_path, formats=["svg"])
        assert (tmp_path / "test_fig.svg").is_file()
        plt.close(fig)

    def test_saves_multiple_formats(self, tmp_path: Path) -> None:
        """Saves in multiple formats at once."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        save_figure(fig, "multi", output_dir=tmp_path, formats=["png", "svg"])
        assert (tmp_path / "multi.png").is_file()
        assert (tmp_path / "multi.svg").is_file()
        plt.close(fig)

    def test_saves_json_data(self, tmp_path: Path) -> None:
        """Saves reproducibility JSON alongside figure."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        data = {"x": [1, 2, 3], "y": [1, 2, 3], "label": "test"}
        save_figure(fig, "with_data", output_dir=tmp_path, formats=["png"], data=data)

        json_path = tmp_path / "with_data.json"
        assert json_path.is_file()

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["x"] == [1, 2, 3]
        plt.close(fig)

    def test_default_formats(self, tmp_path: Path) -> None:
        """Default formats include png."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        save_figure(fig, "default", output_dir=tmp_path)
        assert (tmp_path / "default.png").is_file()
        plt.close(fig)
