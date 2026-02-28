"""Tests for centreline extraction utility.

Covers Issue #116: skeletonization + graph extraction → CentrelineGraph.
TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Helpers: synthetic 3D structures
# ---------------------------------------------------------------------------


def _make_tube(
    shape: tuple[int, int, int],
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    radius: float,
) -> np.ndarray:
    """Create a binary 3D tube (cylinder) mask along a line segment."""
    mask = np.zeros(shape, dtype=np.uint8)
    t_steps = max(shape) * 3
    for t in np.linspace(0, 1, t_steps):
        cz = start[0] + t * (end[0] - start[0])
        cy = start[1] + t * (end[1] - start[1])
        cx = start[2] + t * (end[2] - start[2])
        zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
        dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        mask[dist <= radius] = 1
    return mask


def _make_y_bifurcation(shape: tuple[int, int, int], radius: float = 2) -> np.ndarray:
    """Create a Y-shaped bifurcation mask (1 junction, 3 endpoints)."""
    mid_z = shape[0] // 2
    mid_y = shape[1] // 2
    mid_x = shape[2] // 2

    trunk = _make_tube(shape, (0, mid_y, mid_x), (mid_z, mid_y, mid_x), radius=radius)
    left = _make_tube(
        shape,
        (mid_z, mid_y, mid_x),
        (shape[0] - 1, shape[1] // 4, shape[2] // 4),
        radius=radius,
    )
    right = _make_tube(
        shape,
        (mid_z, mid_y, mid_x),
        (shape[0] - 1, 3 * shape[1] // 4, 3 * shape[2] // 4),
        radius=radius,
    )
    return np.clip(trunk + left + right, 0, 1).astype(np.uint8)


# ===========================================================================
# T4: Centreline Extraction — Issue #116
# ===========================================================================


class TestExtractCentreline:
    """Tests for extract_centreline() and CentrelineGraph dataclass."""

    def test_extract_centreline_returns_dataclass(self) -> None:
        from minivess.pipeline.centreline_extraction import (
            CentrelineGraph,
            extract_centreline,
        )

        tube = _make_tube((24, 24, 24), (4, 12, 12), (20, 12, 12), radius=3)
        graph = extract_centreline(tube)
        assert isinstance(graph, CentrelineGraph)

    def test_simple_tube_has_two_endpoints_no_junctions(self) -> None:
        from minivess.pipeline.centreline_extraction import extract_centreline

        tube = _make_tube((24, 24, 24), (4, 12, 12), (20, 12, 12), radius=3)
        graph = extract_centreline(tube)
        assert len(graph.endpoint_coords) == 2
        assert len(graph.junction_coords) == 0

    def test_y_bifurcation_has_three_endpoints_one_junction(self) -> None:
        from minivess.pipeline.centreline_extraction import extract_centreline

        mask = _make_y_bifurcation((32, 32, 32), radius=3)
        graph = extract_centreline(mask)
        assert len(graph.endpoint_coords) == 3
        assert len(graph.junction_coords) >= 1

    def test_edge_lengths_match_synthetic_tube(self) -> None:
        from minivess.pipeline.centreline_extraction import extract_centreline

        tube = _make_tube((24, 24, 24), (4, 12, 12), (20, 12, 12), radius=3)
        graph = extract_centreline(tube)
        # Tube is ~16 voxels long; total edge length should be close
        total_length = sum(e.length for e in graph.edges)
        assert total_length > 10  # should be roughly 16 but skeleton may differ
        assert total_length < 25

    def test_node_radii_from_distance_transform(self) -> None:
        from minivess.pipeline.centreline_extraction import extract_centreline

        radius = 3
        tube = _make_tube((24, 24, 24), (4, 12, 12), (20, 12, 12), radius=radius)
        graph = extract_centreline(tube)
        # Interior nodes should have radii close to the tube radius
        if graph.nodes:
            radii = [n.radius for n in graph.nodes]
            mean_radius = sum(radii) / len(radii)
            assert (
                mean_radius > 1.5
            )  # should be close to 3 but skeleton artifacts exist
            assert mean_radius < 5.0

    def test_empty_mask_returns_empty_graph(self) -> None:
        from minivess.pipeline.centreline_extraction import extract_centreline

        mask = np.zeros((16, 16, 16), dtype=np.uint8)
        graph = extract_centreline(mask)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_single_voxel_mask(self) -> None:
        from minivess.pipeline.centreline_extraction import extract_centreline

        mask = np.zeros((16, 16, 16), dtype=np.uint8)
        mask[8, 8, 8] = 1
        graph = extract_centreline(mask)
        # Single voxel should produce 1 node, 0 edges
        assert len(graph.nodes) <= 1
        assert len(graph.edges) == 0

    def test_skeleton_is_subset_of_mask(self) -> None:
        from minivess.pipeline.centreline_extraction import extract_centreline

        tube = _make_tube((24, 24, 24), (4, 12, 12), (20, 12, 12), radius=3)
        graph = extract_centreline(tube)
        # All node coordinates should be inside the mask
        for node in graph.nodes:
            z, y, x = node.z, node.y, node.x
            if 0 <= z < 24 and 0 <= y < 24 and 0 <= x < 24:
                assert tube[z, y, x] == 1, f"Node ({z},{y},{x}) outside mask"
