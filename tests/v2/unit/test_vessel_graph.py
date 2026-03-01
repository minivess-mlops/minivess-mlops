"""Tests for vessel graph extraction pipeline (#121).

Covers: binary mask → skeletonize → skan graph extraction → pruning
→ radius estimation → NetworkX graph → export (GraphML, SWC).

Uses skan (BSD-3) for skeleton analysis and networkx for graph output.
TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def _make_tube_mask(
    shape: tuple[int, int, int] = (32, 32, 32),
    center_yx: tuple[int, int] = (16, 16),
    radius: int = 3,
    z_range: tuple[int, int] = (4, 28),
) -> np.ndarray:
    """Create a cylindrical tube mask along the z-axis."""
    mask = np.zeros(shape, dtype=bool)
    cy, cx = center_yx
    for z in range(z_range[0], z_range[1]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                if (y - cy) ** 2 + (x - cx) ** 2 <= radius**2:
                    mask[z, y, x] = True
    return mask


def _make_y_bifurcation_mask(
    shape: tuple[int, int, int] = (32, 32, 32),
) -> np.ndarray:
    """Create a Y-shaped bifurcation mask (trunk + two branches).

    Trunk and branches overlap at the junction zone to ensure
    skeletonize produces a connected skeleton with a true junction node.
    """
    mask = np.zeros(shape, dtype=bool)
    # Trunk along z from 4 to 20 (extends into branching zone)
    for z in range(4, 20):
        for y in range(13, 19):
            for x in range(13, 19):
                if (y - 16) ** 2 + (x - 16) ** 2 <= 4:
                    mask[z, y, x] = True
    # Branch 1: starts at z=16 (overlaps with trunk)
    for z in range(16, 28):
        bz = z - 16  # branch progress
        cy = 16 - bz * 0.4  # drift toward y=11
        cx = 16 - bz * 0.4  # drift toward x=11
        for y in range(shape[1]):
            for x in range(shape[2]):
                if (y - cy) ** 2 + (x - cx) ** 2 <= 4:
                    mask[z, y, x] = True
    # Branch 2: starts at z=16 (overlaps with trunk)
    for z in range(16, 28):
        bz = z - 16
        cy = 16 + bz * 0.4  # drift toward y=21
        cx = 16 + bz * 0.4  # drift toward x=21
        for y in range(shape[1]):
            for x in range(shape[2]):
                if (y - cy) ** 2 + (x - cx) ** 2 <= 4:
                    mask[z, y, x] = True
    return mask


class TestVesselGraphExtraction:
    """Tests for extract_vessel_graph and vessel graph pipeline."""

    def test_extract_vessel_graph_returns_networkx_graph(self) -> None:
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_tube_mask()
        graph = extract_vessel_graph(mask)
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() > 0

    def test_vessel_graph_node_attributes(self) -> None:
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_tube_mask()
        graph = extract_vessel_graph(mask)

        for _, data in graph.nodes(data=True):
            # Each node should have 3D coordinates and radius
            assert "z" in data
            assert "y" in data
            assert "x" in data
            assert "radius" in data
            assert data["radius"] >= 0
            assert "node_type" in data  # endpoint, junction, or intermediate

    def test_vessel_graph_edge_attributes(self) -> None:
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_tube_mask()
        graph = extract_vessel_graph(mask)

        assert graph.number_of_edges() > 0
        for _, _, data in graph.edges(data=True):
            assert "length" in data
            assert data["length"] > 0
            assert "mean_radius" in data
            assert data["mean_radius"] >= 0

    def test_vessel_graph_pruning_removes_short_branches(self) -> None:
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_y_bifurcation_mask()
        # Unpruned graph
        graph_full = extract_vessel_graph(mask, min_branch_length=0)
        # Heavily pruned
        graph_pruned = extract_vessel_graph(mask, min_branch_length=100)
        # Pruning should remove short branches (fewer edges)
        assert graph_pruned.number_of_edges() <= graph_full.number_of_edges()

    def test_vessel_graph_pruning_preserves_long_branches(self) -> None:
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_tube_mask()
        # Pruning with small threshold should preserve the main tube
        graph = extract_vessel_graph(mask, min_branch_length=2)
        assert graph.number_of_edges() >= 1

    def test_vessel_graph_radius_from_distance_transform(self) -> None:
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_tube_mask(radius=3)
        graph = extract_vessel_graph(mask)

        # Radius at centreline should be close to tube radius (within tolerance)
        radii = [d["radius"] for _, d in graph.nodes(data=True)]
        max_radius = max(radii)
        # Lee94 skeleton should be near center, so radius ~ tube radius
        assert max_radius >= 1.5, f"Max radius {max_radius} too small for tube radius 3"

    def test_vessel_graph_empty_mask(self) -> None:
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = np.zeros((16, 16, 16), dtype=bool)
        graph = extract_vessel_graph(mask)
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_vessel_graph_single_tube(self) -> None:
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_tube_mask()
        graph = extract_vessel_graph(mask)
        # Single tube should have exactly 2 endpoints
        endpoints = [
            n for n, d in graph.nodes(data=True) if d.get("node_type") == "endpoint"
        ]
        assert len(endpoints) == 2

    def test_vessel_graph_y_bifurcation(self) -> None:
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_y_bifurcation_mask()
        graph = extract_vessel_graph(mask)
        # Y-bifurcation should have at least 1 junction
        junctions = [
            n for n, d in graph.nodes(data=True) if d.get("node_type") == "junction"
        ]
        assert len(junctions) >= 1
        # And at least 3 endpoints (1 trunk end + 2 branch ends)
        endpoints = [
            n for n, d in graph.nodes(data=True) if d.get("node_type") == "endpoint"
        ]
        assert len(endpoints) >= 3


class TestGraphExport:
    """Tests for GraphML and SWC export."""

    def test_export_graphml_valid_xml(self, tmp_path: Path) -> None:
        from minivess.pipeline.graph_export import export_graphml
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_tube_mask()
        graph = extract_vessel_graph(mask)
        out_path = tmp_path / "test.graphml"
        export_graphml(graph, out_path)
        assert out_path.exists()
        # Should be valid XML
        tree = ET.parse(out_path)  # noqa: S314
        root = tree.getroot()
        assert root is not None

    def test_export_swc_valid_format(self, tmp_path: Path) -> None:
        from minivess.pipeline.graph_export import export_swc
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_tube_mask()
        graph = extract_vessel_graph(mask)
        out_path = tmp_path / "test.swc"
        export_swc(graph, out_path)
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        lines = [
            line for line in content.strip().split("\n") if not line.startswith("#")
        ]
        assert len(lines) > 0
        # Each line should have 7 fields: id type x y z radius parent
        for line in lines:
            fields = line.strip().split()
            assert len(fields) == 7, (
                f"SWC line should have 7 fields, got {len(fields)}: {line}"
            )

    def test_export_roundtrip_graphml(self, tmp_path: Path) -> None:
        from minivess.pipeline.graph_export import export_graphml
        from minivess.pipeline.vessel_graph import extract_vessel_graph

        mask = _make_tube_mask()
        graph = extract_vessel_graph(mask)
        out_path = tmp_path / "roundtrip.graphml"
        export_graphml(graph, out_path)
        # Read back
        loaded = nx.read_graphml(out_path)
        assert loaded.number_of_nodes() == graph.number_of_nodes()
        assert loaded.number_of_edges() == graph.number_of_edges()
