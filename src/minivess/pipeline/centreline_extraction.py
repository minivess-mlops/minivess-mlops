"""Centreline extraction utility for 3D vascular masks.

Converts binary 3D segmentation masks into skeleton-based graphs
via morphological thinning and 26-connectivity graph tracing.

Pipeline:
  1. skimage.morphology.skeletonize (Lee94 algorithm)
  2. Distance transform for radius estimation at skeleton voxels
  3. 26-connectivity graph tracing → CentrelineGraph dataclass
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CentrelineNode:
    """A node in the centreline graph."""

    z: int
    y: int
    x: int
    radius: float
    node_type: str  # "junction", "endpoint", or "intermediate"


@dataclasses.dataclass(frozen=True)
class CentrelineEdge:
    """An edge in the centreline graph."""

    source_idx: int
    target_idx: int
    length: float
    mean_radius: float
    voxel_count: int


@dataclasses.dataclass
class CentrelineGraph:
    """Graph representation of a vascular centreline."""

    nodes: list[CentrelineNode]
    edges: list[CentrelineEdge]

    @property
    def junction_coords(self) -> list[tuple[int, int, int]]:
        """Return coordinates of all junction nodes (degree >= 3)."""
        return [(n.z, n.y, n.x) for n in self.nodes if n.node_type == "junction"]

    @property
    def endpoint_coords(self) -> list[tuple[int, int, int]]:
        """Return coordinates of all endpoint nodes (degree == 1)."""
        return [(n.z, n.y, n.x) for n in self.nodes if n.node_type == "endpoint"]


# ---------------------------------------------------------------------------
# 26-connectivity neighborhood
# ---------------------------------------------------------------------------

_NEIGHBORS_26 = []
for dz in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dz == 0 and dy == 0 and dx == 0:
                continue
            _NEIGHBORS_26.append((dz, dy, dx))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_centreline(mask: np.ndarray) -> CentrelineGraph:
    """Extract centreline graph from a binary 3D mask.

    Args:
        mask: Binary 3D mask (D, H, W), dtype uint8 or bool.

    Returns:
        CentrelineGraph with nodes (xyz + radius + type) and edges.
    """
    mask_bin = (mask > 0).astype(np.uint8)

    if mask_bin.sum() == 0:
        return CentrelineGraph(nodes=[], edges=[])

    # Step 1: Skeletonize
    skeleton = skeletonize(mask_bin).astype(np.uint8)

    if skeleton.sum() == 0:
        return CentrelineGraph(nodes=[], edges=[])

    # Step 2: Distance transform for radius estimation
    dt = ndimage.distance_transform_edt(mask_bin)

    # Step 3: Classify skeleton voxels by neighbor count
    skel_coords = np.argwhere(skeleton > 0)

    if len(skel_coords) == 0:
        return CentrelineGraph(nodes=[], edges=[])

    # Build coordinate set for fast lookup
    skel_set = set(map(tuple, skel_coords))

    # Count neighbors for each skeleton voxel
    neighbor_counts: dict[tuple[int, int, int], int] = {}
    shape = skeleton.shape
    for coord in skel_set:
        count = 0
        for dz, dy, dx in _NEIGHBORS_26:
            nz, ny, nx = coord[0] + dz, coord[1] + dy, coord[2] + dx
            if (
                0 <= nz < shape[0]
                and 0 <= ny < shape[1]
                and 0 <= nx < shape[2]
                and (nz, ny, nx) in skel_set
            ):
                count += 1
        neighbor_counts[coord] = count

    # Classify: endpoint (1 neighbor), junction (3+), intermediate (2)
    endpoints = {c for c, n in neighbor_counts.items() if n == 1}
    junctions = {c for c, n in neighbor_counts.items() if n >= 3}

    # Step 4: Build graph via edge tracing
    # Key nodes = endpoints + junctions; edges traced between them
    key_nodes = endpoints | junctions

    # Handle single-voxel skeleton
    if len(skel_set) == 1:
        coord = next(iter(skel_set))
        node = CentrelineNode(
            z=coord[0],
            y=coord[1],
            x=coord[2],
            radius=float(dt[coord]),
            node_type="endpoint",
        )
        return CentrelineGraph(nodes=[node], edges=[])

    # If no key nodes found (e.g., a simple loop), pick an arbitrary point
    if not key_nodes:
        arbitrary = next(iter(skel_set))
        key_nodes = {arbitrary}
        endpoints = {arbitrary}

    # Trace edges between key nodes
    nodes_list, edges_list = _trace_edges(
        skel_set, key_nodes, endpoints, junctions, dt, shape
    )

    return CentrelineGraph(nodes=nodes_list, edges=edges_list)


def _trace_edges(
    skel_set: set[tuple[int, int, int]],
    key_nodes: set[tuple[int, int, int]],
    endpoints: set[tuple[int, int, int]],
    junctions: set[tuple[int, int, int]],
    dt: np.ndarray,
    shape: tuple[int, ...],
) -> tuple[list[CentrelineNode], list[CentrelineEdge]]:
    """Trace edges between key nodes along the skeleton."""
    # Assign indices to key nodes
    coord_to_idx: dict[tuple[int, int, int], int] = {}
    nodes_list: list[CentrelineNode] = []

    for coord in sorted(key_nodes):
        idx = len(nodes_list)
        coord_to_idx[coord] = idx
        if coord in junctions:
            ntype = "junction"
        elif coord in endpoints:
            ntype = "endpoint"
        else:
            ntype = "intermediate"
        nodes_list.append(
            CentrelineNode(
                z=coord[0],
                y=coord[1],
                x=coord[2],
                radius=float(dt[coord]),
                node_type=ntype,
            )
        )

    # Trace from each key node along each branch
    edges_list: list[CentrelineEdge] = []
    visited_edges: set[tuple[int, int]] = set()

    for start in key_nodes:
        neighbors = _get_skel_neighbors(start, skel_set, shape)
        for first_step in neighbors:
            # Trace along this branch until we hit another key node
            path = [start, first_step]
            current = first_step
            prev = start

            while current not in key_nodes or current == start:
                if current in key_nodes and current != start:
                    break
                next_steps = [
                    n
                    for n in _get_skel_neighbors(current, skel_set, shape)
                    if n != prev
                ]
                if not next_steps:
                    break
                # Follow the branch (pick first available)
                prev = current
                current = next_steps[0]
                path.append(current)

            # Create edge if we reached another key node
            end = path[-1]
            if end in key_nodes and end != start:
                src_idx = coord_to_idx[start]
                tgt_idx = coord_to_idx[end]
                edge_key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    length = _path_length(path)
                    radii = [float(dt[c]) for c in path]
                    mean_r = sum(radii) / len(radii) if radii else 0.0
                    edges_list.append(
                        CentrelineEdge(
                            source_idx=src_idx,
                            target_idx=tgt_idx,
                            length=length,
                            mean_radius=mean_r,
                            voxel_count=len(path),
                        )
                    )

    return nodes_list, edges_list


def _get_skel_neighbors(
    coord: tuple[int, int, int],
    skel_set: set[tuple[int, int, int]],
    shape: tuple[int, ...],
) -> list[tuple[int, int, int]]:
    """Get 26-connected skeleton neighbors of a coordinate."""
    neighbors = []
    for dz, dy, dx in _NEIGHBORS_26:
        nz, ny, nx = coord[0] + dz, coord[1] + dy, coord[2] + dx
        if (
            0 <= nz < shape[0]
            and 0 <= ny < shape[1]
            and 0 <= nx < shape[2]
            and (nz, ny, nx) in skel_set
        ):
            neighbors.append((nz, ny, nx))
    return neighbors


def _path_length(path: list[tuple[int, int, int]]) -> float:
    """Compute Euclidean path length along a list of coordinates."""
    total = 0.0
    for i in range(len(path) - 1):
        dz = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        dx = path[i + 1][2] - path[i][2]
        total += (dz**2 + dy**2 + dx**2) ** 0.5
    return total
