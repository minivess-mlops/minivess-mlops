"""Disconnect-to-Connect 3D data augmentation for topology-aware training.

Reference: Valverde et al. (2025) — topology-preserving data augmentation.
Operates on FULL 3D volumes BEFORE cropping. Reuses centreline_extraction
for junction detection (no reimplementation).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from minivess.pipeline.centreline_extraction import (
    CentrelineGraph,
    extract_centreline,
)

logger = logging.getLogger(__name__)


class DisconnectToConnectd:
    """MONAI-compatible MapTransform that disconnects vessel segments at junctions.

    During training, randomly identifies vessel junction points via skeleton
    extraction, removes short connecting segments from the INPUT image, and
    keeps the GROUND TRUTH intact. Forces the network to predict full
    connectivity from partial evidence.

    Args:
        keys: Keys of images to corrupt (typically ["image"]).
        label_key: Key for ground truth label used for skeleton extraction.
        prob: Probability of applying the transform per call.
        mode: Corruption mode — "zero" sets voxels to 0, "noise" adds Gaussian noise.
        max_segment_length: Max voxels to trace along skeleton branch for disconnection.
        max_junctions: Max number of junctions to disconnect per call.
        dilation_radius: Radius of spherical dilation around skeleton path.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        keys: list[str],
        label_key: str = "label",
        prob: float = 0.3,
        mode: str = "zero",
        max_segment_length: int = 15,
        max_junctions: int = 3,
        dilation_radius: int = 2,
        seed: int | None = None,
    ) -> None:
        self.keys = keys
        self.label_key = label_key
        self.prob = prob
        self.mode = mode
        self.max_segment_length = max_segment_length
        self.max_junctions = max_junctions
        self.dilation_radius = dilation_radius
        self._rng = np.random.default_rng(seed)

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply disconnection augmentation to the data dict."""
        if self._rng.random() > self.prob:
            return data

        label = data[self.label_key]
        if label.sum() == 0:
            return data

        # Extract centreline graph and find junctions
        graph = extract_centreline(label)
        junctions = graph.junction_coords

        if not junctions:
            return data

        # Select random subset of junctions to disconnect
        n_junctions = min(
            len(junctions),
            self._rng.integers(1, self.max_junctions + 1),
        )
        selected_indices = self._rng.choice(
            len(junctions), size=n_junctions, replace=False
        )
        selected_junctions = [junctions[i] for i in selected_indices]

        # Build disconnection mask
        disconnection_mask = self._build_disconnection_mask(
            graph, selected_junctions, label.shape
        )

        if disconnection_mask.sum() == 0:
            return data

        # Apply corruption to specified image keys only
        result = dict(data)
        for key in self.keys:
            if key in result:
                result[key] = self._corrupt(result[key], disconnection_mask)

        return result

    def _build_disconnection_mask(
        self,
        graph: CentrelineGraph,
        selected_junctions: list[tuple[int, int, int]],
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """Build a binary mask of voxels to disconnect."""
        disconnect_voxels = np.zeros(shape, dtype=bool)

        for junc_coord in selected_junctions:
            # Find neighboring skeleton nodes connected to this junction
            junc_idx = None
            for i, node in enumerate(graph.nodes):
                if (node.z, node.y, node.x) == junc_coord:
                    junc_idx = i
                    break

            if junc_idx is None:
                continue

            # Find edges incident to this junction
            incident_edges = [
                e
                for e in graph.edges
                if e.source_idx == junc_idx or e.target_idx == junc_idx
            ]

            if not incident_edges:
                continue

            # Pick a random incident edge to disconnect
            edge = incident_edges[self._rng.integers(0, len(incident_edges))]
            other_idx = (
                edge.target_idx if edge.source_idx == junc_idx else edge.source_idx
            )
            other_node = graph.nodes[other_idx]

            # Create a line of voxels from junction toward the other node
            # (up to max_segment_length)
            branch_voxels = self._trace_branch_voxels(
                junc_coord,
                (other_node.z, other_node.y, other_node.x),
                shape,
            )

            for voxel in branch_voxels:
                disconnect_voxels[voxel] = True

        # Dilate the disconnection path
        if disconnect_voxels.sum() > 0 and self.dilation_radius > 0:
            struct = generate_binary_structure(3, 1)
            for _ in range(self.dilation_radius):
                disconnect_voxels = binary_dilation(disconnect_voxels, struct)

        return disconnect_voxels.astype(np.uint8)

    def _trace_branch_voxels(
        self,
        start: tuple[int, int, int],
        end: tuple[int, int, int],
        shape: tuple[int, ...],
    ) -> list[tuple[int, int, int]]:
        """Trace voxels from start toward end along a straight line, up to max_segment_length."""
        voxels = []
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return [start]

        direction = direction / length
        n_steps = min(int(length), self.max_segment_length)

        for step in range(n_steps + 1):
            pos = np.array(start) + direction * step
            voxel = tuple(np.clip(pos.round().astype(int), 0, np.array(shape) - 1))
            if voxel not in voxels:
                voxels.append(voxel)

        return voxels

    def _corrupt(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply corruption to image at mask locations."""
        result = image.copy()
        if self.mode == "zero":
            result[mask > 0] = 0.0
        elif self.mode == "noise":
            noise = self._rng.standard_normal(image.shape).astype(image.dtype)
            result[mask > 0] = noise[mask > 0] * 0.1
        return result
