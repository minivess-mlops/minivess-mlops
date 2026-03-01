"""Vessel graph extraction pipeline.

Converts binary 3D segmentation masks into NetworkX graphs with
node/edge attributes (coordinates, radius, branch type).

Pipeline:
  1. skimage.morphology.skeletonize (Lee94 algorithm)
  2. scipy.ndimage.distance_transform_edt for radius estimation
  3. skan.Skeleton for graph extraction + branch analysis
  4. Pruning of short branches
  5. NetworkX graph with full attributes

Uses established libraries (skimage, scipy, skan, networkx) rather than
custom implementations. See CLAUDE.md Critical Rule #3.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from scipy import ndimage
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


def extract_vessel_graph(
    mask: np.ndarray,
    *,
    min_branch_length: float = 0.0,
) -> nx.Graph:
    """Extract vessel graph from a binary 3D mask.

    Parameters
    ----------
    mask:
        Binary 3D mask (D, H, W), dtype bool or uint8.
    min_branch_length:
        Remove branches shorter than this threshold (voxels).
        Only endpoint-attached branches are pruned.

    Returns
    -------
    nx.Graph with node attributes (z, y, x, radius, node_type) and
    edge attributes (length, mean_radius, branch_type).
    """
    mask_bin = np.asarray(mask, dtype=bool)

    if not mask_bin.any():
        return nx.Graph()

    # Step 1: Skeletonize via skimage Lee94
    skeleton = skeletonize(mask_bin).astype(np.uint8)

    if skeleton.sum() == 0:
        return nx.Graph()

    # Step 2: Distance transform for radius estimation
    dt = ndimage.distance_transform_edt(mask_bin)

    # Step 3: Build skeleton graph via skan
    skan_skel = Skeleton(skeleton, source_image=dt, spacing=1)
    summary_df = summarize(skan_skel, separator="-")

    # Step 4: Prune short branches
    if min_branch_length > 0:
        skan_skel, summary_df = _prune_short_branches(
            skan_skel, summary_df, min_branch_length
        )
        if skan_skel is None:
            return nx.Graph()

    # Step 5: Convert to NetworkX with attributes
    graph = _skan_to_attributed_graph(skan_skel, summary_df, dt)

    return graph


def _prune_short_branches(
    skan_skel: Skeleton,
    summary_df: object,
    min_length: float,
) -> tuple[Skeleton | None, object]:
    """Remove short endpoint-attached branches via skan's prune_paths.

    Only prunes branches where at least one end is an endpoint (degree 1).
    Junction-to-junction branches are always preserved.
    """
    import pandas as pd

    df = (
        pd.DataFrame(summary_df)
        if not isinstance(summary_df, pd.DataFrame)
        else summary_df
    )

    # Branch type 1 = endpoint-to-junction, type 0 = endpoint-to-endpoint
    # Type 2 = junction-to-junction (never prune)
    short_mask = (df["branch-distance"] < min_length) & (df["branch-type"] != 2)
    short_indices = df.index[short_mask].tolist()

    if not short_indices:
        return skan_skel, df

    remaining_indices = [i for i in range(len(df)) if i not in short_indices]

    if not remaining_indices:
        return None, df.iloc[0:0]

    pruned_skel = skan_skel.prune_paths(remaining_indices)

    if pruned_skel.coordinates.shape[0] == 0:
        return None, df.iloc[0:0]

    pruned_summary = summarize(pruned_skel, separator="-")
    return pruned_skel, pruned_summary


def _skan_to_attributed_graph(
    skan_skel: Skeleton,
    summary_df: object,
    dt: np.ndarray,
) -> nx.Graph:
    """Convert skan Skeleton to attributed NetworkX graph.

    Enriches the basic skan graph with:
    - Node attributes: z, y, x, radius, node_type
    - Edge attributes: length, mean_radius, branch_type
    """
    import pandas as pd

    df = (
        pd.DataFrame(summary_df)
        if not isinstance(summary_df, pd.DataFrame)
        else summary_df
    )

    # Build attributed graph
    G = nx.Graph()

    # Track node degrees from edge structure for type classification
    node_degrees: dict[int, int] = {}

    for _, row in df.iterrows():
        # Get path coordinates for this branch
        branch_idx = int(row.name) if hasattr(row, "name") else 0

        # Node IDs from skan path endpoints
        node_src = int(row.get("node-id-src", branch_idx * 2))
        node_dst = int(row.get("node-id-dst", branch_idx * 2 + 1))

        # Get coordinates from skan skeleton
        path_coords = skan_skel.path_coordinates(branch_idx)

        if len(path_coords) == 0:
            continue

        src_coord = path_coords[0]
        dst_coord = path_coords[-1]

        # Add nodes with attributes
        for node_id, coord in [(node_src, src_coord), (node_dst, dst_coord)]:
            iz, iy, ix = (
                int(round(coord[0])),
                int(round(coord[1])),
                int(round(coord[2])),
            )
            # Clamp to valid range
            iz = min(max(iz, 0), dt.shape[0] - 1)
            iy = min(max(iy, 0), dt.shape[1] - 1)
            ix = min(max(ix, 0), dt.shape[2] - 1)
            radius = float(dt[iz, iy, ix])

            if node_id not in G:
                G.add_node(
                    node_id, z=iz, y=iy, x=ix, radius=radius, node_type="intermediate"
                )
            node_degrees[node_id] = node_degrees.get(node_id, 0) + 1

        # Add edge with attributes
        # Compute mean radius along the path
        path_radii = []
        for coord in path_coords:
            pz = min(max(int(round(coord[0])), 0), dt.shape[0] - 1)
            py = min(max(int(round(coord[1])), 0), dt.shape[1] - 1)
            px = min(max(int(round(coord[2])), 0), dt.shape[2] - 1)
            path_radii.append(float(dt[pz, py, px]))

        mean_radius = float(np.mean(path_radii)) if path_radii else 0.0
        branch_type_val = int(row.get("branch-type", 0))

        G.add_edge(
            node_src,
            node_dst,
            length=float(row["branch-distance"]),
            mean_radius=mean_radius,
            branch_type=branch_type_val,
        )

    # Classify node types based on degree
    for node_id in G.nodes:
        degree = G.degree(node_id)
        if degree == 1:
            G.nodes[node_id]["node_type"] = "endpoint"
        elif degree >= 3:
            G.nodes[node_id]["node_type"] = "junction"
        else:
            G.nodes[node_id]["node_type"] = "intermediate"

    return G
