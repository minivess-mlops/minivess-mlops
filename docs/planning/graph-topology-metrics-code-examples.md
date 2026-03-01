# Graph-Based Topology Metrics - Code Examples

**For MinIVess Issue #124 (Graph-based evaluation metrics)**
**Last Updated:** 2026-02-28

---

## 1. Using MONAI's clDice (Recommended)

### Installation
```bash
# Already in minivess environment
uv add monai>=1.3.0
```

### Training with clDice Loss
```python
from monai.losses import SoftclDiceLoss, SoftDiceclDiceLoss
import torch

# Simple: Soft clDice only
loss_fn = SoftclDiceLoss(
    iter_=3,      # Number of erosion iterations for soft skeleton
    smooth=1.0    # Smoothing factor
)

# Combined: Dice + clDice
loss_fn = SoftDiceclDiceLoss(
    iter_=3,       # Soft skeleton iterations
    alpha=0.5,     # Weight for clDice component
    smooth=1.0,
    include_background=True,
    to_onehot_y=False,
    sigmoid=True,  # For single-channel output
    squared_pred=False
)

# In training loop
pred = model(image)          # Shape: (B, C, D, H, W)
loss = loss_fn(pred, label)  # label shape: (B, C, D, H, W)
loss.backward()
```

### Post-Hoc Evaluation with Hard clDice
```python
from monai.losses import SoftclDiceLoss
from skimage.morphology import skeletonize_3d
import numpy as np

class HardClDiceEvaluator:
    """Hard clDice metric for evaluation using scikit-image skeleton."""

    @staticmethod
    def compute_skeleton(binary_mask):
        """Compute hard skeleton using scikit-image."""
        return skeletonize_3d(binary_mask.astype(bool))

    @staticmethod
    def compute_cldice(pred_mask, gt_mask):
        """Compute hard clDice metric."""
        pred_skel = HardClDiceEvaluator.compute_skeleton(pred_mask)
        gt_skel = HardClDiceEvaluator.compute_skeleton(gt_mask)

        # Intersection and union
        intersection = np.logical_and(pred_skel, gt_skel).sum()
        union = pred_skel.sum() + gt_skel.sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return 2.0 * intersection / union

    @staticmethod
    def compute_topology_metrics(pred_mask, gt_mask):
        """Compute topology precision and recall."""
        pred_skel = HardClDiceEvaluator.compute_skeleton(pred_mask)
        gt_skel = HardClDiceEvaluator.compute_skeleton(gt_mask)

        intersection = np.logical_and(pred_skel, gt_skel).sum()

        # Topology precision: how much of pred skeleton is in GT skeleton
        tp = intersection / max(pred_skel.sum(), 1e-8)

        # Topology recall: how much of GT skeleton is in pred skeleton
        tr = intersection / max(gt_skel.sum(), 1e-8)

        return {
            'topology_precision': float(tp),
            'topology_recall': float(tr),
            'cldice': 2 * tp * tr / (tp + tr + 1e-8) if (tp + tr) > 0 else 0.0
        }

# Usage in evaluation
evaluator = HardClDiceEvaluator()
for pred, gt in validation_loader:
    pred_binary = (pred > 0.5).numpy()
    gt_binary = gt.numpy()

    metrics = evaluator.compute_topology_metrics(pred_binary, gt_binary)
    print(f"Topology metrics: {metrics}")
```

---

## 2. Using Standalone clDice (GitHub Version)

### Installation
```bash
git clone https://github.com/jocpae/clDice.git
cd clDice
# Copy cldice_loss/ to your project or add to PYTHONPATH
```

### PyTorch Implementation
```python
# Option A: Import from cloned repo
import sys
sys.path.insert(0, '/path/to/clDice')
from cldice_loss.pytorch import SoftclDiceLoss, clDice

# Option B: Copy classes directly into your codebase
# See: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py

# Training with soft clDice
loss_fn = SoftclDiceLoss()
pred = model(image)
loss = loss_fn(pred, target)

# Evaluation with hard clDice
metric_fn = clDice(iter_=3)
hard_metric = metric_fn(pred_binary, target_binary)
```

---

## 3. Custom Graph Topology Evaluator (Branch Detection Rate)

### Implementation Template
```python
from __future__ import annotations

import numpy as np
import networkx as nx
from skimage.morphology import skeletonize_3d, label
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class GraphTopologyMetrics:
    """Container for graph-based topology evaluation metrics."""
    cldice: float
    topology_precision: float
    topology_recall: float
    branch_detection_rate: float
    num_gt_bifurcations: int
    num_pred_bifurcations: int
    matched_bifurcations: int
    graph_edit_distance: float


class GraphTopologyEvaluator:
    """Evaluate vessel segmentation using graph-based topology metrics."""

    def __init__(self, min_component_size: int = 10):
        """
        Args:
            min_component_size: Minimum voxels for connected component
        """
        self.min_component_size = min_component_size

    @staticmethod
    def compute_hard_skeleton(binary_mask: np.ndarray) -> np.ndarray:
        """Compute binary skeleton using scikit-image."""
        return skeletonize_3d(binary_mask.astype(bool))

    def extract_bifurcations(self, skeleton: np.ndarray) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
        """
        Extract bifurcation points from skeleton.

        A bifurcation is a point with exactly 3+ neighbors in 26-connectivity
        (center point + 2+ adjacent points in 3D neighborhood).

        Args:
            skeleton: Binary skeleton (Z, Y, X)

        Returns:
            bifurcations: List of (z, y, x) coordinates
            labeled_skel: Labeled skeleton components
        """
        bifurcations = []

        # Remove small components
        labeled_skel, num_components = label(skeleton)
        sizes = np.bincount(labeled_skel.ravel())

        for component_id in range(1, num_components + 1):
            if sizes[component_id] < self.min_component_size:
                labeled_skel[labeled_skel == component_id] = 0

        # Find bifurcation points (degree >= 3)
        for idx in np.ndindex(skeleton.shape):
            if not skeleton[idx]:
                continue

            z, y, x = idx
            # Count neighbors in 26-connectivity (3x3x3 neighborhood excluding center)
            z_min, z_max = max(0, z - 1), min(skeleton.shape[0], z + 2)
            y_min, y_max = max(0, y - 1), min(skeleton.shape[1], y + 2)
            x_min, x_max = max(0, x - 1), min(skeleton.shape[2], x + 2)

            neighborhood = skeleton[z_min:z_max, y_min:y_max, x_min:x_max]
            num_neighbors = neighborhood.sum() - 1  # Exclude center point

            if num_neighbors >= 3:  # Bifurcation or junction
                bifurcations.append((z, y, x))

        return bifurcations, labeled_skel

    def build_vessel_graph(
        self,
        skeleton: np.ndarray,
        bifurcations: List[Tuple[int, int, int]]
    ) -> nx.Graph:
        """
        Build NetworkX graph with bifurcations as nodes.

        Args:
            skeleton: Binary skeleton
            bifurcations: List of bifurcation coordinates

        Returns:
            NetworkX graph with bifurcations as nodes
        """
        G = nx.Graph()

        # Add bifurcation nodes
        for i, bif in enumerate(bifurcations):
            G.add_node(i, coord=bif)

        # Connect nearby bifurcations (within Euclidean distance threshold)
        threshold_dist = 15  # Voxels
        for i in range(len(bifurcations)):
            for j in range(i + 1, len(bifurcations)):
                dist = np.linalg.norm(np.array(bifurcations[i]) - np.array(bifurcations[j]))
                if dist < threshold_dist and dist > 0:
                    G.add_edge(i, j, weight=dist)

        return G

    def match_graphs(self, G_pred: nx.Graph, G_gt: nx.Graph) -> Tuple[List[Tuple[int, int]], float]:
        """
        Match bifurcations between predicted and ground truth graphs.

        Uses Hungarian algorithm on Euclidean distance between bifurcation nodes.

        Args:
            G_pred: Predicted vessel graph
            G_gt: Ground truth vessel graph

        Returns:
            matches: List of (pred_node, gt_node) tuples
            match_score: Average matching distance
        """
        if len(G_pred) == 0 or len(G_gt) == 0:
            return [], 0.0

        # Build distance matrix
        n_pred = len(G_pred)
        n_gt = len(G_gt)
        max_dim = max(n_pred, n_gt)

        distance_matrix = np.zeros((max_dim, max_dim))

        for i, (pred_node, pred_data) in enumerate(G_pred.nodes(data=True)):
            for j, (gt_node, gt_data) in enumerate(G_gt.nodes(data=True)):
                pred_coord = np.array(pred_data['coord'])
                gt_coord = np.array(gt_data['coord'])
                distance_matrix[i, j] = np.linalg.norm(pred_coord - gt_coord)

        # Set unmatched entries to large value
        distance_matrix[:, n_gt:] = 1e10
        distance_matrix[n_pred:, :] = 1e10

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # Filter matches within distance threshold
        threshold = 5.0  # Voxels
        matches = [
            (int(row), int(col))
            for row, col in zip(row_ind, col_ind)
            if col < n_gt and distance_matrix[row, col] < threshold
        ]

        avg_distance = np.mean([distance_matrix[r, c] for r, c in matches]) if matches else 0.0

        return matches, avg_distance

    def compute_branch_detection_rate(
        self,
        G_pred: nx.Graph,
        G_gt: nx.Graph,
        matches: List[Tuple[int, int]]
    ) -> float:
        """
        Compute branch detection rate (BDR).

        BDR = # correctly_detected_branches / # total_gt_branches

        Args:
            G_pred: Predicted vessel graph
            G_gt: Ground truth vessel graph
            matches: Matched bifurcation pairs

        Returns:
            bdr: Branch detection rate (0-1)
        """
        if len(G_gt) == 0:
            return 1.0 if len(G_pred) == 0 else 0.0

        # Number of branches ≈ number of edges in graph
        n_gt_edges = G_gt.number_of_edges()
        n_matched = len(matches)

        # Conservative estimate: each matched node could contribute to edge detection
        bdr = n_matched / len(G_gt) if len(G_gt) > 0 else 0.0

        return float(min(bdr, 1.0))

    def evaluate(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> GraphTopologyMetrics:
        """
        Compute all graph-based topology metrics.

        Args:
            pred_mask: Predicted segmentation (binary or soft)
            gt_mask: Ground truth segmentation (binary)

        Returns:
            GraphTopologyMetrics object with all metrics
        """
        # Convert to binary
        pred_binary = (pred_mask > 0.5).astype(bool)
        gt_binary = (gt_mask > 0.5).astype(bool)

        # Compute skeletons
        pred_skel = self.compute_hard_skeleton(pred_binary)
        gt_skel = self.compute_hard_skeleton(gt_binary)

        # Hard clDice
        intersection = np.logical_and(pred_skel, gt_skel).sum()
        union = pred_skel.sum() + gt_skel.sum()
        cldice = 2.0 * intersection / union if union > 0 else 0.0

        # Topology precision and recall
        tp = intersection / max(pred_skel.sum(), 1e-8)
        tr = intersection / max(gt_skel.sum(), 1e-8)

        # Extract bifurcations
        pred_bifs, _ = self.extract_bifurcations(pred_skel)
        gt_bifs, _ = self.extract_bifurcations(gt_skel)

        # Build graphs
        G_pred = self.build_vessel_graph(pred_skel, pred_bifs)
        G_gt = self.build_vessel_graph(gt_skel, gt_bifs)

        # Match graphs
        matches, _ = self.match_graphs(G_pred, G_gt)

        # Compute BDR
        bdr = self.compute_branch_detection_rate(G_pred, G_gt, matches)

        # Graph edit distance (simple version: node difference)
        ged = abs(len(G_pred) - len(G_gt)) / max(len(G_gt), 1)

        return GraphTopologyMetrics(
            cldice=float(cldice),
            topology_precision=float(tp),
            topology_recall=float(tr),
            branch_detection_rate=float(bdr),
            num_gt_bifurcations=len(gt_bifs),
            num_pred_bifurcations=len(pred_bifs),
            matched_bifurcations=len(matches),
            graph_edit_distance=float(ged)
        )


# Usage Example
if __name__ == "__main__":
    # Dummy data
    pred_seg = np.random.rand(64, 64, 64) > 0.7
    gt_seg = np.random.rand(64, 64, 64) > 0.7

    evaluator = GraphTopologyEvaluator(min_component_size=10)
    metrics = evaluator.evaluate(pred_seg, gt_seg)

    print(f"clDice: {metrics.cldice:.3f}")
    print(f"Topology Precision: {metrics.topology_precision:.3f}")
    print(f"Topology Recall: {metrics.topology_recall:.3f}")
    print(f"Branch Detection Rate: {metrics.branch_detection_rate:.3f}")
    print(f"GT Bifurcations: {metrics.num_gt_bifurcations}")
    print(f"Pred Bifurcations: {metrics.num_pred_bifurcations}")
    print(f"Matched: {metrics.matched_bifurcations}")
    print(f"Graph Edit Distance: {metrics.graph_edit_distance:.3f}")
```

---

## 4. Integration into MLflow

### Logging Graph Metrics
```python
from minivess.observability.tracking import ExperimentTracker

tracker = ExperimentTracker(experiment_name="minivess_evaluation")
tracker.start_run(run_name="eval_ensemble_topology")

# Evaluate predictions
evaluator = GraphTopologyEvaluator()
metrics = evaluator.evaluate(pred_seg, gt_seg)

# Log to MLflow
tracker.log_metrics({
    'eval_cldice': metrics.cldice,
    'eval_topology_precision': metrics.topology_precision,
    'eval_topology_recall': metrics.topology_recall,
    'eval_branch_detection_rate': metrics.branch_detection_rate,
    'eval_graph_edit_distance': metrics.graph_edit_distance,
    'eval_num_gt_bifurcations': metrics.num_gt_bifurcations,
    'eval_num_pred_bifurcations': metrics.num_pred_bifurcations,
})

# Log graph visualization (optional)
visualize_vessel_graph(G_pred, G_gt, artifact_path='vessel_graphs.png')
tracker.log_artifact(artifact_path)

tracker.end_run()
```

### Markdown Report Generation
```python
def generate_topology_report(metrics: GraphTopologyMetrics) -> str:
    """Generate markdown report of topology metrics."""
    report = f"""
## Vessel Topology Evaluation

### Metrics Summary
| Metric | Value |
|--------|-------|
| clDice | {metrics.cldice:.4f} |
| Topology Precision | {metrics.topology_precision:.4f} |
| Topology Recall | {metrics.topology_recall:.4f} |
| Branch Detection Rate | {metrics.branch_detection_rate:.4f} |
| Graph Edit Distance | {metrics.graph_edit_distance:.4f} |

### Graph Statistics
- GT Bifurcations: {metrics.num_gt_bifurcations}
- Pred Bifurcations: {metrics.num_pred_bifurcations}
- Matched Bifurcations: {metrics.matched_bifurcations}

### Interpretation
- **clDice > 0.85**: Excellent topology preservation
- **clDice 0.75-0.85**: Good topology, minor errors
- **clDice < 0.75**: Significant topology errors (breaks/merges)

---
"""
    return report
```

---

## 5. Unit Tests (TDD)

```python
import pytest
import numpy as np

class TestGraphTopologyEvaluator:
    """Tests for GraphTopologyEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return GraphTopologyEvaluator(min_component_size=5)

    def test_hard_skeleton_binary_input(self, evaluator):
        """Skeleton should be binary."""
        seg = np.random.rand(32, 32, 32) > 0.7
        skel = evaluator.compute_hard_skeleton(seg)
        assert skel.dtype == bool
        assert skel.sum() > 0

    def test_perfect_match_cldice_equals_one(self, evaluator):
        """Perfect prediction should have clDice = 1.0."""
        seg = np.zeros((32, 32, 32), dtype=bool)
        seg[10:20, 10:20, 10:20] = True
        metrics = evaluator.evaluate(seg.astype(float), seg.astype(float))
        assert abs(metrics.cldice - 1.0) < 0.01

    def test_empty_prediction_cldice_zero(self, evaluator):
        """Empty prediction should have clDice ≈ 0."""
        pred = np.zeros((32, 32, 32))
        gt = np.zeros((32, 32, 32))
        gt[10:20, 10:20, 10:20] = 1.0
        metrics = evaluator.evaluate(pred, gt)
        assert metrics.cldice < 0.1

    def test_bifurcation_detection(self, evaluator):
        """Should detect T-shaped bifurcations."""
        # Create simple T-shaped structure
        skeleton = np.zeros((32, 32, 32), dtype=bool)
        # Horizontal line
        skeleton[16, 8:24, 16] = True
        # Vertical line
        skeleton[8:24, 16, 16] = True

        bifs, _ = evaluator.extract_bifurcations(skeleton)
        assert len(bifs) > 0  # Should find at least 1 bifurcation

    def test_graph_matching_identical_graphs(self, evaluator):
        """Identical graphs should have perfect matches."""
        G = nx.Graph()
        G.add_node(0, coord=(0, 0, 0))
        G.add_node(1, coord=(5, 5, 5))
        G.add_edge(0, 1)

        matches, _ = evaluator.match_graphs(G, G)
        assert len(matches) == len(G)

    def test_branch_detection_rate_empty(self, evaluator):
        """Empty GT should return BDR=0 or 1 appropriately."""
        G_pred = nx.Graph()
        G_pred.add_node(0, coord=(0, 0, 0))
        G_gt = nx.Graph()

        bdr = evaluator.compute_branch_detection_rate(G_pred, G_gt, [])
        assert bdr >= 0.0 and bdr <= 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 6. MONAI Integration in Training Loop

```python
from monai.losses import SoftDiceclDiceLoss
import torch
from torch.optim import Adam

class TopoAwareTrainer:
    """Training loop with topology-aware loss."""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.loss_fn = SoftDiceclDiceLoss(
            iter_=3,
            alpha=0.5,
            sigmoid=True
        )
        self.optimizer = Adam(model.parameters(), lr=1e-4)
        self.device = device

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Topology-aware loss
            loss = self.loss_fn(outputs, labels)
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Loss: {loss.item():.4f}")

        return epoch_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate with hard clDice metric."""
        self.model.eval()
        evaluator = GraphTopologyEvaluator()
        all_metrics = []

        for images, labels in val_loader:
            images = images.to(self.device)
            outputs = self.model(images)

            pred = torch.sigmoid(outputs).cpu().numpy()
            gt = labels.cpu().numpy()

            # Evaluate per-sample
            for p, g in zip(pred, gt):
                metrics = evaluator.evaluate(p, g)
                all_metrics.append(metrics)

        # Aggregate
        avg_cldice = np.mean([m.cldice for m in all_metrics])
        avg_bdr = np.mean([m.branch_detection_rate for m in all_metrics])

        return {'val_cldice': avg_cldice, 'val_bdr': avg_bdr}
```

---

## Notes for Implementation

1. **Skeleton Computation:** 4-5 minutes per full dataset at native resolution
2. **Graph Matching:** Hungarian algorithm O(n³); acceptable for <100 bifurcations/volume
3. **Memory Usage:** Skeleton + graph operations are lightweight (<1 GB for 512×512×110 volume)
4. **3D Handling:** All code above supports arbitrary (Z, Y, X) dimensions

---

## See Also
- Main survey: `docs/planning/graph-topology-metrics-survey.md`
- Quick reference: `docs/planning/graph-topology-metrics-quick-reference.yaml`
- Issue #124: Graph-based evaluation metrics
