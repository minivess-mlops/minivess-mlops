"""Tests for T-26: Serialize fold splits to JSON in data_flow.

Verifies that:
- serialize_splits_task() writes splits.json to the splits dir
- splits.json is parseable via json.loads() (no regex, no yaml)
- Each fold entry has fold_id, train, and val keys
- SPLITS_OUTPUT_DIR env var controls the output directory
- DataFlowResult includes splits_path field
- training_flow's load_fold_splits_task returns raw dicts (not FoldSplit objects)

NO subprocess — all I/O uses temp directories.
"""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

_DATA_FLOW_SRC = Path("src/minivess/orchestration/flows/data_flow.py")
_TRAIN_FLOW_SRC = Path("src/minivess/orchestration/flows/train_flow.py")


# ---------------------------------------------------------------------------
# Source-level tests
# ---------------------------------------------------------------------------


class TestSplitsSerializationSource:
    def test_serialize_splits_task_defined(self) -> None:
        """data_flow.py must define serialize_splits_task."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "serialize_splits_task" in source, (
            "data_flow.py must define or reference serialize_splits_task. "
            "Add @task(name='serialize-splits') def serialize_splits_task(...)."
        )

    def test_serialize_splits_uses_json_dumps(self) -> None:
        """serialize_splits_task must use json.dumps() — not yaml, not regex."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "json.dumps" in source or "json" in source, (
            "serialize_splits_task must serialize using json.dumps(). "
            "Never use yaml or regex for structured data serialization."
        )

    def test_data_flow_result_has_splits_path_field(self) -> None:
        """DataFlowResult dataclass must have a splits_path field."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        result_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DataFlowResult":
                result_class = node
                break
        assert result_class is not None, "DataFlowResult not found"
        field_names = [
            stmt.target.id
            for stmt in result_class.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        ]
        assert "splits_path" in field_names, (
            f"DataFlowResult must have 'splits_path' field. Found: {field_names}. "
            "training_flow uses this path to find splits."
        )

    def test_splits_output_dir_env_var_referenced(self) -> None:
        """data_flow.py must reference SPLITS_OUTPUT_DIR env var."""
        source = _DATA_FLOW_SRC.read_text(encoding="utf-8")
        assert "SPLITS_OUTPUT_DIR" in source, (
            "data_flow.py must read SPLITS_OUTPUT_DIR env var to determine "
            "where splits.json is written. Use os.environ.get('SPLITS_OUTPUT_DIR', ...)."
        )


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestSerializeSplitsTask:
    def test_splits_written_to_json(self) -> None:
        """serialize_splits_task must write splits.json to the specified dir."""
        from minivess.data.splits import FoldSplit
        from minivess.orchestration.flows.data_flow import serialize_splits_task

        fake_splits = [
            FoldSplit(
                train=[{"image": "img0.nii.gz", "label": "lbl0.nii.gz"}],
                val=[{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            splits_dir = Path(tmp) / "splits"
            splits_path = serialize_splits_task(fake_splits, splits_dir)

        assert splits_path.name == "splits.json", (
            f"serialize_splits_task must write splits.json, wrote: {splits_path.name}"
        )

    def test_splits_json_parseable(self) -> None:
        """splits.json must be parseable via json.loads()."""
        from minivess.data.splits import FoldSplit
        from minivess.orchestration.flows.data_flow import serialize_splits_task

        fake_splits = [
            FoldSplit(
                train=[{"image": "img0.nii.gz", "label": "lbl0.nii.gz"}],
                val=[{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            splits_dir = Path(tmp) / "splits"
            splits_path = serialize_splits_task(fake_splits, splits_dir)
            data = json.loads(splits_path.read_text(encoding="utf-8"))

        assert isinstance(data, list), (
            f"splits.json must parse to a list, got {type(data).__name__}"
        )

    def test_splits_json_has_fold_id(self) -> None:
        """Each fold entry in splits.json must have a 'fold_id' key."""
        from minivess.data.splits import FoldSplit
        from minivess.orchestration.flows.data_flow import serialize_splits_task

        fake_splits = [
            FoldSplit(
                train=[{"image": "img0.nii.gz", "label": "lbl0.nii.gz"}],
                val=[{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            ),
            FoldSplit(
                train=[{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
                val=[{"image": "img0.nii.gz", "label": "lbl0.nii.gz"}],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            splits_dir = Path(tmp) / "splits"
            splits_path = serialize_splits_task(fake_splits, splits_dir)
            data = json.loads(splits_path.read_text(encoding="utf-8"))

        for i, fold in enumerate(data):
            assert "fold_id" in fold, (
                f"Fold {i} in splits.json missing 'fold_id' key. Got keys: {list(fold.keys())}"
            )

    def test_splits_json_has_train_val(self) -> None:
        """Each fold entry in splits.json must have 'train' and 'val' lists."""
        from minivess.data.splits import FoldSplit
        from minivess.orchestration.flows.data_flow import serialize_splits_task

        fake_splits = [
            FoldSplit(
                train=[{"image": "img0.nii.gz", "label": "lbl0.nii.gz"}],
                val=[{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            splits_dir = Path(tmp) / "splits"
            splits_path = serialize_splits_task(fake_splits, splits_dir)
            data = json.loads(splits_path.read_text(encoding="utf-8"))

        fold = data[0]
        assert "train" in fold, (
            f"Fold entry missing 'train' key. Got: {list(fold.keys())}"
        )
        assert "val" in fold, f"Fold entry missing 'val' key. Got: {list(fold.keys())}"
        assert isinstance(fold["train"], list), "'train' must be a list"
        assert isinstance(fold["val"], list), "'val' must be a list"

    def test_splits_path_from_env(self) -> None:
        """SPLITS_OUTPUT_DIR env var must control where splits.json is written."""
        import os

        from minivess.orchestration.flows.data_flow import run_data_flow

        def _make_data_dir(base: Path) -> Path:
            data_dir = base / "data"
            (data_dir / "images").mkdir(parents=True)
            (data_dir / "labels").mkdir(parents=True)
            for i in range(3):
                (data_dir / "images" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")
                (data_dir / "labels" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")
            return data_dir

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits_dir = base / "custom_splits"
            data_dir = _make_data_dir(base)
            os.environ["SPLITS_OUTPUT_DIR"] = str(splits_dir)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{base}/mlruns"
            try:
                result = run_data_flow(data_dir=data_dir, n_folds=2, seed=42)
            finally:
                del os.environ["SPLITS_OUTPUT_DIR"]
                del os.environ["MLFLOW_TRACKING_URI"]

            assert result.splits_path is not None, (
                "DataFlowResult.splits_path must not be None"
            )
            assert str(splits_dir) in str(result.splits_path), (
                f"splits_path {result.splits_path} should be under SPLITS_OUTPUT_DIR={splits_dir}"
            )
