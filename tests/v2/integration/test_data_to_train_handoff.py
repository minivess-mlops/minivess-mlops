"""Tests for T-27: data_flow → training_flow artifact handoff via volumes.

Verifies that:
- run_data_flow() writes splits.json to SPLITS_OUTPUT_DIR
- training_flow's load_fold_splits_task() can read it from SPLITS_DIR
- The JSON round-trips with correct fold_id, train, val structure
- training_flow reads from SPLITS_DIR (not a hardcoded path)

Uses actual file I/O via tempfile.TemporaryDirectory — no mocks.
Does NOT run full training (too heavy for integration test).
Tests the file-handoff contract between the two flows.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _make_data_dir(base: Path) -> Path:
    """Create a minimal data directory with images/labels subdirs."""
    data_dir = base / "data"
    (data_dir / "images").mkdir(parents=True)
    (data_dir / "labels").mkdir(parents=True)
    for i in range(4):
        (data_dir / "images" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")
        (data_dir / "labels" / f"vol_{i:02d}.nii.gz").write_bytes(b"fake")
    return data_dir


class TestDataToTrainHandoff:
    def test_data_flow_writes_splits_json(self) -> None:
        """run_data_flow() must write splits.json to SPLITS_OUTPUT_DIR."""
        from minivess.orchestration.flows.data_flow import run_data_flow

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits_dir = base / "splits"
            data_dir = _make_data_dir(base)
            os.environ["SPLITS_OUTPUT_DIR"] = str(splits_dir)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{base}/mlruns"
            try:
                result = run_data_flow(data_dir=data_dir, n_folds=2)
            finally:
                del os.environ["SPLITS_OUTPUT_DIR"]
                del os.environ["MLFLOW_TRACKING_URI"]

            assert result.splits_path is not None, "splits_path must not be None"
            assert result.splits_path.exists(), (
                f"splits.json was not written at {result.splits_path}"
            )

    def test_splits_json_readable_by_load_fold_splits_task(self) -> None:
        """load_fold_splits_task() must read splits.json written by run_data_flow()."""
        from minivess.orchestration.flows.data_flow import run_data_flow
        from minivess.orchestration.flows.train_flow import load_fold_splits_task

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits_dir = base / "splits"
            data_dir = _make_data_dir(base)
            os.environ["SPLITS_OUTPUT_DIR"] = str(splits_dir)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{base}/mlruns"
            try:
                run_data_flow(data_dir=data_dir, n_folds=2)
            finally:
                del os.environ["SPLITS_OUTPUT_DIR"]
                del os.environ["MLFLOW_TRACKING_URI"]

            # training_flow reads from SPLITS_DIR env var
            loaded = load_fold_splits_task(splits_dir)
            assert isinstance(loaded, list), (
                f"load_fold_splits_task must return a list, got {type(loaded).__name__}"
            )
            assert len(loaded) > 0, (
                "load_fold_splits_task must return at least one fold"
            )

    def test_splits_json_round_trips_fold_structure(self) -> None:
        """Splits written by data_flow must round-trip with train/val lists."""
        from minivess.data.splits import FoldSplit
        from minivess.orchestration.flows.data_flow import run_data_flow
        from minivess.orchestration.flows.train_flow import load_fold_splits_task

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits_dir = base / "splits"
            data_dir = _make_data_dir(base)
            os.environ["SPLITS_OUTPUT_DIR"] = str(splits_dir)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{base}/mlruns"
            try:
                run_data_flow(data_dir=data_dir, n_folds=2)
            finally:
                del os.environ["SPLITS_OUTPUT_DIR"]
                del os.environ["MLFLOW_TRACKING_URI"]

            loaded = load_fold_splits_task(splits_dir)
            # load_fold_splits_task returns FoldSplit objects (deserialized from JSON)
            for i, fold in enumerate(loaded):
                assert isinstance(fold, FoldSplit), (
                    f"Fold {i} must be a FoldSplit, got {type(fold).__name__}"
                )
                assert isinstance(fold.train, list), "train must be a list"
                assert isinstance(fold.val, list), "val must be a list"

    def test_load_fold_splits_task_reads_from_splits_dir_env(self) -> None:
        """load_fold_splits_task must read SPLITS_DIR, not a hardcoded path."""
        import ast

        train_flow_src = Path("src/minivess/orchestration/flows/train_flow.py")
        source = train_flow_src.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find training_flow function and check it reads SPLITS_DIR env var
        found_splits_dir_env = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "training_flow":
                func_src = ast.unparse(node)
                if "SPLITS_DIR" in func_src:
                    found_splits_dir_env = True
                    break

        assert found_splits_dir_env, (
            "training_flow() must read SPLITS_DIR env var to locate splits.json. "
            "Use: splits_dir = Path(os.environ.get('SPLITS_DIR', 'configs/splits'))"
        )

    def test_data_flow_splits_path_under_splits_output_dir(self) -> None:
        """DataFlowResult.splits_path must be under SPLITS_OUTPUT_DIR."""
        from minivess.orchestration.flows.data_flow import run_data_flow

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits_dir = base / "custom_splits_dir"
            data_dir = _make_data_dir(base)
            os.environ["SPLITS_OUTPUT_DIR"] = str(splits_dir)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{base}/mlruns"
            try:
                result = run_data_flow(data_dir=data_dir, n_folds=2)
            finally:
                del os.environ["SPLITS_OUTPUT_DIR"]
                del os.environ["MLFLOW_TRACKING_URI"]

            assert result.splits_path is not None
            assert str(splits_dir) in str(result.splits_path), (
                f"splits_path {result.splits_path} must be under "
                f"SPLITS_OUTPUT_DIR={splits_dir}"
            )

    def test_splits_json_contains_correct_number_of_folds(self) -> None:
        """splits.json must contain exactly n_folds entries."""
        from minivess.orchestration.flows.data_flow import run_data_flow
        from minivess.orchestration.flows.train_flow import load_fold_splits_task

        n_folds = 2
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits_dir = base / "splits"
            data_dir = _make_data_dir(base)
            os.environ["SPLITS_OUTPUT_DIR"] = str(splits_dir)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{base}/mlruns"
            try:
                run_data_flow(data_dir=data_dir, n_folds=n_folds)
            finally:
                del os.environ["SPLITS_OUTPUT_DIR"]
                del os.environ["MLFLOW_TRACKING_URI"]

            loaded = load_fold_splits_task(splits_dir)
            assert len(loaded) == n_folds, (
                f"splits.json must contain {n_folds} folds, got {len(loaded)}"
            )

    def test_handoff_json_is_parseable_without_import(self) -> None:
        """splits.json must be parseable with json.loads() — no special imports needed."""
        from minivess.orchestration.flows.data_flow import run_data_flow

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits_dir = base / "splits"
            data_dir = _make_data_dir(base)
            os.environ["SPLITS_OUTPUT_DIR"] = str(splits_dir)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{base}/mlruns"
            try:
                result = run_data_flow(data_dir=data_dir, n_folds=2)
            finally:
                del os.environ["SPLITS_OUTPUT_DIR"]
                del os.environ["MLFLOW_TRACKING_URI"]

            assert result.splits_path is not None
            raw = result.splits_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            assert isinstance(parsed, list), (
                f"splits.json must parse to list, got {type(parsed).__name__}"
            )
