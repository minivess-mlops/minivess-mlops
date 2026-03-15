"""Tests for atomic checkpoint writes.

T0.5: Verify torch.save uses tmp + os.replace pattern for atomicity.
B1: Verify wiring — save_metric_checkpoint and trainer epoch_latest use atomic saves.
B2: SHA256 checkpoint integrity verification.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


class TestAtomicSave:
    """_atomic_torch_save should write to tmp then atomically rename."""

    def test_produces_valid_checkpoint(self, tmp_path: Path) -> None:
        """Atomic save produces a loadable checkpoint file."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        state = {"weight": torch.randn(4, 4), "epoch": 10}
        path = tmp_path / "model.pth"
        atomic_torch_save(state, path)

        assert path.exists()
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 10
        assert torch.equal(loaded["weight"], state["weight"])

    def test_no_tmp_file_after_success(self, tmp_path: Path) -> None:
        """Temporary file is cleaned up after successful save."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        path = tmp_path / "model.pth"
        atomic_torch_save({"x": 1}, path)

        tmp_path_check = path.with_suffix(".pth.tmp")
        assert not tmp_path_check.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """atomic_torch_save creates parent directories if missing."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        path = tmp_path / "deep" / "dir" / "model.pth"
        atomic_torch_save({"x": 1}, path)
        assert path.exists()

    def test_original_preserved_on_failure(self, tmp_path: Path) -> None:
        """If save fails, original file is untouched."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        path = tmp_path / "model.pth"
        # Write original
        atomic_torch_save({"epoch": 1}, path)

        # Force a failure during save
        with (
            patch("torch.save", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            atomic_torch_save({"epoch": 2}, path)

        # Original should still be readable with epoch=1
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 1


class TestAtomicTextWrite:
    """atomic_text_write should write text atomically (tmp + os.replace)."""

    def test_produces_valid_text_file(self, tmp_path: Path) -> None:
        """Atomic text write produces a readable file."""
        from minivess.pipeline.checkpoint_utils import atomic_text_write

        path = tmp_path / "state.yaml"
        atomic_text_write("epoch: 5\n", path)

        assert path.exists()
        assert path.read_text(encoding="utf-8") == "epoch: 5\n"

    def test_no_tmp_file_after_success(self, tmp_path: Path) -> None:
        """Temporary file is cleaned up after successful write."""
        from minivess.pipeline.checkpoint_utils import atomic_text_write

        path = tmp_path / "state.yaml"
        atomic_text_write("hello", path)

        tmp_check = path.with_suffix(".yaml.tmp")
        assert not tmp_check.exists()

    def test_original_preserved_on_failure(self, tmp_path: Path) -> None:
        """If write fails, original file is untouched."""
        from minivess.pipeline.checkpoint_utils import atomic_text_write

        path = tmp_path / "state.yaml"
        atomic_text_write("epoch: 1\n", path)

        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            atomic_text_write("epoch: 2\n", path)

        assert path.read_text(encoding="utf-8") == "epoch: 1\n"


class TestWiringSaveMetricCheckpoint:
    """B1: save_metric_checkpoint must use atomic_torch_save, not raw torch.save."""

    def test_save_metric_checkpoint_uses_atomic(self) -> None:
        """AST check: save_metric_checkpoint calls atomic_torch_save, not torch.save."""
        src = Path("src/minivess/pipeline/multi_metric_tracker.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "save_metric_checkpoint"
            ):
                body_src = ast.dump(node)
                assert "atomic_torch_save" in body_src, (
                    "save_metric_checkpoint must call atomic_torch_save"
                )
                # Ensure no raw torch.save call
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "save"
                        and isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "torch"
                    ):
                        pytest.fail(
                            "save_metric_checkpoint must NOT call torch.save directly"
                        )
                return
        pytest.fail("save_metric_checkpoint function not found")

    def test_module_imports_atomic_torch_save(self) -> None:
        """multi_metric_tracker.py must import atomic_torch_save."""
        src = Path("src/minivess/pipeline/multi_metric_tracker.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "checkpoint_utils" in node.module
            ):
                names = [alias.name for alias in node.names]
                if "atomic_torch_save" in names:
                    return
        pytest.fail(
            "multi_metric_tracker.py must import atomic_torch_save from checkpoint_utils"
        )


class TestWiringTrainerEpochLatest:
    """B1: trainer.py epoch_latest saves must use atomic writes."""

    def test_trainer_uses_atomic_for_epoch_pth(self) -> None:
        """AST check: trainer.py uses atomic_torch_save for epoch_latest.pth."""
        src = Path("src/minivess/pipeline/trainer.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        # Find the fit method and check for atomic_torch_save usage
        found_atomic = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "fit":
                body_dump = ast.dump(node)
                if "atomic_torch_save" in body_dump:
                    found_atomic = True
                break

        assert found_atomic, (
            "trainer.py fit() must use atomic_torch_save for epoch_latest.pth"
        )

    def test_trainer_uses_atomic_for_epoch_yaml(self) -> None:
        """AST check: trainer.py uses atomic_text_write for epoch_latest.yaml."""
        src = Path("src/minivess/pipeline/trainer.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        found_atomic = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "fit":
                body_dump = ast.dump(node)
                if "atomic_text_write" in body_dump:
                    found_atomic = True
                break

        assert found_atomic, (
            "trainer.py fit() must use atomic_text_write for epoch_latest.yaml"
        )

    def test_trainer_imports_checkpoint_utils(self) -> None:
        """trainer.py must import atomic_torch_save and atomic_text_write."""
        src = Path("src/minivess/pipeline/trainer.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "checkpoint_utils" in node.module
            ):
                for alias in node.names:
                    imported_names.add(alias.name)

        assert "atomic_torch_save" in imported_names, "Must import atomic_torch_save"
        assert "atomic_text_write" in imported_names, "Must import atomic_text_write"


class TestCheckpointSHA256:
    """B2: SHA256 sidecar file written alongside checkpoints."""

    def test_compute_sha256(self, tmp_path: Path) -> None:
        """compute_checkpoint_sha256 returns correct SHA256 hex digest."""
        from minivess.pipeline.checkpoint_integrity import compute_checkpoint_sha256

        path = tmp_path / "test.pth"
        content = b"test checkpoint data"
        path.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        assert compute_checkpoint_sha256(path) == expected

    def test_write_sha256_sidecar(self, tmp_path: Path) -> None:
        """write_sha256_sidecar creates .sha256 file with correct hash."""
        from minivess.pipeline.checkpoint_integrity import write_sha256_sidecar

        path = tmp_path / "model.pth"
        path.write_bytes(b"model weights")

        write_sha256_sidecar(path)

        sidecar = path.with_suffix(".pth.sha256")
        assert sidecar.exists()
        expected = hashlib.sha256(b"model weights").hexdigest()
        assert sidecar.read_text(encoding="utf-8").strip() == expected

    def test_verify_checkpoint_sha256_valid(self, tmp_path: Path) -> None:
        """verify_checkpoint_sha256 returns True for uncorrupted file."""
        from minivess.pipeline.checkpoint_integrity import (
            verify_checkpoint_sha256,
            write_sha256_sidecar,
        )

        path = tmp_path / "model.pth"
        path.write_bytes(b"valid data")
        write_sha256_sidecar(path)

        assert verify_checkpoint_sha256(path) is True

    def test_verify_checkpoint_sha256_corrupted(self, tmp_path: Path) -> None:
        """verify_checkpoint_sha256 returns False for corrupted file."""
        from minivess.pipeline.checkpoint_integrity import (
            verify_checkpoint_sha256,
            write_sha256_sidecar,
        )

        path = tmp_path / "model.pth"
        path.write_bytes(b"original data")
        write_sha256_sidecar(path)

        # Corrupt the checkpoint
        path.write_bytes(b"corrupted data")

        assert verify_checkpoint_sha256(path) is False

    def test_verify_missing_sidecar_returns_false(self, tmp_path: Path) -> None:
        """verify_checkpoint_sha256 returns False when no .sha256 file exists."""
        from minivess.pipeline.checkpoint_integrity import verify_checkpoint_sha256

        path = tmp_path / "model.pth"
        path.write_bytes(b"data")

        assert verify_checkpoint_sha256(path) is False

    def test_verify_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """verify_checkpoint_sha256 raises FileNotFoundError for missing checkpoint."""
        from minivess.pipeline.checkpoint_integrity import verify_checkpoint_sha256

        path = tmp_path / "nonexistent.pth"
        with pytest.raises(FileNotFoundError):
            verify_checkpoint_sha256(path)

    def test_save_metric_checkpoint_writes_sidecar(self, tmp_path: Path) -> None:
        """save_metric_checkpoint writes .sha256 sidecar alongside .pth."""
        from minivess.pipeline.multi_metric_tracker import (
            MetricCheckpoint,
            save_metric_checkpoint,
        )

        path = tmp_path / "best_val_loss.pth"
        ckpt = MetricCheckpoint(
            epoch=5,
            metrics={"val_loss": 0.3},
            metric_name="val_loss",
            metric_value=0.3,
            metric_direction="minimize",
            train_loss=0.2,
            val_loss=0.3,
            wall_time_sec=120.0,
            config_snapshot={"lr": 0.001},
        )
        save_metric_checkpoint(
            path=path,
            model_state_dict={"w": torch.randn(2, 2)},
            optimizer_state_dict={"lr": 0.001},
            scheduler_state_dict={"step": 1},
            checkpoint=ckpt,
        )

        sidecar = path.with_suffix(".pth.sha256")
        assert sidecar.exists(), "save_metric_checkpoint must write SHA256 sidecar"
