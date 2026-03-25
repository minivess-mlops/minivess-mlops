"""Tests for atomic checkpoint writes and corruption fallback.

T0.5: Verify torch.save uses tmp + os.replace pattern for atomicity.
B1: Verify wiring — save_metric_checkpoint and trainer epoch_latest use atomic saves.
B2: SHA256 checkpoint integrity verification.
2.D: Atomic save + corruption fallback (Issue #949).
"""

from __future__ import annotations

import ast
import hashlib
import logging
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


class TestAtomicSaveIdentity:
    """2.D: atomic_torch_save produces file identical to what torch.load expects."""

    def test_atomic_save_produces_identical_file(self, tmp_path: Path) -> None:
        """Save with atomic_torch_save, load, compare state_dict tensors."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        original = {
            "layer.weight": torch.randn(8, 8),
            "layer.bias": torch.randn(8),
            "epoch": 42,
        }
        path = tmp_path / "ckpt.pth"
        atomic_torch_save(original, path)

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 42
        assert torch.equal(loaded["layer.weight"], original["layer.weight"])
        assert torch.equal(loaded["layer.bias"], original["layer.bias"])


class TestAtomicSaveNoCrashPartial:
    """2.D: on rename failure, no .pth file remains — only .tmp (or nothing)."""

    def test_atomic_save_no_partial_on_crash(self, tmp_path: Path) -> None:
        """Mock os.replace to raise after torch.save; verify no .pth file remains."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        path = tmp_path / "model.pth"

        with (
            patch(
                "minivess.pipeline.checkpoint_utils.os.replace",
                side_effect=OSError("rename failed"),
            ),
            pytest.raises(OSError, match="rename failed"),
        ):
            atomic_torch_save({"epoch": 1}, path)

        # The final .pth must NOT exist (rename was never completed)
        assert not path.exists(), "Final .pth must not exist when rename fails"
        # The .tmp file should also be cleaned up by the except clause
        tmp_file = path.with_suffix(".pth.tmp")
        assert not tmp_file.exists(), ".tmp file should be cleaned up on failure"


class TestFallbackLoadsPrevious:
    """2.D: load_checkpoint_with_fallback tries primary then falls back."""

    def _write_valid_checkpoint(self, path: Path, epoch: int) -> dict:
        """Helper: write a valid checkpoint and return the state dict."""
        state = {"epoch": epoch, "weight": torch.randn(4, 4)}
        torch.save(state, path)
        return state

    def test_fallback_loads_previous_when_latest_truncated(
        self, tmp_path: Path
    ) -> None:
        """Write truncated bytes as 'latest', valid as 'previous' -> fallback."""
        from minivess.pipeline.checkpoint_utils import load_checkpoint_with_fallback

        primary = tmp_path / "epoch_latest.pth"
        fallback = tmp_path / "epoch_previous.pth"

        # Write valid fallback
        self._write_valid_checkpoint(fallback, epoch=5)
        # Write truncated primary (not valid pickle/torch data)
        primary.write_bytes(b"\x80\x02truncated_garbage")

        result = load_checkpoint_with_fallback(primary, fallback=fallback)
        assert result["epoch"] == 5

    def test_fallback_loads_previous_when_latest_not_pickle(
        self, tmp_path: Path
    ) -> None:
        """Write JSON bytes as 'latest.pth' -> torch.load fails -> fallback."""
        from minivess.pipeline.checkpoint_utils import load_checkpoint_with_fallback

        primary = tmp_path / "latest.pth"
        fallback = tmp_path / "previous.pth"

        # Write valid fallback
        self._write_valid_checkpoint(fallback, epoch=10)
        # Write non-pickle data as primary
        primary.write_bytes(b'{"not": "a checkpoint"}')

        result = load_checkpoint_with_fallback(primary, fallback=fallback)
        assert result["epoch"] == 10


class TestDoubleCorruption:
    """2.D: both primary and fallback corrupt -> ValueError (Rule #25)."""

    def test_double_corruption_raises_value_error(self, tmp_path: Path) -> None:
        """Both latest and previous are truncated -> ValueError raised."""
        from minivess.pipeline.checkpoint_utils import load_checkpoint_with_fallback

        primary = tmp_path / "latest.pth"
        fallback = tmp_path / "previous.pth"

        # Write garbage to both
        primary.write_bytes(b"corrupt_primary")
        fallback.write_bytes(b"corrupt_fallback")

        with pytest.raises(ValueError, match="Both primary.*and fallback.*corrupt"):
            load_checkpoint_with_fallback(primary, fallback=fallback)


class TestFallbackLogging:
    """2.D: fallback emits a warning with the fallback path string."""

    def test_logs_warning_with_fallback_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """caplog captures warning with fallback path string."""
        from minivess.pipeline.checkpoint_utils import load_checkpoint_with_fallback

        primary = tmp_path / "latest.pth"
        fallback = tmp_path / "previous.pth"

        # Corrupt primary, valid fallback
        primary.write_bytes(b"corrupt")
        state = {"epoch": 3, "w": torch.randn(2, 2)}
        torch.save(state, fallback)

        with caplog.at_level(logging.WARNING, logger="minivess.pipeline.checkpoint_utils"):
            result = load_checkpoint_with_fallback(primary, fallback=fallback)

        assert result["epoch"] == 3
        # Check that a warning was logged mentioning the fallback path
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(str(fallback) in msg for msg in warning_messages), (
            f"Expected warning mentioning fallback path {fallback}, "
            f"got: {warning_messages}"
        )


class TestMissingFallback:
    """2.D: latest corrupt + no fallback file -> FileNotFoundError."""

    def test_missing_fallback_raises_file_not_found(self, tmp_path: Path) -> None:
        """Latest corrupt, no fallback file -> FileNotFoundError."""
        from minivess.pipeline.checkpoint_utils import load_checkpoint_with_fallback

        primary = tmp_path / "latest.pth"
        primary.write_bytes(b"corrupt")

        # No fallback provided
        with pytest.raises(FileNotFoundError, match="Primary corrupt and no fallback"):
            load_checkpoint_with_fallback(primary, fallback=None)

    def test_missing_fallback_file_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """Latest corrupt, fallback path specified but file doesn't exist."""
        from minivess.pipeline.checkpoint_utils import load_checkpoint_with_fallback

        primary = tmp_path / "latest.pth"
        primary.write_bytes(b"corrupt")
        fallback = tmp_path / "nonexistent.pth"

        with pytest.raises(FileNotFoundError, match="Primary corrupt and no fallback"):
            load_checkpoint_with_fallback(primary, fallback=fallback)
