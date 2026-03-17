"""Tests for factorial post-training execution loop.

Validates run_factorial_post_training() that systematically applies
post-training methods across all checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch


def _make_temp_checkpoint(tmp_path: Path, name: str = "ckpt.pt") -> Path:
    """Create a temporary checkpoint file."""
    ckpt_path = tmp_path / name
    state = {"model_state_dict": {"layer.weight": torch.randn(4, 4)}}
    torch.save(state, ckpt_path)
    return ckpt_path


class TestFactorialExecutionNonePassthrough:
    """T4: 'none' method returns checkpoint unchanged."""

    def test_factorial_execution_none_passthrough(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        ckpt = _make_temp_checkpoint(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        results = run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["none"],
            output_dir=output_dir,
        )

        assert len(results) == 1
        result = results[0]
        assert result["method"] == "none"
        assert Path(result["output_path"]).exists()
        # None method should be a copy / symlink — same state_dict
        loaded = torch.load(Path(result["output_path"]), weights_only=True)
        original = torch.load(ckpt, weights_only=True)
        for key in original["model_state_dict"]:
            assert torch.equal(
                loaded["model_state_dict"][key],
                original["model_state_dict"][key],
            )


class TestFactorialExecutionSWA:
    """T4: SWA method averages weights."""

    def test_factorial_execution_swa(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        # Need at least 2 checkpoints for SWA to average
        ckpt1 = _make_temp_checkpoint(tmp_path, "ckpt1.pt")
        ckpt2 = _make_temp_checkpoint(tmp_path, "ckpt2.pt")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        results = run_factorial_post_training(
            checkpoint_paths=[ckpt1, ckpt2],
            methods=["swa"],
            output_dir=output_dir,
        )

        assert len(results) == 1
        result = results[0]
        assert result["method"] == "swa"
        assert Path(result["output_path"]).exists()


class TestFactorialExecutionMultiSWA:
    """T4: Multi-SWA produces n_models variants."""

    def test_factorial_execution_multi_swa(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        ckpts = [_make_temp_checkpoint(tmp_path, f"ckpt{i}.pt") for i in range(4)]
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        results = run_factorial_post_training(
            checkpoint_paths=ckpts,
            methods=["multi_swa"],
            output_dir=output_dir,
            n_multi_swa_models=2,
        )

        assert len(results) == 1
        result = results[0]
        assert result["method"] == "multi_swa"
        assert Path(result["output_path"]).exists()


class TestFactorialExecutionAllMethods:
    """T4: Running all methods produces 3 results."""

    def test_factorial_execution_all_methods(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        ckpts = [_make_temp_checkpoint(tmp_path, f"ckpt{i}.pt") for i in range(3)]
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        results = run_factorial_post_training(
            checkpoint_paths=ckpts,
            methods=["none", "swa", "multi_swa"],
            output_dir=output_dir,
        )

        assert len(results) == 3
        methods = {r["method"] for r in results}
        assert methods == {"none", "swa", "multi_swa"}


class TestFactorialExecutionMlflowTags:
    """T4: Each variant is tagged with post_training_method."""

    def test_factorial_execution_mlflow_tags(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        ckpt = _make_temp_checkpoint(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        results = run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["none", "swa"],
            output_dir=output_dir,
        )

        for result in results:
            assert "post_training_method" in result
            assert result["post_training_method"] in {"none", "swa"}


class TestFactorialExecutionCheckpointNaming:
    """T4: Output checkpoints follow naming convention."""

    def test_factorial_execution_checkpoint_naming(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        ckpt = _make_temp_checkpoint(tmp_path, "run123.pt")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        results = run_factorial_post_training(
            checkpoint_paths=[ckpt],
            methods=["none"],
            output_dir=output_dir,
        )

        output_path = Path(results[0]["output_path"])
        assert output_path.suffix == ".pt"
        assert "none" in output_path.stem
