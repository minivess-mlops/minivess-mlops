"""Tests for Multi-SWA post-training plugin.

Phase 3 of post-training plugin architecture (#317).
"""

from __future__ import annotations

from pathlib import Path

import torch

from minivess.pipeline.post_training_plugin import (
    PluginInput,
    PostTrainingPlugin,
)


def _make_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "conv.weight": torch.randn(4, 1, 3, 3, 3, generator=gen),
        "conv.bias": torch.randn(4, generator=gen),
    }


def _write_checkpoint(path: Path, seed: int = 0) -> Path:
    sd = _make_state_dict(seed)
    torch.save({"state_dict": sd}, path)
    return path


class TestMultiSWAPlugin:
    """Multi-SWA plugin should produce M independent SWA models."""

    def test_implements_protocol(self) -> None:
        from minivess.pipeline.post_training_plugins.multi_swa import MultiSWAPlugin

        assert isinstance(MultiSWAPlugin(), PostTrainingPlugin)

    def test_name_property(self) -> None:
        from minivess.pipeline.post_training_plugins.multi_swa import MultiSWAPlugin

        assert MultiSWAPlugin().name == "multi_swa"

    def test_produces_n_models(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.multi_swa import MultiSWAPlugin

        # Create 6 checkpoints
        ckpts = [_write_checkpoint(tmp_path / f"ckpt{i}.pt", seed=i) for i in range(6)]

        pi = PluginInput(
            checkpoint_paths=ckpts,
            config={
                "n_models": 3,
                "subsample_fraction": 0.5,
                "seed": 42,
                "output_dir": str(tmp_path),
            },
        )
        result = MultiSWAPlugin().execute(pi)
        assert len(result.model_paths) == 3

    def test_different_seeds_different_subsets(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.multi_swa import MultiSWAPlugin

        ckpts = [_write_checkpoint(tmp_path / f"ckpt{i}.pt", seed=i) for i in range(10)]

        pi1 = PluginInput(
            checkpoint_paths=ckpts,
            config={
                "n_models": 2,
                "subsample_fraction": 0.5,
                "seed": 1,
                "output_dir": str(tmp_path / "run1"),
            },
        )
        pi2 = PluginInput(
            checkpoint_paths=ckpts,
            config={
                "n_models": 2,
                "subsample_fraction": 0.5,
                "seed": 99,
                "output_dir": str(tmp_path / "run2"),
            },
        )
        r1 = MultiSWAPlugin().execute(pi1)
        r2 = MultiSWAPlugin().execute(pi2)

        # Different seeds should produce different averaged state dicts
        sd1 = torch.load(r1.model_paths[0], weights_only=True)
        sd2 = torch.load(r2.model_paths[0], weights_only=True)
        # At least one weight should differ
        any_diff = any(
            not torch.equal(sd1[k], sd2[k]) for k in sd1 if sd1[k].is_floating_point()
        )
        assert any_diff

    def test_reproducible_with_same_seed(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.multi_swa import MultiSWAPlugin

        ckpts = [_write_checkpoint(tmp_path / f"ckpt{i}.pt", seed=i) for i in range(6)]

        common_config = {"n_models": 2, "subsample_fraction": 0.5, "seed": 42}

        pi1 = PluginInput(
            checkpoint_paths=ckpts,
            config={**common_config, "output_dir": str(tmp_path / "r1")},
        )
        pi2 = PluginInput(
            checkpoint_paths=ckpts,
            config={**common_config, "output_dir": str(tmp_path / "r2")},
        )

        r1 = MultiSWAPlugin().execute(pi1)
        r2 = MultiSWAPlugin().execute(pi2)

        for p1, p2 in zip(r1.model_paths, r2.model_paths, strict=True):
            sd1 = torch.load(p1, weights_only=True)
            sd2 = torch.load(p2, weights_only=True)
            for k in sd1:
                assert torch.equal(sd1[k], sd2[k])

    def test_subsample_fraction_1_equals_single_swa(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.multi_swa import MultiSWAPlugin

        ckpts = [_write_checkpoint(tmp_path / f"ckpt{i}.pt", seed=i) for i in range(4)]

        pi = PluginInput(
            checkpoint_paths=ckpts,
            config={
                "n_models": 2,
                "subsample_fraction": 1.0,
                "seed": 42,
                "output_dir": str(tmp_path),
            },
        )
        result = MultiSWAPlugin().execute(pi)
        # With fraction=1.0, all models use all checkpoints → all identical
        sd0 = torch.load(result.model_paths[0], weights_only=True)
        sd1 = torch.load(result.model_paths[1], weights_only=True)
        for k in sd0:
            assert torch.equal(sd0[k], sd1[k])

    def test_validate_inputs_too_few_checkpoints(self) -> None:
        from minivess.pipeline.post_training_plugins.multi_swa import MultiSWAPlugin

        pi = PluginInput(
            checkpoint_paths=[Path("/tmp/single.pt")],
            config={"n_models": 3, "subsample_fraction": 0.5},
        )
        errors = MultiSWAPlugin().validate_inputs(pi)
        assert len(errors) > 0
