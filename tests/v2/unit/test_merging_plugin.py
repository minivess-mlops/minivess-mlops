"""Tests for model merging post-training plugin.

Phase 4 of post-training plugin architecture (#318).
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


class TestModelMergingPlugin:
    """Model merging plugin should wrap linear/slerp/layer-wise merge."""

    def test_implements_protocol(self) -> None:
        from minivess.pipeline.post_training_plugins.model_merging import (
            ModelMergingPlugin,
        )

        assert isinstance(ModelMergingPlugin(), PostTrainingPlugin)

    def test_name_property(self) -> None:
        from minivess.pipeline.post_training_plugins.model_merging import (
            ModelMergingPlugin,
        )

        assert ModelMergingPlugin().name == "model_merging"

    def test_does_not_require_calibration_data(self) -> None:
        from minivess.pipeline.post_training_plugins.model_merging import (
            ModelMergingPlugin,
        )

        assert ModelMergingPlugin().requires_calibration_data is False

    def test_linear_merge(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.model_merging import (
            ModelMergingPlugin,
        )

        ckpt1 = _write_checkpoint(tmp_path / "ckpt1.pt", seed=1)
        ckpt2 = _write_checkpoint(tmp_path / "ckpt2.pt", seed=2)

        pi = PluginInput(
            checkpoint_paths=[ckpt1, ckpt2],
            config={"method": "linear", "t": 0.5, "output_dir": str(tmp_path)},
            run_metadata=[
                {"champion_category": "topology"},
                {"champion_category": "overlap"},
            ],
        )
        result = ModelMergingPlugin().execute(pi)
        assert len(result.model_paths) == 1

        # Verify merged weights are between the two inputs
        sd1 = _make_state_dict(1)
        sd2 = _make_state_dict(2)
        merged = torch.load(result.model_paths[0], weights_only=True)
        for key in sd1:
            if sd1[key].is_floating_point():
                expected = 0.5 * sd1[key] + 0.5 * sd2[key]
                assert torch.allclose(merged[key], expected, atol=1e-5)

    def test_slerp_merge(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.model_merging import (
            ModelMergingPlugin,
        )

        ckpt1 = _write_checkpoint(tmp_path / "ckpt1.pt", seed=1)
        ckpt2 = _write_checkpoint(tmp_path / "ckpt2.pt", seed=2)

        pi = PluginInput(
            checkpoint_paths=[ckpt1, ckpt2],
            config={"method": "slerp", "t": 0.5, "output_dir": str(tmp_path)},
            run_metadata=[
                {"champion_category": "topology"},
                {"champion_category": "overlap"},
            ],
        )
        result = ModelMergingPlugin().execute(pi)
        assert len(result.model_paths) == 1
        merged = torch.load(result.model_paths[0], weights_only=True)
        assert "conv.weight" in merged

    def test_validate_inputs_needs_exactly_two(self) -> None:
        from minivess.pipeline.post_training_plugins.model_merging import (
            ModelMergingPlugin,
        )

        pi = PluginInput(
            checkpoint_paths=[Path("/tmp/one.pt")],
            config={"method": "slerp"},
        )
        errors = ModelMergingPlugin().validate_inputs(pi)
        assert len(errors) > 0
        assert (
            "2" in errors[0]
            or "two" in errors[0].lower()
            or "pair" in errors[0].lower()
        )

    def test_merge_multiple_pairs(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.model_merging import (
            ModelMergingPlugin,
        )

        ckpt1 = _write_checkpoint(tmp_path / "ckpt1.pt", seed=1)
        ckpt2 = _write_checkpoint(tmp_path / "ckpt2.pt", seed=2)
        ckpt3 = _write_checkpoint(tmp_path / "ckpt3.pt", seed=3)

        pi = PluginInput(
            checkpoint_paths=[ckpt1, ckpt2, ckpt3],
            config={
                "method": "linear",
                "t": 0.5,
                "output_dir": str(tmp_path),
                "merge_pairs": [["topology", "overlap"], ["topology", "balanced"]],
            },
            run_metadata=[
                {"champion_category": "topology"},
                {"champion_category": "overlap"},
                {"champion_category": "balanced"},
            ],
        )
        result = ModelMergingPlugin().execute(pi)
        assert len(result.model_paths) == 2

    def test_invalid_method_in_validate(self) -> None:
        from minivess.pipeline.post_training_plugins.model_merging import (
            ModelMergingPlugin,
        )

        pi = PluginInput(
            checkpoint_paths=[Path("/tmp/a.pt"), Path("/tmp/b.pt")],
            config={"method": "unknown_method"},
        )
        errors = ModelMergingPlugin().validate_inputs(pi)
        assert any("method" in e.lower() for e in errors)
