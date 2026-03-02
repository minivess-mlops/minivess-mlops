"""Tests for model-agnostic training infrastructure (T1).

Verifies that train_monitored.py supports --model-family argument and
uses build_adapter() instead of hardcoded DynUNetAdapter.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


# ---------------------------------------------------------------------------
# Test parse_args accepts --model-family
# ---------------------------------------------------------------------------
class TestParseArgsModelFamily:
    """Test that parse_args handles --model-family argument."""

    def test_model_family_default_is_dynunet(self) -> None:
        """Default model_family should be 'dynunet' when not specified."""
        from scripts.train_monitored import parse_args

        args = parse_args(
            [
                "--loss",
                "dice_ce",
                "--data-dir",
                "data/raw/minivess",
            ]
        )
        assert args.model_family == "dynunet"

    def test_model_family_sam3_vanilla(self) -> None:
        """--model-family sam3_vanilla should be accepted."""
        from scripts.train_monitored import parse_args

        args = parse_args(
            [
                "--model-family",
                "sam3_vanilla",
                "--loss",
                "dice_ce",
            ]
        )
        assert args.model_family == "sam3_vanilla"

    def test_model_family_sam3_topolora(self) -> None:
        from scripts.train_monitored import parse_args

        args = parse_args(
            ["--model-family", "sam3_topolora", "--loss", "cbdice_cldice"]
        )
        assert args.model_family == "sam3_topolora"

    def test_model_family_sam3_hybrid(self) -> None:
        from scripts.train_monitored import parse_args

        args = parse_args(["--model-family", "sam3_hybrid", "--loss", "cbdice_cldice"])
        assert args.model_family == "sam3_hybrid"


# ---------------------------------------------------------------------------
# Test _build_configs creates correct ModelConfig from model_family
# ---------------------------------------------------------------------------
class TestBuildConfigsModelFamily:
    """Test _build_configs creates correct ModelConfig for each model family."""

    def _make_args(
        self, model_family: str = "dynunet", **kwargs: object
    ) -> argparse.Namespace:
        """Create a minimal args namespace for _build_configs."""
        from scripts.train_monitored import parse_args

        argv = [
            "--model-family",
            model_family,
            "--loss",
            "dice_ce",
            "--data-dir",
            "data/raw/minivess",
            "--compute",
            "cpu",
            "--debug",
        ]
        args = parse_args(argv)
        # Inject any extra attributes (architecture_params, etc.)
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args

    def test_dynunet_default(self) -> None:
        """Default model_family=dynunet creates MONAI_DYNUNET ModelConfig."""
        from scripts.train_monitored import _build_configs

        args = self._make_args("dynunet")
        data_config, model_config, _training_config = _build_configs(args)
        assert model_config.family.value == "dynunet"
        assert model_config.name == "dynunet"

    def test_sam3_vanilla_creates_correct_family(self) -> None:
        """model_family=sam3_vanilla creates SAM3_VANILLA ModelConfig."""
        from scripts.train_monitored import _build_configs

        args = self._make_args("sam3_vanilla")
        _dc, model_config, _tc = _build_configs(args)
        assert model_config.family.value == "sam3_vanilla"
        assert model_config.name == "sam3_vanilla"

    def test_sam3_vanilla_not_dynunet(self) -> None:
        """NEGATIVE: sam3_vanilla must NOT create MONAI_DYNUNET."""
        from scripts.train_monitored import _build_configs

        args = self._make_args("sam3_vanilla")
        _dc, model_config, _tc = _build_configs(args)
        assert model_config.family.value != "dynunet"

    def test_sam3_hybrid_with_architecture_params(self) -> None:
        """Hybrid config passes architecture_params including filters."""
        from scripts.train_monitored import _build_configs

        args = self._make_args(
            "sam3_hybrid",
            architecture_params={
                "backbone": "vit_32l",
                "filters": [32, 64, 128, 256],
                "fusion_gate_init": 0.0,
            },
        )
        _dc, model_config, _tc = _build_configs(args)
        assert model_config.family.value == "sam3_hybrid"
        assert model_config.architecture_params["filters"] == [32, 64, 128, 256]

    def test_sam3_topolora_extracts_lora_params(self) -> None:
        """TopoLoRA config extracts lora_rank/lora_alpha from architecture_params."""
        from scripts.train_monitored import _build_configs

        args = self._make_args(
            "sam3_topolora",
            architecture_params={
                "backbone": "vit_32l",
                "lora_rank": 8,
                "lora_alpha": 16.0,
                "lora_dropout": 0.05,
            },
        )
        _dc, model_config, _tc = _build_configs(args)
        assert model_config.family.value == "sam3_topolora"
        assert model_config.lora_rank == 8
        assert model_config.lora_alpha == 16.0


# ---------------------------------------------------------------------------
# Test: no hardcoded DynUNetAdapter in run_fold_safe (static analysis)
# ---------------------------------------------------------------------------
class TestNoDynUNetHardcode:
    """Verify DynUNetAdapter is not directly used in run_fold_safe."""

    def test_run_fold_safe_no_dynunet_adapter_call(self) -> None:
        """run_fold_safe must not contain 'DynUNetAdapter(' in source."""
        import scripts.train_monitored as mod

        source = inspect.getsource(mod.run_fold_safe)
        assert "DynUNetAdapter(" not in source, (
            "run_fold_safe still hardcodes DynUNetAdapter. "
            "Should use build_adapter(model_config) instead."
        )

    def test_module_imports_build_adapter(self) -> None:
        """train_monitored should import build_adapter."""
        import scripts.train_monitored as mod

        # Check that build_adapter is accessible (imported or defined)
        source = inspect.getsource(mod)
        assert "build_adapter" in source, (
            "train_monitored.py does not reference build_adapter. "
            "Should import from minivess.adapters.model_builder."
        )
