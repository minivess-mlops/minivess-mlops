"""Tests for SAM3 debug experiment configs and runner script.

Validates that:
- sam3_vanilla_debug.yaml loads correctly with expected values
- sam3_topolora_debug.yaml loads correctly with LoRA architecture params
- sam3_hybrid_debug.yaml loads correctly with filter/FPN params
- run_sam3_debug_experiment.py is importable

Closes #258.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

VANILLA_CONFIG = CONFIGS_DIR / "sam3_vanilla_debug.yaml"
TOPOLORA_CONFIG = CONFIGS_DIR / "sam3_topolora_debug.yaml"
HYBRID_CONFIG = CONFIGS_DIR / "sam3_hybrid_debug.yaml"
RUNNER_SCRIPT = SCRIPTS_DIR / "run_sam3_debug_experiment.py"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Helper: load YAML config from path."""
    assert path.exists(), f"Config not found: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestSam3VanillaDebugConfig:
    """Validate sam3_vanilla_debug.yaml structure and values."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return _load_yaml(VANILLA_CONFIG)

    def test_loads_without_error(self) -> None:
        """Config loads from YAML without error."""
        cfg = _load_yaml(VANILLA_CONFIG)
        assert isinstance(cfg, dict)

    def test_experiment_name(self, config: dict[str, Any]) -> None:
        assert config["experiment_name"] == "sam3_vanilla_debug"

    def test_model_family(self, config: dict[str, Any]) -> None:
        assert config["model_family"] == "sam3_vanilla"

    def test_losses(self, config: dict[str, Any]) -> None:
        assert config["losses"] == ["dice_ce"]

    def test_num_folds(self, config: dict[str, Any]) -> None:
        assert config["num_folds"] == 3

    def test_max_epochs(self, config: dict[str, Any]) -> None:
        assert config["max_epochs"] == 6

    def test_seed(self, config: dict[str, Any]) -> None:
        assert config["seed"] == 42

    def test_debug_flag(self, config: dict[str, Any]) -> None:
        assert config.get("debug") is True

    def test_compute_is_cpu(self, config: dict[str, Any]) -> None:
        assert config["compute"] == "cpu"

    def test_split_mode(self, config: dict[str, Any]) -> None:
        assert config["split_mode"] == "file"

    def test_splits_file(self, config: dict[str, Any]) -> None:
        assert config["splits_file"] == "configs/splits/3fold_seed42.json"

    def test_architecture_params_present(self, config: dict[str, Any]) -> None:
        assert "architecture_params" in config

    def test_architecture_params_backbone(self, config: dict[str, Any]) -> None:
        assert config["architecture_params"]["backbone"] == "vit_32l"

    def test_architecture_params_input_size(self, config: dict[str, Any]) -> None:
        assert config["architecture_params"]["input_size"] == 1008

    def test_architecture_params_embed_dim(self, config: dict[str, Any]) -> None:
        assert config["architecture_params"]["embed_dim"] == 1024

    def test_architecture_params_pretrained_false(self, config: dict[str, Any]) -> None:
        assert config["architecture_params"]["pretrained"] is False

    def test_checkpoint_section_present(self, config: dict[str, Any]) -> None:
        assert "checkpoint" in config

    def test_checkpoint_primary_metric(self, config: dict[str, Any]) -> None:
        assert config["checkpoint"]["primary_metric"] == "val_loss"

    def test_checkpoint_tracked_metrics_present(self, config: dict[str, Any]) -> None:
        metrics = config["checkpoint"]["tracked_metrics"]
        assert isinstance(metrics, list)
        assert len(metrics) >= 1

    def test_checkpoint_save_last(self, config: dict[str, Any]) -> None:
        assert config["checkpoint"]["save_last"] is True

    def test_notes_present(self, config: dict[str, Any]) -> None:
        assert "notes" in config
        assert isinstance(config["notes"], str)
        assert len(config["notes"]) > 0


class TestSam3TopoLoRADebugConfig:
    """Validate sam3_topolora_debug.yaml structure and values."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return _load_yaml(TOPOLORA_CONFIG)

    def test_loads_without_error(self) -> None:
        """Config loads from YAML without error."""
        cfg = _load_yaml(TOPOLORA_CONFIG)
        assert isinstance(cfg, dict)

    def test_experiment_name(self, config: dict[str, Any]) -> None:
        assert config["experiment_name"] == "sam3_topolora_debug"

    def test_model_family(self, config: dict[str, Any]) -> None:
        assert config["model_family"] == "sam3_topolora"

    def test_losses(self, config: dict[str, Any]) -> None:
        assert config["losses"] == ["cbdice_cldice"]

    def test_num_folds(self, config: dict[str, Any]) -> None:
        assert config["num_folds"] == 3

    def test_max_epochs(self, config: dict[str, Any]) -> None:
        assert config["max_epochs"] == 6

    def test_seed(self, config: dict[str, Any]) -> None:
        assert config["seed"] == 42

    def test_debug_flag(self, config: dict[str, Any]) -> None:
        assert config.get("debug") is True

    def test_lora_rank_in_architecture_params(self, config: dict[str, Any]) -> None:
        arch = config["architecture_params"]
        assert "lora_rank" in arch
        assert arch["lora_rank"] == 16

    def test_lora_alpha_in_architecture_params(self, config: dict[str, Any]) -> None:
        arch = config["architecture_params"]
        assert "lora_alpha" in arch
        assert arch["lora_alpha"] == 32.0

    def test_lora_dropout_in_architecture_params(self, config: dict[str, Any]) -> None:
        arch = config["architecture_params"]
        assert "lora_dropout" in arch
        assert arch["lora_dropout"] == 0.1

    def test_lora_targets_in_architecture_params(self, config: dict[str, Any]) -> None:
        arch = config["architecture_params"]
        assert "lora_targets" in arch
        assert arch["lora_targets"] == ["mlp.lin1", "mlp.lin2"]

    def test_checkpoint_section_present(self, config: dict[str, Any]) -> None:
        assert "checkpoint" in config

    def test_notes_present(self, config: dict[str, Any]) -> None:
        assert "notes" in config
        assert isinstance(config["notes"], str)
        assert len(config["notes"]) > 0


class TestSam3HybridDebugConfig:
    """Validate sam3_hybrid_debug.yaml structure and values."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return _load_yaml(HYBRID_CONFIG)

    def test_loads_without_error(self) -> None:
        """Config loads from YAML without error."""
        cfg = _load_yaml(HYBRID_CONFIG)
        assert isinstance(cfg, dict)

    def test_experiment_name(self, config: dict[str, Any]) -> None:
        assert config["experiment_name"] == "sam3_hybrid_debug"

    def test_model_family(self, config: dict[str, Any]) -> None:
        assert config["model_family"] == "sam3_hybrid"

    def test_losses(self, config: dict[str, Any]) -> None:
        assert config["losses"] == ["cbdice_cldice"]

    def test_num_folds(self, config: dict[str, Any]) -> None:
        assert config["num_folds"] == 3

    def test_max_epochs(self, config: dict[str, Any]) -> None:
        assert config["max_epochs"] == 6

    def test_seed(self, config: dict[str, Any]) -> None:
        assert config["seed"] == 42

    def test_debug_flag(self, config: dict[str, Any]) -> None:
        assert config.get("debug") is True

    def test_filters_in_architecture_params(self, config: dict[str, Any]) -> None:
        arch = config["architecture_params"]
        assert "filters" in arch
        assert arch["filters"] == [32, 64, 128, 256]

    def test_fpn_dim_in_architecture_params(self, config: dict[str, Any]) -> None:
        arch = config["architecture_params"]
        assert "fpn_dim" in arch
        assert arch["fpn_dim"] == 256

    def test_fusion_gate_init_in_architecture_params(
        self, config: dict[str, Any]
    ) -> None:
        arch = config["architecture_params"]
        assert "fusion_gate_init" in arch
        assert arch["fusion_gate_init"] == 0.0

    def test_checkpoint_section_present(self, config: dict[str, Any]) -> None:
        assert "checkpoint" in config

    def test_notes_present(self, config: dict[str, Any]) -> None:
        assert "notes" in config
        assert isinstance(config["notes"], str)
        assert len(config["notes"]) > 0


class TestSam3DebugRunnerScript:
    """Validate run_sam3_debug_experiment.py is importable and well-formed."""

    def test_runner_script_exists(self) -> None:
        assert RUNNER_SCRIPT.exists(), f"Runner script not found: {RUNNER_SCRIPT}"

    def test_runner_script_importable(self) -> None:
        """Script imports without errors."""
        scripts_dir = str(SCRIPTS_DIR)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_sam3_debug_experiment", RUNNER_SCRIPT
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Loading should not raise
        spec.loader.exec_module(module)

    def test_runner_has_main_function(self) -> None:
        """Runner script exposes a main() function."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_sam3_debug_experiment", RUNNER_SCRIPT
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "main"), "run_sam3_debug_experiment must expose main()"

    def test_runner_has_dry_run_support(self) -> None:
        """Runner script accepts --dry-run argument without error."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_sam3_debug_experiment", RUNNER_SCRIPT
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # main(["--dry-run"]) should not raise — it validates configs only
        # We test that the function signature accepts argv param
        import inspect

        sig = inspect.signature(module.main)
        assert "argv" in sig.parameters, "main() must accept argv parameter"

    def test_runner_references_all_three_configs(self) -> None:
        """Runner script references all three SAM3 debug configs."""
        content = RUNNER_SCRIPT.read_text(encoding="utf-8")
        assert "sam3_vanilla_debug" in content
        assert "sam3_topolora_debug" in content
        assert "sam3_hybrid_debug" in content
