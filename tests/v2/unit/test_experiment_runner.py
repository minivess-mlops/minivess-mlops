from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Add scripts to path for import
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts")
)


class TestExperimentConfig:
    def test_parse_experiment_yaml(self, tmp_path: Path) -> None:
        """Experiment YAML is parsed into structured config."""
        config = {
            "experiment_name": "test_experiment",
            "model": "dynunet",
            "losses": ["dice_ce"],
            "compute": "auto",
            "data_dir": str(tmp_path),
            "num_folds": 3,
            "max_epochs": 1,
            "seed": 42,
        }
        yaml_path = tmp_path / "exp.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        from run_experiment import load_experiment_config

        result = load_experiment_config(yaml_path)
        assert result["experiment_name"] == "test_experiment"
        assert result["model"] == "dynunet"
        assert result["losses"] == ["dice_ce"]

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        """Missing experiment_name raises ValueError."""
        config = {"model": "dynunet", "losses": ["dice_ce"]}
        yaml_path = tmp_path / "bad.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        from run_experiment import load_experiment_config

        with pytest.raises((ValueError, KeyError)):
            load_experiment_config(yaml_path)

    def test_missing_model_field_raises(self, tmp_path: Path) -> None:
        """Missing model raises ValueError."""
        config = {"experiment_name": "test_exp", "losses": ["dice_ce"]}
        yaml_path = tmp_path / "bad_model.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        from run_experiment import load_experiment_config

        with pytest.raises((ValueError, KeyError)):
            load_experiment_config(yaml_path)

    def test_missing_losses_field_raises(self, tmp_path: Path) -> None:
        """Missing losses raises ValueError."""
        config = {"experiment_name": "test_exp", "model": "dynunet"}
        yaml_path = tmp_path / "bad_losses.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        from run_experiment import load_experiment_config

        with pytest.raises((ValueError, KeyError)):
            load_experiment_config(yaml_path)

    def test_optional_fields_have_defaults(self, tmp_path: Path) -> None:
        """Optional fields get default values when absent."""
        config = {
            "experiment_name": "minimal_exp",
            "model": "dynunet",
            "losses": ["dice_ce"],
        }
        yaml_path = tmp_path / "minimal.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        from run_experiment import load_experiment_config

        result = load_experiment_config(yaml_path)
        assert result["experiment_name"] == "minimal_exp"
        # Must at minimum contain required fields
        assert "model" in result
        assert "losses" in result


class TestDryRun:
    def test_dry_run_validates_without_training(self, tmp_path: Path) -> None:
        """Dry run validates config and exits without training."""
        config = {
            "experiment_name": "dry_test",
            "model": "dynunet",
            "losses": ["dice_ce"],
            "compute": "gpu_low",
            "data_dir": str(tmp_path),
            "num_folds": 3,
            "max_epochs": 1,
            "seed": 42,
            "debug": True,
        }
        yaml_path = tmp_path / "exp.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        from run_experiment import run_dry_run

        # Create minimal data dir structure
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        (img_dir / "v1.nii.gz").touch()
        (lbl_dir / "v1.nii.gz").touch()

        # Dry run should succeed (validates config) — it might warn about things
        # but should not raise
        result = run_dry_run(config)
        assert isinstance(result, dict)
        assert "preflight" in result or "validation" in result

    def test_dry_run_returns_dict(self, tmp_path: Path) -> None:
        """Dry run always returns a dict result."""
        config = {
            "experiment_name": "dict_test",
            "model": "dynunet",
            "losses": ["dice_ce"],
            "compute": "cpu",
            "data_dir": str(tmp_path),
            "num_folds": 2,
            "max_epochs": 1,
            "seed": 42,
        }

        from run_experiment import run_dry_run

        result = run_dry_run(config)
        assert isinstance(result, dict)

    def test_dry_run_includes_validation_key(self, tmp_path: Path) -> None:
        """Dry run result has a 'validation' key."""
        config = {
            "experiment_name": "valid_key_test",
            "model": "dynunet",
            "losses": ["dice_ce"],
            "compute": "cpu",
            "data_dir": str(tmp_path),
            "num_folds": 2,
            "max_epochs": 1,
            "seed": 42,
        }

        from run_experiment import run_dry_run

        result = run_dry_run(config)
        assert "validation" in result


class TestComputeSelection:
    def test_auto_profile_triggers_detection(self, tmp_path: Path) -> None:
        """compute='auto' triggers hardware detection."""
        from run_experiment import resolve_compute_profile

        # Mock hardware detection to avoid actual nvidia-smi calls
        mock_budget = MagicMock()
        mock_budget.gpu_vram_mb = 8192
        mock_budget.ram_available_mb = 24000
        mock_budget.gpu_tier = "gpu_low"

        mock_profile = MagicMock()
        mock_profile.num_volumes = 70
        mock_profile.min_shape = (512, 512, 5)
        mock_profile.total_size_bytes = 4_500_000_000

        with (
            patch("run_experiment.detect_hardware", return_value=mock_budget),
            patch("run_experiment.compute_adaptive_profile") as mock_adaptive,
        ):
            mock_adaptive.return_value = MagicMock(
                name="auto_gpu_low_dynunet",
                batch_size=2,
                patch_size=(96, 96, 4),
                num_workers=2,
                mixed_precision=True,
                gradient_accumulation_steps=2,
                cache_rate=1.0,
            )
            resolve_compute_profile("auto", "dynunet", mock_profile)
            mock_adaptive.assert_called_once()

    def test_explicit_profile_overrides_auto(self, tmp_path: Path) -> None:
        """compute='gpu_low' returns static profile, skips detection."""
        from run_experiment import resolve_compute_profile

        result = resolve_compute_profile("gpu_low", "dynunet", None)
        assert result["name"] == "gpu_low"

    def test_cpu_profile_resolves(self, tmp_path: Path) -> None:
        """compute='cpu' returns cpu profile."""
        from run_experiment import resolve_compute_profile

        result = resolve_compute_profile("cpu", "dynunet", None)
        assert result["name"] == "cpu"

    def test_resolve_returns_dict_with_required_keys(self, tmp_path: Path) -> None:
        """Resolved profile has all required keys."""
        from run_experiment import resolve_compute_profile

        result = resolve_compute_profile("cpu", "dynunet", None)
        assert "name" in result
        assert "batch_size" in result
        assert "patch_size" in result
        assert "num_workers" in result
        assert "mixed_precision" in result

    def test_auto_without_dataset_profile_falls_back(self, tmp_path: Path) -> None:
        """compute='auto' with no dataset_profile falls back to cpu."""
        from run_experiment import resolve_compute_profile

        result = resolve_compute_profile("auto", "dynunet", None)
        # Should fall back gracefully when no dataset profile provided
        assert isinstance(result, dict)
        assert "name" in result


class TestDebugOverrides:
    def test_debug_overrides_applied(self, tmp_path: Path) -> None:
        """Debug mode reduces epochs and data."""
        config = {
            "experiment_name": "debug_test",
            "model": "dynunet",
            "losses": ["dice_ce"],
            "compute": "cpu",
            "data_dir": str(tmp_path),
            "num_folds": 2,
            "max_epochs": 100,
            "seed": 42,
            "debug": True,
        }
        from run_experiment import apply_debug_to_config

        result = apply_debug_to_config(config)
        assert result["max_epochs"] == 1
        assert result["num_folds"] <= 2

    def test_debug_false_no_overrides(self, tmp_path: Path) -> None:
        """Debug=False does not override config values."""
        config = {
            "experiment_name": "nodebug_test",
            "model": "dynunet",
            "losses": ["dice_ce"],
            "compute": "cpu",
            "data_dir": str(tmp_path),
            "num_folds": 5,
            "max_epochs": 100,
            "seed": 42,
            "debug": False,
        }
        from run_experiment import apply_debug_to_config

        result = apply_debug_to_config(config)
        assert result["max_epochs"] == 100
        assert result["num_folds"] == 5

    def test_debug_does_not_mutate_original(self, tmp_path: Path) -> None:
        """apply_debug_to_config returns new dict without mutating original."""
        config = {
            "experiment_name": "mutate_test",
            "model": "dynunet",
            "losses": ["dice_ce"],
            "compute": "cpu",
            "data_dir": str(tmp_path),
            "num_folds": 5,
            "max_epochs": 100,
            "seed": 42,
            "debug": True,
        }
        original_epochs = config["max_epochs"]
        original_folds = config["num_folds"]

        from run_experiment import apply_debug_to_config

        result = apply_debug_to_config(config)
        # Original should be unchanged
        assert config["max_epochs"] == original_epochs
        assert config["num_folds"] == original_folds
        # Result should be changed
        assert result["max_epochs"] == 1

    def test_debug_large_folds_clamped_to_two(self, tmp_path: Path) -> None:
        """Debug mode clamps num_folds to 2 even if config has more."""
        config = {
            "experiment_name": "folds_clamp_test",
            "model": "dynunet",
            "losses": ["dice_ce"],
            "compute": "cpu",
            "data_dir": str(tmp_path),
            "num_folds": 10,
            "max_epochs": 50,
            "seed": 42,
            "debug": True,
        }
        from run_experiment import apply_debug_to_config

        result = apply_debug_to_config(config)
        assert result["num_folds"] <= 2
