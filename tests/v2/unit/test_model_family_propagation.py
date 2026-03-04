"""Tests for model_family propagation from YAML config through run_experiment.py.

Covers T2: _build_train_argv() helper that centralises argv construction
and includes --model-family so both _run_losses_mode() and
_run_single_condition() pass the correct model family to train_monitored.py.

Closes #257
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Add scripts directory to path so run_experiment can be imported directly
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid losses-mode config, optionally overridden."""
    base: dict[str, Any] = {
        "experiment_name": "test_exp",
        "model": "dynunet",
        "losses": ["dice_ce"],
        "compute": "cpu",
        "data_dir": "data/raw",
        "num_folds": 3,
        "seed": 42,
    }
    base.update(overrides)
    return base


def _minimal_profile() -> dict[str, Any]:
    """Return a minimal resolved compute profile."""
    return {
        "name": "cpu",
        "batch_size": 1,
        "patch_size": (64, 64, 16),
        "num_workers": 0,
        "mixed_precision": False,
        "gradient_accumulation_steps": 1,
        "cache_rate": 0.0,
    }


# ---------------------------------------------------------------------------
# TestBuildTrainArgvExists — _build_train_argv() is importable
# ---------------------------------------------------------------------------


class TestBuildTrainArgvExists:
    def test_function_is_importable(self) -> None:
        """_build_train_argv() must be importable from run_experiment."""
        from run_experiment import _build_train_argv  # noqa: F401

        assert callable(_build_train_argv)

    def test_returns_list(self) -> None:
        """_build_train_argv() must return a list."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert isinstance(result, list)

    def test_all_elements_are_strings(self) -> None:
        """Every element of the returned argv must be a str."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        for item in result:
            assert isinstance(item, str), f"Non-str element: {item!r}"


# ---------------------------------------------------------------------------
# TestModelFamilyInArgv — --model-family flag is always present
# ---------------------------------------------------------------------------


class TestModelFamilyInArgv:
    def test_model_family_flag_present(self) -> None:
        """--model-family flag must be in argv output."""
        from run_experiment import _build_train_argv

        config = _minimal_config(model_family="dynunet")
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--model-family" in result

    def test_model_family_value_dynunet(self) -> None:
        """--model-family value must be 'dynunet' when config says dynunet."""
        from run_experiment import _build_train_argv

        config = _minimal_config(model_family="dynunet")
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        idx = result.index("--model-family")
        assert result[idx + 1] == "dynunet"

    def test_model_family_value_sam3_vanilla(self) -> None:
        """--model-family value must propagate 'sam3_vanilla' correctly."""
        from run_experiment import _build_train_argv

        config = _minimal_config(model_family="sam3_vanilla")
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        idx = result.index("--model-family")
        assert result[idx + 1] == "sam3_vanilla"

    def test_model_family_value_sam3_topolora(self) -> None:
        """--model-family value must propagate 'sam3_topolora' correctly."""
        from run_experiment import _build_train_argv

        config = _minimal_config(model_family="sam3_topolora")
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        idx = result.index("--model-family")
        assert result[idx + 1] == "sam3_topolora"

    def test_model_family_value_sam3_hybrid(self) -> None:
        """--model-family value must propagate 'sam3_hybrid' correctly."""
        from run_experiment import _build_train_argv

        config = _minimal_config(model_family="sam3_hybrid")
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        idx = result.index("--model-family")
        assert result[idx + 1] == "sam3_hybrid"


# ---------------------------------------------------------------------------
# TestDefaultModelFamily — default is "dynunet" when key absent from config
# ---------------------------------------------------------------------------


class TestDefaultModelFamily:
    def test_default_is_dynunet_when_key_absent(self) -> None:
        """When 'model_family' is absent from config, default must be 'dynunet'."""
        from run_experiment import _build_train_argv

        config = _minimal_config()  # no model_family key
        assert "model_family" not in config
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        idx = result.index("--model-family")
        assert result[idx + 1] == "dynunet"

    def test_default_dynunet_when_none(self) -> None:
        """When 'model_family' is None in config, default must be 'dynunet'."""
        from run_experiment import _build_train_argv

        config = _minimal_config(model_family=None)
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        idx = result.index("--model-family")
        assert result[idx + 1] == "dynunet"


# ---------------------------------------------------------------------------
# TestStandardFlagsPresent — core flags always present in argv
# ---------------------------------------------------------------------------


class TestStandardFlagsPresent:
    def test_compute_flag_present(self) -> None:
        """--compute flag must be in argv."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--compute" in result

    def test_loss_flag_present(self) -> None:
        """--loss flag must be in argv."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--loss" in result

    def test_loss_value_matches_argument(self) -> None:
        """--loss value must match the loss_name argument passed to helper."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "cbdice_cldice", "test_exp")
        idx = result.index("--loss")
        assert result[idx + 1] == "cbdice_cldice"

    def test_data_dir_flag_present(self) -> None:
        """--data-dir flag must be in argv."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--data-dir" in result

    def test_num_folds_flag_present(self) -> None:
        """--num-folds flag must be in argv."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--num-folds" in result

    def test_seed_flag_present(self) -> None:
        """--seed flag must be in argv."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--seed" in result

    def test_experiment_name_flag_present(self) -> None:
        """--experiment-name flag must be in argv."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--experiment-name" in result

    def test_experiment_name_value_matches_argument(self) -> None:
        """--experiment-name value must match the experiment_name argument."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "my_experiment")
        idx = result.index("--experiment-name")
        assert result[idx + 1] == "my_experiment"

    def test_splits_file_flag_present(self) -> None:
        """--splits-file flag must be in argv."""
        from run_experiment import _build_train_argv

        config = _minimal_config()
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--splits-file" in result

    def test_debug_flag_included_when_config_debug_true(self) -> None:
        """--debug flag must be in argv when config['debug'] is True."""
        from run_experiment import _build_train_argv

        config = _minimal_config(debug=True)
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--debug" in result

    def test_debug_flag_absent_when_config_debug_false(self) -> None:
        """--debug flag must NOT be in argv when config['debug'] is False."""
        from run_experiment import _build_train_argv

        config = _minimal_config(debug=False)
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--debug" not in result

    def test_max_epochs_flag_included_when_set(self) -> None:
        """--max-epochs flag must be in argv when config['max_epochs'] is set."""
        from run_experiment import _build_train_argv

        config = _minimal_config(max_epochs=10)
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--max-epochs" in result
        idx = result.index("--max-epochs")
        assert result[idx + 1] == "10"

    def test_max_epochs_absent_when_not_set(self) -> None:
        """--max-epochs flag must NOT be in argv when config['max_epochs'] is None."""
        from run_experiment import _build_train_argv

        config = _minimal_config()  # no max_epochs key
        profile = _minimal_profile()
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        assert "--max-epochs" not in result


# ---------------------------------------------------------------------------
# TestInvalidModelFamily — ValueError for invalid model_family
# ---------------------------------------------------------------------------


class TestInvalidModelFamily:
    def test_invalid_model_family_raises_value_error(self) -> None:
        """_build_train_argv() must raise ValueError for unknown model_family."""
        from run_experiment import _build_train_argv

        config = _minimal_config(model_family="totally_invalid_family_xyz")
        profile = _minimal_profile()
        with pytest.raises(ValueError, match="totally_invalid_family_xyz"):
            _build_train_argv(config, profile, "dice_ce", "test_exp")

    def test_invalid_model_family_typo_raises(self) -> None:
        """A common typo ('dynunet2') must raise ValueError."""
        from run_experiment import _build_train_argv

        config = _minimal_config(model_family="dynunet2")
        profile = _minimal_profile()
        with pytest.raises(ValueError):
            _build_train_argv(config, profile, "dice_ce", "test_exp")

    def test_valid_model_families_do_not_raise(self) -> None:
        """All valid ModelFamily values must not raise ValueError."""
        from run_experiment import _build_train_argv

        valid_families = [
            "dynunet",
            "vesselfm",
            "comma_mamba",
            "sam3_lora",
            "sam3_vanilla",
            "sam3_topolora",
            "sam3_hybrid",
            "multitask_dynunet",
            "custom",
        ]
        profile = _minimal_profile()
        for family in valid_families:
            config = _minimal_config(model_family=family)
            # Should not raise
            result = _build_train_argv(config, profile, "dice_ce", "test_exp")
            idx = result.index("--model-family")
            assert result[idx + 1] == family


# ---------------------------------------------------------------------------
# TestAutoComputeProfile — --compute handling with "auto" mode
# ---------------------------------------------------------------------------


class TestAutoComputeProfile:
    def test_auto_compute_falls_back_to_cpu_in_argv(self) -> None:
        """When compute='auto', argv --compute must be 'cpu' (profile name fallback)."""
        from run_experiment import _build_train_argv

        config = _minimal_config(compute="auto")
        # Profile with name "cpu" simulates the fallback resolved by resolve_compute_profile
        profile = _minimal_profile()  # name="cpu"
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        idx = result.index("--compute")
        # When compute is "auto", the profile name should be passed as "cpu" fallback
        assert result[idx + 1] == "cpu"

    def test_explicit_gpu_low_compute_in_argv(self) -> None:
        """When compute='gpu_low', argv --compute must be 'gpu_low'."""
        from run_experiment import _build_train_argv

        config = _minimal_config(compute="gpu_low")
        profile = {**_minimal_profile(), "name": "gpu_low"}
        result = _build_train_argv(config, profile, "dice_ce", "test_exp")
        idx = result.index("--compute")
        assert result[idx + 1] == "gpu_low"
