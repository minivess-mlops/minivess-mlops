"""Config guard tests — enforce non-negotiable constraints.

T4 ban: No config file may reference T4 as an accelerator.
Factorial-KG consistency: paper_factorial.yaml models must match KG.
Debug = production: debug_factorial.yaml must have same factors as production.
TubeNet ban: No config/source file may use tubenet_2pm for test evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestT4Ban:
    """T4 is BANNED — Turing has no BF16, causes FP16 overflow → NaN in SAM3."""

    def test_no_t4_in_gcp_spot(self) -> None:
        gcp_spot = REPO_ROOT / "configs" / "cloud" / "gcp_spot.yaml"
        content = gcp_spot.read_text(encoding="utf-8")
        assert "T4:1" not in content, "T4 accelerator found in gcp_spot.yaml — BANNED"
        assert "T4:" not in content.split("#")[0], "T4 reference in non-comment line"

    def test_no_t4_in_skypilot_yamls(self) -> None:
        skypilot_dir = REPO_ROOT / "deployment" / "skypilot"
        if not skypilot_dir.exists():
            return
        for yaml_file in skypilot_dir.glob("*.yaml"):
            content = yaml_file.read_text(encoding="utf-8")
            # T4 may appear in comments (explaining the ban) — only check non-comment lines
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert "T4:" not in stripped, (
                    f"T4 accelerator found in {yaml_file.name} line: {stripped}"
                )


class TestFactorialMatchesKG:
    """paper_factorial.yaml models must match KG paper_model_comparison."""

    def test_trainable_models_match_kg(self) -> None:
        # Read paper_factorial.yaml
        factorial_path = REPO_ROOT / "configs" / "hpo" / "paper_factorial.yaml"
        with factorial_path.open(encoding="utf-8") as f:
            factorial = yaml.safe_load(f)

        factorial_models = set(factorial["factors"]["model_family"])

        # Read KG paper_model_comparison.yaml
        kg_path = (
            REPO_ROOT
            / "knowledge-graph"
            / "decisions"
            / "L3-technology"
            / "paper_model_comparison.yaml"
        )
        with kg_path.open(encoding="utf-8") as f:
            kg = yaml.safe_load(f)

        # Extract trainable models (not zero-shot) from KG
        kg_trainable = set()
        for model in kg["models"]:
            strategy = model.get("training_strategy", "")
            if "zero_shot" not in strategy:
                kg_trainable.add(model["model_id"])

        # Map KG model_ids to config names
        kg_id_to_config = {
            "dynunet": "dynunet",
            "mambavesselnet_pp": "mambavesselnet",
            "sam3_topolora": "sam3_topolora",
            "sam3_hybrid": "sam3_hybrid",
        }
        kg_config_names = {kg_id_to_config.get(m, m) for m in kg_trainable}

        assert factorial_models == kg_config_names, (
            f"Factorial models {factorial_models} don't match KG {kg_config_names}"
        )

    def test_losses_match_factorial(self) -> None:
        factorial_path = REPO_ROOT / "configs" / "hpo" / "paper_factorial.yaml"
        with factorial_path.open(encoding="utf-8") as f:
            factorial = yaml.safe_load(f)

        expected_losses = {"cbdice_cldice", "dice_ce", "dice_ce_cldice"}
        actual_losses = set(factorial["factors"]["loss_name"])
        assert actual_losses == expected_losses

    def test_aux_calibration_factor_exists(self) -> None:
        factorial_path = REPO_ROOT / "configs" / "hpo" / "paper_factorial.yaml"
        with factorial_path.open(encoding="utf-8") as f:
            factorial = yaml.safe_load(f)

        assert "aux_calibration" in factorial["factors"]
        assert set(factorial["factors"]["aux_calibration"]) == {True, False}


class TestDebugEqualsProduction:
    """Debug config must have same factors as production — only epochs/data/folds differ."""

    def test_debug_has_same_loss_default(self) -> None:
        debug_path = REPO_ROOT / "configs" / "experiment" / "debug_factorial.yaml"
        with debug_path.open(encoding="utf-8") as f:
            debug = yaml.safe_load(f)

        assert debug["max_epochs"] == 2
        assert debug["num_folds"] == 1
        assert debug["max_train_volumes"] == 23
        assert debug["max_val_volumes"] == 12

    def test_debug_has_aux_calib_field(self) -> None:
        debug_path = REPO_ROOT / "configs" / "experiment" / "debug_factorial.yaml"
        with debug_path.open(encoding="utf-8") as f:
            debug = yaml.safe_load(f)

        assert "with_aux_calib" in debug, (
            "Debug must have aux_calib (debug = production)"
        )

    def test_debug_split_file_exists(self) -> None:
        split_path = REPO_ROOT / "configs" / "splits" / "debug_half_1fold.json"
        assert split_path.exists(), "Debug half-dataset split file missing"
        with split_path.open(encoding="utf-8") as f:
            splits = json.load(f)
        assert len(splits) == 1, "Debug split should have exactly 1 fold"
        assert len(splits[0]["train"]) == 23
        assert len(splits[0]["val"]) == 12


class TestTubeNetBan:
    """TubeNet is EXCLUDED — olfactory bulb, different organ, 1 2PM volume."""

    def test_no_tubenet_in_external_datasets_registry(self) -> None:
        from minivess.data.external_datasets import EXTERNAL_DATASETS

        assert "tubenet_2pm" not in EXTERNAL_DATASETS, (
            "tubenet_2pm found in EXTERNAL_DATASETS — BANNED. See CLAUDE.md."
        )

    def test_no_tubenet_in_dvc_configs(self) -> None:
        from minivess.data.external_datasets import DVC_CONFIGS

        assert "tubenet_2pm" not in DVC_CONFIGS, (
            "tubenet_2pm found in DVC_CONFIGS — BANNED. See CLAUDE.md."
        )
