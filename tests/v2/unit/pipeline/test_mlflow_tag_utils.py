"""Tests for MLflow factorial tag utilities — Plan Task 1.1.

Verifies:
- set_factorial_tags() applies correct tags to MLflow runs
- Tag mapping from experiment config name to factorial metadata
- Training flow logs with_aux_calib tag on new runs

Pure unit tests — no Docker, no Prefect, no DagsHub.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient


@pytest.fixture
def mlflow_tmp(tmp_path: Path) -> tuple[str, MlflowClient]:
    """Create a temporary MLflow tracking directory and return (uri, client)."""
    tracking_uri = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    return tracking_uri, client


class TestSetFactorialTags:
    """Tests for set_factorial_tags() — retroactive tag application."""

    def test_applies_with_aux_calib_true(
        self, mlflow_tmp: tuple[str, MlflowClient]
    ) -> None:
        """Runs with 'auxcalib' in config name get with_aux_calib=true."""
        from minivess.pipeline.mlflow_tag_utils import set_factorial_tags

        _, client = mlflow_tmp
        exp_id = client.create_experiment("test_exp")
        run = client.create_run(exp_id)
        run_id = run.info.run_id

        config_map = {
            run_id: {
                "config_name": "local_dynunet_dice_ce_auxcalib",
                "loss_function": "dice_ce",
                "model_family": "dynunet",
            }
        }

        set_factorial_tags(client, config_map)

        tags = client.get_run(run_id).data.tags
        assert tags["with_aux_calib"] == "true"
        assert tags["loss_function"] == "dice_ce"
        assert tags["model_family"] == "dynunet"

    def test_applies_with_aux_calib_false(
        self, mlflow_tmp: tuple[str, MlflowClient]
    ) -> None:
        """Runs WITHOUT 'auxcalib' in config name get with_aux_calib=false."""
        from minivess.pipeline.mlflow_tag_utils import set_factorial_tags

        _, client = mlflow_tmp
        exp_id = client.create_experiment("test_exp2")
        run = client.create_run(exp_id)
        run_id = run.info.run_id

        config_map = {
            run_id: {
                "config_name": "local_dynunet_cbdice_cldice",
                "loss_function": "cbdice_cldice",
                "model_family": "dynunet",
            }
        }

        set_factorial_tags(client, config_map)

        tags = client.get_run(run_id).data.tags
        assert tags["with_aux_calib"] == "false"
        assert tags["loss_function"] == "cbdice_cldice"

    def test_applies_layer_bc_defaults(
        self, mlflow_tmp: tuple[str, MlflowClient]
    ) -> None:
        """Layer B+C tags get correct defaults when not specified."""
        from minivess.pipeline.mlflow_tag_utils import set_factorial_tags

        _, client = mlflow_tmp
        exp_id = client.create_experiment("test_exp3")
        run = client.create_run(exp_id)
        run_id = run.info.run_id

        config_map = {
            run_id: {
                "config_name": "local_dynunet_dice_ce",
                "loss_function": "dice_ce",
                "model_family": "dynunet",
            }
        }

        set_factorial_tags(client, config_map)

        tags = client.get_run(run_id).data.tags
        assert tags["post_training_method"] == "none"
        assert tags["recalibration"] == "none"
        assert tags["ensemble_strategy"] == "none"
        assert tags["is_zero_shot"] == "false"

    def test_multiple_runs_tagged(
        self, mlflow_tmp: tuple[str, MlflowClient]
    ) -> None:
        """All runs in config_map get tagged."""
        from minivess.pipeline.mlflow_tag_utils import set_factorial_tags

        _, client = mlflow_tmp
        exp_id = client.create_experiment("test_exp_multi")

        run_ids = []
        for _ in range(4):
            run = client.create_run(exp_id)
            run_ids.append(run.info.run_id)

        config_map = {
            run_ids[0]: {
                "config_name": "local_dynunet_dice_ce",
                "loss_function": "dice_ce",
                "model_family": "dynunet",
            },
            run_ids[1]: {
                "config_name": "local_dynunet_dice_ce_auxcalib",
                "loss_function": "dice_ce",
                "model_family": "dynunet",
            },
            run_ids[2]: {
                "config_name": "local_dynunet_cbdice_cldice",
                "loss_function": "cbdice_cldice",
                "model_family": "dynunet",
            },
            run_ids[3]: {
                "config_name": "local_dynunet_cbdice_cldice_auxcalib",
                "loss_function": "cbdice_cldice",
                "model_family": "dynunet",
            },
        }

        set_factorial_tags(client, config_map)

        # Verify each run has correct tags
        tags_0 = client.get_run(run_ids[0]).data.tags
        tags_1 = client.get_run(run_ids[1]).data.tags
        tags_2 = client.get_run(run_ids[2]).data.tags
        tags_3 = client.get_run(run_ids[3]).data.tags

        assert tags_0["with_aux_calib"] == "false"
        assert tags_1["with_aux_calib"] == "true"
        assert tags_2["with_aux_calib"] == "false"
        assert tags_3["with_aux_calib"] == "true"

        assert tags_0["loss_function"] == "dice_ce"
        assert tags_2["loss_function"] == "cbdice_cldice"

    def test_custom_layer_bc_values(
        self, mlflow_tmp: tuple[str, MlflowClient]
    ) -> None:
        """Layer B+C tags can be overridden from config_map."""
        from minivess.pipeline.mlflow_tag_utils import set_factorial_tags

        _, client = mlflow_tmp
        exp_id = client.create_experiment("test_exp_custom")
        run = client.create_run(exp_id)
        run_id = run.info.run_id

        config_map = {
            run_id: {
                "config_name": "local_dynunet_dice_ce",
                "loss_function": "dice_ce",
                "model_family": "dynunet",
                "post_training_method": "swag",
                "recalibration": "temperature_scaling",
                "ensemble_strategy": "per_loss_single_best",
                "is_zero_shot": "true",
            }
        }

        set_factorial_tags(client, config_map)

        tags = client.get_run(run_id).data.tags
        assert tags["post_training_method"] == "swag"
        assert tags["recalibration"] == "temperature_scaling"
        assert tags["ensemble_strategy"] == "per_loss_single_best"
        assert tags["is_zero_shot"] == "true"


class TestDiscoveryAfterTagging:
    """Test that discover_source_runs_from_api sees tagged runs correctly."""

    def test_discovery_finds_four_conditions_after_tagging(
        self, mlflow_tmp: tuple[str, MlflowClient]
    ) -> None:
        """After tagging, discovery should find 4 distinct conditions."""
        from minivess.pipeline.mlflow_tag_utils import set_factorial_tags

        tracking_uri, client = mlflow_tmp
        exp_name = "local_dynunet_mechanics_training"
        exp_id = client.create_experiment(exp_name)

        # Create 8 runs: 4 conditions × 2 folds
        configs = [
            ("local_dynunet_dice_ce", "dice_ce", False),
            ("local_dynunet_dice_ce_auxcalib", "dice_ce", True),
            ("local_dynunet_cbdice_cldice", "cbdice_cldice", False),
            ("local_dynunet_cbdice_cldice_auxcalib", "cbdice_cldice", True),
        ]

        config_map = {}
        for config_name, loss, _aux in configs:
            for fold_id in range(2):
                run = client.create_run(exp_id)
                run_id = run.info.run_id
                # Set fold_id tag (training flow already does this)
                client.set_tag(run_id, "fold_id", str(fold_id))
                # Terminate as FINISHED
                client.set_terminated(run_id, status="FINISHED")

                config_map[run_id] = {
                    "config_name": config_name,
                    "loss_function": loss,
                    "model_family": "dynunet",
                }

        set_factorial_tags(client, config_map)

        # Now test discovery
        from minivess.pipeline.biostatistics_discovery import (
            discover_source_runs_from_api,
        )

        manifest = discover_source_runs_from_api(
            experiment_names=[exp_name],
            tracking_uri=tracking_uri,
        )

        assert len(manifest.runs) == 8

        # Check 4 distinct conditions
        conditions = {
            (r.loss_function, r.with_aux_calib) for r in manifest.runs
        }
        assert len(conditions) == 4
        assert ("dice_ce", False) in conditions
        assert ("dice_ce", True) in conditions
        assert ("cbdice_cldice", False) in conditions
        assert ("cbdice_cldice", True) in conditions
