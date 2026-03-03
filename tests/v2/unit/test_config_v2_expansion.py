from __future__ import annotations

import pytest

from minivess.pipeline.loss_functions import build_loss_function


@pytest.fixture()
def experiment_config():
    """Load the dynunet_losses experiment config via Hydra compose."""
    from minivess.config.compose import compose_experiment_config

    return compose_experiment_config(experiment_name="dynunet_losses")


class TestConfigV2Expansion:
    """Tests for the v2 experiment config with expanded metrics."""

    def test_yaml_loads_6_tracked_metrics(self, experiment_config):
        """Config must have 6 tracked metrics."""
        metrics = experiment_config["checkpoint"]["tracked_metrics"]
        assert len(metrics) == 6

    def test_primary_metric_is_compound(self, experiment_config):
        """Primary metric must be val_compound_nsd_cldice (updated from masd-based compound)."""
        assert (
            experiment_config["checkpoint"]["primary_metric"]
            == "val_compound_nsd_cldice"
        )

    def test_cbdice_cldice_in_loss_list(self, experiment_config):
        """cbdice_cldice must be in the loss list."""
        assert "cbdice_cldice" in experiment_config["losses"]

    def test_warp_not_in_loss_list(self, experiment_config):
        """warp must NOT be in the loss list (replaced by cbdice_cldice)."""
        assert "warp" not in experiment_config["losses"]

    def test_build_loss_for_all_configured_losses(self, experiment_config):
        """All configured losses must resolve in the factory."""
        for loss_name in experiment_config["losses"]:
            loss_fn = build_loss_function(loss_name)
            assert loss_fn is not None, f"Factory returned None for {loss_name}"

    def test_tracked_metric_names(self, experiment_config):
        """Verify exact set of tracked metric names."""
        names = {m["name"] for m in experiment_config["checkpoint"]["tracked_metrics"]}
        expected = {
            "val_loss",
            "val_dice",
            "val_f1_foreground",
            "val_cldice",
            "val_masd",
            "val_compound_nsd_cldice",
        }
        assert names == expected

    def test_four_losses_configured(self, experiment_config):
        """Exactly 4 losses must be configured."""
        assert len(experiment_config["losses"]) == 4
