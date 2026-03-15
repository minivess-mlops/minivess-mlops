"""Tests for smoke_vesselfm experiment config composition (T0.6).

Validates:
- compose_experiment_config("smoke_vesselfm") succeeds
- Resolved config has correct model_family, epochs, folds
- Debug flag is set
- Addresses failure hypothesis H6 (Hydra config composition fails)
"""

from __future__ import annotations

import pytest


@pytest.mark.model_loading
class TestSmokeVesselfmConfig:
    """Verify smoke_vesselfm Hydra experiment config composes correctly."""

    def test_smoke_vesselfm_composes_without_error(self) -> None:
        """compose_experiment_config('smoke_vesselfm') must not raise."""
        from minivess.config.compose import compose_experiment_config

        config = compose_experiment_config("smoke_vesselfm")
        assert isinstance(config, dict)

    def test_smoke_vesselfm_has_correct_model_family(self) -> None:
        """Resolved config must specify model_family=vesselfm."""
        from minivess.config.compose import compose_experiment_config

        config = compose_experiment_config("smoke_vesselfm")
        assert config.get("model_family") == "vesselfm"

    def test_smoke_vesselfm_has_correct_epochs_and_folds(self) -> None:
        """Resolved config must have max_epochs=2 and num_folds=1."""
        from minivess.config.compose import compose_experiment_config

        config = compose_experiment_config("smoke_vesselfm")
        assert config.get("max_epochs") == 2
        assert config.get("num_folds") == 1

    def test_smoke_vesselfm_has_debug_flag(self) -> None:
        """Resolved config must have debug=true."""
        from minivess.config.compose import compose_experiment_config

        config = compose_experiment_config("smoke_vesselfm")
        assert config.get("debug") is True

    def test_smoke_vesselfm_has_splits_file(self) -> None:
        """Resolved config must reference the smoke test splits file."""
        from minivess.config.compose import compose_experiment_config

        config = compose_experiment_config("smoke_vesselfm")
        splits_file = config.get("splits_file", "")
        assert "smoke_test_1fold_4vol" in splits_file
