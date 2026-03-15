"""T07 — RED phase: model profile YAML + experiment config tests.

Tests verify that 4 config files exist and have correct content:
- configs/model_profiles/mambavesselnet.yaml
- configs/experiment/debug_mambavesselnet.yaml
- configs/experiment/smoke_mambavesselnet.yaml
- configs/experiment/smoke_mambavesselnet_cloud.yaml
"""

from __future__ import annotations

from pathlib import Path


class TestModelProfileExists:
    """T07: configs/model_profiles/mambavesselnet.yaml must exist and be valid."""

    def test_profile_file_exists(self) -> None:
        assert Path("configs/model_profiles/mambavesselnet.yaml").exists()

    def test_load_model_profile_succeeds(self) -> None:
        from minivess.config.model_profiles import load_model_profile

        profile = load_model_profile("mambavesselnet")
        assert profile is not None

    def test_profile_divisor_is_16(self) -> None:
        from minivess.config.model_profiles import load_model_profile

        profile = load_model_profile("mambavesselnet")
        assert profile.divisor == 16

    def test_profile_has_vram_field(self) -> None:
        from minivess.config.model_profiles import load_model_profile

        profile = load_model_profile("mambavesselnet")
        assert profile.vram is not None

    def test_profile_default_patch_xy(self) -> None:
        from minivess.config.model_profiles import load_model_profile

        profile = load_model_profile("mambavesselnet")
        assert profile.default_patch_xy == 64


class TestDebugConfigExists:
    """T07: configs/experiment/debug_mambavesselnet.yaml must exist and compose."""

    def test_debug_config_file_exists(self) -> None:
        assert Path("configs/experiment/debug_mambavesselnet.yaml").exists()

    def test_debug_config_has_mixed_precision_val_false(self) -> None:
        """AMP val must be disabled per D04 (MONAI #4243)."""
        import yaml

        cfg_path = Path("configs/experiment/debug_mambavesselnet.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert (
            cfg.get("mixed_precision_val") is False
            or cfg.get("mixed_precision") is False
        )

    def test_debug_config_model_family(self) -> None:
        import yaml

        cfg_path = Path("configs/experiment/debug_mambavesselnet.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg.get("model_family") == "mambavesselnet"

    def test_debug_config_max_epochs_1(self) -> None:
        import yaml

        cfg_path = Path("configs/experiment/debug_mambavesselnet.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg.get("max_epochs") == 1


class TestSmokeConfigExists:
    """T07: configs/experiment/smoke_mambavesselnet.yaml must exist."""

    def test_smoke_config_file_exists(self) -> None:
        assert Path("configs/experiment/smoke_mambavesselnet.yaml").exists()

    def test_smoke_config_model_family(self) -> None:
        import yaml

        cfg_path = Path("configs/experiment/smoke_mambavesselnet.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg.get("model_family") == "mambavesselnet"

    def test_smoke_config_max_epochs_at_least_2(self) -> None:
        import yaml

        cfg_path = Path("configs/experiment/smoke_mambavesselnet.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg.get("max_epochs", 0) >= 2

    def test_smoke_config_debug_true(self) -> None:
        import yaml

        cfg_path = Path("configs/experiment/smoke_mambavesselnet.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg.get("debug") is True


class TestSmokeCloudConfigExists:
    """T07: configs/experiment/smoke_mambavesselnet_cloud.yaml must exist."""

    def test_smoke_cloud_config_file_exists(self) -> None:
        assert Path("configs/experiment/smoke_mambavesselnet_cloud.yaml").exists()

    def test_smoke_cloud_config_model_family(self) -> None:
        import yaml

        cfg_path = Path("configs/experiment/smoke_mambavesselnet_cloud.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg.get("model_family") == "mambavesselnet"

    def test_smoke_cloud_val_interval_1(self) -> None:
        """Cloud config must enable validation (val_interval=1)."""
        import yaml

        cfg_path = Path("configs/experiment/smoke_mambavesselnet_cloud.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg.get("val_interval") == 1
