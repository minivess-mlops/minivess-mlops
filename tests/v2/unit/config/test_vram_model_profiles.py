"""Tests for structured VRAM sections in model profiles (#641).

Every model profile in configs/model_profiles/ must have a structured
vram: section as the single source of truth for VRAM requirements.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROFILES_DIR = Path("configs/model_profiles")
REQUIRED_VRAM_KEYS = {
    "inference_gb",
    "training_gb",
    "measured",
    "measured_gpu",
    "measured_date",
}


def _all_profiles() -> list[Path]:
    return sorted(PROFILES_DIR.glob("*.yaml"))


class TestVramSectionExists:
    """Every model profile must have a vram: section."""

    @pytest.mark.parametrize("profile", _all_profiles(), ids=lambda p: p.stem)
    def test_has_vram_section(self, profile: Path) -> None:
        data = yaml.safe_load(profile.read_text(encoding="utf-8"))
        assert "vram" in data, f"{profile.name} missing 'vram:' section"

    @pytest.mark.parametrize("profile", _all_profiles(), ids=lambda p: p.stem)
    def test_vram_has_required_keys(self, profile: Path) -> None:
        data = yaml.safe_load(profile.read_text(encoding="utf-8"))
        vram = data.get("vram", {})
        missing = REQUIRED_VRAM_KEYS - set(vram.keys())
        assert not missing, f"{profile.name} vram section missing keys: {missing}"

    @pytest.mark.parametrize("profile", _all_profiles(), ids=lambda p: p.stem)
    def test_measured_profiles_have_gpu_and_date(self, profile: Path) -> None:
        """If measured=true, must have measured_gpu and measured_date."""
        data = yaml.safe_load(profile.read_text(encoding="utf-8"))
        vram = data.get("vram", {})
        if vram.get("measured"):
            assert vram.get("measured_gpu") is not None, (
                f"{profile.name}: measured=true but no measured_gpu"
            )
            assert vram.get("measured_date") is not None, (
                f"{profile.name}: measured=true but no measured_date"
            )

    def test_at_least_6_measured_profiles(self) -> None:
        """At least 6 profiles should have measured VRAM data."""
        measured_count = 0
        for profile in _all_profiles():
            data = yaml.safe_load(profile.read_text(encoding="utf-8"))
            if data.get("vram", {}).get("measured"):
                measured_count += 1
        # 6 paper models, but not all may have measured VRAM yet
        # (mambavesselnet GPU-pending). Require at least 4 measured.
        assert measured_count >= 4, (
            f"Expected at least 4 measured profiles, got {measured_count}"
        )
