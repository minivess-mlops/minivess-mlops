"""Verify every model has a corresponding smoke test experiment config.

Ensures all ModelAdapter implementations in the registry can be tested
on RunPod via SkyPilot smoke tests. Each model profile in
configs/model_profiles/ must have a matching configs/experiment/smoke_{name}.yaml.

Models that require special hardware (>24 GB VRAM) or are not yet implemented
may be excluded via the EXCLUDED_MODELS set.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_PROFILES_DIR = Path("configs/model_profiles")
_EXPERIMENTS_DIR = Path("configs/experiment")

# Models excluded from smoke test requirement:
# - example_custom: template, not a real model
# - sam3_topolora: requires ≥16 GB training VRAM, no separate smoke test
#   (tested via sam3_vanilla with LoRA wrapper)
# - comma_mamba / comma: duplicate profiles (comma is alias)
# - ulike_mamba / mamba: duplicate profiles (mamba is alias)
_EXCLUDED_PROFILES = {
    "example_custom",
    "sam3_topolora",
    "comma",  # alias for comma_mamba
    "mamba",  # alias for ulike_mamba
}


def _model_profile_names() -> list[str]:
    """Return model names from configs/model_profiles/, minus exclusions."""
    names = []
    for p in sorted(_PROFILES_DIR.glob("*.yaml")):
        name = p.stem
        if name not in _EXCLUDED_PROFILES:
            names.append(name)
    return names


class TestSmokeConfigsComplete:
    """Ensure every model profile has a matching smoke experiment config."""

    def test_every_model_has_smoke_config(self) -> None:
        """Each model_profiles/*.yaml must have a matching smoke_*.yaml."""
        missing: list[str] = []
        for name in _model_profile_names():
            smoke_path = _EXPERIMENTS_DIR / f"smoke_{name}.yaml"
            if not smoke_path.exists():
                missing.append(name)

        assert not missing, (
            f"Missing smoke test configs for models: {missing}. "
            f"Create configs/experiment/smoke_{{name}}.yaml for each."
        )

    def test_smoke_configs_reference_valid_models(self) -> None:
        """Each smoke_*.yaml must have a model field matching a profile."""
        profile_names = {p.stem for p in _PROFILES_DIR.glob("*.yaml")}
        invalid: list[str] = []

        for smoke_path in sorted(_EXPERIMENTS_DIR.glob("smoke_*.yaml")):
            config = yaml.safe_load(smoke_path.read_text(encoding="utf-8"))
            model_name = config.get("model", config.get("model_family", ""))
            if model_name and model_name not in profile_names:
                invalid.append(f"{smoke_path.name}: model={model_name}")

        assert not invalid, (
            f"Smoke configs reference unknown models: {invalid}. "
            "Model must exist in configs/model_profiles/."
        )

    def test_smoke_configs_have_required_fields(self) -> None:
        """Each smoke_*.yaml must have experiment_name, max_epochs, num_folds."""
        issues: list[str] = []
        for smoke_path in sorted(_EXPERIMENTS_DIR.glob("smoke_*.yaml")):
            config = yaml.safe_load(smoke_path.read_text(encoding="utf-8"))
            for field in ("experiment_name", "max_epochs", "num_folds"):
                if field not in config:
                    issues.append(f"{smoke_path.name}: missing {field}")
            # Smoke tests must be fast
            if config.get("max_epochs", 100) > 5:
                issues.append(
                    f"{smoke_path.name}: max_epochs={config['max_epochs']} "
                    "(must be ≤5 for smoke test)"
                )

        assert not issues, "Smoke config issues:\n  " + "\n  ".join(issues)
