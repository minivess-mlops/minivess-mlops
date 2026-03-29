"""Tests for config layer boundary enforcement.

Module B of the QA config scanner: Hydra keys must stay in Hydra YAMLs,
Dynaconf keys must stay in TOML, .env.example keys stay in .env.

Three config systems serve different layers:
  Hydra (training): seed, max_epochs, batch_size, model, losses
  Dynaconf (deployment): agent_provider, langfuse_enabled, environment
  .env.example (infrastructure): ports, hostnames, credentials, paths

Cross-contamination (e.g., seed in Dynaconf TOML) creates maintenance nightmares
where changing a parameter requires knowing which of 3 systems to update.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIGS = _REPO_ROOT / "configs"

# Keys that belong ONLY in the Hydra config chain (training/model parameters)
_HYDRA_ONLY_KEYS = frozenset({
    "seed", "max_epochs", "batch_size", "learning_rate", "loss_name",
    "model_family", "model_name", "patch_size", "in_channels", "out_channels",
    "optimizer", "weight_decay", "warmup_epochs", "val_interval",
    "gradient_accumulation_steps", "gradient_clip_val", "mixed_precision",
})

# Keys that belong ONLY in Dynaconf (deployment/integration parameters)
_DYNACONF_ONLY_KEYS = frozenset({
    "agent_provider", "langfuse_enabled", "langfuse_secret_key", "langfuse_public_key",
    "langfuse_host", "braintrust_api_key", "braintrust_enabled",
})


def _flatten_toml_keys(data: dict, prefix: str = "") -> set[str]:
    """Flatten nested TOML dict to dot-separated keys."""
    keys: set[str] = set()
    for k, v in data.items():
        full_key = f"{prefix}.{k}" if prefix else k
        keys.add(k)  # Add leaf key name (for cross-layer check)
        if isinstance(v, dict):
            keys.update(_flatten_toml_keys(v, full_key))
    return keys


def _flatten_yaml_keys(data: dict, prefix: str = "") -> set[str]:
    """Flatten nested YAML dict to leaf keys."""
    keys: set[str] = set()
    if not isinstance(data, dict):
        return keys
    for k, v in data.items():
        keys.add(str(k))
        if isinstance(v, dict):
            keys.update(_flatten_yaml_keys(v, f"{prefix}.{k}" if prefix else str(k)))
    return keys


class TestDynaconfNoBrainHydraKeys:
    """Dynaconf TOML must not define training/model parameters."""

    def test_settings_toml_no_hydra_keys(self) -> None:
        """configs/deployment/settings.toml must not contain Hydra-layer keys."""
        settings_path = _CONFIGS / "deployment" / "settings.toml"
        if not settings_path.exists():
            return

        with settings_path.open("rb") as f:
            data = tomllib.load(f)

        toml_keys = _flatten_toml_keys(data)
        violations = toml_keys & _HYDRA_ONLY_KEYS

        assert not violations, (
            f"Dynaconf settings.toml contains Hydra-layer keys: {sorted(violations)}\n"
            "These belong in configs/training/*.yaml or configs/model/*.yaml, not in Dynaconf.\n"
            "Hydra handles: seed, max_epochs, batch_size, learning_rate, model.\n"
            "Dynaconf handles: deployment environment, agent provider, monitoring toggles."
        )


class TestHydraNoDeploymentKeys:
    """Hydra YAML configs must not define deployment-layer parameters."""

    def test_training_configs_no_dynaconf_keys(self) -> None:
        """configs/training/*.yaml must not contain Dynaconf-layer keys."""
        training_dir = _CONFIGS / "training"
        if not training_dir.exists():
            return

        violations: list[str] = []
        for yaml_file in training_dir.glob("*.yaml"):
            with yaml_file.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            yaml_keys = _flatten_yaml_keys(data)
            found = yaml_keys & _DYNACONF_ONLY_KEYS
            if found:
                violations.append(f"{yaml_file.name}: {sorted(found)}")

        assert not violations, (
            "Hydra training configs contain Dynaconf-layer keys:\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\nThese belong in configs/deployment/settings.toml, not in Hydra YAMLs."
        )

    def test_model_configs_no_dynaconf_keys(self) -> None:
        """configs/model/*.yaml must not contain Dynaconf-layer keys."""
        model_dir = _CONFIGS / "model"
        if not model_dir.exists():
            return

        violations: list[str] = []
        for yaml_file in model_dir.glob("*.yaml"):
            with yaml_file.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            yaml_keys = _flatten_yaml_keys(data)
            found = yaml_keys & _DYNACONF_ONLY_KEYS
            if found:
                violations.append(f"{yaml_file.name}: {sorted(found)}")

        assert not violations, (
            "Hydra model configs contain Dynaconf-layer keys:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


class TestFactorialConfigsUseHydraKeys:
    """Factorial configs should only use Hydra-layer or infrastructure keys."""

    def test_factorial_configs_no_dynaconf_keys(self) -> None:
        """configs/factorial/*.yaml must not contain Dynaconf-layer keys."""
        factorial_dir = _CONFIGS / "factorial"
        if not factorial_dir.exists():
            return

        violations: list[str] = []
        for yaml_file in factorial_dir.glob("*.yaml"):
            with yaml_file.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            yaml_keys = _flatten_yaml_keys(data)
            found = yaml_keys & _DYNACONF_ONLY_KEYS
            if found:
                violations.append(f"{yaml_file.name}: {sorted(found)}")

        assert not violations, (
            "Factorial configs contain Dynaconf-layer keys:\n"
            + "\n".join(f"  {v}" for v in violations)
        )
