"""Tests for HPO grid script and experiment YAML configs.

T-04.4a: Verify that scripts/train_all_hyperparam_combos.sh exists and
implements the correct patterns (Docker-based, YAML-driven, dry-run capable).

CLAUDE.md Rule #16: Shell script is scanned with str.split() / str.partition(),
NOT regex. We check for presence/absence of specific string tokens.
"""

from __future__ import annotations

from pathlib import Path

import yaml

GRID_SCRIPT = Path("scripts/train_all_hyperparam_combos.sh")
EXPERIMENTS_DIR = Path("configs/experiment")
HPO_DIR = Path("configs/hpo")


# ---------------------------------------------------------------------------
# Grid script existence and structure
# ---------------------------------------------------------------------------


def test_grid_script_exists() -> None:
    """scripts/train_all_hyperparam_combos.sh must exist."""
    assert GRID_SCRIPT.exists(), f"Grid script not found: {GRID_SCRIPT}"


def test_grid_script_uses_docker() -> None:
    """Grid script must invoke 'docker compose' (not bare python) for training."""
    source = GRID_SCRIPT.read_text(encoding="utf-8")
    assert "docker compose" in source or "docker-compose" in source, (
        "Grid script must use 'docker compose' to launch training. "
        "CLAUDE.md Rule #17: never use bare uv run python for training."
    )


def test_grid_script_no_bare_python_training() -> None:
    """Grid script must NOT use 'uv run python' for training invocation.

    Uses str.splitlines() — regex is banned (CLAUDE.md Rule #16).
    """
    source = GRID_SCRIPT.read_text(encoding="utf-8")
    bad_lines = []
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        if "uv run python" in stripped and "train" in stripped.lower():
            bad_lines.append(f"line {i}: {stripped}")

    assert not bad_lines, (
        "Grid script uses 'uv run python' for training — "
        "must use Docker instead:\n" + "\n".join(bad_lines)
    )


def test_grid_script_reads_yaml() -> None:
    """Grid script must reference configs/experiments/ (reads YAML config)."""
    source = GRID_SCRIPT.read_text(encoding="utf-8")
    assert "configs/experiments" in source, (
        "Grid script must reference 'configs/experiments/' "
        "to read YAML hyperparameter configs."
    )


def test_grid_script_has_dry_run() -> None:
    """Grid script must handle a --dry-run flag."""
    source = GRID_SCRIPT.read_text(encoding="utf-8")
    assert "--dry-run" in source or "dry_run" in source or "DRY_RUN" in source, (
        "Grid script must support --dry-run to print grid without launching training."
    )


def test_grid_script_is_not_placeholder() -> None:
    """Grid script must NOT be the old placeholder (which exits 1 with error)."""
    source = GRID_SCRIPT.read_text(encoding="utf-8")
    # Old placeholder contained this exact string
    assert "PLACEHOLDER" not in source, (
        "Grid script is still the old placeholder — implement it!"
    )


# ---------------------------------------------------------------------------
# Experiment YAML configs
# ---------------------------------------------------------------------------


def test_experiment_configs_exist() -> None:
    """configs/experiments/ must contain at least one YAML file."""
    assert EXPERIMENTS_DIR.exists(), f"Directory not found: {EXPERIMENTS_DIR}"
    yaml_files = list(EXPERIMENTS_DIR.glob("*.yaml"))
    assert yaml_files, f"No YAML files found in {EXPERIMENTS_DIR}"


def test_dynunet_grid_config_exists() -> None:
    """configs/hpo/dynunet_grid.yaml must exist."""
    assert (HPO_DIR / "dynunet_grid.yaml").exists()


def test_smoke_test_config_exists() -> None:
    """configs/hpo/smoke_test.yaml must exist."""
    assert (HPO_DIR / "smoke_test.yaml").exists()


def test_hpo_configs_valid_yaml() -> None:
    """Every config in configs/hpo/ must be valid YAML."""
    for yaml_path in sorted(HPO_DIR.glob("*.yaml")):
        content = yaml_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), (
            f"{yaml_path} did not parse to a dict — got {type(parsed)}"
        )


def test_hpo_grid_configs_have_required_keys() -> None:
    """Grid-sweep HPO configs must have: experiment_name, model_family, hyperparameters, fixed.

    Only checks configs that use the grid schema (contain ``hyperparameters`` key).
    Optuna-style configs (``search_space`` key) have a different schema and are
    validated separately.
    """
    required_keys = {"experiment_name", "model_family", "hyperparameters", "fixed"}
    grid_configs_found = False
    for yaml_path in sorted(HPO_DIR.glob("*.yaml")):
        content = yaml_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        # Skip Optuna-style configs (they use search_space, not hyperparameters)
        if "search_space" in parsed:
            continue
        grid_configs_found = True
        missing = required_keys - set(parsed.keys())
        assert not missing, f"{yaml_path} missing required keys: {missing}"
    assert grid_configs_found, "No grid-schema HPO configs found in configs/hpo/"


def test_dynunet_grid_hyperparameters() -> None:
    """dynunet_grid.yaml must have the expected hyperparameter lists."""
    config_path = HPO_DIR / "dynunet_grid.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    hp = config["hyperparameters"]
    assert "loss_name" in hp, "dynunet_grid.yaml must have loss_name hyperparameter"
    assert isinstance(hp["loss_name"], list), "loss_name must be a list"
    assert len(hp["loss_name"]) >= 2, "dynunet_grid must sweep at least 2 losses"


def test_smoke_test_is_minimal() -> None:
    """smoke_test.yaml must have max_epochs=1 and num_folds=1."""
    config_path = HPO_DIR / "smoke_test.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    fixed = config["fixed"]
    assert fixed.get("max_epochs") == 1, (
        f"smoke_test must have max_epochs=1, got {fixed.get('max_epochs')}"
    )
    assert fixed.get("num_folds") == 1, (
        f"smoke_test must have num_folds=1, got {fixed.get('num_folds')}"
    )


def test_hpo_grid_configs_mlflow_experiment_key() -> None:
    """Grid-sweep HPO configs must have an mlflow_experiment key."""
    for yaml_path in sorted(HPO_DIR.glob("*.yaml")):
        content = yaml_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        # Skip Optuna-style configs (they don't use mlflow_experiment)
        if "search_space" in parsed:
            continue
        assert "mlflow_experiment" in parsed, (
            f"{yaml_path} missing 'mlflow_experiment' key"
        )
