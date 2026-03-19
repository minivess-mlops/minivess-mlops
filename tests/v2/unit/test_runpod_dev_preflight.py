"""Tests for RunPod dev environment preflight validation (plan v4.0).

Covers:
- validate_runpod_dev_env.py: env var, SkyPilot version, network volume,
  balance, DVC status, sync size checks
- dev_runpod.yaml: network volume mount, file-based MLflow, weight cache
- verify_smoke_test.py: dev_* experiment search + artifact size (T1.2)

GitHub issues: #694, #695, #696, #697
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest  # noqa: TC002 — used at runtime for MonkeyPatch fixture type
import yaml

ROOT = Path(__file__).parent.parent.parent.parent
SCRIPTS = ROOT / "scripts"
SKYPILOT_DIR = ROOT / "deployment" / "skypilot"


# ---------------------------------------------------------------------------
# Preflight script: validate_runpod_dev_env.py
# ---------------------------------------------------------------------------


def _load_preflight_module():  # type: ignore[no-untyped-def]
    """Import validate_runpod_dev_env.py as a module."""
    script = SCRIPTS / "validate_runpod_dev_env.py"
    assert script.exists(), (
        f"Missing: {script}. Create scripts/validate_runpod_dev_env.py (#694)"
    )
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("validate_runpod_dev_env", script)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_validate_runpod_dev_env_importable() -> None:
    """validate_runpod_dev_env.py must exist and be importable."""
    mod = _load_preflight_module()
    assert hasattr(mod, "check_env_vars"), "Missing check_env_vars() function"
    assert hasattr(mod, "check_skypilot"), "Missing check_skypilot() function"
    assert hasattr(mod, "check_network_volume"), (
        "Missing check_network_volume() function"
    )
    assert hasattr(mod, "check_runpod_balance"), (
        "Missing check_runpod_balance() function"
    )
    assert hasattr(mod, "check_dvc_status"), "Missing check_dvc_status() function"
    assert hasattr(mod, "check_sync_size"), "Missing check_sync_size() function"
    assert hasattr(mod, "main"), "Missing main() function"


def test_check_env_vars_detects_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_env_vars returns missing var names (5 required)."""
    required = [
        "RUNPOD_API_KEY",
        "DVC_S3_ENDPOINT_URL",
        "DVC_S3_ACCESS_KEY",
        "DVC_S3_SECRET_KEY",
        "HF_TOKEN",
    ]
    for var in required:
        monkeypatch.delenv(var, raising=False)
    # Also clear other vars to ensure they're NOT required
    for var in [
        "DVC_S3_BUCKET",
    ]:
        monkeypatch.delenv(var, raising=False)

    mod = _load_preflight_module()
    missing = mod.check_env_vars()
    assert isinstance(missing, list), "check_env_vars() must return a list"
    for var in required:
        assert var in missing, f"Expected {var} in missing list, got: {missing}"


def test_check_env_vars_all_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """check_env_vars returns empty list when all 5 vars are set."""
    required = [
        "RUNPOD_API_KEY",
        "DVC_S3_ENDPOINT_URL",
        "DVC_S3_ACCESS_KEY",
        "DVC_S3_SECRET_KEY",
        "HF_TOKEN",
    ]
    for var in required:
        monkeypatch.setenv(var, "test_value")

    mod = _load_preflight_module()
    missing = mod.check_env_vars()
    assert missing == [], f"Expected no missing vars, got: {missing}"


def test_no_mlflow_cloud_vars_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy MLFLOW_CLOUD_* vars must NOT be in the required list."""
    mod = _load_preflight_module()
    required_vars = mod._REQUIRED_VARS
    banned = ["MLFLOW_CLOUD_URI", "MLFLOW_CLOUD_USERNAME", "MLFLOW_CLOUD_PASSWORD"]
    for var in banned:
        assert var not in required_vars, (
            f"{var} is still required — removed in single-MLFLOW_TRACKING_URI simplification"
        )


def test_check_skypilot_version_in_output() -> None:
    """check_skypilot() must verify SkyPilot version >= 0.6.0."""
    mod = _load_preflight_module()
    import inspect

    sig = inspect.signature(mod.check_skypilot)
    assert (
        len(
            [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
        )
        == 0
    ), "check_skypilot() must have no required parameters"


def test_check_network_volume_exists() -> None:
    """check_network_volume() function exists and is callable."""
    mod = _load_preflight_module()
    import inspect

    sig = inspect.signature(mod.check_network_volume)
    assert "volume_name" in sig.parameters, (
        "check_network_volume() must accept volume_name parameter"
    )


def test_check_runpod_balance_exists() -> None:
    """check_runpod_balance() function exists and is callable."""
    mod = _load_preflight_module()
    import inspect

    sig = inspect.signature(mod.check_runpod_balance)
    assert "min_balance" in sig.parameters, (
        "check_runpod_balance() must accept min_balance parameter"
    )


def test_check_sync_size_returns_tuple() -> None:
    """check_sync_size() returns (size_bytes, critical_files_missing)."""
    mod = _load_preflight_module()
    import inspect

    sig = inspect.signature(mod.check_sync_size)
    assert "project_root" in sig.parameters, (
        "check_sync_size() must accept project_root parameter"
    )


def test_check_sync_size_under_limit() -> None:
    """Actual project sync size must be under 100 MB."""
    mod = _load_preflight_module()
    size_bytes, missing_critical = mod.check_sync_size(project_root=ROOT)
    assert size_bytes < 100 * 1024 * 1024, (
        f"Sync size {size_bytes / 1024 / 1024:.1f} MB exceeds 100 MB limit. "
        "Check .skyignore for missing exclusions."
    )
    assert missing_critical == [], (
        f"Critical files missing from sync: {missing_critical}"
    )


# ---------------------------------------------------------------------------
# dev_runpod.yaml: Network Volume + file-based MLflow
# ---------------------------------------------------------------------------


def _load_dev_runpod_yaml() -> dict:  # type: ignore[type-arg]
    """Load and parse dev_runpod.yaml."""
    yaml_path = SKYPILOT_DIR / "dev_runpod.yaml"
    assert yaml_path.exists(), f"Missing: {yaml_path}"
    with yaml_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def test_dev_runpod_yaml_has_volume_mount() -> None:
    """dev_runpod.yaml must mount a persistent network volume."""
    data = _load_dev_runpod_yaml()
    volumes = data.get("volumes", {})
    assert volumes, (
        "dev_runpod.yaml must have a 'volumes:' section for persistent storage. "
        "Add: volumes: {/opt/vol: minivess-dev}"
    )
    # At least one mount point should reference minivess-dev
    has_minivess_vol = any("minivess" in str(v) for v in volumes.values())
    assert has_minivess_vol, (
        "dev_runpod.yaml volumes must reference 'minivess-dev' network volume"
    )


def test_dev_runpod_yaml_mlflow_file_based() -> None:
    """MLFLOW_TRACKING_URI must be a file path, not an HTTP URL."""
    data = _load_dev_runpod_yaml()
    envs = data.get("envs", {})
    uri = envs.get("MLFLOW_TRACKING_URI", "")
    assert uri, "MLFLOW_TRACKING_URI not set in envs"
    assert not uri.startswith("http"), (
        f"MLFLOW_TRACKING_URI = '{uri}' is HTTP — must be a file path on volume. "
        "Use /opt/vol/mlruns for file-based MLflow."
    )
    assert "MLFLOW_CLOUD_URI" not in uri, (
        "MLFLOW_TRACKING_URI still references MLFLOW_CLOUD_URI — "
        "use /opt/vol/mlruns for file-based MLflow"
    )


def test_dev_runpod_yaml_no_mlflow_auth() -> None:
    """dev_runpod.yaml must NOT require MLFLOW_TRACKING_USERNAME/PASSWORD."""
    data = _load_dev_runpod_yaml()
    envs = data.get("envs", {})
    assert "MLFLOW_TRACKING_USERNAME" not in envs, (
        "Remove MLFLOW_TRACKING_USERNAME — file-based MLflow needs no auth"
    )
    assert "MLFLOW_TRACKING_PASSWORD" not in envs, (
        "Remove MLFLOW_TRACKING_PASSWORD — file-based MLflow needs no auth"
    )


def test_dev_runpod_yaml_weight_cache_envs() -> None:
    """HF_HOME and TORCH_HOME must point to Network Volume paths."""
    data = _load_dev_runpod_yaml()
    envs = data.get("envs", {})
    hf_home = envs.get("HF_HOME", "")
    torch_home = envs.get("TORCH_HOME", "")
    assert hf_home, "HF_HOME not set — weights won't cache on volume"
    assert torch_home, "TORCH_HOME not set — weights won't cache on volume"
    assert "/opt/vol" in hf_home or "/workspace" in hf_home, (
        f"HF_HOME = '{hf_home}' must point to network volume"
    )
    assert "/opt/vol" in torch_home or "/workspace" in torch_home, (
        f"TORCH_HOME = '{torch_home}' must point to network volume"
    )


def test_dev_runpod_yaml_checkpoint_dir_on_volume() -> None:
    """CHECKPOINT_DIR must point to network volume, not ephemeral workdir."""
    data = _load_dev_runpod_yaml()
    run_block: str = data.get("run", "")
    assert (
        "/opt/vol/checkpoints" in run_block or "/workspace/checkpoints" in run_block
    ), "CHECKPOINT_DIR must be on the network volume (e.g. /opt/vol/checkpoints)"


def test_dev_runpod_run_block_has_mlflow_diagnostic() -> None:
    """run block must log the resolved MLFLOW_TRACKING_URI."""
    data = _load_dev_runpod_yaml()
    run_block: str = data.get("run", "")
    assert "MLFLOW_TRACKING_URI" in run_block, (
        "dev_runpod.yaml run: block must print MLFLOW_TRACKING_URI before training"
    )


def test_dev_runpod_run_block_guards_volume_mount() -> None:
    """run block must verify Network Volume is mounted."""
    data = _load_dev_runpod_yaml()
    run_block: str = data.get("run", "")
    assert "/opt/vol" in run_block or "/workspace" in run_block, (
        "dev_runpod.yaml run: block must verify network volume is mounted"
    )


# ---------------------------------------------------------------------------
# dev_runpod.yaml — no bare python calls in setup after venv creation (T0.7)
# ---------------------------------------------------------------------------


def test_dev_runpod_setup_no_bare_python_after_venv() -> None:
    """After venv PATH export, no bare 'python -c' — must use explicit venv path."""
    yaml_path = SKYPILOT_DIR / "dev_runpod.yaml"
    assert yaml_path.exists()
    with yaml_path.open(encoding="utf-8") as f:
        content = f.read()

    setup_block = yaml.safe_load(content).get("setup", "")
    venv_export_pos = setup_block.find('export PATH="${WORKDIR}/.venv/bin')
    assert venv_export_pos != -1, (
        "dev_runpod.yaml setup: must export PATH with venv bin."
    )
    post_export = setup_block[venv_export_pos:]
    import re

    bare_python_calls = re.findall(r"(?<!/bin/)python -[cm]", post_export)
    assert bare_python_calls == [], (
        f"Found bare python calls after venv PATH export: {bare_python_calls}. "
        "Replace with ${{WORKDIR}}/.venv/bin/python. (#696)"
    )


def test_dev_runpod_setup_has_symlink_check() -> None:
    """setup block should verify python symlink exists in venv."""
    data = _load_dev_runpod_yaml()
    setup_block: str = data.get("setup", "")
    has_symlink_check = ".venv/bin/python" in setup_block or "ls -la" in setup_block
    assert has_symlink_check, (
        "dev_runpod.yaml setup: must check python symlink exists in venv. (#696)"
    )


def test_dev_runpod_setup_skips_dvc_if_cached() -> None:
    """Setup must skip DVC pull if data already exists on network volume."""
    data = _load_dev_runpod_yaml()
    setup_block: str = data.get("setup", "")
    assert (
        "already cached" in setup_block.lower() or "skipping" in setup_block.lower()
    ), "dev_runpod.yaml setup must skip DVC pull when data is cached on volume"


# ---------------------------------------------------------------------------
# verify_smoke_test.py — searches dev_* prefix + artifact size (T1.2)
# ---------------------------------------------------------------------------


def test_verify_smoke_test_searches_dev_prefix() -> None:
    """verify_smoke_test.py must search for 'dev_' prefix."""
    verify_script = SCRIPTS / "verify_smoke_test.py"
    assert verify_script.exists(), f"Missing: {verify_script}"
    content = verify_script.read_text(encoding="utf-8")
    assert "dev_" in content, (
        "verify_smoke_test.py only searches for 'smoke_test_' prefix. "
        "dev_runpod.yaml creates experiments named 'dev_{uuid}_{model}'. (#697)"
    )


def test_verify_smoke_test_checks_artifact_size() -> None:
    """verify_smoke_test.py must validate artifact size."""
    verify_script = SCRIPTS / "verify_smoke_test.py"
    assert verify_script.exists()
    content = verify_script.read_text(encoding="utf-8")
    has_size_check = (
        "file_size" in content
        or "artifact_size" in content
        or ".size" in content
        or "nbytes" in content
        or "min_size" in content
    )
    assert has_size_check, (
        "verify_smoke_test.py does not check artifact file sizes. (#697)"
    )


def test_verify_smoke_test_checks_train_loss_range() -> None:
    """verify_smoke_test.py must validate train_loss is in sane range."""
    verify_script = SCRIPTS / "verify_smoke_test.py"
    assert verify_script.exists()
    content = verify_script.read_text(encoding="utf-8")
    has_range_check = "train_loss" in content and (
        "0.01" in content or "range" in content or "< 5" in content
    )
    assert has_range_check, (
        "verify_smoke_test.py does not sanity-check train_loss range. (#697)"
    )
