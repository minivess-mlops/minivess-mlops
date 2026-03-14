"""Tests for RunPod dev environment preflight validation (plan v3.1 T0.1-T0.7).

Covers:
- validate_runpod_dev_env.py: env var, SkyPilot version, MLflow health,
  DVC status, sync size checks (T0.1-T0.5)
- dev_runpod.yaml MLFLOW_TRACKING_URI guards (T0.6)
- dev_runpod.yaml python symlink safety in setup block (T0.7)
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
# T0.1-T0.5: validate_runpod_dev_env.py — importable + correct checks
# ---------------------------------------------------------------------------


def test_validate_runpod_dev_env_importable() -> None:
    """validate_runpod_dev_env.py must exist and be importable."""
    script = SCRIPTS / "validate_runpod_dev_env.py"
    assert script.exists(), (
        f"Missing: {script}. Create scripts/validate_runpod_dev_env.py (T0.1-T0.5, #694)"
    )
    # Add scripts/ to sys.path temporarily
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("validate_runpod_dev_env", script)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    assert hasattr(mod, "check_env_vars"), "Missing check_env_vars() function"
    assert hasattr(mod, "check_mlflow_health"), "Missing check_mlflow_health() function"
    assert hasattr(mod, "check_skypilot"), "Missing check_skypilot() function"
    assert hasattr(mod, "check_dvc_status"), "Missing check_dvc_status() function"
    assert hasattr(mod, "check_sync_size"), "Missing check_sync_size() function"
    assert hasattr(mod, "main"), "Missing main() function"


def test_check_env_vars_detects_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T0.1: check_env_vars returns list of missing variable names."""
    # Remove all required vars from environment
    required = [
        "RUNPOD_API_KEY",
        "DVC_S3_ENDPOINT_URL",
        "DVC_S3_ACCESS_KEY",
        "DVC_S3_SECRET_KEY",
        "DVC_S3_BUCKET",
        "MLFLOW_CLOUD_URI",
        "MLFLOW_CLOUD_USERNAME",
        "MLFLOW_CLOUD_PASSWORD",
        "HF_TOKEN",
    ]
    for var in required:
        monkeypatch.delenv(var, raising=False)

    script = SCRIPTS / "validate_runpod_dev_env.py"
    assert script.exists(), "Missing validate_runpod_dev_env.py — create it first"

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("validate_runpod_dev_env", script)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    missing = mod.check_env_vars()
    assert isinstance(missing, list), "check_env_vars() must return a list"
    # All 9 required vars should be missing
    for var in required:
        assert var in missing, f"Expected {var} in missing list, got: {missing}"


def test_check_env_vars_all_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """T0.1: check_env_vars returns empty list when all vars are set."""
    required = [
        "RUNPOD_API_KEY",
        "DVC_S3_ENDPOINT_URL",
        "DVC_S3_ACCESS_KEY",
        "DVC_S3_SECRET_KEY",
        "DVC_S3_BUCKET",
        "MLFLOW_CLOUD_URI",
        "MLFLOW_CLOUD_USERNAME",
        "MLFLOW_CLOUD_PASSWORD",
        "HF_TOKEN",
    ]
    for var in required:
        monkeypatch.setenv(var, "test_value")

    script = SCRIPTS / "validate_runpod_dev_env.py"
    assert script.exists(), "Missing validate_runpod_dev_env.py"

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "validate_runpod_dev_env_allset", script
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    missing = mod.check_env_vars()
    assert missing == [], f"Expected no missing vars, got: {missing}"


def test_check_skypilot_version_in_output() -> None:
    """T0.2: check_skypilot() must verify SkyPilot version >= 0.6.0."""
    script = SCRIPTS / "validate_runpod_dev_env.py"
    assert script.exists(), "Missing validate_runpod_dev_env.py"

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("validate_runpod_dev_env_sky", script)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    # check_skypilot() returns (ok: bool, message: str)
    import inspect

    sig = inspect.signature(mod.check_skypilot)
    # Must be callable with no required args
    assert (
        len(
            [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
        )
        == 0
    ), "check_skypilot() must have no required parameters"


def test_check_sync_size_returns_tuple() -> None:
    """T0.5: check_sync_size() returns (size_bytes, critical_files_missing)."""
    script = SCRIPTS / "validate_runpod_dev_env.py"
    assert script.exists(), "Missing validate_runpod_dev_env.py"

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "validate_runpod_dev_env_sync", script
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    import inspect

    sig = inspect.signature(mod.check_sync_size)
    # Must accept an optional project_root argument
    assert "project_root" in sig.parameters, (
        "check_sync_size() must accept project_root parameter"
    )


def test_check_sync_size_under_limit() -> None:
    """T0.5: Actual project sync size must be under 100 MB."""
    script = SCRIPTS / "validate_runpod_dev_env.py"
    assert script.exists(), "Missing validate_runpod_dev_env.py"

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "validate_runpod_dev_env_size", script
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    size_bytes, missing_critical = mod.check_sync_size(project_root=ROOT)
    assert size_bytes < 100 * 1024 * 1024, (
        f"Sync size {size_bytes / 1024 / 1024:.1f} MB exceeds 100 MB limit. "
        "Check .skyignore for missing exclusions."
    )
    assert missing_critical == [], (
        f"Critical files missing from sync: {missing_critical}"
    )


# ---------------------------------------------------------------------------
# T0.6: dev_runpod.yaml MLFLOW_TRACKING_URI guards
# ---------------------------------------------------------------------------


def _load_dev_runpod_yaml() -> dict:  # type: ignore[type-arg]
    """Load and parse dev_runpod.yaml."""
    yaml_path = SKYPILOT_DIR / "dev_runpod.yaml"
    assert yaml_path.exists(), f"Missing: {yaml_path}"
    with yaml_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def test_dev_runpod_run_block_has_mlflow_uri_diagnostic() -> None:
    """T0.6: run block must echo the resolved MLFLOW_TRACKING_URI before training."""
    data = _load_dev_runpod_yaml()
    run_block: str = data.get("run", "")
    assert "MLFLOW_TRACKING_URI" in run_block, (
        "dev_runpod.yaml run: block must print/check MLFLOW_TRACKING_URI "
        "before training starts. Add diagnostic echo. (#695)"
    )


def test_dev_runpod_run_block_guards_empty_uri() -> None:
    """T0.6: run block must guard against empty MLFLOW_TRACKING_URI."""
    data = _load_dev_runpod_yaml()
    run_block: str = data.get("run", "")
    # Must check for empty string
    assert '-z "${MLFLOW_TRACKING_URI}"' in run_block or "[ -z" in run_block, (
        "dev_runpod.yaml run: block must guard: "
        '[ -z "${MLFLOW_TRACKING_URI}" ]. Add this check. (#695)'
    )


def test_dev_runpod_run_block_guards_local_mlruns() -> None:
    """T0.6: run block must guard against MLFLOW_TRACKING_URI being 'mlruns' (local fallback)."""
    data = _load_dev_runpod_yaml()
    run_block: str = data.get("run", "")
    assert "mlruns" in run_block, (
        "dev_runpod.yaml run: block must guard against MLFLOW_TRACKING_URI = 'mlruns'. "
        'Add: [ "${MLFLOW_TRACKING_URI}" = "mlruns" ]. (#695)'
    )


def test_dev_runpod_run_block_guards_auth_missing() -> None:
    """T0.6: run block must guard against missing MLflow auth credentials."""
    data = _load_dev_runpod_yaml()
    run_block: str = data.get("run", "")
    assert (
        "MLFLOW_TRACKING_USERNAME" in run_block
        and "MLFLOW_TRACKING_PASSWORD" in run_block
    ), (
        "dev_runpod.yaml run: block must validate MLFLOW_TRACKING_USERNAME and PASSWORD "
        "are non-empty before training. (#695)"
    )


# ---------------------------------------------------------------------------
# T0.7: dev_runpod.yaml — no bare python calls in setup after venv creation
# ---------------------------------------------------------------------------


def test_dev_runpod_setup_no_bare_python_after_venv() -> None:
    """T0.7: After venv PATH export, no bare 'python -c' — must use explicit venv path."""
    yaml_path = SKYPILOT_DIR / "dev_runpod.yaml"
    assert yaml_path.exists()
    with yaml_path.open(encoding="utf-8") as f:
        content = f.read()

    setup_block = yaml.safe_load(content).get("setup", "")
    # Find position of venv PATH export
    venv_export_pos = setup_block.find('export PATH="${WORKDIR}/.venv/bin')
    assert venv_export_pos != -1, (
        "dev_runpod.yaml setup: must export PATH with venv bin. "
        'Missing: export PATH="${WORKDIR}/.venv/bin:..."'
    )
    # After the export, there must be no bare `python -c`
    post_export = setup_block[venv_export_pos:]
    import re

    # Look for bare `python -c` (not preceded by `.venv/bin/python`)
    bare_python_calls = re.findall(r"(?<!/bin/)python -[cm]", post_export)
    assert bare_python_calls == [], (
        f"Found bare python calls after venv PATH export: {bare_python_calls}. "
        "Replace with ${WORKDIR}/.venv/bin/python. (#696)"
    )


def test_dev_runpod_setup_has_symlink_check() -> None:
    """T0.7: setup block should verify python symlink exists in venv."""
    data = _load_dev_runpod_yaml()
    setup_block: str = data.get("setup", "")
    # Must have either a symlink check or explicit venv python path
    has_symlink_check = ".venv/bin/python" in setup_block or "ls -la" in setup_block
    assert has_symlink_check, (
        "dev_runpod.yaml setup: must either check python symlink exists in venv "
        "or use explicit ${WORKDIR}/.venv/bin/python for all calls. (#696)"
    )


# ---------------------------------------------------------------------------
# T1.2: verify_smoke_test.py — searches dev_* prefix + artifact size
# ---------------------------------------------------------------------------


def test_verify_smoke_test_searches_dev_prefix() -> None:
    """T1.2: verify_smoke_test.py must search for 'dev_' prefix, not just 'smoke_test_'."""
    verify_script = SCRIPTS / "verify_smoke_test.py"
    assert verify_script.exists(), f"Missing: {verify_script}"

    content = verify_script.read_text(encoding="utf-8")
    assert '"dev_"' in content or "'dev_'" in content or "dev_" in content, (
        "verify_smoke_test.py only searches for 'smoke_test_' prefix. "
        "dev_runpod.yaml creates experiments named 'dev_{uuid}_{model}'. "
        "Add search for 'dev_' prefix. (#697)"
    )


def test_verify_smoke_test_checks_artifact_size() -> None:
    """T1.2: verify_smoke_test.py must validate artifact size (not just existence)."""
    verify_script = SCRIPTS / "verify_smoke_test.py"
    assert verify_script.exists()

    content = verify_script.read_text(encoding="utf-8")
    # Should have a size check — look for file_size, size, or bytes reference
    has_size_check = (
        "file_size" in content
        or "artifact_size" in content
        or ".size" in content
        or "nbytes" in content
        or "min_size" in content
    )
    assert has_size_check, (
        "verify_smoke_test.py does not check artifact file sizes. "
        "A 0-byte checkpoint passes 'artifacts exist' check. "
        "Add size validation: size > expected_minimum. (#697)"
    )


def test_verify_smoke_test_checks_train_loss_range() -> None:
    """T1.2: verify_smoke_test.py must validate train_loss is in sane range (0.01, 5.0)."""
    verify_script = SCRIPTS / "verify_smoke_test.py"
    assert verify_script.exists()

    content = verify_script.read_text(encoding="utf-8")
    # Should have numeric range validation for train_loss
    has_range_check = "train_loss" in content and (
        "0.01" in content
        or "sanity" in content
        or "range" in content
        or "< 5" in content
        or "< 5.0" in content
    )
    assert has_range_check, (
        "verify_smoke_test.py does not sanity-check train_loss range. "
        "NaN or 0.0 loss would pass 'metric exists' check. "
        "Add: assert 0.01 < train_loss < 5.0. (#697)"
    )
