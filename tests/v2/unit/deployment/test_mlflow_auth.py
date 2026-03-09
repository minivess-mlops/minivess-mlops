"""Tests for MLflow basic-auth opt-in — T-04.1 / issue #553.

MLflow auth is a non-breaking opt-in via docker compose profile "secure".
Default (no auth) remains the solo-dev workflow.

Verifies:
- deployment/mlflow/auth.ini.example exists with [mlflow] section
- deployment/mlflow/auth.ini is gitignored
- scripts/init_mlflow_auth.sh exists, uses --app-name basic-auth (not --users-db-uri)
- .env.example has MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
- docker-compose.yml has mlflow-secure service under profile "secure"
- mlflow-secure command contains --app-name basic-auth
- mutual exclusivity is documented (port collision warning)
- resolve_tracking_uri() injects credentials when MLFLOW_TRACKING_USERNAME is set

Rule #16: yaml.safe_load(), str methods, pathlib.Path — no regex.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
AUTH_INI_EXAMPLE = ROOT / "deployment" / "mlflow" / "auth.ini.example"
INIT_AUTH_SCRIPT = ROOT / "scripts" / "init_mlflow_auth.sh"
COMPOSE_YML = ROOT / "deployment" / "docker-compose.yml"
ENV_EXAMPLE = ROOT / ".env.example"
GITIGNORE = ROOT / ".gitignore"


def test_mlflow_auth_ini_example_exists() -> None:
    assert AUTH_INI_EXAMPLE.exists(), (
        "deployment/mlflow/auth.ini.example not found. "
        "Create template for MLflow basic-auth config."
    )


def test_mlflow_auth_ini_example_has_mlflow_section() -> None:
    content = AUTH_INI_EXAMPLE.read_text(encoding="utf-8")
    assert "[mlflow]" in content, (
        "deployment/mlflow/auth.ini.example must have [mlflow] section. "
        "MLflow basic-auth reads this INI file format."
    )


def test_mlflow_auth_ini_example_has_required_keys() -> None:
    content = AUTH_INI_EXAMPLE.read_text(encoding="utf-8")
    assert "default_permission" in content, (
        "auth.ini.example missing 'default_permission'"
    )
    assert "admin_username" in content or "username" in content, (
        "auth.ini.example missing admin username field"
    )


def test_mlflow_auth_ini_is_gitignored() -> None:
    """auth.ini may contain hashed passwords — must never be committed."""
    gitignore = GITIGNORE.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in gitignore.splitlines()
        if not line.strip().startswith("#")
    ]
    assert any("auth.ini" in line for line in lines), (
        "deployment/mlflow/auth.ini must be in .gitignore. "
        "The file may contain hashed passwords."
    )


def test_env_example_has_mlflow_auth_vars() -> None:
    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "MLFLOW_TRACKING_USERNAME" in content, (
        ".env.example must define MLFLOW_TRACKING_USERNAME (per Rule #22)."
    )
    assert "MLFLOW_TRACKING_PASSWORD" in content, (
        ".env.example must define MLFLOW_TRACKING_PASSWORD (per Rule #22)."
    )


def test_init_mlflow_auth_script_exists() -> None:
    assert INIT_AUTH_SCRIPT.exists(), (
        "scripts/init_mlflow_auth.sh not found. "
        "Create script documenting post-startup auth setup steps."
    )


def test_init_mlflow_auth_uses_app_name_basic_auth() -> None:
    content = INIT_AUTH_SCRIPT.read_text(encoding="utf-8")
    assert "basic-auth" in content, (
        "scripts/init_mlflow_auth.sh must reference '--app-name basic-auth'. "
        "This is the correct MLflow 3.x flag for enabling authentication."
    )


def test_init_mlflow_auth_does_not_use_users_db_uri() -> None:
    """--users-db-uri is NOT a valid MLflow 3.x flag."""
    content = INIT_AUTH_SCRIPT.read_text(encoding="utf-8")
    assert "users-db-uri" not in content, (
        "--users-db-uri is not a valid MLflow 3.x flag. "
        "Use --app-name basic-auth. MLflow auto-creates basic_auth.db."
    )


def test_compose_has_mlflow_secure_service() -> None:
    compose = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    assert "mlflow-secure" in compose.get("services", {}), (
        "docker-compose.yml missing 'mlflow-secure' service. "
        "Add under profile 'secure' with --app-name basic-auth."
    )


def test_mlflow_secure_is_under_secure_profile() -> None:
    compose = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    secure_svc = compose["services"]["mlflow-secure"]
    profiles = secure_svc.get("profiles", [])
    assert "secure" in profiles, (
        "mlflow-secure service must be under profile 'secure'. "
        "Default docker compose up must NOT start it."
    )


def test_mlflow_secure_command_has_basic_auth() -> None:
    compose = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    secure_svc = compose["services"]["mlflow-secure"]
    command = str(secure_svc.get("command", ""))
    assert "basic-auth" in command, (
        "mlflow-secure command must include --app-name basic-auth."
    )


def test_compose_documents_mutual_exclusivity() -> None:
    """Both mlflow and mlflow-secure bind the same port — must be documented."""
    content = COMPOSE_YML.read_text(encoding="utf-8")
    has_warning = (
        "mutually exclusive" in content.lower()
        or "cannot run simultaneously" in content.lower()
        or "stop mlflow" in content.lower()
    )
    assert has_warning, (
        "docker-compose.yml must document that mlflow and mlflow-secure "
        "cannot run simultaneously (port collision). Add a comment."
    )


def test_resolve_tracking_uri_injects_auth_credentials() -> None:
    """When MLFLOW_TRACKING_USERNAME/PASSWORD are set, URI includes credentials."""
    from minivess.observability.tracking import resolve_tracking_uri

    with patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
            "MLFLOW_TRACKING_USERNAME": "admin",
            "MLFLOW_TRACKING_PASSWORD": "secret123",
        },
    ):
        uri = resolve_tracking_uri(use_dynaconf=False)
    assert "admin" in uri, f"Username not injected into URI: {uri}"
    assert "secret123" in uri, f"Password not injected into URI: {uri}"
    assert uri.startswith("http://admin:secret123@"), (
        f"Expected http://admin:secret123@..., got: {uri}"
    )


def test_resolve_tracking_uri_unchanged_without_auth_vars() -> None:
    """Without auth vars, resolve_tracking_uri must return URI unchanged."""
    from minivess.observability.tracking import resolve_tracking_uri

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
    }
    env["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    with patch.dict(os.environ, env, clear=True):
        uri = resolve_tracking_uri(use_dynaconf=False)
    assert uri == "http://localhost:5000", f"URI unexpectedly modified: {uri}"
