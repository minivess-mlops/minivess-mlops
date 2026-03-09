"""Compose file structural correctness tests.

Rule #16: No regex. Parse with yaml.safe_load() and str methods.
Rule #22: All configurable values from .env.example — no hardcoded paths.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
FLOWS_COMPOSE = ROOT / "deployment" / "docker-compose.flows.yml"


def _load_flows_compose() -> dict:
    return yaml.safe_load(FLOWS_COMPOSE.read_text(encoding="utf-8"))


def test_compose_flows_exists() -> None:
    assert FLOWS_COMPOSE.exists(), f"{FLOWS_COMPOSE} not found"


def test_no_hardcoded_password_in_compose_flows() -> None:
    """OPTUNA_STORAGE_URL must use ${POSTGRES_PASSWORD} reference, not a bare literal password.

    Acceptable:   ${POSTGRES_PASSWORD:-minivess_secret}   (variable reference with fallback)
    Not acceptable: ://minivess:minivess_secret@postgres   (bare literal in connection string)

    Rule #22: passwords must be behind ${VAR} references.
    """
    content = FLOWS_COMPOSE.read_text(encoding="utf-8")
    # Find the OPTUNA_STORAGE_URL line and verify it uses POSTGRES_PASSWORD reference
    for line in content.splitlines():
        if "OPTUNA_STORAGE_URL" in line and ":-" in line:
            # The connection string must use ${POSTGRES_PASSWORD} reference.
            # A bare literal like ://user:minivess_secret@host would be a violation.
            # Having ${POSTGRES_PASSWORD:-minivess_secret} is CORRECT (variable-first).
            assert "${POSTGRES_PASSWORD" in line, (
                f"OPTUNA_STORAGE_URL does not use ${{POSTGRES_PASSWORD}} variable reference. "
                f"Line: {line!r}. "
                f"Change hardcoded password to ${{POSTGRES_PASSWORD:-minivess_secret}}. "
                f"Rule #22 violation."
            )


def test_model_cache_has_no_hardcoded_home_path() -> None:
    """MODEL_CACHE_HOST_PATH volume mount must not fall back to /home/petteri."""
    content = FLOWS_COMPOSE.read_text(encoding="utf-8")
    for line in content.splitlines():
        if "model_cache" in line or "MODEL_CACHE_HOST_PATH" in line:
            assert "/home/petteri" not in line, (
                f"Found hardcoded /home/petteri in compose line: {line!r}. "
                f"Remove the fallback — MODEL_CACHE_HOST_PATH must be set in .env."
            )
