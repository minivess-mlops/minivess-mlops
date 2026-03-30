"""Enforcement tests for CLAUDE.md Rule #22: .env.example is the single source of truth.

These tests FAIL when anyone violates the rule by:
- Hardcoding MLFLOW_TRACKING_URI in a Dockerfile
- Defining mlflow_tracking_uri in a Dynaconf TOML file
- Using a hardcoded URL in docker-compose without ${VAR} substitution
- Using os.environ.get("MLFLOW_TRACKING_URI", "mlruns") instead of resolve_tracking_uri()

Run: uv run pytest tests/v2/unit/test_env_single_source.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _env_example_vars() -> set[str]:
    """Parse .env.example and return all defined variable names."""
    vars_: set[str] = set()
    for line in _read(ROOT / ".env.example").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            vars_.add(key)
    return vars_


# ---------------------------------------------------------------------------
# .env.example completeness
# ---------------------------------------------------------------------------


def test_env_example_has_mlflow_tracking_uri() -> None:
    """MLFLOW_TRACKING_URI must be defined in .env.example."""
    assert "MLFLOW_TRACKING_URI" in _env_example_vars()


def test_env_example_has_docker_service_vars() -> None:
    """Docker-internal service name vars must be in .env.example.

    MLFLOW_DOCKER_HOST and MLFLOW_SKYPILOT_HOST were removed in the
    single-MLFLOW_TRACKING_URI simplification. Docker Compose hardcodes
    the MLflow service name; SkyPilot YAMLs set MLFLOW_TRACKING_URI directly.
    """
    vars_ = _env_example_vars()
    for var in (
        "MLFLOW_PORT",
        "PREFECT_DOCKER_HOST",
        "PREFECT_PORT",
        "MINIO_DOCKER_HOST",
        "POSTGRES_DOCKER_HOST",
    ):
        assert var in vars_, f"{var} missing from .env.example"


def test_env_example_removed_mlflow_docker_host() -> None:
    """MLFLOW_DOCKER_HOST must NOT be in .env.example.

    Removed in single-MLFLOW_TRACKING_URI simplification. Docker Compose
    hardcodes the service name 'minivess-mlflow' directly.
    """
    vars_ = _env_example_vars()
    assert "MLFLOW_DOCKER_HOST" not in vars_, (
        "MLFLOW_DOCKER_HOST should be removed from .env.example — "
        "Docker Compose hardcodes 'minivess-mlflow' service name directly."
    )


def test_env_example_removed_mlflow_skypilot_host() -> None:
    """MLFLOW_SKYPILOT_HOST must NOT be in .env.example.

    Removed in single-MLFLOW_TRACKING_URI simplification. SkyPilot YAMLs
    set MLFLOW_TRACKING_URI directly (file-based for RunPod, Cloud Run for GCP).
    """
    vars_ = _env_example_vars()
    assert "MLFLOW_SKYPILOT_HOST" not in vars_, (
        "MLFLOW_SKYPILOT_HOST should be removed from .env.example — "
        "SkyPilot YAMLs set MLFLOW_TRACKING_URI directly."
    )


def test_env_example_has_per_service_db_names() -> None:
    """Each service database name must be independently defined."""
    vars_ = _env_example_vars()
    for var in ("POSTGRES_DB_MLFLOW", "POSTGRES_DB_PREFECT", "POSTGRES_DB_LANGFUSE"):
        assert var in vars_, f"{var} missing from .env.example"


def test_env_example_has_bentoml_home() -> None:
    vars_ = _env_example_vars()
    assert "BENTOML_HOME" in vars_


def test_env_example_has_dvc_s3_vars() -> None:
    """DVC S3 cloud storage vars must be in .env.example (#630, T0.1)."""
    vars_ = _env_example_vars()
    for var in (
        "DVC_S3_ENDPOINT_URL",
        "DVC_S3_ACCESS_KEY",
        "DVC_S3_SECRET_KEY",
        "DVC_S3_BUCKET",
        "DVC_REMOTE",
    ):
        assert var in vars_, f"{var} missing from .env.example"


def test_env_example_has_mlflow_server_version() -> None:
    """MLFLOW_SERVER_VERSION must be pinned in .env.example.

    This prevents the v2.20 vs v3.10 mismatch that cost 8+ hours of debugging.
    Both Pulumi stacks (UpCloud + GCP) must reference this variable instead of
    hardcoding the MLflow Docker image version.
    See: .claude/metalearning/2026-03-14-mlflow-version-mismatch-fuckup.md
    """
    vars_ = _env_example_vars()
    assert "MLFLOW_SERVER_VERSION" in vars_, (
        "MLFLOW_SERVER_VERSION missing from .env.example — this caused the v2.20 vs v3.10 "
        "mismatch that broke artifact uploads. Pin it as single source of truth."
    )


def test_env_example_has_gcp_vars() -> None:
    """GCP project and region must be defined in .env.example.

    GCP is the primary cloud for staging+prod. All GCP config goes through
    .env.example as single source of truth (CLAUDE.md Rule #22).
    """
    vars_ = _env_example_vars()
    for var in (
        "GCP_PROJECT",
        "GCP_REGION",
    ):
        assert var in vars_, f"{var} missing from .env.example"


def test_env_example_has_gcs_bucket_vars() -> None:
    """GCS bucket names must be defined in .env.example.

    Three buckets: MLflow artifacts, DVC data, checkpoints.
    All in the same region for same-region artifact uploads.
    """
    vars_ = _env_example_vars()
    for var in (
        "GCS_MLFLOW_BUCKET",
        "GCS_DVC_BUCKET",
        "GCS_CHECKPOINT_BUCKET",
    ):
        assert var in vars_, f"{var} missing from .env.example"


# ---------------------------------------------------------------------------
# Dockerfiles: no ENV MLFLOW_TRACKING_URI
# ---------------------------------------------------------------------------


def _collect_dockerfiles() -> list[Path]:
    return list((ROOT / "deployment" / "docker").glob("Dockerfile*"))


def test_dockerfiles_do_not_hardcode_mlflow_tracking_uri() -> None:
    """Dockerfiles must NOT contain ENV MLFLOW_TRACKING_URI.

    The tracking URI is injected at runtime by docker-compose x-common-env,
    not baked into the image. (CLAUDE.md Rule #22)
    """
    for df in _collect_dockerfiles():
        content = _read(df)
        for line in content.splitlines():
            stripped = line.strip()
            assert not (
                stripped.startswith("ENV") and "MLFLOW_TRACKING_URI" in stripped
            ), (
                f"{df.name} line '{stripped}': MLFLOW_TRACKING_URI must not be in a Dockerfile ENV. "
                "Set it in docker-compose.flows.yml x-common-env."
            )


# ---------------------------------------------------------------------------
# Dynaconf TOMLs: no mlflow_tracking_uri key
# ---------------------------------------------------------------------------


def _collect_toml_files() -> list[Path]:
    return list((ROOT / "configs" / "deployment").glob("*.toml"))


def test_dynaconf_tomls_do_not_define_mlflow_tracking_uri() -> None:
    """Dynaconf TOML files must NOT define mlflow_tracking_uri.

    The MLflow URI comes from the MLFLOW_TRACKING_URI env var read by
    resolve_tracking_uri(). A TOML entry creates a hidden second source. (Rule #22)
    """
    for toml_path in _collect_toml_files():
        content = _read(toml_path)
        for line in content.splitlines():
            stripped = line.strip()
            # Skip commented-out lines
            if stripped.startswith("#"):
                continue
            assert "mlflow_tracking_uri" not in stripped.lower(), (
                f"{toml_path.name}: 'mlflow_tracking_uri' must not be defined in Dynaconf TOML. "
                "Use MLFLOW_TRACKING_URI env var (.env.example)."
            )


# ---------------------------------------------------------------------------
# docker-compose.flows.yml: x-common-env uses variable substitution
# ---------------------------------------------------------------------------


def test_flows_compose_mlflow_uri_uses_hardcoded_service_name() -> None:
    """docker-compose.flows.yml MLFLOW_TRACKING_URI must use minivess-mlflow service name.

    After the single-MLFLOW_TRACKING_URI simplification, Docker Compose hardcodes
    the service name directly: http://minivess-mlflow:${MLFLOW_PORT:-5000}.
    No need for ${MLFLOW_DOCKER_HOST} indirection.
    """
    compose = _read(ROOT / "deployment" / "docker-compose.flows.yml")
    for line in compose.splitlines():
        if "MLFLOW_TRACKING_URI" in line and not line.strip().startswith("#"):
            assert "minivess-mlflow" in line, (
                f"MLFLOW_TRACKING_URI in docker-compose.flows.yml: '{line.strip()}'. "
                "Must use hardcoded service name: http://minivess-mlflow:${MLFLOW_PORT:-5000}."
            )


def test_flows_compose_prefect_url_uses_substitution() -> None:
    """docker-compose.flows.yml PREFECT_API_URL must use ${PREFECT_DOCKER_HOST}."""
    compose = _read(ROOT / "deployment" / "docker-compose.flows.yml")
    for line in compose.splitlines():
        if "PREFECT_API_URL" in line and not line.strip().startswith("#"):
            assert "${PREFECT_DOCKER_HOST" in line, (
                f"Hardcoded PREFECT_API_URL in docker-compose.flows.yml: '{line.strip()}'. "
                "Must use ${PREFECT_DOCKER_HOST:-minivess-prefect}."
            )


def test_flows_compose_minio_endpoint_uses_substitution() -> None:
    """docker-compose.flows.yml MLFLOW_S3_ENDPOINT_URL must use ${MINIO_DOCKER_HOST}."""
    compose = _read(ROOT / "deployment" / "docker-compose.flows.yml")
    for line in compose.splitlines():
        if "MLFLOW_S3_ENDPOINT_URL" in line and not line.strip().startswith("#"):
            assert "${MINIO_DOCKER_HOST" in line, (
                f"Hardcoded MLFLOW_S3_ENDPOINT_URL in docker-compose.flows.yml: '{line.strip()}'. "
                "Must use ${MINIO_DOCKER_HOST:-minio}."
            )


# ---------------------------------------------------------------------------
# Python flow files: no hardcoded os.environ.get("MLFLOW_TRACKING_URI", ...) fallback
# ---------------------------------------------------------------------------


_FLOW_FILES = [
    "train_flow",
    "data_flow",
    "analysis_flow",
    "deploy_flow",
    "acquisition_flow",
    "dashboard_flow",
]


def _get_flow_path(name: str) -> Path:
    return ROOT / "src" / "minivess" / "orchestration" / "flows" / f"{name}.py"


def test_flow_files_do_not_hardcode_mlflow_tracking_uri_fallback() -> None:
    """Flow .py files must not use os.environ.get('MLFLOW_TRACKING_URI', 'mlruns').

    Use resolve_tracking_uri() instead. (CLAUDE.md Rule #22)
    """
    banned = 'os.environ.get("MLFLOW_TRACKING_URI"'
    for name in _FLOW_FILES:
        path = _get_flow_path(name)
        if not path.exists():
            continue
        content = _read(path)
        assert banned not in content, (
            f"{name}.py: found banned pattern '{banned}'. "
            "Use 'from minivess.observability.tracking import resolve_tracking_uri' "
            "and call resolve_tracking_uri() instead."
        )


def test_flow_files_import_resolve_tracking_uri() -> None:
    """Flow files that call resolve_tracking_uri() must import it at module level.

    Ensures the import is not buried inside a function (would fail at call time).
    """
    for name in _FLOW_FILES:
        path = _get_flow_path(name)
        if not path.exists():
            continue
        tree = ast.parse(_read(path), filename=str(path))
        # Find all module-level ImportFrom nodes
        module_imports: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    module_imports.add(alias.name)
        # Only check files that actually call resolve_tracking_uri()
        if "resolve_tracking_uri()" in _read(path):
            assert "resolve_tracking_uri" in module_imports, (
                f"{name}.py calls resolve_tracking_uri() but does not import it "
                "at module level. Add: "
                "'from minivess.observability.tracking import resolve_tracking_uri'"
            )


# ---------------------------------------------------------------------------
# .gitignore: mlruns-docker is ignored
# ---------------------------------------------------------------------------


def test_gitignore_includes_mlruns_docker() -> None:
    """mlruns-docker/ must be gitignored — it is a transient Docker artifact."""
    gitignore = _read(ROOT / ".gitignore")
    assert "mlruns-docker" in gitignore, (
        ".gitignore missing 'mlruns-docker' entry. "
        "Add '/mlruns-docker/*' to prevent committing Docker bind-mount artifacts."
    )
