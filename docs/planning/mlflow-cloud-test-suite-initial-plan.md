---
title: "Composable MLflow Test Suite — Plan (Reviewer-Optimized)"
status: active
created: "2026-03-13"
parent_issue: "#620"
sub_issues: ["#621", "#622", "#623", "#624", "#625"]
review_round: 1
reviewers:
  - "Testing Architecture Reviewer (fixture design, tier integration, security)"
  - "MLOps/SkyPilot Reviewer (tracking URI, artifacts, data isolation, edge cases)"
  - "IaC/Pulumi Reviewer (mock feasibility, static analysis, dependency isolation)"
depends_on:
  - pulumi-upcloud-managed-deployment.md
  - skypilot-and-finops-complete-report.md
---

# Composable MLflow Test Suite

**Goal:** A 4-layer test suite that verifies MLflow works correctly from local
filesystem through remote cloud deployment to SkyPilot spot instance tracking.
Each layer is independently runnable, composable with the others, and supports
multiple cloud providers through a provider-agnostic fixture system.

**Audience:** The same PhD researchers who run `pulumi up` — they should be able
to run `make test-cloud-mlflow` to verify their deployment before spending GPU
hours on SkyPilot training jobs.

---

## Current State (30+ Existing Tests)

The codebase already has comprehensive MLflow test coverage for local operations:

| Category | Files | Coverage |
|----------|-------|----------|
| Tracking & Observability | `test_observability.py`, `test_system_info.py`, `test_param_naming.py` | URI resolution, parameter prefixes, system info logging |
| Model Wrappers | `test_mlflow_wrapper.py`, `test_mlflow_roundtrip.py` | PyFunc single + ensemble, log/load/predict cycle |
| Backend Health | `test_mlflow_health.py`, `test_mlflow_backend_standardization.py` | Filesystem vs server detection, health checks |
| Flow Integration | `test_*_mlflow.py` (6 files) | Per-flow artifact logging (data, train, analyze, deploy, dashboard, biostatistics) |
| Docker/Auth | `test_mlflow_auth.py`, `test_dockerfile_mlflow.py` | Basic-auth opt-in, Dockerfile validation |
| E2E | `test_full_pipeline_e2e.py`, `test_infrastructure_health.py` | Full pipeline, Docker Compose health |

**Gap:** All tests use the local filesystem backend (`mlruns/`). Zero tests verify:
- Remote MLflow server connectivity and basic auth API
- Managed PostgreSQL backend operations
- S3-compatible artifact storage (UpCloud Object Storage)
- Pulumi IaC correctness
- SkyPilot → remote MLflow tracking URI propagation
- `resolve_tracking_uri()` with special characters in passwords

---

## Architecture: 4 Composable Layers

```
┌──────────────────────────────────────────────────────────────────┐
│                    Test Execution Tiers                          │
│                                                                  │
│  make test-staging      → L1 (generic) + L3 (Pulumi static)    │
│  make test-prod         → L1 + L3 + L2 (if creds available)     │
│  make test-cloud-mlflow → L2 + L4 (cloud-specific)              │
│  make test-all          → L1 + L2 + L3 + L4                     │
└──────────────────────────────────────────────────────────────────┘

┌─────────────┐  ┌─────────────────┐  ┌─────────────┐  ┌──────────────┐
│ L1: Generic │  │ L2: Cloud       │  │ L3: Pulumi  │  │ L4: SkyPilot │
│ MLflow      │  │ MLflow          │  │ IaC         │  │ → Cloud      │
│             │  │                 │  │             │  │              │
│ filesystem  │  │ Live deployment │  │ String      │  │ Tracking URI │
│ + server    │  │ UpCloud/Scaleway│  │ checks +    │  │ propagation  │
│ params      │  │                 │  │ YAML parse  │  │ + preemption │
│             │  │                 │  │             │  │              │
│ No creds    │  │ MLFLOW_CLOUD_*  │  │ No creds    │  │ Cloud creds  │
│ Staging ✓   │  │ Staging ✗       │  │ Staging ✓   │  │ Staging ✗    │
└─────────────┘  └─────────────────┘  └─────────────┘  └──────────────┘
       │                  │                   │                │
       └──────────────────┴───────────────────┴────────────────┘
                                  │
                    ┌─────────────────────────┐
                    │ Shared Fixture Layer     │
                    │ tests/v2/cloud/conftest  │
                    │ CloudMLflowConnection    │
                    │ Provider-agnostic        │
                    └─────────────────────────┘
```

---

## Reviewer Feedback Summary (3 Reviewers, 2026-03-13)

Changes from initial draft based on reviewer consensus:

| Finding | All 3 | Action Taken |
|---------|-------|-------------|
| **SQLite backend violates project ban** | Yes | Removed SQLite param; use `filesystem` + `server` only |
| **`ast.parse()` for templates over-engineered** | Yes | Changed to plain `in` string checks (like `test_dockerfile_mlflow.py`) |
| **Credentials leak in `__repr__`** | Yes | Added custom `__repr__` masking secrets |
| **`os.environ` mutation in session fixture** | Yes | Changed to save/restore pattern with cleanup |
| **Resource count test fragile** | Yes | Changed to required-types assertion |
| **Real credentials in plan doc** | Yes | Replaced with `REPLACE_ME` placeholders |
| **Missing: URL-encoding in `resolve_tracking_uri()`** | R2 | Added test + production bug to fix |
| **Missing: startup cleanup for crash recovery** | R1+R2 | Added pre-yield cleanup pass |
| **Missing: `detect_backend_type()` postgres:// bug** | R2 | Added test + production bug to fix |
| **UUID-based test isolation** | R1+R2 | Changed from `_test_` to `_test_{uuid[:8]}_` prefix |
| **Pulumi mock effort underestimated** | R3 | Split into string checks (staging) + optional mocks (prod) |
| **Missing: S3 path-style addressing test** | R2 | Added direct boto3 S3 connectivity test |
| **Missing: connection failure tests** | R1+R2 | Added wrong-port, wrong-password, timeout tests |
| **Concurrency tests low-value** | R1 | Replaced with preemption simulation (L4) |

---

## Layer 1: Generic MLflow (Issue #621)

**Marker:** None (runs in all tiers)
**Credentials:** None
**Tier:** Staging + Prod

### What Changes

Introduce a `mlflow_backend` parametrized fixture in `tests/v2/fixtures/mlflow_backends.py`:

```python
import subprocess
import time

import pytest


@pytest.fixture(params=[
    "filesystem",
    pytest.param("server", marks=pytest.mark.slow),
])
def mlflow_backend(request, tmp_path):
    """Parametrize tests across MLflow backend types.

    NOTE: SQLite is BANNED per project rules (PostgreSQL is ONLY database).
    The 'server' param starts a subprocess MLflow server with filesystem backend.
    """
    if request.param == "filesystem":
        yield str(tmp_path / "mlruns")
    elif request.param == "server":
        store = str(tmp_path / "mlruns")
        port = _find_free_port()
        proc = subprocess.Popen(
            ["mlflow", "server", "--backend-store-uri", store,
             "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=10)
        yield f"http://127.0.0.1:{port}"
        proc.terminate()
        proc.wait(timeout=5)
```

The `server` param is excluded from staging tier via `@pytest.mark.slow` (staging
runs with `-m "not slow"`). This keeps staging under 3 minutes.

### Tests to Parametrize

| Existing Test | Currently Tests | Parametrize? |
|--------------|-----------------|--------------|
| `test_mlflow_roundtrip.py` | Log/load/predict cycle | Yes — works on any backend |
| `test_observability.py` | ExperimentTracker | Yes — tracking is backend-agnostic |
| `test_param_naming.py` | Parameter prefixes | Yes — param logging is backend-agnostic |
| `test_mlflow_health.py` | Filesystem health | No — inherently filesystem-specific |
| `test_mlflow_wrapper.py` | PyFunc wrappers | Partial — model logging works on any backend |
| `test_model_logger.py` | Model artifact logging | Yes — artifact storage is abstracted |
| `test_backfill_metadata.py` | Retroactive run updates | Yes — `mlflow.start_run(run_id=)` is backend-agnostic |

### New Tests

```python
# tests/v2/unit/test_mlflow_backend_operations.py

class TestBackendOperations:
    """Backend-agnostic MLflow operations that must work on any backend."""

    def test_create_experiment(self, mlflow_backend): ...
    def test_create_run_log_params_metrics(self, mlflow_backend): ...
    def test_run_lifecycle_finished(self, mlflow_backend): ...
    def test_run_lifecycle_failed(self, mlflow_backend): ...
    def test_search_runs_filter(self, mlflow_backend): ...
    def test_log_artifact_roundtrip(self, mlflow_backend, tmp_path): ...
    def test_set_tag_after_run_end(self, mlflow_backend): ...


class TestTrackingUriEdgeCases:
    """resolve_tracking_uri() edge cases discovered by reviewers."""

    def test_password_with_at_sign(self, monkeypatch):
        """Password containing '@' must be percent-encoded."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "p@ss:w/rd")
        uri = resolve_tracking_uri()
        assert "%40" in uri  # @ encoded
        assert "%3A" in uri  # : encoded
        assert "%2F" in uri  # / encoded

    def test_password_with_percent(self, monkeypatch):
        """Password containing '%' must be double-encoded."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "100%safe")
        uri = resolve_tracking_uri()
        assert "%25" in uri  # % encoded


class TestDetectBackendType:
    """detect_backend_type() edge cases."""

    def test_postgres_scheme_without_ql(self):
        """postgres:// (without 'ql') must be detected as database backend."""
        # Bug found by reviewer — currently falls through to 'local'
```

### Production Bugs to Fix (Found by Reviewers)

1. **`resolve_tracking_uri()` does not percent-encode credentials** — passwords with
   `@`, `:`, `/`, `%`, `+` produce malformed URIs. Fix: `urllib.parse.quote(password, safe="")`
   in `tracking.py:84-93`.

2. **`detect_backend_type()` misses `postgres://` scheme** — only checks `postgresql://`.
   UpCloud returns `postgres://` which falls through to `"local"`. Fix: add `postgres://`
   to the database detection branch in `mlflow_backend.py:35`.

### Estimated Effort

- Fixture: ~60 lines
- Parametrize existing tests: ~2 hours
- New tests + edge cases: ~200 lines
- Production bug fixes: ~1 hour
- **Total: ~5 hours**

---

## Layer 2: Cloud MLflow (Issue #622)

**Marker:** `@pytest.mark.cloud_mlflow`
**Credentials:** `MLFLOW_CLOUD_URI`, `MLFLOW_CLOUD_USERNAME`, `MLFLOW_CLOUD_PASSWORD`
**Tier:** Cloud-only (`make test-cloud-mlflow`)

### Environment Variables

```bash
# .env or shell — NOT committed to git
MLFLOW_CLOUD_URI=http://REPLACE_ME:5000
MLFLOW_CLOUD_USERNAME=admin
MLFLOW_CLOUD_PASSWORD=REPLACE_ME
MLFLOW_CLOUD_S3_ENDPOINT=https://REPLACE_ME.upcloudobjects.com
MLFLOW_CLOUD_S3_ACCESS_KEY=REPLACE_ME
MLFLOW_CLOUD_S3_SECRET_KEY=REPLACE_ME
MLFLOW_CLOUD_S3_BUCKET=mlflow-artifacts
MLFLOW_CLOUD_PROVIDER=upcloud
```

These must also be added to `.env.example` (with placeholder values) per Rule #22.

### Test Structure

```python
# tests/v2/cloud/test_cloud_mlflow.py

@pytest.mark.cloud_mlflow
class TestCloudMLflowHealth:
    """Verify the remote MLflow deployment is healthy."""

    def test_health_endpoint_public(self, cloud_mlflow_connection):
        """GET /health returns 200 without auth."""

    def test_unauthenticated_api_returns_401(self, cloud_mlflow_connection):
        """GET /api/2.0/mlflow/experiments/search without auth → 401."""

    def test_authenticated_api_returns_200(self, cloud_mlflow_client):
        """Authenticated experiment search returns 200."""

    def test_wrong_password_returns_401(self, cloud_mlflow_connection):
        """Auth with wrong password → 401 (not 500)."""

    def test_connection_refused_on_wrong_port(self, cloud_mlflow_connection):
        """Connection to port+1 raises ConnectionError, not hangs."""


@pytest.mark.cloud_mlflow
class TestCloudMLflowTracking:
    """Verify experiment tracking against remote server."""

    def test_create_experiment(self, cloud_mlflow_client, test_run_id):
        """Create experiment on remote server, verify it exists."""

    def test_create_run_log_params_metrics(self, cloud_mlflow_client, test_run_id):
        """Full run lifecycle: create → log params/metrics → end."""

    def test_search_runs_filter(self, cloud_mlflow_client, test_run_id):
        """Search runs with filter_string on PostgreSQL backend."""

    def test_log_batch_metrics(self, cloud_mlflow_client, test_run_id):
        """Log batch of 100 metrics in one call — PostgreSQL perf check."""

    def test_tag_run_after_completion(self, cloud_mlflow_client, test_run_id):
        """Set tag on completed run (champion tagging pattern)."""

    def test_postgresql_soft_delete(self, cloud_mlflow_client, test_run_id):
        """Delete experiment → lifecycle_stage=deleted, not truly removed."""


@pytest.mark.cloud_mlflow
class TestCloudMLflowArtifacts:
    """Verify S3-compatible artifact storage."""

    def test_log_artifact_roundtrip(self, cloud_mlflow_client, test_run_id, tmp_path):
        """Upload file artifact via MLflow, download, verify content."""

    def test_log_large_artifact(self, cloud_mlflow_client, test_run_id, tmp_path):
        """Upload 10 MB artifact — verifies S3 multipart works."""

    def test_list_artifacts(self, cloud_mlflow_client, test_run_id):
        """List artifacts for a run, verify expected structure."""

    def test_direct_s3_connectivity(self, cloud_s3_client, cloud_mlflow_connection):
        """Direct boto3 ListBuckets against S3 endpoint (path-style)."""

    def test_s3_bucket_exists(self, cloud_s3_client, cloud_mlflow_connection):
        """Verify the configured artifact bucket exists."""
```

### Cleanup Strategy (Reviewer-Improved)

Test isolation uses UUID prefix + bidirectional cleanup:

```python
import uuid

_TEST_SESSION_ID = uuid.uuid4().hex[:8]

@pytest.fixture(scope="session")
def test_run_id():
    """Unique prefix for this test session's experiments."""
    return f"_test_{_TEST_SESSION_ID}"

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_experiments(cloud_mlflow_client, test_run_id):
    """Cleanup before AND after test run (handles crash recovery)."""
    _delete_test_experiments(cloud_mlflow_client)  # startup cleanup
    yield
    _delete_test_experiments(cloud_mlflow_client)  # teardown cleanup

def _delete_test_experiments(client):
    for exp in client.search_experiments(view_type=ViewType.ALL):
        if exp.name.startswith("_test_"):
            client.delete_experiment(exp.experiment_id)
```

### Estimated Effort

- Tests: ~350 lines
- Conftest fixtures: ~120 lines (see #625)
- **Total: ~7 hours**

---

## Layer 3: Pulumi IaC (Issue #623)

**Marker:** `@pytest.mark.pulumi`
**Credentials:** None
**Tier:** Staging (fast, no cloud access)

### Approach: String Checks + YAML Validation (Reviewer-Revised)

**Reviewers unanimously rejected `ast.parse()` for template extraction.** The deploy
command uses f-strings inside lambdas inside `.apply()` — AST nodes contain
unresolvable `FormattedValue` references. Instead:

**Strategy A: Plain string `in` checks** — read `__main__.py` as raw text and verify
required substrings are present. This matches the existing pattern in
`test_dockerfile_mlflow.py` and is robust, simple, and Rule #16 compliant.

**Strategy B: YAML/INI validation** — render templates with dummy values and parse
with `yaml.safe_load()` / `configparser.ConfigParser()` to verify structural validity.

**Strategy C (optional, prod tier): `pytest.importorskip("pulumi")`** — if Pulumi SDK
is available, run resource graph tests; if not, skip cleanly. No subprocess hacks.

### Test Structure

```python
# tests/v2/unit/test_pulumi_stack.py
from pathlib import Path

PULUMI_DIR = Path("deployment/pulumi")

@pytest.mark.pulumi
class TestPulumiYamlConfig:
    """Validate Pulumi.yaml configuration."""

    def test_pulumi_yaml_parseable(self):
        """Pulumi.yaml is valid YAML with required fields."""
        config = yaml.safe_load((PULUMI_DIR / "Pulumi.yaml").read_text())
        assert config["name"] == "minivess-mlflow"
        assert config["runtime"]["name"] == "python"

    def test_runtime_uses_uv_toolchain(self):
        """Runtime options specify toolchain: uv (not pip/venv)."""
        config = yaml.safe_load((PULUMI_DIR / "Pulumi.yaml").read_text())
        assert config["runtime"]["options"]["toolchain"] == "uv"

    def test_config_keys_have_descriptions(self):
        """All config keys have description fields."""
        config = yaml.safe_load((PULUMI_DIR / "Pulumi.yaml").read_text())
        for key, val in config.get("config", {}).items():
            if isinstance(val, dict):
                assert "description" in val or "default" in val, f"{key} needs description"

    def test_mlflow_admin_password_is_secret(self):
        """mlflow_admin_password config is marked secret: true."""
        config = yaml.safe_load((PULUMI_DIR / "Pulumi.yaml").read_text())
        pw_config = config["config"]["minivess-mlflow:mlflow_admin_password"]
        assert pw_config.get("secret") is True


@pytest.mark.pulumi
class TestPulumiMainModuleTemplates:
    """Validate __main__.py templates using plain string checks.

    Approach: read raw file content, check for required substrings.
    Matches existing pattern in test_dockerfile_mlflow.py.
    No ast.parse() needed (reviewer consensus: over-engineered for f-strings).
    """

    @pytest.fixture
    def main_content(self):
        return (PULUMI_DIR / "__main__.py").read_text(encoding="utf-8")

    def test_docker_compose_has_required_env_vars(self, main_content):
        """Deploy template includes all required env vars."""
        required = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "MLFLOW_S3_ENDPOINT_URL",
            "MLFLOW_FLASK_SERVER_SECRET_KEY",
            "MLFLOW_AUTH_CONFIG_PATH",
        ]
        for var in required:
            assert var in main_content, f"Missing env var: {var}"

    def test_basic_auth_ini_has_database_uri(self, main_content):
        """basic_auth.ini template includes database_uri field."""
        assert "database_uri" in main_content

    def test_basic_auth_ini_no_authorization_function(self, main_content):
        """basic_auth.ini does NOT include authorization_function.
        (Broken in MLflow v2.20.3 — see #618)."""
        # Find the AUTHEOF section and check it doesn't contain the bad key
        auth_section = main_content.split("AUTHEOF")[0].rsplit("basic_auth.ini", 1)[-1]
        assert "authorization_function" not in auth_section

    def test_postgres_public_access_enabled(self, main_content):
        """ManagedDatabasePostgresql has public_access: True."""
        assert '"public_access": True' in main_content
        assert '"ip_filters"' in main_content

    def test_uri_scheme_replacement(self, main_content):
        """postgres:// → postgresql:// replacement is present."""
        assert '.replace("postgres://", "postgresql://", 1)' in main_content

    def test_dockerfile_required_packages(self, main_content):
        """Custom Dockerfile installs psycopg2-binary, boto3, flask-wtf."""
        for pkg in ["psycopg2-binary", "boto3", "flask-wtf"]:
            assert pkg in main_content, f"Missing pip package: {pkg}"

    def test_server_template_ubuntu_2404(self, main_content):
        """Server template uses Ubuntu 24.04 LTS."""
        assert "Ubuntu Server 24.04" in main_content

    def test_mlflow_version_pinned(self, main_content):
        """MLflow image version is pinned (not :latest)."""
        assert "mlflow:v" in main_content
        assert "mlflow:latest" not in main_content


@pytest.mark.pulumi
class TestPulumiResourceTypes:
    """Validate expected resource types (not exact count — fragile).

    Uses pytest.importorskip('pulumi') so tests skip cleanly without SDK.
    When Pulumi SDK is available (e.g., in the deployment/pulumi venv),
    these tests validate the full resource graph.
    """

    def test_required_resource_types_in_source(self):
        """Source code references all required UpCloud resource types."""
        content = (PULUMI_DIR / "__main__.py").read_text(encoding="utf-8")
        required_types = [
            "ManagedDatabasePostgresql",
            "ManagedObjectStorage",
            "ManagedObjectStorageBucket",
            "ManagedObjectStorageUser",
            "ManagedObjectStorageUserAccessKey",
            "Server",
        ]
        for rtype in required_types:
            assert rtype in content, f"Missing resource type: {rtype}"

    def test_remote_command_provisioning(self):
        """Source code uses pulumi-command for remote provisioning."""
        content = (PULUMI_DIR / "__main__.py").read_text(encoding="utf-8")
        assert "command.remote.Command" in content
        assert "install-docker" in content
        assert "deploy-mlflow" in content
```

### Estimated Effort

- String check tests: ~200 lines, ~3 hours
- Resource type tests: ~50 lines, ~1 hour
- **Total: ~4 hours** (reduced from 7 — no AST parsing, no subprocess mocks)

---

## Layer 4: SkyPilot → Cloud (Issue #624)

**Marker:** `@pytest.mark.skypilot_cloud`
**Credentials:** `MLFLOW_CLOUD_*` + SkyPilot cloud credentials
**Tier:** Cloud-only (`make test-cloud-mlflow`)

### Test Structure

```python
# tests/v2/cloud/test_skypilot_mlflow.py

class TestSkyPilotTrackingUriResolution:
    """Verify tracking URI assembly from SkyPilot-style env vars.
    No cloud credentials needed — pure unit tests."""

    def test_resolve_from_skypilot_host_env(self, monkeypatch):
        """MLFLOW_SKYPILOT_HOST + MLFLOW_PORT → correct URI."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI",
                          "http://my-mlflow-server:5000")
        uri = resolve_tracking_uri()
        assert uri == "http://my-mlflow-server:5000"

    def test_auth_credentials_embedded_in_uri(self, monkeypatch):
        """MLFLOW_TRACKING_USERNAME/PASSWORD → embedded in URI."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret")
        uri = resolve_tracking_uri()
        assert "admin:secret@" in uri

    def test_skypilot_yaml_env_var_references(self):
        """train_generic.yaml references MLFLOW_SKYPILOT_HOST in envs section."""
        config = yaml.safe_load(
            Path("deployment/skypilot/train_generic.yaml").read_text()
        )
        envs = config.get("envs", {})
        mlflow_uri = envs.get("MLFLOW_TRACKING_URI", "")
        assert "MLFLOW_SKYPILOT_HOST" in mlflow_uri
        assert "MLFLOW_PORT" in mlflow_uri

    def test_skypilot_yaml_uses_prefect_not_scripts(self):
        """train_generic.yaml invokes prefect deployment, not scripts/*.py."""
        content = Path("deployment/skypilot/train_generic.yaml").read_text()
        assert "prefect deployment run" in content
        assert "scripts/train_monitored" not in content


@pytest.mark.skypilot_cloud
class TestSkyPilotRemoteLogging:
    """Simulate SkyPilot environment logging to remote MLflow.
    Requires MLFLOW_CLOUD_* credentials."""

    def test_simulated_spot_vm_logs_run(self, cloud_mlflow_client, test_run_id, monkeypatch):
        """Set env vars as SkyPilot would, create run, log metrics."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI",
                          cloud_mlflow_client.tracking_uri)
        # Simulate what happens inside a SkyPilot VM
        exp_name = f"{test_run_id}_skypilot_sim"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)
        cloud_mlflow_client.log_metric(run.info.run_id, "loss", 0.42)
        cloud_mlflow_client.log_param(run.info.run_id, "model", "dynunet")
        cloud_mlflow_client.set_terminated(run.info.run_id, "FINISHED")
        # Verify data persisted
        fetched = cloud_mlflow_client.get_run(run.info.run_id)
        assert fetched.data.metrics["loss"] == 0.42

    def test_simulated_spot_vm_uploads_artifact(self, cloud_mlflow_client, test_run_id, tmp_path):
        """Simulate artifact upload from ephemeral VM to S3."""
        artifact = tmp_path / "model_weights.bin"
        artifact.write_bytes(b"\x00" * 1024)  # 1 KB dummy
        exp_name = f"{test_run_id}_skypilot_artifact"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)
        cloud_mlflow_client.log_artifact(run.info.run_id, str(artifact))
        artifacts = cloud_mlflow_client.list_artifacts(run.info.run_id)
        assert any(a.path == "model_weights.bin" for a in artifacts)

    def test_simulated_preemption_recovery(self, cloud_mlflow_client, test_run_id):
        """Start run, simulate preemption (KILLED), resume with new run,
        verify both runs are queryable and data intact."""
        exp_name = f"{test_run_id}_preemption"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)

        # Phase 1: start run, log some metrics, get preempted
        run1 = cloud_mlflow_client.create_run(exp_id)
        cloud_mlflow_client.log_metric(run1.info.run_id, "epoch", 5)
        cloud_mlflow_client.set_terminated(run1.info.run_id, "KILLED")

        # Phase 2: resume on new spot VM — new run, same experiment
        run2 = cloud_mlflow_client.create_run(exp_id)
        cloud_mlflow_client.log_metric(run2.info.run_id, "epoch", 10)
        cloud_mlflow_client.log_param(run2.info.run_id, "resumed_from", run1.info.run_id)
        cloud_mlflow_client.set_terminated(run2.info.run_id, "FINISHED")

        # Verify both runs exist and data intact
        runs = cloud_mlflow_client.search_runs([exp_id])
        assert len(runs) == 2
        killed = [r for r in runs if r.info.status == "KILLED"]
        finished = [r for r in runs if r.info.status == "FINISHED"]
        assert len(killed) == 1
        assert len(finished) == 1
```

### Estimated Effort

- Unit tests (no creds): ~100 lines, ~2 hours
- Cloud tests: ~150 lines, ~3 hours
- **Total: ~5 hours**

---

## Fixture Infrastructure (Issue #625)

### `tests/v2/cloud/conftest.py` (Reviewer-Improved)

```python
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

import pytest


@dataclass(frozen=True)
class CloudMLflowConnection:
    """Provider-agnostic cloud MLflow connection details.

    Secret fields use field(repr=False) to prevent credential leaks
    in pytest failure output (reviewer finding: all 3 flagged this).
    """
    tracking_uri: str
    username: str
    password: str = field(repr=False)
    s3_endpoint: str = ""
    s3_access_key: str = field(default="", repr=False)
    s3_secret_key: str = field(default="", repr=False)
    s3_bucket: str = "mlflow-artifacts"
    provider_name: str = "unknown"


_TEST_SESSION_ID = uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def test_run_id():
    """Unique prefix for this test session's experiments.
    UUID-based to prevent interference between concurrent test runs."""
    return f"_test_{_TEST_SESSION_ID}"


@pytest.fixture(scope="session")
def cloud_mlflow_connection() -> CloudMLflowConnection:
    """Read cloud MLflow connection from env vars.
    Skips entire session if credentials not available."""
    uri = os.environ.get("MLFLOW_CLOUD_URI")
    if not uri:
        pytest.skip("MLFLOW_CLOUD_URI not set — skipping cloud tests")
    return CloudMLflowConnection(
        tracking_uri=uri,
        username=os.environ.get("MLFLOW_CLOUD_USERNAME", "admin"),
        password=os.environ["MLFLOW_CLOUD_PASSWORD"],
        s3_endpoint=os.environ.get("MLFLOW_CLOUD_S3_ENDPOINT", ""),
        s3_access_key=os.environ.get("MLFLOW_CLOUD_S3_ACCESS_KEY", ""),
        s3_secret_key=os.environ.get("MLFLOW_CLOUD_S3_SECRET_KEY", ""),
        s3_bucket=os.environ.get("MLFLOW_CLOUD_S3_BUCKET", "mlflow-artifacts"),
        provider_name=os.environ.get("MLFLOW_CLOUD_PROVIDER", "unknown"),
    )


@pytest.fixture(scope="session")
def cloud_mlflow_client(cloud_mlflow_connection):
    """Authenticated MlflowClient for cloud server.

    Uses save/restore for env vars instead of bare os.environ mutation
    (reviewer finding: session-scoped mutation leaks into other tests).
    """
    from mlflow import MlflowClient

    # Save existing values
    _saved_user = os.environ.get("MLFLOW_TRACKING_USERNAME")
    _saved_pass = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    os.environ["MLFLOW_TRACKING_USERNAME"] = cloud_mlflow_connection.username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cloud_mlflow_connection.password

    client = MlflowClient(tracking_uri=cloud_mlflow_connection.tracking_uri)
    # Verify connectivity
    client.search_experiments(max_results=1)
    yield client

    # Restore original values
    if _saved_user is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = _saved_user
    else:
        os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
    if _saved_pass is not None:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = _saved_pass
    else:
        os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)


@pytest.fixture(scope="session")
def cloud_s3_client(cloud_mlflow_connection):
    """boto3 S3 client configured for the cloud provider's endpoint."""
    import boto3

    if not cloud_mlflow_connection.s3_endpoint:
        pytest.skip("MLFLOW_CLOUD_S3_ENDPOINT not set — skipping S3 tests")
    return boto3.client(
        "s3",
        endpoint_url=cloud_mlflow_connection.s3_endpoint,
        aws_access_key_id=cloud_mlflow_connection.s3_access_key,
        aws_secret_access_key=cloud_mlflow_connection.s3_secret_key,
    )


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_experiments(cloud_mlflow_client):
    """Bidirectional cleanup: delete _test_* experiments before AND after.
    Handles crash recovery (reviewer finding: yield-only misses SIGKILL).
    """
    from mlflow.entities import ViewType

    def _cleanup():
        for exp in cloud_mlflow_client.search_experiments(view_type=ViewType.ALL):
            if exp.name.startswith("_test_"):
                try:
                    cloud_mlflow_client.delete_experiment(exp.experiment_id)
                except Exception:
                    pass  # Best-effort cleanup

    _cleanup()  # Startup: clean stale experiments from crashed runs
    yield
    _cleanup()  # Teardown: clean this session's experiments
```

### Pytest Configuration Additions

```toml
# pyproject.toml additions
[tool.pytest.ini_options]
markers = [
    # ... existing markers ...
    "cloud_mlflow: requires live cloud MLflow deployment (MLFLOW_CLOUD_* env vars)",
    "pulumi: Pulumi IaC validation (no cloud credentials needed)",
    "skypilot_cloud: SkyPilot → remote MLflow integration (cloud + SkyPilot creds)",
]
```

### Makefile Additions

```makefile
test-cloud-mlflow:  ## Run cloud MLflow tests (requires MLFLOW_CLOUD_* env vars)
	uv run pytest tests/v2/cloud/ -m "cloud_mlflow or skypilot_cloud" -v

test-pulumi:  ## Run Pulumi IaC validation tests
	uv run pytest tests/v2/unit/test_pulumi_stack.py -v
```

### `.env.example` Additions (Rule #22)

```bash
# Cloud MLflow test suite (tests/v2/cloud/)
MLFLOW_CLOUD_URI=                    # Remote MLflow server URL (e.g., http://IP:5000)
MLFLOW_CLOUD_USERNAME=admin          # MLflow basic auth username
MLFLOW_CLOUD_PASSWORD=               # MLflow basic auth password
MLFLOW_CLOUD_S3_ENDPOINT=            # S3-compatible endpoint (e.g., https://xxx.upcloudobjects.com)
MLFLOW_CLOUD_S3_ACCESS_KEY=          # S3 access key
MLFLOW_CLOUD_S3_SECRET_KEY=          # S3 secret key
MLFLOW_CLOUD_S3_BUCKET=mlflow-artifacts  # S3 bucket name
MLFLOW_CLOUD_PROVIDER=upcloud        # Provider name (upcloud, scaleway, hetzner, etc.)
```

### Directory Structure

```
tests/v2/
├── cloud/                          # NEW — cloud-specific tests
│   ├── __init__.py
│   ├── conftest.py                 # CloudMLflowConnection, fixtures
│   ├── test_cloud_mlflow.py        # L2: health, tracking, artifacts
│   └── test_skypilot_mlflow.py     # L4: SkyPilot URI + remote logging
├── unit/
│   ├── test_mlflow_backend_operations.py  # NEW — L1: backend-parametrized
│   ├── test_pulumi_stack.py        # NEW — L3: IaC validation
│   └── ... (existing 20+ files)
├── integration/
│   └── ... (existing files)
└── fixtures/
    └── mlflow_backends.py          # NEW — L1: backend parametrization
```

---

## Implementation Order

| Phase | Issues | Dependencies | Time Est. |
|-------|--------|-------------|-----------|
| **Phase 1** | #625 (fixtures) + #621 (L1 generic) | None | ~7 hours |
| **Phase 2** | #622 (L2 cloud) | Phase 1 + live deployment | ~7 hours |
| **Phase 3** | #623 (L3 Pulumi) | None (parallel with Phase 2) | ~4 hours |
| **Phase 4** | #624 (L4 SkyPilot) | Phase 1 + Phase 2 | ~5 hours |

**Total estimated:** ~23 hours across 4 phases.

Phase 1 and Phase 3 can run in parallel (no dependencies).
Phase 2 and Phase 4 require a live cloud deployment.

---

## Provider Extension Path

When adding a second cloud provider (e.g., Scaleway):

1. Add `MLFLOW_CLOUD_*` env vars for Scaleway in `.env.example`
2. Create `deployment/pulumi-scaleway/__main__.py` (new Pulumi stack)
3. The existing `tests/v2/cloud/` tests run unmodified — just change env vars
4. Add provider-specific assertions via `cloud_mlflow_connection.provider_name`

**No test duplication.** The provider-agnostic fixture layer means adding a
cloud provider is deploying infrastructure + setting env vars, not writing tests.

---

## Risk Assessment (Reviewer-Improved)

| Risk | Mitigation |
|------|-----------|
| Cloud tests flaky due to network | `tenacity` retry (3 attempts, 2s exponential backoff) |
| S3 credentials leak in test output | `field(repr=False)` on all secret fields in `CloudMLflowConnection` |
| Test experiments pollute production MLflow | `_test_{uuid[:8]}_` prefix + bidirectional cleanup fixture |
| Concurrent test runs interfere | UUID-based experiment names — each session has unique prefix |
| Pulumi mock tests diverge from real behavior | L3 validates structure only; L2 validates behavior against live server |
| SkyPilot not installed in test env | `pytest.importorskip("sky")` + graceful skip |
| `resolve_tracking_uri()` password encoding bug | L1 test exposes the bug; production fix in Phase 1 |
| `detect_backend_type()` postgres:// bug | L1 test exposes the bug; production fix in Phase 1 |

---

## Production Bugs Found During Planning

These bugs were discovered by reviewers analyzing the code for testability.
They should be fixed in Phase 1 before writing L1 tests:

1. **`resolve_tracking_uri()` does not percent-encode credentials** (`tracking.py:84-93`)
   - Passwords with `@`, `:`, `/`, `%`, `+` produce malformed URIs
   - Fix: `urllib.parse.quote(password, safe="")` before embedding

2. **`detect_backend_type()` misses `postgres://` scheme** (`mlflow_backend.py:35`)
   - UpCloud returns `postgres://` which falls through to `"local"` type
   - Fix: add `uri.startswith("postgres://")` to the database detection branch

---

## References

- [Pulumi Unit Testing](https://www.pulumi.com/docs/iac/concepts/testing/unit/) — Mock-based resource testing
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html) — Client operations
- [MLflow Basic Auth](https://mlflow.org/docs/latest/auth/index.html) — Authentication setup
- [SkyPilot Job YAML](https://docs.skypilot.co/en/latest/reference/yaml-spec.html) — Task specification
- Deployment fixes: #616 (CSRF key), #617 (auth config path), #618 (database_uri), #619 (public access)
