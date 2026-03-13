---
title: "S3 Mounting, Testing & Simulation Plan"
status: active
priority: P0
created: "2026-03-13"
depends_on:
  - pulumi-upcloud-managed-deployment.md
  - cloud-tutorial.md
  - upcloud-runpod-skypilot-mlflow-integration-testing.xml
related_issues: []
---

# S3 Mounting, Testing & Simulation Plan

## Abstract

This document presents a comprehensive engineering plan for S3-compatible object storage
provisioning, multi-environment access, and local simulation within the MinIVess MLOps
platform. The plan addresses six interconnected architectural decisions: (D1) Pulumi
resource abstraction strategy, (D2) credential injection mechanisms, (D3) local S3
simulation approaches, (D4) test isolation for three distinct execution environments,
(D5) unified vs. separate S3 backends for MLflow and DVC, and (D6) multi-cloud provider
extensibility. Each decision is evaluated via a multi-hypothesis decision matrix with
explicit scoring criteria derived from the project's design goals: zero-config start
for PhD researchers, Docker-per-flow isolation, and MONAI ecosystem extensibility.

The plan builds on the existing Pulumi stack (`deployment/pulumi/__main__.py`) which
already provisions UpCloud Managed Object Storage with two buckets (`mlflow-artifacts`
and `minivess-dvc-data`), and the existing DVC remote configuration (`.dvc/config`)
which defines both local MinIO and cloud UpCloud remotes. The proposed architecture
introduces a `Pulumi ComponentResource` abstraction (`S3StorageStack`) that encapsulates
provider-specific resources behind a uniform output interface, enabling future expansion
to AWS, GCP, or Scaleway without modifying consumer code.

---

## Table of Contents

1. [Background: Current Architecture](#1-background-current-architecture)
2. [Requirements Analysis](#2-requirements-analysis)
3. [Decision D1: Pulumi Resource Abstraction](#3-decision-d1-pulumi-resource-abstraction)
4. [Decision D2: Credential Injection Strategy](#4-decision-d2-credential-injection-strategy)
5. [Decision D3: Local S3 Simulation](#5-decision-d3-local-s3-simulation)
6. [Decision D4: Test Isolation Strategy](#6-decision-d4-test-isolation-strategy)
7. [Decision D5: MLflow + DVC Unified S3 Configuration](#7-decision-d5-mlflow--dvc-unified-s3-configuration)
8. [Decision D6: Multi-Cloud Provider Extensibility](#8-decision-d6-multi-cloud-provider-extensibility)
9. [Implementation Phases](#9-implementation-phases)
10. [Test Matrix](#10-test-matrix)
11. [GitHub Issue Breakdown](#11-github-issue-breakdown)
12. [Risk Analysis](#12-risk-analysis)
13. [References](#13-references)
14. [Appendix A: User Prompt (Verbatim)](#appendix-a-user-prompt-verbatim)
15. [Appendix B: Existing Infrastructure Inventory](#appendix-b-existing-infrastructure-inventory)

---

## 1. Background: Current Architecture

### 1.1 S3 Storage Topology

The MinIVess MLOps platform uses S3-compatible object storage in two distinct roles:

| Role | Consumer | Local Backend | Cloud Backend | Bucket |
|------|----------|---------------|---------------|--------|
| **MLflow Artifact Store** | MLflow server (tracking + serving) | MinIO container (`minivess-minio:9000`) | UpCloud Managed Object Storage | `mlflow-artifacts` |
| **DVC Data Remote** | DVC CLI (`dvc push/pull`) | MinIO container (`minivess-minio:9000`) | UpCloud Managed Object Storage | `minivess-dvc-data` |

Both consumers converge on the same UpCloud Managed Object Storage instance in cloud
deployments, but use separate buckets for namespace isolation.

### 1.2 Existing Pulumi Stack

The Pulumi stack (`deployment/pulumi/__main__.py`) provisions:

```
ManagedObjectStorage("mlflow-s3")
  ├── ManagedObjectStorageBucket("mlflow-artifacts")
  ├── ManagedObjectStorageBucket("minivess-dvc-data")
  ├── ManagedObjectStorageUser("mlflow")
  └── ManagedObjectStorageUserAccessKey("mlflow-s3-key")
```

Outputs: `s3_endpoint`, `s3_bucket`, `dvc_bucket`, `dvc_s3_endpoint`.

This stack is UpCloud-specific. No abstraction layer exists for multi-cloud S3 provisioning.

### 1.3 Existing DVC Configuration

`.dvc/config` defines four remotes:

| Remote | URL | Credential Source | Status |
|--------|-----|-------------------|--------|
| `minio` (default) | `s3://dvc-data` at `localhost:9000` | Hardcoded in `.dvc/config` | Active (local dev) |
| `upcloud` | `s3://minivess-dvc-data` at UpCloud endpoint | `.dvc/config.local` (gitignored) | Active (cloud) |
| `remote_storage` | `s3://minivessdataset` (AWS) | AWS credentials | Legacy |
| `remote_readonly` | HTTP static S3 website | None (public) | Legacy |

`scripts/configure_dvc_remote.py` writes UpCloud credentials to `.dvc/config.local`
from `DVC_S3_*` environment variables.

### 1.4 Existing Docker Compose S3 Configuration

**Infrastructure stack** (`deployment/docker-compose.yml`):
- MinIO service exposes ports `9000` (API) and `9001` (console)
- `minio-init` service auto-creates `mlflow-artifacts` bucket on startup
- MLflow service receives `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY` from `${MINIO_*}` env vars

**Flow services** (`deployment/docker-compose.flows.yml`):
- Common env anchor `x-common-env` injects S3 credentials to all flow containers:
  ```yaml
  AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER:-minioadmin}
  AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD:-minioadmin_secret}
  MLFLOW_S3_ENDPOINT_URL: http://${MINIO_DOCKER_HOST:-minivess-minio}:${MINIO_API_PORT:-9000}
  ```

### 1.5 Existing Test Infrastructure

Cloud tests (`tests/v2/cloud/`) use `@pytest.mark.cloud_mlflow` with credential-gated
skip logic:

| Test File | What It Tests | Skip Condition |
|-----------|--------------|----------------|
| `test_cloud_mlflow.py` | MLflow health, tracking, artifacts, S3 connectivity | `MLFLOW_CLOUD_URI` not set |
| `test_skypilot_mlflow.py` | SkyPilot YAML config + simulated VM logging | Unit: none; Cloud: `MLFLOW_CLOUD_*` not set |
| `test_dvc_cloud_pull.py` | DVC status + boto3 bucket access against UpCloud | `DVC_S3_*` not set |

### 1.6 Three Execution Environments

| Environment | Docker | Compute | S3 Backend | MLflow Backend |
|-------------|--------|---------|------------|----------------|
| **Cloud GPU** (RunPod via SkyPilot) | Native execution (no Docker-in-Docker) | Cloud GPU (RTX 4090, A100) | UpCloud Managed Object Storage | UpCloud MLflow VPS |
| **Local Docker** | Docker Compose (per-flow isolation) | Local GPU | MinIO container | Local MLflow container |
| **Dev (no Docker)** | None | Local GPU or CPU | Local filesystem (symlinked) | `mlruns/` directory |

---

## 2. Requirements Analysis

### 2.1 Functional Requirements

| ID | Requirement | Priority | Source |
|----|------------|----------|--------|
| FR1 | Pulumi programmatically provisions S3-compatible storage on UpCloud | P0 | User prompt |
| FR2 | Multi-cloud S3 abstraction: general interface, provider-specific implementations | P0 | User prompt |
| FR3 | S3 accessible from Docker on RunPod (cloud), Docker locally, dev without Docker | P0 | User prompt |
| FR4 | S3 serves as backend for both DVC (data) and MLflow (artifacts) | P0 | User prompt |
| FR5 | S3 tests skip gracefully when running locally with local dirs | P0 | User prompt |
| FR6 | Local S3 simulation: symlinked data or MinIO container | P1 | User prompt |
| FR7 | Comprehensive test suite verifying S3 access from all 3 environments | P0 | User prompt |
| FR8 | Future providers (AWS, GCP, Scaleway) addable via Pulumi without code changes to consumers | P1 | User prompt |

### 2.2 Non-Functional Requirements

| ID | Requirement | Source |
|----|------------|--------|
| NFR1 | Zero-config start: `just experiment` works on any machine without S3 credentials | CLAUDE.md Design Goal #1 |
| NFR2 | All credentials from `.env.example` (Rule #22: single source of config) | CLAUDE.md Rule #22 |
| NFR3 | No hardcoded URLs in Dockerfiles or Python code | CLAUDE.md Rule #22 |
| NFR4 | TDD mandatory: failing tests before implementation | CLAUDE.md Rule #2 |
| NFR5 | GitHub Actions CI remains disabled; all tests run locally | CLAUDE.md Rule #21 |
| NFR6 | No standalone script execution path; Prefect flows only | CLAUDE.md Rule #17 |

### 2.3 Constraints

| Constraint | Impact |
|-----------|--------|
| UpCloud trial: EUR 250 / 30 days | All provisioning must be within trial budget |
| 8 GB local GPU (RTX 2070 Super) | MinIO container adds ~200 MB RAM overhead |
| GitHub Actions disabled | All S3 integration tests run via `make test-cloud-mlflow` locally |
| Pulumi state in Pulumi Cloud | Free tier: 1 user, unlimited resources |

---

## 3. Decision D1: Pulumi Resource Abstraction

### 3.1 Problem Statement

The current Pulumi stack (`deployment/pulumi/__main__.py`) provisions UpCloud resources
directly at the top level. Adding a second cloud provider (e.g., AWS S3 or Scaleway
Object Storage) would require duplicating resource definitions. The question: what
abstraction level is appropriate for multi-cloud S3 provisioning?

### 3.2 Hypotheses

| ID | Hypothesis | Description |
|----|-----------|-------------|
| H1.1 | **Pulumi ComponentResource per provider** | Create `UpCloudS3Stack(ComponentResource)`, `AWSS3Stack(ComponentResource)`, etc., each exposing a uniform output interface (`endpoint_url`, `access_key_id`, `secret_access_key`, `bucket_names`) |
| H1.2 | **Flat resources with conditional logic** | Single `__main__.py` with `if provider == "upcloud": ... elif provider == "aws": ...` |
| H1.3 | **Pulumi multi-stack with stack references** | Separate Pulumi projects per provider, cross-referencing via `StackReference` |
| H1.4 | **Single stack, provider-agnostic abstraction in Python** | Python ABC `S3ProviderStack` with provider-specific subclasses, outside Pulumi's ComponentResource hierarchy |

### 3.3 Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Code isolation** | 0.25 | Provider code in separate files, no cross-contamination |
| **Uniform output interface** | 0.25 | All providers expose identical Pulumi exports |
| **PhD researcher complexity** | 0.20 | How many files/commands a new user must understand |
| **Extensibility** | 0.15 | Adding a 3rd provider requires editing how many files |
| **Pulumi best practices** | 0.15 | Alignment with [Pulumi component model](https://www.pulumi.com/docs/concepts/resources/components/) |

### 3.4 Decision Matrix

| Criterion (weight) | H1.1 ComponentResource | H1.2 Flat conditional | H1.3 Multi-stack | H1.4 Python ABC |
|--------------------|-----------------------|----------------------|-------------------|-----------------|
| Code isolation (0.25) | 5 — Each provider is a separate class file | 2 — All providers in one file, growing linearly | 5 — Separate projects entirely | 4 — Separate files, but not Pulumi-native |
| Uniform outputs (0.25) | 5 — `ComponentResource` outputs are typed | 3 — Manual discipline required | 4 — StackReference is typed but verbose | 4 — Python dataclass outputs |
| PhD complexity (0.20) | 4 — One `__main__.py` instantiates the right component | 5 — Single file, easy to read | 2 — Multiple projects, multiple `pulumi up` commands | 4 — Similar to H1.1 |
| Extensibility (0.15) | 5 — Add one file, one class | 2 — Edit shared file, risk breakage | 4 — Add one project | 4 — Add one file |
| Pulumi best practices (0.15) | 5 — This is the recommended Pulumi pattern | 2 — Anti-pattern at scale | 4 — Valid for large teams | 2 — Fights the framework |
| **Weighted total** | **4.75** | **2.80** | **3.80** | **3.70** |

### 3.5 Decision

**Selected: H1.1 — Pulumi ComponentResource per provider.**

Each cloud provider gets its own `ComponentResource` subclass in a separate Python file:

```
deployment/pulumi/
  __main__.py              # Entry point: reads provider config, instantiates component
  providers/
    __init__.py
    _base.py               # S3StackOutputs dataclass (uniform output contract)
    upcloud_s3.py           # UpCloudS3Stack(ComponentResource)
    aws_s3.py               # AWSS3Stack(ComponentResource) — future
    scaleway_s3.py           # ScalewayS3Stack(ComponentResource) — future
```

The `__main__.py` reads `pulumi.Config().get("s3_provider")` (defaulting to `"upcloud"`)
and instantiates the appropriate component. All components export the same outputs:

```python
@dataclass
class S3StackOutputs:
    endpoint_url: pulumi.Output[str]
    access_key_id: pulumi.Output[str]
    secret_access_key: pulumi.Output[str]
    mlflow_bucket: pulumi.Output[str]
    dvc_bucket: pulumi.Output[str]
```

### 3.6 Rationale

ComponentResource is Pulumi's recommended abstraction for grouping related resources.
It provides automatic resource grouping in the Pulumi state, makes `pulumi preview`
show provider-specific resources nested under the component, and is the documented
path for reusable multi-cloud infrastructure. The flat conditional (H1.2) is the current
state and will not scale; multi-stack (H1.3) adds operational complexity inappropriate
for a single-developer academic project; Python ABC (H1.4) fights Pulumi's type system.

---

## 4. Decision D2: Credential Injection Strategy

### 4.1 Problem Statement

S3 credentials must reach three distinct consumers across three environments:

| Consumer | Local Docker | Cloud GPU (RunPod) | Dev (no Docker) |
|----------|-------------|-------------------|-----------------|
| **MLflow server** | Docker Compose env vars (`MINIO_ROOT_*`) | Not applicable (MLflow runs on UpCloud VPS) | `MLFLOW_TRACKING_URI=mlruns` (no S3) |
| **MLflow client** (in flow containers) | Docker Compose env vars (`AWS_ACCESS_KEY_ID`, `MLFLOW_S3_ENDPOINT_URL`) | SkyPilot envs (`MLFLOW_CLOUD_*`) | Not applicable (filesystem backend) |
| **DVC** | `.dvc/config` hardcoded MinIO creds | SkyPilot envs (`DVC_S3_*`) + `configure_dvc_remote.py` | `.dvc/config` hardcoded MinIO creds or filesystem |

### 4.2 Hypotheses

| ID | Hypothesis | Description |
|----|-----------|-------------|
| H2.1 | **`.env` + Docker Compose env injection** (current) | All creds in `.env`, injected via `${VAR:-fallback}` in compose files, SkyPilot envs read from `.env` |
| H2.2 | **Pulumi config secrets + stack output** | Store creds as `pulumi config set --secret`, export via `pulumi stack output`, populate `.env` from outputs |
| H2.3 | **OIDC / IAM roles** | Cloud-native identity (AWS IAM roles, GCP Workload Identity) — no static credentials |
| H2.4 | **SOPS + age encrypted `.env`** | Encrypt `.env.secrets` with age, decrypt at runtime via `scripts/setup_dev.sh` |

### 4.3 Decision Matrix

| Criterion (weight) | H2.1 .env + Compose | H2.2 Pulumi secrets | H2.3 OIDC/IAM | H2.4 SOPS+age |
|--------------------|--------------------|--------------------|---------------|---------------|
| Simplicity (0.30) | 5 — One file, copy+paste | 3 — Two-step: pulumi output, then .env | 2 — Cloud-specific IAM setup per provider | 3 — Install age+sops, understand encryption |
| Security (0.25) | 3 — `.env` is gitignored but plaintext on disk | 4 — Encrypted in Pulumi state, plaintext in `.env` | 5 — No static creds at all | 4 — Encrypted at rest, decrypted at runtime |
| Multi-env compat (0.25) | 5 — Works everywhere | 3 — Pulumi required on every machine | 1 — UpCloud has no OIDC; RunPod has no IAM | 4 — Works everywhere with age key |
| Rule #22 compliance (0.20) | 5 — `.env.example` is the contract | 3 — Two sources of truth | 2 — Env vars still needed for non-OIDC cases | 4 — `.env.example` still the contract |
| **Weighted total** | **4.50** | **3.25** | **2.45** | **3.75** |

### 4.4 Decision

**Selected: H2.1 — `.env` + Docker Compose env injection (maintain current pattern).**

Additionally, provide a **convenience script** that reads Pulumi stack outputs and
writes them to `.env`:

```bash
# scripts/pulumi_to_env.sh
# Reads Pulumi outputs and appends S3 credentials to .env
S3_ENDPOINT=$(cd deployment/pulumi && pulumi stack output s3_endpoint)
S3_ACCESS_KEY=$(cd deployment/pulumi && pulumi stack output --show-secrets s3_access_key_id)
S3_SECRET_KEY=$(cd deployment/pulumi && pulumi stack output --show-secrets s3_secret_access_key)

# Write to .env (idempotent: sed-replace if exists, append if not)
...
```

### 4.5 Rationale

The `.env` pattern is already the project's single source of truth (Rule #22). All three
environments already consume credentials from `.env` or `.env`-derived SkyPilot envs.
OIDC (H2.3) is infeasible because UpCloud Managed Object Storage uses static access
keys with no OIDC integration, and RunPod provides no IAM role mechanism. SOPS+age
(H2.4) adds meaningful security but introduces tool dependencies (`age`, `sops`) that
every lab member must install. The Pulumi-to-env convenience script bridges H2.1 and
H2.2, giving Pulumi users automated credential population while keeping the `.env`
contract intact.

---

## 5. Decision D3: Local S3 Simulation

### 5.1 Problem Statement

When running without cloud credentials (local dev, new contributor onboarding), the
platform needs a local S3 substitute. The question: what fidelity of S3 simulation
is required, and what mechanism provides it?

### 5.2 Hypotheses

| ID | Hypothesis | Description |
|----|-----------|-------------|
| H3.1 | **MinIO Docker container** (current) | Full S3-compatible server, already in `docker-compose.yml` |
| H3.2 | **LocalStack** | AWS service emulator; S3 is free tier; supports IAM, STS |
| H3.3 | **Filesystem symlinks** | Symlink `data/` to a local directory; MLflow uses `mlruns/` filesystem backend |
| H3.4 | **MinIO in Docker + filesystem fallback** | MinIO for Docker environments; filesystem for dev-no-Docker |

### 5.3 Decision Matrix

| Criterion (weight) | H3.1 MinIO only | H3.2 LocalStack | H3.3 Filesystem only | H3.4 MinIO + filesystem |
|--------------------|-----------------|-----------------|-----------------------|-------------------------|
| S3 API fidelity (0.25) | 5 — Full S3 API, path-style + virtual-hosted | 5 — Full S3 API + IAM/STS | 1 — No S3 API at all | 5 (Docker) / 1 (dev) |
| Zero-config dev (0.25) | 2 — Requires Docker Compose up | 2 — Requires Docker Compose up | 5 — No extra services | 4 — Dev works without Docker |
| Resource overhead (0.15) | 4 — ~200 MB RAM, minimal CPU | 3 — ~500 MB RAM, Java-based | 5 — Zero overhead | 4/5 — Environment-dependent |
| Parity with production (0.20) | 4 — Same API, different auth model | 3 — S3 is same, but adds AWS-specific behaviors | 1 — Completely different API | 3 — Mixed fidelity |
| Maintenance burden (0.15) | 4 — Already maintained, `minio-init` exists | 2 — Additional service, version pinning, config | 5 — Nothing to maintain | 3 — Two code paths |
| **Weighted total** | **3.75** | **3.15** | **3.05** | **3.95** |

### 5.4 Decision

**Selected: H3.4 — MinIO in Docker + filesystem fallback.**

The two modes are determined by environment detection, not user configuration:

| Detection Signal | Mode | MLflow Backend | DVC Backend |
|-----------------|------|----------------|-------------|
| `MLFLOW_S3_ENDPOINT_URL` is set | **S3 mode** | MinIO or UpCloud S3 | MinIO or UpCloud S3 |
| `MLFLOW_S3_ENDPOINT_URL` is unset | **Filesystem mode** | `mlruns/` directory | Local filesystem (`data/`) |

This aligns with the existing `resolve_tracking_uri()` function in
`src/minivess/observability/tracking.py`, which already handles the MLflow side.
DVC already defaults to its `minio` remote (local) or `upcloud` remote (cloud)
based on `DVC_REMOTE` env var.

### 5.5 Filesystem Mode Specification

When running in dev-no-Docker mode:

1. **MLflow**: `MLFLOW_TRACKING_URI=mlruns` (existing default in `.env.example`)
2. **DVC**: Default remote is `minio` which points to `s3://dvc-data` at `localhost:9000`.
   Without MinIO running, DVC operations fail. The filesystem fallback provides:
   - A new DVC remote `local` with `url = /absolute/path/to/data/raw/minivess`
   - Or: data is already present via manual download / symlink to the data directory
3. **Tests**: S3-specific tests skip; filesystem-based tests run

The filesystem fallback does NOT attempt to emulate S3 semantics. It provides a
degraded but functional mode for researchers who just want to iterate on model code.

---

## 6. Decision D4: Test Isolation Strategy

### 6.1 Problem Statement

Tests must correctly handle three environments, two consumers (MLflow, DVC), and the
presence or absence of S3 credentials. The question: how do we structure test fixtures
and markers to achieve clean isolation without false passes or false skips?

### 6.2 Hypotheses

| ID | Hypothesis | Description |
|----|-----------|-------------|
| H4.1 | **Separate fixtures per environment** | `cloud_s3_fixture`, `local_minio_fixture`, `filesystem_fixture` — tests declare which fixture they need |
| H4.2 | **Environment auto-detection with unified fixture** | Single `s3_client_or_skip` fixture that detects environment and provides the right client or skips |
| H4.3 | **Marker-based gating** | `@pytest.mark.requires_s3_cloud`, `@pytest.mark.requires_s3_local`, `@pytest.mark.requires_s3_any` — conftest checks marker and skips |
| H4.4 | **Parametrized fixtures across environments** | `@pytest.fixture(params=["cloud", "minio", "filesystem"])` — runs each test in all available environments |

### 6.3 Decision Matrix

| Criterion (weight) | H4.1 Separate fixtures | H4.2 Auto-detect unified | H4.3 Marker gating | H4.4 Parametrized |
|--------------------|----------------------|-------------------------|--------------------|--------------------|
| Clarity (0.30) | 5 — Explicit dependencies | 3 — Magic detection, hard to debug | 4 — Markers are visible | 3 — Implicit matrix |
| False-skip prevention (0.25) | 4 — Manual fixture selection | 3 — Detection logic can be wrong | 5 — Skip reason in marker | 3 — Params may unexpectedly skip |
| Test count management (0.20) | 4 — One test per concern | 4 — One test per concern | 4 — One test per concern | 2 — 3x test count inflation |
| Existing pattern alignment (0.25) | 3 — Different from current `cloud_mlflow_connection` | 3 — Would replace existing fixtures | 5 — Matches existing `@pytest.mark.cloud_mlflow` | 2 — No existing parametrized pattern |
| **Weighted total** | **4.10** | **3.25** | **4.60** | **2.55** |

### 6.4 Decision

**Selected: H4.3 — Marker-based gating (extending existing pattern).**

New markers (added to `pyproject.toml`):

```python
markers = [
    # Existing:
    "cloud_mlflow: Requires live cloud MLflow deployment (MLFLOW_CLOUD_* env vars)",
    "skypilot_cloud: SkyPilot to remote MLflow integration",
    # New:
    "requires_s3_cloud: Requires cloud S3 credentials (DVC_S3_* or MLFLOW_CLOUD_S3_*)",
    "requires_s3_local: Requires local MinIO running (docker compose up minio)",
    "requires_s3_any: Requires any S3 backend (cloud, MinIO, or skip)",
]
```

Conftest detection logic:

```python
def _s3_cloud_available() -> bool:
    """Cloud S3 credentials are set."""
    return bool(
        os.environ.get("MLFLOW_CLOUD_S3_ENDPOINT")
        or os.environ.get("DVC_S3_ENDPOINT_URL")
    )

def _s3_local_available() -> bool:
    """Local MinIO is reachable."""
    try:
        import urllib.request
        port = os.environ.get("MINIO_API_PORT", "9000")
        urllib.request.urlopen(f"http://localhost:{port}/minio/health/live", timeout=2)
        return True
    except Exception:
        return False
```

### 6.5 Skip Behavior Summary

| Marker | Cloud GPU (RunPod) | Local Docker | Dev (no Docker) |
|--------|-------------------|-------------|-----------------|
| `@requires_s3_cloud` | RUN | SKIP | SKIP |
| `@requires_s3_local` | SKIP | RUN | SKIP |
| `@requires_s3_any` | RUN | RUN | SKIP |
| (no S3 marker) | RUN | RUN | RUN |

---

## 7. Decision D5: MLflow + DVC Unified S3 Configuration

### 7.1 Problem Statement

MLflow and DVC both use S3 but configure credentials differently:

| Aspect | MLflow | DVC |
|--------|--------|-----|
| **Credential env vars** | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL` | `DVC_S3_ACCESS_KEY` (or AWS_*), `DVC_S3_ENDPOINT_URL` |
| **Config file** | None (env vars only) | `.dvc/config` + `.dvc/config.local` |
| **Bucket** | `mlflow-artifacts` | `minivess-dvc-data` |
| **Usage pattern** | MLflow server writes artifacts; client reads via tracking URI proxy | DVC CLI push/pull directly to bucket |

Should we unify credential configuration, or keep them separate?

### 7.2 Hypotheses

| ID | Hypothesis | Description |
|----|-----------|-------------|
| H5.1 | **Shared credentials, separate buckets** | One set of S3 creds in `.env` (`S3_*` prefix), consumed by both MLflow and DVC. Same endpoint + user, different buckets. |
| H5.2 | **Separate credentials per consumer** (current) | MLflow uses `MINIO_ROOT_*` / `AWS_*`; DVC uses `DVC_S3_*`. Independent lifecycle. |
| H5.3 | **Single S3 user, multiple IAM policies** | One UpCloud user with bucket-scoped policies. Both consumers use the same key pair. |

### 7.3 Decision Matrix

| Criterion (weight) | H5.1 Shared creds | H5.2 Separate creds | H5.3 IAM policies |
|--------------------|-------------------|---------------------|-------------------|
| Simplicity (0.30) | 4 — One set to manage | 3 — Two sets, some overlap | 2 — IAM policy management |
| Security (0.25) | 3 — Blast radius: both consumers compromised | 4 — Credential isolation | 5 — Least-privilege per consumer |
| Operational overhead (0.25) | 5 — One credential rotation | 3 — Two rotations, two configs | 2 — Policy management overhead |
| UpCloud compatibility (0.20) | 4 — Works (same user for both buckets) | 4 — Works (can use same user) | 2 — UpCloud Managed Object Storage has limited IAM |
| **Weighted total** | **4.00** | **3.50** | **2.70** |

### 7.4 Decision

**Selected: H5.1 — Shared credentials, separate buckets.**

In practice, this is already nearly the case. The current Pulumi stack creates a single
`mlflow` user with one access key that has access to both buckets. The `.env` variables
are structured differently for MLflow vs DVC only because they were added at different
times.

**Implementation**: Introduce a common `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`,
`S3_SECRET_ACCESS_KEY` set of env vars in `.env.example`, and derive both
MLflow-specific and DVC-specific variables from them:

```bash
# .env.example (new unified section)
# ─── S3-Compatible Object Storage (shared by MLflow + DVC) ──────────────
# These credentials are for the cloud S3 backend (UpCloud, AWS, etc.)
# Local Docker uses MinIO credentials (MINIO_ROOT_* above) instead.
S3_ENDPOINT_URL=
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=

# Bucket names (separate namespaces on the same S3 endpoint)
MLFLOW_ARTIFACT_BUCKET=mlflow-artifacts
DVC_S3_BUCKET=minivess-dvc-data
```

The existing `DVC_S3_*` vars remain as **aliases** for backward compatibility:

```bash
# Legacy aliases (consumed by configure_dvc_remote.py — prefer S3_* above)
DVC_S3_ENDPOINT_URL=${S3_ENDPOINT_URL}
DVC_S3_ACCESS_KEY=${S3_ACCESS_KEY_ID}
DVC_S3_SECRET_KEY=${S3_SECRET_ACCESS_KEY}
```

### 7.5 Local Docker Override

When running with Docker Compose locally, the `x-common-env` anchor overrides
with MinIO credentials. This means cloud S3 vars are ignored inside Docker.
The flow container always uses whichever S3 backend Docker Compose provides.

---

## 8. Decision D6: Multi-Cloud Provider Extensibility

### 8.1 Problem Statement

The user requested a general S3 class with provider-specific implementations that
enables adding new cloud providers (AWS, GCP, Scaleway) in the future. What is the
appropriate scope of this abstraction given the current project state?

### 8.2 Hypotheses

| ID | Hypothesis | Description |
|----|-----------|-------------|
| H6.1 | **Implement UpCloud only, design for extension** | Ship `UpCloudS3Stack` ComponentResource now; define the `S3StackOutputs` interface but do not implement other providers |
| H6.2 | **Implement UpCloud + AWS immediately** | Ship both providers, using AWS S3 as the validation that the abstraction generalizes |
| H6.3 | **Implement all four providers** (UpCloud, AWS, GCP, Scaleway) | Full multi-cloud from day one |
| H6.4 | **Abstract at Python SDK level, not Pulumi** | Create `src/minivess/storage/s3_client.py` with `S3Client` ABC and provider subclasses, leaving Pulumi UpCloud-only |

### 8.3 Decision Matrix

| Criterion (weight) | H6.1 UpCloud + design | H6.2 UpCloud + AWS | H6.3 All four | H6.4 Python SDK only |
|--------------------|-----------------------|--------------------|-----------------|-----------------------|
| Ship speed (0.30) | 5 — Minimal new code | 3 — AWS requires IAM/VPC decisions | 1 — Massive scope | 4 — Lighter implementation |
| Abstraction validation (0.25) | 3 — Unproven interface | 5 — Two providers validate the interface | 5 — Fully validated | 2 — Wrong layer (SDK, not IaC) |
| YAGNI compliance (0.20) | 5 — Build what you need | 3 — AWS may never be used | 1 — Speculative | 4 — Minimal |
| Future extensibility (0.25) | 4 — Interface exists, untested | 5 — Proven interface | 5 — Complete | 3 — IaC still UpCloud-only |
| **Weighted total** | **4.30** | **3.95** | **2.70** | **3.30** |

### 8.4 Decision

**Selected: H6.1 — Implement UpCloud only, design for extension.**

Rationale: The project currently uses only UpCloud (trial). AWS and GCP are listed in
SkyPilot failover order but have not been configured. YAGNI dictates implementing the
second provider when the first user requests it. The `S3StackOutputs` dataclass and
the `ComponentResource` pattern are sufficient to prove the interface is sound.

A `README.md` in `deployment/pulumi/providers/` documents how to add a new provider:

```markdown
## Adding a New S3 Provider

1. Create `providers/{provider}_s3.py`
2. Implement a class inheriting from `pulumi.ComponentResource`
3. Return `S3StackOutputs` from your constructor
4. Register the provider name in `__main__.py` dispatch dict
5. Add `pulumi-{provider}` to `deployment/pulumi/pyproject.toml`
6. Write tests in `tests/v2/cloud/test_s3_{provider}.py`
```

---

## 9. Implementation Phases

### Phase 1: Pulumi Refactor (ComponentResource Abstraction)

**Goal:** Extract existing UpCloud S3 resources into a `ComponentResource`, define the
uniform output interface, and ensure `pulumi up` still works identically.

**TDD Spec:**

```
RED:   test_upcloud_s3_component_returns_outputs() — verify S3StackOutputs fields
RED:   test_pulumi_main_dispatches_to_upcloud() — verify provider selection
GREEN: Implement UpCloudS3Stack, refactor __main__.py
```

**Files changed:**
- `deployment/pulumi/__main__.py` — refactored to use ComponentResource
- `deployment/pulumi/providers/__init__.py` — new
- `deployment/pulumi/providers/_base.py` — `S3StackOutputs` dataclass
- `deployment/pulumi/providers/upcloud_s3.py` — `UpCloudS3Stack(ComponentResource)`
- `deployment/pulumi/pyproject.toml` — no changes (upcloud dep already present)

**Verification:** `cd deployment/pulumi && pulumi preview` shows no changes to existing
resources (refactor is output-preserving).

### Phase 2: Credential Bridge (Pulumi Outputs to .env)

**Goal:** Automate population of `.env` S3 credentials from Pulumi stack outputs.

**TDD Spec:**

```
RED:   test_pulumi_to_env_writes_s3_credentials() — script populates .env
RED:   test_env_example_has_unified_s3_vars() — .env.example audit
GREEN: Implement scripts/pulumi_to_env.sh, update .env.example
```

**Files changed:**
- `scripts/pulumi_to_env.sh` — new: reads Pulumi outputs, writes to `.env`
- `.env.example` — add unified `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`

### Phase 3: Test Infrastructure (Markers + Fixtures)

**Goal:** Add S3-specific test markers and conftest fixtures that detect environment
and skip appropriately.

**TDD Spec:**

```
RED:   test_marker_requires_s3_cloud_skips_without_creds()
RED:   test_marker_requires_s3_local_skips_without_minio()
RED:   test_s3_environment_detection_cloud()
RED:   test_s3_environment_detection_local()
RED:   test_s3_environment_detection_none()
GREEN: Implement conftest detection, add markers to pyproject.toml
```

**Files changed:**
- `pyproject.toml` — add `requires_s3_cloud`, `requires_s3_local`, `requires_s3_any` markers
- `tests/v2/cloud/conftest.py` — add S3 environment detection fixtures
- `tests/v2/unit/test_s3_markers.py` — new: marker behavior unit tests

### Phase 4: S3 Connectivity Test Suite

**Goal:** Comprehensive tests verifying S3 access from all three environments for both
MLflow and DVC consumers.

**TDD Spec:**

```
RED:   test_cloud_s3_bucket_exists_mlflow()
RED:   test_cloud_s3_bucket_exists_dvc()
RED:   test_cloud_s3_put_get_roundtrip()
RED:   test_cloud_s3_multipart_upload()
RED:   test_local_minio_bucket_exists()
RED:   test_local_minio_put_get_roundtrip()
RED:   test_dvc_push_pull_cloud_roundtrip()
RED:   test_mlflow_artifact_via_cloud_s3()
RED:   test_filesystem_fallback_mlflow()
RED:   test_filesystem_fallback_dvc_status()
GREEN: Implement test bodies
```

**Files changed:**
- `tests/v2/cloud/test_s3_connectivity.py` — new: cloud S3 tests
- `tests/v2/integration/test_s3_local.py` — new: local MinIO tests
- `tests/v2/unit/test_s3_filesystem_fallback.py` — new: filesystem mode tests

### Phase 5: DVC Remote Unification

**Goal:** Update DVC configuration and `configure_dvc_remote.py` to use the unified
`S3_*` env vars while maintaining backward compatibility with `DVC_S3_*`.

**TDD Spec:**

```
RED:   test_configure_dvc_remote_reads_unified_s3_vars()
RED:   test_configure_dvc_remote_falls_back_to_dvc_s3_vars()
RED:   test_dvc_config_has_upcloud_remote_template()
GREEN: Update configure_dvc_remote.py, .dvc/config
```

**Files changed:**
- `scripts/configure_dvc_remote.py` — read `S3_*` with fallback to `DVC_S3_*`
- `.dvc/config` — update `upcloud` remote endpoint URL template

### Phase 6: Documentation and Makefile Targets

**Goal:** Add `make test-s3-cloud` and `make test-s3-local` targets, update
`cloud-tutorial.md` with S3 setup steps.

**Files changed:**
- `Makefile` — add `test-s3-cloud`, `test-s3-local` targets
- `docs/planning/cloud-tutorial.md` — S3 setup section

---

## 10. Test Matrix

### 10.1 Environment x Consumer x Operation Matrix

| Test | Operation | Consumer | Cloud GPU | Local Docker | Dev (no Docker) |
|------|-----------|----------|-----------|-------------|-----------------|
| T1 | `ListBuckets` | boto3 | PASS | PASS (MinIO) | SKIP |
| T2 | `HeadBucket(mlflow-artifacts)` | boto3 | PASS | PASS (MinIO) | SKIP |
| T3 | `HeadBucket(minivess-dvc-data)` | boto3 | PASS | PASS (MinIO) | SKIP |
| T4 | `PutObject` + `GetObject` roundtrip | boto3 | PASS | PASS (MinIO) | SKIP |
| T5 | Multipart upload (10 MB) | boto3 | PASS | PASS (MinIO) | SKIP |
| T6 | `mlflow.log_artifact()` roundtrip | MLflow | PASS | PASS | SKIP |
| T7 | `mlflow.list_artifacts()` | MLflow | PASS | PASS | SKIP |
| T8 | `dvc status -r upcloud` | DVC | PASS | SKIP (no cloud creds) | SKIP |
| T9 | `dvc status -r minio` | DVC | SKIP (no MinIO) | PASS | SKIP |
| T10 | `dvc push/pull` roundtrip | DVC | PASS | PASS (MinIO) | SKIP |
| T11 | MLflow filesystem fallback | MLflow | N/A | N/A | PASS |
| T12 | DVC local remote fallback | DVC | N/A | N/A | PASS |
| T13 | S3 credential env var validation | Unit | PASS | PASS | PASS |
| T14 | `S3StackOutputs` interface test | Unit | PASS | PASS | PASS |

### 10.2 Marker Assignment

| Test | Marker |
|------|--------|
| T1-T5, T8 | `@requires_s3_cloud` |
| T1-T5 (MinIO), T9, T10 | `@requires_s3_local` |
| T6-T7 (cloud) | `@cloud_mlflow` |
| T6-T7 (local) | `@requires_s3_local` |
| T11-T12 | (none — always runs) |
| T13-T14 | (none — always runs) |

### 10.3 Makefile Targets

```makefile
test-s3-cloud:    ## Run S3 tests against cloud backend (requires S3_* env vars)
	uv run pytest tests/v2/cloud/ -m "requires_s3_cloud or cloud_mlflow" -v

test-s3-local:    ## Run S3 tests against local MinIO (requires docker compose up minio)
	uv run pytest tests/v2/integration/ -m "requires_s3_local" -v

test-s3-all:      ## Run all S3 tests that are available in current environment
	uv run pytest tests/v2/ -m "requires_s3_any or requires_s3_cloud or requires_s3_local" -v
```

---

## 11. GitHub Issue Breakdown

### Issue #A: Pulumi ComponentResource Refactor (Phase 1)

**Title:** `feat(pulumi): Extract S3 resources into UpCloudS3Stack ComponentResource`

**Labels:** `P0`, `infra`, `pulumi`

**Acceptance criteria:**
- [ ] `S3StackOutputs` dataclass defined in `providers/_base.py`
- [ ] `UpCloudS3Stack(ComponentResource)` in `providers/upcloud_s3.py`
- [ ] `__main__.py` dispatches to provider based on `pulumi.Config().get("s3_provider")`
- [ ] `pulumi preview` shows zero diff (output-preserving refactor)
- [ ] `README.md` in `providers/` documents adding new providers
- [ ] Tests: `test_s3_stack_outputs_interface()`

**Estimated effort:** 2-3 hours

### Issue #B: Unified S3 Environment Variables (Phase 2)

**Title:** `feat(config): Add unified S3_* env vars + Pulumi-to-env bridge script`

**Labels:** `P0`, `config`, `s3`

**Acceptance criteria:**
- [ ] `.env.example` has `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`
- [ ] `scripts/pulumi_to_env.sh` reads Pulumi outputs, writes to `.env`
- [ ] Backward compatibility: `DVC_S3_*` vars still work
- [ ] `test_env_single_source.py` updated to validate new vars
- [ ] Documentation in `.env.example` explains unified vs. legacy vars

**Estimated effort:** 1-2 hours

### Issue #C: S3 Test Markers and Fixtures (Phase 3)

**Title:** `feat(tests): Add S3 environment detection markers + conftest fixtures`

**Labels:** `P0`, `testing`, `s3`

**Acceptance criteria:**
- [ ] `pyproject.toml` has `requires_s3_cloud`, `requires_s3_local`, `requires_s3_any` markers
- [ ] `conftest.py` has `_s3_cloud_available()`, `_s3_local_available()` detection
- [ ] Auto-skip with descriptive messages ("MinIO not running", "S3 creds not set")
- [ ] Unit tests for marker behavior with `monkeypatch`

**Estimated effort:** 2-3 hours

### Issue #D: S3 Connectivity Test Suite (Phase 4)

**Title:** `test(s3): Comprehensive S3 connectivity tests for cloud + local + fallback`

**Labels:** `P0`, `testing`, `s3`

**Acceptance criteria:**
- [ ] `test_s3_connectivity.py` — 5 cloud tests (T1-T5)
- [ ] `test_s3_local.py` — 5 local MinIO tests
- [ ] `test_s3_filesystem_fallback.py` — 2 filesystem fallback tests
- [ ] All tests pass in their respective environments
- [ ] All tests skip gracefully in other environments

**Estimated effort:** 3-4 hours

### Issue #E: DVC Remote Unification (Phase 5)

**Title:** `refactor(dvc): Update configure_dvc_remote.py to use unified S3_* vars`

**Labels:** `P1`, `dvc`, `s3`

**Acceptance criteria:**
- [ ] `configure_dvc_remote.py` reads `S3_*` with `DVC_S3_*` fallback
- [ ] `.dvc/config` upcloud remote endpoint URL updated
- [ ] Test: `test_configure_dvc_remote_reads_unified_s3_vars()`
- [ ] Backward compat test: `test_configure_dvc_remote_falls_back_to_dvc_s3_vars()`

**Estimated effort:** 1-2 hours

### Issue #F: Documentation and Makefile (Phase 6)

**Title:** `docs(s3): Add S3 setup tutorial + Makefile test targets`

**Labels:** `P1`, `docs`, `testing`

**Acceptance criteria:**
- [ ] `Makefile` has `test-s3-cloud`, `test-s3-local`, `test-s3-all` targets
- [ ] `cloud-tutorial.md` has S3 setup section with Pulumi instructions
- [ ] `deployment/pulumi/providers/README.md` documents adding new providers

**Estimated effort:** 1-2 hours

### Total Estimated Effort: 10-16 hours

---

## 12. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| UpCloud trial expires before testing complete | Medium | High — lose S3 endpoint | Complete Phase 1-4 within first 2 weeks; export credentials early |
| UpCloud Managed Object Storage API differs from AWS S3 in edge cases | Low | Medium — DVC or MLflow incompatibility | UpCloud documents S3 compatibility; existing `test_cloud_mlflow.py` already passes |
| MinIO Docker container conflicts with existing port usage | Low | Low — port already configured | `MINIO_API_PORT` is configurable via `.env` |
| Pulumi state corruption during refactor | Low | High — infrastructure drift | Run `pulumi preview` before every `pulumi up`; keep Pulumi state in cloud |
| DVC credential migration breaks existing workflows | Medium | Medium — data inaccessible | Backward-compatible aliases in `.env.example`; `DVC_S3_*` vars remain |
| RunPod VMs cannot reach UpCloud S3 endpoint | Low | High — cloud training broken | Already verified via `smoke_test_gpu.yaml` (DVC pull from UpCloud works) |

---

## 13. References

- [Pulumi ComponentResource documentation](https://www.pulumi.com/docs/concepts/resources/components/) — Official guide for creating reusable resource groups. Accessed 2026-03-13.
- [UpCloud Managed Object Storage documentation](https://upcloud.com/docs/products/object-storage/) — S3-compatible API, pricing, availability. Accessed 2026-03-13.
- [UpCloud Managed Object Storage API reference](https://developers.upcloud.com/1.4/16-managed-object-storage/) — REST API for programmatic access. Accessed 2026-03-13.
- [MinIO S3 compatibility documentation](https://min.io/docs/minio/linux/operations/checklists/thresholds.html) — Local S3 simulation fidelity. Accessed 2026-03-13.
- [DVC S3 remote configuration](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3) — DVC credential and endpoint configuration. Accessed 2026-03-13.
- [MLflow artifact stores (S3)](https://mlflow.org/docs/latest/tracking/artifacts-stores.html#amazon-s3-and-s3-compatible-storage) — MLflow S3 integration documentation. Accessed 2026-03-13.
- [pytest markers documentation](https://docs.pytest.org/en/stable/how-to/mark.html) — Custom marker registration and conditional skipping. Accessed 2026-03-13.
- [LocalStack S3 free tier](https://docs.localstack.cloud/user-guide/aws/s3/) — LocalStack community edition S3 support. Accessed 2026-03-13.

---

## Appendix A: User Prompt (Verbatim)

> Well create an P0 Issue then on how to set-up the S3 to UpCloud, hopefully all
> programatically. Can we spin the S3 via Pulumi then so that we can build on top of
> Pulumi and then later add other "S3 spin scripts" for other clouds if needed? So some
> general S3 class with special cases for the different providers, all handled via Pulumi.
> And as you can authenticate to my UpCloud, plan then how to achieve this, with proper
> test suite to ensure that we can access that S3 from Docker running e.g. on Runpod,
> docker running on my local machine, and in "dev" environment without any docker. And as
> this S3 is the backend for DVC and MLFlow, then the S3 testing obviously should be
> skipped if someone wants to run this (or just test) all locally using some local dir as
> the mounted S3 artifact store for MLflow, and some other subdir as the Data folder
> simulating S3 that we for example symlink to the data folder inside the repo. Make this
> plan comprehensive as well for the
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/s3-mounting-testing-and-simulation-plan.md
> and optimize with reviewer agents until converging into a perfect plan / background
> research report with multi-hypothesis decision matrix on different options here, all in
> academic format. And start by writing my prompt verbatim to disk as appendix and start
> planning and do quality work. Quality over AI Slop then!

---

## Appendix B: Existing Infrastructure Inventory

### B.1 Pulumi Stack Resources (as of 2026-03-13)

| Pulumi Resource Name | Type | UpCloud Service |
|---------------------|------|-----------------|
| `mlflow-postgres` | `ManagedDatabasePostgresql` | Managed PostgreSQL, `1x1xCPU-2GB-25GB` |
| `mlflow-s3` | `ManagedObjectStorage` | Managed Object Storage, `europe-1` |
| `mlflow-artifacts` | `ManagedObjectStorageBucket` | Bucket: `mlflow-artifacts` |
| `dvc-data` | `ManagedObjectStorageBucket` | Bucket: `minivess-dvc-data` |
| `mlflow-s3-user` | `ManagedObjectStorageUser` | Username: `mlflow` |
| `mlflow-s3-key` | `ManagedObjectStorageUserAccessKey` | Active access key |
| `mlflow-server` | `Server` | VPS: `DEV-1xCPU-2GB`, Helsinki |
| `install-docker` | `remote.Command` | Docker installation on VPS |
| `deploy-mlflow` | `remote.Command` | MLflow container deployment |

### B.2 Pulumi Stack Outputs

| Output | Description |
|--------|------------|
| `server_ip` | VPS public IPv4 address |
| `mlflow_url` | `http://{server_ip}:5000` |
| `mlflow_username` | `admin` |
| `postgres_host` | Managed PostgreSQL hostname |
| `s3_endpoint` | Public S3 endpoint URL (`https://....upcloudobjects.com`) |
| `s3_bucket` | `mlflow-artifacts` |
| `dvc_bucket` | `minivess-dvc-data` |
| `dvc_s3_endpoint` | Same as `s3_endpoint` |
| `ssh_command` | `ssh -i ~/.ssh/upcloud_minivess deploy@{ip}` |

### B.3 DVC Remotes

| Remote | URL | Default | Status |
|--------|-----|---------|--------|
| `minio` | `s3://dvc-data` at `localhost:9000` | Yes | Active (local) |
| `upcloud` | `s3://minivess-dvc-data` at UpCloud | No | Active (cloud) |
| `remote_storage` | `s3://minivessdataset` (AWS) | No | Legacy |
| `remote_readonly` | HTTP static S3 website | No | Legacy |

### B.4 Docker Compose S3 Services

| Service | Image | Ports | Volume |
|---------|-------|-------|--------|
| `minio` | `minio/minio:RELEASE.2025-02-08T19-54-51Z` | `9000`, `9001` | `minio_data:/data` |
| `minio-init` | `minio/mc:RELEASE.2025-02-08T19-54-51Z` | N/A | N/A |

### B.5 Environment Variables (.env.example)

| Variable | Default | Used By |
|----------|---------|---------|
| `MINIO_ROOT_USER` | `minioadmin` | MinIO, MLflow, flow containers |
| `MINIO_ROOT_PASSWORD` | `minioadmin_secret` | MinIO, MLflow, flow containers |
| `MINIO_API_PORT` | `9000` | MinIO, Docker Compose |
| `MINIO_CONSOLE_PORT` | `9001` | MinIO |
| `MINIO_DOCKER_HOST` | `minio` | Docker Compose env vars |
| `MLFLOW_ARTIFACT_BUCKET` | `mlflow-artifacts` | MLflow |
| `DVC_S3_ENDPOINT_URL` | (empty) | DVC cloud remote |
| `DVC_S3_ACCESS_KEY` | (empty) | DVC cloud remote |
| `DVC_S3_SECRET_KEY` | (empty) | DVC cloud remote |
| `DVC_S3_BUCKET` | `minivess-dvc-data` | DVC cloud remote |
| `DVC_REMOTE` | `upcloud` | DVC default remote selection |
| `MLFLOW_CLOUD_S3_ENDPOINT` | (empty) | Cloud test suite |
| `MLFLOW_CLOUD_S3_ACCESS_KEY` | (empty) | Cloud test suite |
| `MLFLOW_CLOUD_S3_SECRET_KEY` | (empty) | Cloud test suite |
| `MLFLOW_CLOUD_S3_BUCKET` | `mlflow-artifacts` | Cloud test suite |
