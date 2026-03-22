# Integration Test Double-Check: Multi-Hypothesis Decision Matrix

**Date**: 2026-03-22
**Branch**: `test/debug-factorial-4th-pass`
**Author**: Claude Code (Opus 4.6) — 4 specialist reviewer agents
**Purpose**: Audit whether the Docker Compose + Prefect architecture delivers on the
promise of full local testability, including SkyPilot via SSH mockup.

---

## Executive Summary

Four specialist reviewer agents independently analyzed the integration test strategy
across four domains: SkyPilot local testing, Docker Compose integration, Prefect flow
orchestration, and MLflow/data pipeline continuity. This document synthesizes their
findings into a unified decision matrix with clear recommendations.

**Bottom line**: The project has excellent *unit-level* infrastructure testing (66 SkyPilot
YAML tests, compose correctness/hardening tests, Prefect structural tests, MLflow backend
tests). However, **zero Docker-runtime integration tests run in any CI gate**. The E2E
tests exist but are excluded from both staging and prod. The gap between "YAML looks correct"
and "services actually start and talk to each other" is the primary risk for the factorial
launch.

### Key Numbers

| Metric | Current | Target |
|--------|---------|--------|
| Staging tests | 5752 passed, 0 skipped | Maintain |
| Prod tests | 6063 passed, 34 skipped, 1 failed | 6097+ passed, 1 acceptable skip (VRAM), 0 failed |
| Integration tests in CI gates | 0 | ≥20 (cross-flow contracts + health checks) |
| SkyPilot tests | 66 passing (YAML-only) | 66 + Task.from_yaml() non-optional |
| Docker runtime tests in gates | 0 | Compose health gate (opt-in) |

### 34 Prod Skip Breakdown

| Count | Root Cause | Category |
|-------|-----------|----------|
| 16 | `MLFLOW_TRACKING_URI not set to remote URL` | Cloud-credential-gated (acceptable) |
| 3 | `MLFLOW_TRACKING_URI not set to remote URL` (training artifacts) | Cloud-credential-gated (acceptable) |
| 3 | `MLFLOW_TRACKING_URI not set to remote URL` (SkyPilot+MLflow) | Cloud-credential-gated (acceptable) |
| 3 | `MLFLOW_TRACKING_URI not set to remote URL` (preflight) | Cloud-credential-gated (acceptable) |
| 3 | `RUNPOD_API_KEY not set` | Cloud-credential-gated (acceptable) |
| 2 | `MLFLOW_TRACKING_URI not set to remote URL` (HF accessibility) | Cloud-credential-gated (acceptable) |
| 3 | `MLFLOW_TRACKING_URI not set to remote URL` (training flow) | Cloud-credential-gated (acceptable) |
| 1 | `SAM3 TopoLoRA requires >= 16 GB VRAM` | Hardware-gated (acceptable: RTX 2070 Super = 8 GB) |

**Verdict**: All 34 skips are legitimate — 33 require cloud credentials (tests/v2/cloud/),
1 requires ≥16 GB VRAM. These should run on GCP or RunPod, not locally. The prod gate
should exclude `tests/v2/cloud/` to avoid counting them as skips.

### 1 Prod Failure

| Test | Root Cause | Fix |
|------|-----------|-----|
| `test_no_colon_directories_in_repo_root` | Spurious `file:` directory created by relative MLflow URI bug | Removed `file:` directory |

---

## Domain 1: SkyPilot Local Testing

### Reviewer: SkyPilot Testing Specialist

**Question**: How to test SkyPilot locally without cloud spend?

### Decision Matrix

| ID | Hypothesis | Effort | Fidelity | Maintenance | Key Limitation |
|----|-----------|--------|----------|-------------|----------------|
| **H1** | YAML assertions (current) | 0 hrs | 3/10 | Low | No runtime validation |
| **H2** | Docker SSH (panubo/sshd) | 8-12 hrs | 2/10 | High | Wrong abstraction layer — SkyPilot doesn't SSH into arbitrary targets |
| **H3** | sky local up + KinD | 6-10 hrs | 6/10 | Medium | No GPU, modified YAML needed, 5-min bootstrap |
| **H4** | Mock sky.launch() | 3-5 hrs | 2/10 | Medium | Tests wrapper, not SkyPilot |
| **H5** | SkyPilot internal mocks | 10-15 hrs | 4/10 | High | Tests SkyPilot internals, not our config |
| **H6** | Custom SkyPilot backend | 40-80 hrs | 7/10 | Extreme | Engineering project, not a test |
| **H7** | Testcontainers ephemeral | 5-8 hrs | 4/10 | Medium | Tests Docker image, not SkyPilot |

### Pros/Cons Summary

**H1 (Current — YAML Assertions)**:
- ✅ Already working (66 tests), zero infrastructure, sub-second execution
- ✅ Catches exact bugs that burned real money ($5-15 per failed launch)
- ❌ Cannot validate SkyPilot schema, setup script execution, env var interpolation
- ❌ Cannot test Docker image boots correctly

**H2 (Docker SSH — panubo/sshd)**:
- ✅ Tests real SSH transport layer
- ❌ SkyPilot does NOT SSH into arbitrary endpoints — massive impedance mismatch
- ❌ Would need to hack SkyPilot internals (breaks on every update)
- ❌ Wrong abstraction: SSH is an implementation detail, not a test interface

**H3 (sky local up + KinD)**:
- ✅ Official SkyPilot feature, tests real orchestration (pod creation, env injection, setup)
- ✅ Closest to production of any non-cloud option
- ❌ No GPU, `cloud: gcp` must be overridden to `kubernetes` (modified YAML)
- ❌ 3-5 min bootstrap, KinD + SkyPilot version coupling
- ❌ No `file_mounts` GCS syntax support

**H4-H7**: Low fidelity, wrong focus, or extreme effort. See detailed analysis in agent output.

### Recommendation: **H1 + one targeted enhancement**

Keep the 66 YAML assertion tests. Add:
1. **Make `sky.Task.from_yaml()` non-optional** — add `skypilot` to dev extras, remove `try/except ImportError` guard
2. **Parametrize over ALL SkyPilot YAMLs** (not just `train_factorial.yaml`)
3. **Bash syntax check**: `bash -n <(setup_script)` to catch shell syntax errors

**Rationale**: Single-researcher academic project. The 66 tests prevent the $5-15 bugs. KinD
(H3) costs 6-10 hours for a weekly check nobody will run. Every hour on test infra is an
hour not on the Nature Protocols paper. H2/H5/H6 are engineering traps that test the wrong
abstraction layer.

---

## Domain 2: Docker Compose Integration

### Reviewer: Docker Compose & Service Integration Specialist

**Question**: Are Docker Compose services actually tested in any CI gate?

### Decision Matrix

| ID | Hypothesis | Effort | Fidelity | Maintenance | Key Limitation |
|----|-----------|--------|----------|-------------|----------------|
| **H1** | Status quo (E2E exists, excluded) | 0 hrs | 3/10 | Low | Zero runtime tests in gates |
| **H2** | Lightweight "test" profile | 8-12 hrs | 6/10 | Medium | ~60s startup, 4 GB RAM |
| **H3** | Testcontainers per-test | 16-24 hrs | 5/10 | High | Does NOT test actual compose config |
| **H4** | Docker in prod gate | 4-6 hrs | 7/10 | Medium | Blows 10-min budget, env-dependent |
| **H5** | Health-check-only tests | 2-4 hrs | 4/10 | Low | Tests liveness, not correctness |
| **H6** | Pre-built snapshots | 24-40 hrs | 5/10 | High | Over-engineered, masks startup bugs |
| **H7** | Compose-less mocking | 8-12 hrs | 2/10 | High | Violates "Docker = execution model" |
| **H8** | Docker-in-Docker | 20-30 hrs | 8/10 | Very High | RAM doubling, DinD complexity |

### Recommendation: **H2 + H5 combined (two-layer strategy)**

**Layer 1 — Static analysis (already in staging gate)**: Keep `test_compose_correctness.py`,
`test_compose_hardening.py`, `test_docker_compose_volumes.py` as-is.

**Layer 2 — `make test-compose-health` (new, opt-in)**:
1. Start `docker compose --profile dev up -d` (4 services: PostgreSQL, MinIO, MLflow, Prefect)
2. Wait for health checks (90s timeout)
3. Run ~10 focused tests: TCP connectivity, HTTP health, MLflow experiment creation, MinIO bucket
4. Tear down with `docker compose --profile dev down -v`
5. Total budget: ~2 minutes

NOT in staging or prod gates — a manual pre-merge check for compose changes.
Documented in `deployment/CLAUDE.md`. Eventually promotable to prod gate.

**What this catches**: PostgreSQL connection failures, MinIO health regressions, MLflow startup
failures (the `MLFLOW_SERVER_ALLOWED_HOSTS` incident), Prefect API errors, env var interpolation,
network creation issues, `minio-init` bucket creation failures.

---

## Domain 3: Prefect Flow Testing

### Reviewer: Prefect Orchestration Testing Specialist

**Question**: Do we ever invoke a @flow through Prefect's engine?

### Decision Matrix

| ID | Hypothesis | Effort | Fidelity | Maintenance | Key Limitation |
|----|-----------|--------|----------|-------------|----------------|
| **H1** | Status quo (.fn() calls) | 0 hrs | 3/10 | Low | Bypasses ALL Prefect machinery |
| **H2** | Full deployment-based | 16-24 hrs | 9/10 | High | Needs worker, 10+ min per test |
| **H3** | Direct flow invocation | 8-12 hrs | 6/10 | Medium | No deployment/container layer |
| **H4** | Docker work pools | 30-40 hrs | 10/10 | Very High | Full production path but enormous effort |
| **H5** | Mock task execution | 10-14 hrs | 5/10 | Medium-High | Tests graph, not implementation |
| **H6** | Subprocess-based | 6-8 hrs | 4/10 | Medium | Violates Rule #17 |
| **H7** | Contract tests only | 4-6 hrs | 2/10 | Low-Medium | No execution at all |

### Recommendation: **H3 + H7, with H5 as follow-up**

**Phase 1**: Strengthen H7 (contract tests) — define typed schemas for each flow's inputs/outputs.
**Phase 2**: Implement H3 (direct flow invocation) — call `@flow`-decorated functions through
Prefect's engine using `prefect_test_harness()`. Start with lightweight flows (data_flow,
deploy_flow, biostatistics_flow).
**Phase 3**: H5 (mock task execution) for heavy flows (train_flow, analysis_flow).

**Tier placement**:
- H7 (contracts): Staging (<5s)
- H3 (light flows): Staging (<60s)
- H3 (heavy flows): Prod (<120s with synthetic data)

---

## Domain 4: MLflow & Data Pipeline

### Reviewer: MLflow & Data Pipeline Testing Specialist

**Question**: Is cross-flow MLflow metric continuity tested?

### Decision Matrix

| ID | Hypothesis | Effort | Fidelity | Maintenance | Key Limitation |
|----|-----------|--------|----------|-------------|----------------|
| **H1** | Status quo (unit MLflow) | 0 hrs | 4/10 | Low | Integration tests always skip |
| **H2** | Expanded file backend | 4-6 hrs | 5/10 | Low | No auth, no S3, no PostgreSQL |
| **H3** | Docker MLflow+MinIO | 8-12 hrs | 9/10 | Medium-High | 30s startup, Docker required |
| **H4** | Cross-flow artifact contracts | 6-8 hrs | 7/10 | Medium | Catches contract bugs, not infra |
| **H5** | DVC data integrity | 3-4 hrs | 6-8/10 | Low | Limited to data existence |
| **H6** | Metric continuity tests | 4-5 hrs | 6/10 | Low | Narrow scope (keys only) |
| **H7** | Snapshot-based mlruns | 6-8 hrs | 3/10 | HIGH | Maintenance trap, MLflow version coupling |

### Recommendation: **Phase 1: H4+H6, Phase 2: H3, Phase 3: H5**

**Critical gap**: `FlowContract.log_flow_completion()` and `FlowContract.find_upstream_run()`
are NEVER tested together. If a tag name changes in one but not the other, the entire 5-flow
pipeline silently breaks.

**Phase 1** (Staging, 8-10 hrs): Cross-flow contract + metric continuity tests against file backend.
Simulate train→post_training→analysis→biostatistics chain. Verify checkpoint discovery,
metric key queryability, tag matching.

**Phase 2** (Prod, 10-14 hrs): Docker-based MLflow+MinIO integration. Activate existing
`test_mlflow_minio_contract.py`. Tests MinIO bucket creation, S3 artifact paths, auth.

**Phase 3** (Cloud, 3-4 hrs): DVC data integrity. Extend `test_dvc_remote_sync.py` with
actual `dvc pull` verification.

**Reject H7 (snapshots)**: Coupling tests to MLflow's internal file format is a maintenance
trap that breaks on every MLflow version bump.

---

## Consolidated Recommendations

### Immediate Actions (before factorial launch)

| # | Action | Effort | Gate Impact |
|---|--------|--------|-------------|
| 1 | Remove spurious `file:` directory | Done | Fixes 1 prod failure |
| 2 | Exclude `tests/v2/cloud/` from prod gate (already excluded from staging) | 5 min | Removes 33 "skips" from prod report |
| 3 | Add MLFLOW_TRACKING_URI to SkyPilot required envs test | 10 min | Staging |
| 4 | Add `job_recovery` absence test | 10 min | Staging |
| 5 | Add `disk_size >= 100` test | 10 min | Staging |

### Short-term (this sprint, 20-30 hours total)

| # | Action | Domain | Effort | Fidelity Gain |
|---|--------|--------|--------|---------------|
| 6 | Cross-flow FlowContract tests (H4+H6) | MLflow | 8-10 hrs | 4/10 → 7/10 |
| 7 | Make `sky.Task.from_yaml()` non-optional | SkyPilot | 2-3 hrs | 3/10 → 5/10 |
| 8 | Bash syntax check for setup scripts | SkyPilot | 1 hr | Catch shell errors |
| 9 | `make test-compose-health` target | Docker | 8-12 hrs | 3/10 → 6/10 |
| 10 | Direct flow invocation for light flows | Prefect | 8-12 hrs | 3/10 → 6/10 |

### Deferred (post-paper)

| # | Action | Rationale for deferral |
|---|--------|----------------------|
| 11 | KinD-based SkyPilot testing | 6-10 hrs for a weekly check nobody will run |
| 12 | Docker work pool Prefect tests | 30-40 hrs, needs dedicated integration env |
| 13 | Docker MLflow+MinIO in prod gate | Requires Docker running for prod gate |
| 14 | Mock SSH (panubo/sshd) | Wrong abstraction layer for SkyPilot |

### Explicitly Rejected

| Approach | Reason |
|----------|--------|
| Custom SkyPilot backend (H6-Sky) | 40-80 hrs, engineering project not a test |
| Docker-in-Docker (H8-Docker) | RAM doubling, DinD complexity |
| MLflow snapshot fixtures (H7-MLflow) | Maintenance trap, breaks on MLflow upgrades |
| Compose-less mocking (H7-Docker) | Violates "Docker = execution model" principle |
| SkyPilot internal mocks (H5-Sky) | Tests SkyPilot's code, not ours |

---

## Architecture: What Full Local Testability Actually Means

The user's vision — "every service is locally available, including SkyPilot testing via SSH
mockup for a mock on-prem server" — maps to this realized architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                         STAGING TIER (<3 min)                    │
│  ├── YAML assertion tests (SkyPilot, Compose, Dockerfile)       │
│  ├── sky.Task.from_yaml() schema validation (all YAMLs)         │
│  ├── Bash syntax checks (setup/run scripts)                     │
│  ├── Cross-flow FlowContract tests (file backend MLflow)        │
│  ├── Metric continuity tests (MetricKeys constants)             │
│  ├── Prefect direct flow invocation (light flows)               │
│  └── Structural/AST verification (Docker gate, constants)       │
├─────────────────────────────────────────────────────────────────┤
│                          PROD TIER (~10 min)                     │
│  ├── Everything in staging +                                     │
│  ├── Model loading tests (DynUNet, SegResNet, Mamba)            │
│  ├── Slow tests (>30s)                                           │
│  └── Heavy flow invocations (train with synthetic data)         │
├─────────────────────────────────────────────────────────────────┤
│                    COMPOSE HEALTH (opt-in, ~2 min)               │
│  ├── docker compose --profile dev up                             │
│  ├── PostgreSQL, MinIO, MLflow, Prefect health checks           │
│  ├── MLflow experiment CRUD against real server                  │
│  └── docker compose --profile dev down -v                        │
├─────────────────────────────────────────────────────────────────┤
│                      CLOUD TIER (with credentials)               │
│  ├── Remote MLflow CRUD tests                                    │
│  ├── SkyPilot RunPod smoke tests                                 │
│  ├── DVC pull verification                                       │
│  └── Preflight GCP checks                                        │
├─────────────────────────────────────────────────────────────────┤
│                       GPU TIER (RunPod/GCP only)                 │
│  └── SAM3 forward passes, model training integration             │
└─────────────────────────────────────────────────────────────────┘
```

**The SSH mockup question**: The reviewers unanimously agree that mock SSH is the wrong
abstraction for SkyPilot testing. SkyPilot's SSH layer is an internal implementation detail.
The right approach is:
1. **YAML validation** (Task.from_yaml()) — catches schema/config bugs
2. **KinD** (sky local up) — catches orchestration bugs, deferred post-paper
3. **Real cloud** — catches provisioning bugs, via cloud tier tests

Full local testability for SkyPilot means "validate everything we can without cloud spend"
— which is YAML parsing + setup script syntax checking + env var assertions. The orchestration
layer is tested on real cloud (where it runs in production) because SkyPilot does not have
a meaningful local execution mode for GCP/RunPod backends.

---

## Cross-References

- `skypilot-fake-mock-ssh-test-suite-plan.md` — Original 6-approach decision matrix (Approach F recommended, aligned with this review)
- `dvc-skypilot-factorial-monitor-double-check.xml` — QA audit identifying 9 SkyPilot YAML test gaps
- `docker-base-improvement-plan.md` — 3-tier Docker hierarchy (implemented)
- `docker-improvements-for-debug-training.md` — 17 Docker bugs catalog
- `prefect-docker-optimization-and-monai-consolidation.md` — Prefect/Docker integration plan
- `skypilot-and-finops-complete-report.md` — SkyPilot architecture + MLflow accessibility
- `skypilot-spot-resume.md` — 5 spot recovery gaps
- `skypilot-observability-for-factorial-monitor.md` — Factorial monitor upgrade plan
- `skypilot-spot-preemption-checkpoint-research-report.md` — MOUNT_CACHED + checkpoint research
- `docker-pull-runpod-provisioning-mlflow-cloud-run-multipart-upload-report.md` — 12 Docker pull hypotheses
- `skypilot-pulumi-gcp-runpod-mlflow-dvc-docker-registry-summary.md` — Full infrastructure audit
