# SkyPilot Local Test Suite — Mock SSH & YAML Validation Plan

**Date**: 2026-03-22
**Issue**: #908 (P1)
**Status**: PLANNED

---

## Problem

Every SkyPilot launch on GCP costs real credits. Bugs in YAML configs (wrong
splits file, unsupported fields, missing env vars) are only discovered when jobs
fail on cloud VMs — after spending $5-15 on provisioning + setup.

In this session alone, we hit 3 launch failures before a single job ran:
1. `sky` binary not on PATH (shell scripting bug)
2. `job_recovery` field unsupported in SkyPilot v1.0 (YAML schema bug)
3. Wrong splits file hardcoded in setup (logic bug)

All three would have been caught by local tests.

---

## Decision Matrix: 6 Approaches

| Criterion | A: `--dryrun` | B: `sky local up` + KinD | C: Docker SSH | D: Python Mock SSH | E: SkyPilot's Own Tests | **F: Hybrid (REC.)** |
|-----------|--------------|--------------------------|---------------|-------------------|------------------------|---------------------|
| Setup complexity | 1 (trivial) | 2 | 3 | 2 | 2 | **1** |
| Fidelity to cloud | 2 | **3** | 3 | 1 | 2 | 2.5 |
| GPU simulation | No | **Partial** | No | No | No | Partial |
| Cost | $0 | $0 | $0 | $0 | $0 | **$0** |
| CI/CD suitable | Yes | Yes (DinD) | Yes | Yes | Yes | **Yes** |
| Maintenance | Low | Medium | Medium | Low-Med | Low | **Low** |
| Tests YAML parsing | **Yes** | **Yes** | Partial | No | **Yes** | **Yes** |
| Tests Docker pull | No | **Yes** | No | No | No | No |
| Tests env var passing | Partial | **Yes** | **Yes** | Partial | Partial | **Yes** |
| Time to first result | 1 min | 5-10 min | 5 min | 1 min | 2 min | **<1 min** |

---

## Approach Details

### A: SkyPilot `--dryrun` + `Task.from_yaml()`

SkyPilot provides two built-in validation mechanisms:
- `sky launch --dryrun task.yaml` — parses YAML, runs optimizer, but does NOT provision
- `sky.Task.from_yaml('task.yaml')` — Python API for YAML validation

**Pros**: Zero infrastructure, validates YAML schema + resource specs.
**Cons**: Does NOT test setup/run scripts, Docker images, or SSH connectivity.

### B: `sky local up` + KinD (Local Kubernetes)

SkyPilot has `sky local up` ([issue #1165, March 2024](https://github.com/skypilot-org/skypilot/issues/1165))
which creates a KinD (Kubernetes-in-Docker) cluster locally. SkyPilot treats it as
a Kubernetes backend. Tasks are launched as pods.

GPU simulation possible via [kind-gpu-sim](https://github.com/maryamtahhan/kind-gpu-sim)
or [fake-gpu-operator](https://github.com/run-ai/fake-gpu-operator).

**Pros**: Highest fidelity — real pod scheduling, real Docker pull, real SSH into pod.
**Cons**: KinD bootstrap slow (~5 min), requires Docker-in-Docker for CI, `cloud: gcp`
must be overridden to `kubernetes`.

### C: Docker SSH Container (panubo/docker-sshd)

Run [panubo/sshd](https://github.com/panubo/docker-sshd) (476 stars, v1.10.0 Jan 2026)
as a mock cloud VM. Configure SkyPilot to SSH into it.

**Pros**: Tests actual SSH transport layer.
**Cons**: Significant plumbing to make SkyPilot treat a container as a node.
SkyPilot's cloud backends use cloud APIs for provisioning — SSH is only the
execution layer. Tests the wrong abstraction level.

### D: Python Mock SSH Server

Libraries: [mock-ssh-server](https://github.com/carletes/mock-ssh-server) (57 stars),
[fake-ssh](https://pypi.org/project/fake-ssh/),
[paramiko-mock](https://github.com/ghhwer/paramiko-ssh-mock) (v2.0.1, active).

**Pros**: Pure Python, pytest-native, no Docker.
**Cons**: Tests SSH transport, not SkyPilot orchestration. mock-ssh-server unmaintained
since 2021.

### E: SkyPilot's Own Test Patterns

SkyPilot's unit tests (`tests/unit_tests/`) use `@mock.patch` to mock `sky.catalog.*`
and `sky.clouds.*`. Run WITHOUT cloud access. Test resource parsing, DAG construction,
config validation.

**Pros**: Battle-tested patterns from SkyPilot itself.
**Cons**: Tests SkyPilot internals, not our config correctness.

### F: Hybrid `Task.from_yaml()` + Custom Assertions (RECOMMENDED)

Combine SkyPilot's `Task.from_yaml()` for structural validation with custom pytest
tests for project-specific invariants.

Example tests:
```python
import sky

def test_factorial_yaml_parses():
    task = sky.Task.from_yaml("deployment/skypilot/train_factorial.yaml")
    assert task is not None

def test_no_t4_in_accelerators():
    task = sky.Task.from_yaml("deployment/skypilot/train_factorial.yaml")
    for r in task.resources:
        if r.accelerators:
            assert "T4" not in r.accelerators

def test_spot_enabled():
    task = sky.Task.from_yaml("deployment/skypilot/train_factorial.yaml")
    for r in task.resources:
        assert r.use_spot is True

def test_required_envs_declared():
    task = sky.Task.from_yaml("deployment/skypilot/train_factorial.yaml")
    required = {"MODEL_FAMILY", "LOSS_NAME", "FOLD_ID", "EXPERIMENT_NAME",
                "POST_TRAINING_METHODS", "MAX_EPOCHS"}
    assert required.issubset(set(task.envs.keys()))

def test_no_apt_get_in_setup():
    task = sky.Task.from_yaml("deployment/skypilot/train_factorial.yaml")
    assert "apt-get" not in task.setup

def test_docker_image_id_set():
    task = sky.Task.from_yaml("deployment/skypilot/train_factorial.yaml")
    for r in task.resources:
        assert r.image_id is not None
        assert r.image_id.startswith("docker:")
```

---

## Recommended Strategy: Two Tiers

### Tier 1: Daily (`make test-staging`) — Approach F

- File: `tests/v2/unit/deployment/test_skypilot_yamls.py`
- Tests: YAML parsing, T4 ban, spot enabled, required envs, Docker image URI,
  no `apt-get`/`uv sync` in setup, GCS paths in file_mounts
- Runs in <1 minute, $0, no cloud credentials
- Add to staging test gate

### Tier 2: Weekly (manual) — Approach B

- `sky local up` creates KinD cluster
- Run modified YAML (remove `cloud: gcp`, use lightweight test image)
- Validates full SkyPilot flow: pod creation, env injection, setup execution
- 5-10 minutes, $0
- NOT in CI (too slow), used for pre-launch validation

---

## What These Tests Would Have Caught

| Bug from this session | Caught by Tier 1? | Caught by Tier 2? |
|----------------------|-------------------|-------------------|
| `sky` not on PATH | No (shell, not YAML) | No |
| `job_recovery` unsupported field | **Yes** (`Task.from_yaml()` raises) | **Yes** |
| Wrong splits file in setup | **Yes** (custom assertion on setup script content) | **Yes** (execution fails) |
| Missing `POST_TRAINING_METHODS` env for zero-shot | **Yes** (env var assertion) | **Yes** |

---

## Implementation Plan

1. Create `tests/v2/unit/deployment/test_skypilot_yamls.py` with Tier 1 tests
2. Add `sky.Task.from_yaml()` based validation for ALL SkyPilot YAMLs in `deployment/skypilot/`
3. Add to `make test-staging` gate
4. Document `sky local up` workflow in `deployment/CLAUDE.md`

## References

- [SkyPilot Python SDK](https://docs.skypilot.co/en/latest/reference/api.html)
- [SkyPilot CLI `--dryrun`](https://docs.skypilot.co/en/latest/reference/cli.html)
- [SkyPilot K8s Getting Started](https://docs.skypilot.co/en/latest/reference/kubernetes/kubernetes-getting-started.html)
- [SkyPilot Issue #1165 — `sky local up`](https://github.com/skypilot-org/skypilot/issues/1165)
- [kind-gpu-sim](https://github.com/maryamtahhan/kind-gpu-sim)
- [panubo/docker-sshd](https://github.com/panubo/docker-sshd) (476 stars, v1.10.0)
- [mock-ssh-server](https://github.com/carletes/mock-ssh-server)
