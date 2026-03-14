# Hydra Config System Verification Report

> **Date**: 2026-03-14 (updated)
> **Status**: Config system correctly divided; SkyPilot GPU config is the gap
> **Purpose**: Verify config composability, single-source-of-truth, and lab/user override architecture

---

## User Requirement (Verbatim)

> "This allowed instances should be configured in some .yaml related to Lambda Labs
> linked to some high-level SkyPilot YAML under the composable hydra-zen config system,
> right? With the option to have the defaults defined in one place, and then users
> allowing to easily override these with like lab-XYZ-defaults.yaml as remember that
> this repo should work with heterogeneous research labs with them each allowing easy
> customization of configs without actually having to change the defaults if they do
> not want. Similarly within the lab, different users might have very different settings,
> so it is beneficial to have this lab-XYZ-personABC-defaults.yaml type of override
> possibility."

---

## 1. Three-Config System: Correctly Divided

| Domain | Tool | Files | Purpose | Composable? |
|--------|------|-------|---------|-------------|
| **Training** | Hydra-zen | `configs/base.yaml` + groups | Experiment hyperparams | YES |
| **Deployment** | Dynaconf | `configs/deployment/*.toml` | Environment-specific (dev/staging/prod) | YES |
| **Infrastructure** | `.env` | `.env.example` | Secrets, URLs, cloud creds | YES |

## 2. Three-Environment GPU Strategy

| Environment | Provider | GPU Default | Docker | Config File |
|-------------|----------|-------------|--------|-------------|
| **dev** | RunPod | RTX 4090/5090/3090 (consumer, ≥24 GB) | NO | `dev_runpod.yaml` |
| **staging** | Lambda Labs | A10 ($0.86/hr, 24 GB) | YES | `smoke_test_lambda.yaml` |
| **prod** | Lambda / GCP / AWS | A10, A100, L4 spot | YES | `train_production.yaml` |

**RunPod = dev only.** RunPod pods ARE containers — Docker-in-Docker is impossible.
The dev environment installs deps via `uv sync` in the SkyPilot `setup:` block.
Quick prototyping for researchers without local GPUs.

**Lambda/GCP/AWS = staging + prod.** VM-based providers support SkyPilot's
`image_id: docker:` natively. Full Docker pipeline.

## 3. What's Composable Today

### Hydra-zen (Training) — WORKING

```
configs/base.yaml  →  defaults list (model, training, data, checkpoint, profiling)
  ↓ experiment override (e.g., smoke_sam3_vanilla.yaml)
  ↓ runtime override (HYDRA_OVERRIDES env var)
  → compose_experiment_config() → resolved dict → MLflow artifact
```

**Lab/user override for training**: A lab adds `configs/experiment/juntendo_default.yaml`.
A user adds `configs/experiment/petteri_debug.yaml`. No defaults change required.

### Dynaconf (Deployment) — WORKING

```
configs/deployment/settings.toml    →  shared defaults
  ↓ ENV_FOR_DYNACONF=production  →  production.toml overrides
```

### .env (Secrets) — WORKING

Single source of truth (CLAUDE.md Rule #22). 370+ lines in `.env.example`.

## 4. What's NOT Composable: SkyPilot GPU Config

### Current: GPU Types Hardcoded in 5 YAML Files

| File | GPU List | Provider |
|------|----------|----------|
| `dev_runpod.yaml` | `{RTX4090: 1, RTX5090: 1, RTX3090: 1}` | RunPod |
| `smoke_test_gpu.yaml` | `{RTX4090: 1, RTX5090: 1, RTX3090: 1}` | RunPod |
| `smoke_test_lambda.yaml` | `{A10: 1, A100: 1, GH200: 1, H100: 1}` | Lambda |
| `train_production.yaml` | `RTX4090:1` + multi-cloud `any_of` | Multi-cloud |
| `train_hpo.yaml` | `A100:1` | Multi-cloud |

GPU types are **sensible defaults** (commented as overridable, not mandated),
but they ARE duplicated and NOT integrated into Hydra.

### Why SkyPilot YAML Can't Use ${} in resources

SkyPilot's `resources:` block does NOT support `${VAR}` interpolation for
`accelerators`. `envs:` supports it, but `resources:` is parsed by SkyPilot's
Python code, not shell. GPU preferences must be resolved before YAML parsing.

## 5. Single-Source-of-Truth Audit

### Training Config: Hydra-zen — CORRECT

| Setting | Defined In | Single Source? |
|---------|-----------|----------------|
| `max_epochs` | `configs/training/default.yaml` | YES |
| `losses` | `configs/base.yaml` | YES |
| `model` | `configs/model/*.yaml` | YES |
| `patch_size` | `configs/base.yaml` / model profile | YES |

### Infrastructure Config: .env — CORRECT

| Setting | Defined In | Single Source? |
|---------|-----------|----------------|
| `MLFLOW_CLOUD_URI` | `.env.example` → `.env` | YES |
| `DVC_S3_*` | `.env.example` → `.env` | YES |
| `RUNPOD_API_KEY` | `.env.example` → `.env` | YES |

### SkyPilot Config: YAML — VIOLATION

| Setting | Defined In | Single Source? |
|---------|-----------|----------------|
| `accelerators` | Each SkyPilot YAML | **NO — duplicated** |
| `image_id: docker:ghcr.io/...` | Each SkyPilot YAML | **NO — hardcoded** |
| DVC pull setup script | Each YAML (setup:) | **NO — copy-pasted** |

## 6. Recommended Fix: Phased Approach

### Phase 1: .env GPU Defaults (Minimal Effort)

```bash
# .env.example — add these
SKYPILOT_DEV_GPUS=RTX4090,RTX5090,RTX3090       # RunPod dev defaults
SKYPILOT_PROD_GPUS=A10,A100,GH200,H100          # Lambda/AWS/GCP prod defaults
```

Launcher scripts (`launch_dev_runpod.py`, `launch_smoke_test.py`) read these
and override `task.resources` dynamically. YAML keeps current defaults as fallback.

### Phase 2: Hydra Cloud Config Group (Before 2nd Lab Onboards)

```
configs/cloud/
├── lambda.yaml              # Lambda Labs: A10, A100, no spot
├── gcp_spot.yaml            # GCP: L4, T4 spot ($0.14-0.22/hr)
├── runpod_dev.yaml          # RunPod: RTX 4090/5090/3090, no Docker
└── local.yaml               # No cloud (local GPU only)
```

Add `cloud: lambda` to `base.yaml` defaults list.

### Phase 3: Lab/User Override Hierarchy (Before 3rd Lab Onboards)

```
configs/lab/
├── default.yaml             # No lab-specific overrides
├── example_lab.yaml         # Template for new labs
└── .gitignore               # Lab configs are gitignored

configs/user/
├── default.yaml             # No user-specific overrides
├── example_user.yaml        # Template
└── .gitignore               # User configs are gitignored
```

Hydra resolution order:
```
base.yaml → cloud/lambda.yaml → lab/juntendo.yaml → user/jane.yaml → experiment.yaml → HYDRA_OVERRIDES
```

### Phase 4: Setup/Run Deduplication

Extract duplicated setup/run scripts from 5 SkyPilot YAMLs into shared scripts:
```yaml
setup: |
  cd /app && bash scripts/cloud_setup.sh
run: |
  cd /app && bash scripts/cloud_run.sh
```

## 7. Summary

| Aspect | Status |
|--------|--------|
| Hydra composition (training) | **WORKING** |
| Dynaconf (deployment) | **CORRECT** |
| .env (secrets) | **CORRECT** |
| SkyPilot cloud config | **VIOLATION** — hardcoded, duplicated |
| Lab/user override system | **MISSING** — Hydra supports it, not yet implemented |
| Setup/run deduplication | **MISSING** — copy-pasted across 5 YAMLs |

**Bottom Line**: The training config system is properly composable. The gap is that
SkyPilot cloud configuration is not integrated into Hydra. GPU types in YAML files
are sensible defaults that users can freely edit, but they should eventually be
configurable via `configs/cloud/*.yaml` + lab/user overrides to support heterogeneous
research labs without changing defaults.
