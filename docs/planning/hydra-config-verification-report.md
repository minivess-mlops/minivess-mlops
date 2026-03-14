# Hydra Config System Verification Report

> **Date**: 2026-03-14
> **Status**: Assessment complete, gaps identified
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

## 1. Current Config System Architecture

### Three Config Domains (Correctly Separated)

| Domain | Tool | Files | Purpose |
|--------|------|-------|---------|
| **Training** | Hydra-zen | `configs/base.yaml` + groups | Experiment hyperparams |
| **Deployment** | Dynaconf | `configs/deployment/*.toml` | Environment-specific (dev/staging/prod) |
| **Infrastructure** | `.env` + env vars | `.env.example` | Secrets, URLs, cloud creds |

### Hydra Composition Flow (WORKING)

```
configs/base.yaml
    ├── defaults:
    │     ├── data: minivess        → configs/data/minivess.yaml
    │     ├── model: dynunet        → configs/model/dynunet.yaml
    │     ├── training: default     → configs/training/default.yaml
    │     ├── checkpoint: standard  → configs/checkpoint/standard.yaml
    │     └── profiling: default    → configs/profiling/default.yaml
    ├── experiment_name: unnamed
    ├── losses: [dice_ce]
    └── debug: false

    ↓ experiment override (e.g., smoke_sam3_vanilla.yaml)

    # @package _global_
    defaults:
      - override /model: sam3_vanilla
      - override /checkpoint: lightweight
    experiment_name: smoke_sam3_vanilla
    max_epochs: 2
    ...

    ↓ runtime overrides (HYDRA_OVERRIDES env var)

    compose_experiment_config() → resolved dict → MLflow artifact
```

### What's Working

| Aspect | Status | Evidence |
|--------|--------|---------|
| Base + defaults composition | **OK** | `base.yaml` with 5 config groups |
| Experiment overrides | **OK** | 30+ experiment YAMLs in `configs/experiment/` |
| Model config groups | **OK** | 5 models in `configs/model/` |
| Config → MLflow artifact | **OK** | `log_hydra_config()` in tracking.py |
| Runtime overrides via env | **OK** | `HYDRA_OVERRIDES` env var in SkyPilot YAML |
| Model profiles (VRAM) | **OK** | 12 profiles in `configs/model_profiles/` |
| Debug configs | **OK** | `debug_*.yaml` in experiment dir |

---

## 2. What's MISSING: SkyPilot Config Integration

### Gap 1: SkyPilot Cloud Config Is Not in Hydra

Currently, SkyPilot cloud/GPU configuration is **hardcoded in YAML files**:

```yaml
# deployment/skypilot/smoke_test_lambda.yaml (CURRENT — hardcoded)
resources:
  accelerators: {A10: 1, A100: 1, GH200: 1, H100: 1}
  cloud: lambda
  use_spot: false
  disk_size: 40
```

This should be a **Hydra config group** so labs can override:

```yaml
# configs/cloud/lambda.yaml (PROPOSED)
cloud: lambda
accelerators: {A10: 1, A100: 1, GH200: 1, H100: 1}
use_spot: false
disk_size: 40
docker_image: ghcr.io/petteriteikari/minivess-base:latest
regions_priority:
  - europe-south-1
  - europe-central-1
  - us-east-1
```

### Gap 2: No Lab/User Override Mechanism

There is no `configs/lab/` or `configs/user/` directory. No way for a lab to say
"we have AWS credits, use A100 spot on us-east-2" without editing the defaults.

### Gap 3: SkyPilot YAML Duplication

There are 5 SkyPilot YAMLs with duplicated setup/run sections:
- `smoke_test_gpu.yaml` (RunPod)
- `smoke_test_lambda.yaml` (Lambda)
- `dev_runpod.yaml` (RunPod dev)
- `train_production.yaml` (multi-cloud)
- `train_hpo.yaml` (HPO sweep)

The DVC pull, pre-flight check, and run logic is copy-pasted across all 5.

### Gap 4: Docker Registry Not Configurable via Hydra

The Docker image registry is in `.env.example` but the SkyPilot YAML hardcodes it:
```yaml
image_id: docker:ghcr.io/petteriteikari/minivess-base:latest
```

A lab using ECR or GAR must edit every YAML file.

---

## 3. Proposed Architecture: Cloud Config as Hydra Group

### New Config Groups

```
configs/
├── base.yaml                    # Add: cloud defaults
├── cloud/                       # NEW: Cloud provider configs
│   ├── lambda.yaml              # Lambda Labs defaults
│   ├── gcp_spot.yaml            # GCP spot defaults
│   ├── gcp_ondemand.yaml        # GCP on-demand
│   ├── aws_spot.yaml            # AWS spot defaults
│   ├── azure_spot.yaml          # Azure spot defaults
│   ├── runpod_dev.yaml          # RunPod for dev (non-Docker)
│   └── local.yaml               # Local GPU (no cloud)
├── lab/                         # NEW: Lab-specific overrides
│   ├── default.yaml             # Default lab config
│   ├── example_lab.yaml         # Template for new labs
│   └── .gitignore               # Labs can add their own (gitignored)
├── user/                        # NEW: Per-user overrides
│   ├── default.yaml             # Default user config
│   ├── example_user.yaml        # Template
│   └── .gitignore               # Users add their own (gitignored)
└── registry/                    # NEW: Docker registry configs
    ├── ghcr.yaml                # GitHub Container Registry
    ├── gar.yaml                 # Google Artifact Registry
    ├── ecr.yaml                 # AWS ECR
    └── dockerhub.yaml           # Docker Hub
```

### Updated base.yaml

```yaml
defaults:
  - data: minivess
  - model: dynunet
  - training: default
  - checkpoint: standard
  - profiling: default
  - cloud: lambda            # NEW: default cloud provider
  - registry: ghcr           # NEW: default Docker registry
  - lab: default             # NEW: lab-level overrides
  - user: default            # NEW: user-level overrides
  - _self_

experiment_name: unnamed
losses:
  - dice_ce
debug: false
patch_size: null
```

### Example: configs/cloud/lambda.yaml

```yaml
# Lambda Labs — VM-based, Docker works natively
cloud:
  provider: lambda
  accelerators:
    - A10
    - A100
    - GH200
    - H100
  use_spot: false          # Lambda has no spot
  disk_size: 40
  docker_image: ${registry.image_uri}
  regions_priority:
    # Unpopular first (better availability)
    - europe-south-1
    - europe-central-1
    - me-west-1
    - us-midwest-1
    - us-east-1            # Most popular last
```

### Example: configs/cloud/gcp_spot.yaml

```yaml
# GCP Spot — same-region everything, 60-91% cheaper
cloud:
  provider: gcp
  accelerators:
    - L4              # $0.22/hr spot, 24 GB (like RTX 4090)
    - T4              # $0.14/hr spot, 16 GB (cheapest)
    - A100            # $1.15/hr spot, 40 GB
    - H100            # $2.25/hr spot, 80 GB
  use_spot: true
  disk_size: 50
  disk_tier: best     # SSD for MOUNT_CACHED checkpointing
  docker_image: ${registry.image_uri}
  checkpoint_bucket: minivess-checkpoints
  checkpoint_mode: MOUNT_CACHED
  regions_priority:
    - us-central1     # Best availability + same as GCS/Cloud SQL
```

### Example: configs/lab/example_lab.yaml

```yaml
# @package _global_
# Lab: Acme University Biomedical Imaging
# Override cloud and registry for lab-specific AWS setup
defaults:
  - override /cloud: aws_spot
  - override /registry: ecr

cloud:
  accelerators:
    - A100            # Lab has AWS A100 credits
  regions_priority:
    - us-west-2       # Lab's AWS region

# Lab-specific DVC remote
dvc_remote: s3://acme-lab-minivess-data

# Lab-wide training defaults
training:
  compute: gpu_high   # Lab has A100s
  memory_limit_gb: 40
```

### Example: configs/user/example_user.yaml

```yaml
# @package _global_
# User: Jane (PhD student in Acme Lab)
# Inherits lab defaults, overrides for personal workflow
defaults:
  - override /lab: acme_university

# Jane prefers cheaper GPUs for development
cloud:
  accelerators:
    - T4              # Budget-friendly for dev
    - L4              # When more VRAM needed

# Personal experiment preferences
training:
  seed: 123           # Jane's lucky seed
  max_epochs: 50      # Shorter for faster iteration
```

### Override Hierarchy (Hydra Resolution Order)

```
1. configs/base.yaml                     (repo defaults)
2. configs/{data,model,training,...}      (config groups)
3. configs/cloud/lambda.yaml             (cloud defaults)
4. configs/registry/ghcr.yaml            (registry defaults)
5. configs/lab/acme_university.yaml      (lab overrides)
6. configs/user/jane.yaml                (user overrides)
7. configs/experiment/smoke_sam3.yaml     (experiment overrides)
8. HYDRA_OVERRIDES env var               (runtime overrides)
```

Later entries override earlier ones. A user config overrides lab config,
which overrides repo defaults. Hydra handles this natively.

---

## 4. Current Single-Source-of-Truth Audit

### Training Config: Hydra-zen (**CORRECT**)

| Setting | Defined In | Single Source? |
|---------|-----------|----------------|
| `max_epochs` | `configs/training/default.yaml` | YES |
| `seed` | `configs/training/default.yaml` | YES |
| `losses` | `configs/base.yaml` | YES |
| `model` | `configs/model/*.yaml` | YES |
| `patch_size` | `configs/base.yaml` / model profile | YES |
| `experiment_name` | `configs/experiment/*.yaml` | YES |

### Deployment Config: Dynaconf (**CORRECT**)

| Setting | Defined In | Single Source? |
|---------|-----------|----------------|
| MLflow URI | `.env` → `resolve_tracking_uri()` | YES |
| PostgreSQL URL | `.env` → Dynaconf/env | YES |
| MinIO credentials | `.env` → Docker Compose | YES |

### Infrastructure Config: .env (**CORRECT**)

| Setting | Defined In | Single Source? |
|---------|-----------|----------------|
| `LAMBDA_API_KEY` | `.env.example` → `.env` | YES |
| `RUNPOD_API_KEY` | `.env.example` → `.env` | YES |
| `GITHUB_TOKEN` | `.env.example` → `.env` | YES |
| `DVC_S3_*` | `.env.example` → `.env` | YES |

### SkyPilot Config: Hardcoded in YAML (**VIOLATION**)

| Setting | Defined In | Single Source? |
|---------|-----------|----------------|
| `cloud: lambda` | Each SkyPilot YAML | **NO** — duplicated |
| `accelerators: {A10: 1, ...}` | Each SkyPilot YAML | **NO** — duplicated |
| `image_id: docker:ghcr.io/...` | Each SkyPilot YAML | **NO** — duplicated |
| `disk_size: 40` | Each SkyPilot YAML | **NO** — duplicated |
| DVC pull setup | Each SkyPilot YAML (setup:) | **NO** — copy-pasted |

### Docker Registry: Mixed Sources (**PARTIAL VIOLATION**)

| Setting | Defined In | Single Source? |
|---------|-----------|----------------|
| `DOCKER_REGISTRY` | `.env.example` | YES (env var) |
| `image_id: docker:ghcr.io/...` | SkyPilot YAMLs | **NO** — hardcoded |

---

## 5. SkyPilot YAML Templating Solution

The setup/run scripts are duplicated across 5 SkyPilot YAMLs. Since SkyPilot
doesn't support YAML includes or templates, the options are:

### Option A: Python-Generated YAML (Recommended)

Generate SkyPilot YAML from Hydra config at launch time:

```python
# In launch_smoke_test.py
cfg = compose_experiment_config(experiment_name="smoke_sam3_vanilla")
cloud_cfg = cfg["cloud"]

task = sky.Task(
    setup=_generate_setup_script(cfg),
    run=_generate_run_script(cfg),
)
task.set_resources(sky.Resources(
    cloud=_resolve_cloud(cloud_cfg["provider"]),
    accelerators=_format_accelerators(cloud_cfg["accelerators"]),
    image_id=f"docker:{cfg['registry']['image_uri']}",
    use_spot=cloud_cfg["use_spot"],
    disk_size=cloud_cfg["disk_size"],
))
```

This makes SkyPilot YAMLs generated from Hydra config → single source of truth.

### Option B: Shared Setup Script (Simpler)

Extract the duplicated setup/run to a shared script in the Docker image:

```yaml
# deployment/skypilot/smoke_test_lambda.yaml
setup: |
  cd /app && bash scripts/cloud_setup.sh
run: |
  cd /app && bash scripts/cloud_run.sh
```

The cloud-specific config (GPU, region, etc.) stays in the YAML but the
setup/run logic is defined once in `scripts/cloud_setup.sh`.

---

## 6. Implementation Roadmap

### Phase 1: Cloud Config Group (Small, No Breaking Changes)

1. Create `configs/cloud/` directory with provider YAMLs
2. Add `cloud` to `base.yaml` defaults list
3. Modify `launch_smoke_test.py` to read cloud config from Hydra
4. **Does NOT change SkyPilot YAMLs** — they become generated

### Phase 2: Lab/User Override Structure

1. Create `configs/lab/` and `configs/user/` with examples
2. Add `lab` and `user` to `base.yaml` defaults list
3. `.gitignore` lab/user files (except examples)
4. Document in `configs/README.md`

### Phase 3: Registry Abstraction

1. Create `configs/registry/` with GHCR/GAR/ECR/DockerHub configs
2. SkyPilot `image_id` generated from registry config
3. `make push-registry` reads from Hydra config

### Phase 4: Setup/Run Deduplication

1. Extract shared setup to `scripts/cloud_setup.sh`
2. Shared run script: `scripts/cloud_run.sh`
3. SkyPilot YAMLs reference scripts instead of inlining

---

## 7. Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Hydra composition | **WORKING** | base + groups + experiments + overrides |
| Training config single-source | **CORRECT** | All in Hydra YAML |
| Deployment config single-source | **CORRECT** | Dynaconf TOML |
| Infrastructure config single-source | **CORRECT** | .env.example |
| SkyPilot cloud config | **VIOLATION** | Hardcoded in each YAML |
| Docker registry config | **PARTIAL VIOLATION** | .env has it but YAMLs hardcode |
| Lab/user override system | **MISSING** | No configs/lab/ or configs/user/ |
| Setup/run deduplication | **MISSING** | Copy-pasted across 5 YAMLs |
| Config composability for labs | **ARCHITECTURAL GAP** | Hydra supports it, not yet implemented |

### Bottom Line

The training config system (Hydra-zen) is **correctly architected and composable**.
The deployment config (Dynaconf) is **correct**. The infrastructure config (.env) is
**correct**. The gap is that **SkyPilot cloud configuration is not integrated into
Hydra** — it's hardcoded in separate YAML files. Adding `configs/cloud/`, `configs/lab/`,
and `configs/user/` groups would close this gap and enable the heterogeneous lab
customization the user needs.
