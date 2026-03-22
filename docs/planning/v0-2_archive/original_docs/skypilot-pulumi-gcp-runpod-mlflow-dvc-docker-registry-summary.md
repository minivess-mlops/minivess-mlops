# Infrastructure System Status: Complete Audit

**Date**: 2026-03-15
**Purpose**: Single source of truth for what WORKS and what's BROKEN across
the entire Pulumi+SkyPilot+GCP+RunPod+MLflow+DVC+Docker system.

## Architecture

```
NEW USER EXPERIENCE (must work from scratch):

1. git clone → uv sync --all-extras
2. pulumi up (GCP stack)           → Cloud SQL, GCS, GAR, Cloud Run MLflow, IAM
3. docker build + push to GAR      → Docker image with all deps
4. dvc pull                        → MiniVess data from UpCloud S3
5. sky jobs launch train.yaml      → SkyPilot provisions GCP L4 spot, pulls Docker, trains
6. MLflow Cloud Run                → Experiment tracking, artifact storage in GCS
7. sky rsync / gsutil              → Sync mlruns back to local for analysis
```

## Component Status Matrix

| # | Component | Verified? | Last Test | Blocker | Fix |
|---|-----------|-----------|-----------|---------|-----|
| 1 | `pulumi up` GCP stack | YES (2026-03-15) | Phase A | None | — |
| 2 | Cloud SQL PostgreSQL | YES | Phase A | None | — |
| 3 | GCS buckets (3) | YES | Phase A | None | — |
| 4 | GAR Docker registry | YES | Phase A | None | — |
| 5 | Cloud Run MLflow 3.10.0 | YES | Phase A | None | — |
| 6 | SA key signing for GCS URLs | YES | Phase A | None | — |
| 7 | Docker build (21 GB) | YES | Multiple | None | — |
| 8 | Docker push to GAR | YES | Multiple | None | — |
| 9 | DVC pull from UpCloud S3 | YES | SkyPilot setup | None | — |
| 10 | SkyPilot GCP L4 spot launch | YES | Run 8 | None | — |
| 11 | Training: sam3_vanilla | YES | val_loss=0.632 | None | — |
| 12 | Training: dynunet | YES | val_loss=0.749 | None | — |
| 13 | Training: sam3_hybrid | YES | val_loss=0.679 (Run 8) | None | — |
| 14 | Training: vesselfm | **NO** | SSH error (transient) | Retry needed | Relaunch |
| 15 | Training: sam3_topolora | **NO** | OOM on RTX 4090 (24 GB) | Needs A100 (40 GB+) | Hardware |
| 16 | MLflow artifact upload (multipart) | YES | 910 MB .pth | None | — |
| 17 | Checkpoint SHA256 sidecar | YES (code) | Unit tests | Not verified on cloud | Cloud test |
| 18 | Spot preemption recovery | **NO** | Never tested | MOUNT source: vs name: | Test |
| 19 | `pulumi destroy` + `pulumi up` (from scratch) | **NO** | Never tested E2E | Unknown | **MUST TEST** |
| 20 | RunPod dev workflow (E2E) | PARTIAL | 3/5 models | mlruns sync missing | #716 |
| 21 | RunPod → local mlruns sync | **NO** | Never built | No script | #716 |
| 22 | Cost estimation (epoch-0) | YES (code) | Unit tests | Not wired to trainer | Wire |

## Previous Plans (26 documents, 18 months of iteration)

### RunPod Plans (10 docs)
| Doc | Outcome |
|-----|---------|
| runpod-for-quick-dev-env-use.md | Planning only, never executed |
| runpod-post-alignment-qa-decisions | Identified GHCR blocker |
| runpod-vs-lambda-vs-rest | **KEY**: RunPod pods ARE containers, not VMs |
| runpod-dev-verification-plan.xml | Never executed |
| runpod-dev-verification-plan-for-realz-maybe.xml | **EXECUTED**: 3/4 models pass |
| runpod-debug-profiling.xml | Bare-VM approach (BANNED) |
| runpod-debug-profiling-execution-final.xml | 29 hypotheses, Phase 3 never reached |
| runpod-debug-profiling-execution-final-debug-plan.xml | 6 systemic fixes |
| runpod-debug-profiling-execution-final-debug-plan-post-alignment.xml | Docker mandate correction |
| dev-runpod-doublecheck-code-review.xml | 5 critical adapter bugs found |

### GCP Plans (7 docs)
| Doc | Outcome |
|-----|---------|
| gcp-setup-tutorial.md | Reference guide |
| gcp-phase-b-d-cold-start.xml | Phase B-D: atomic checkpoints, resume, MOUNT |
| gcp-vs-lambda-availability.md | GCP >> Lambda for availability |
| gcp-spot-with-skypilot-and-pulumi-up-plan.xml | Pulumi stack deployed |
| prod-staging-gcp-doublecheck-code-review.xml | Code review plan |
| remaining-runpod-gcp-qa.xml | BF16, AMP, cost estimator |
| sam3-val-loss-final-report.md | Root cause: wrong config selection |

### Infra Plans (5 docs)
| Doc | Outcome |
|-----|---------|
| pulumi-iac-implementation-guide.md | Reference |
| pulumi-upcloud-managed-deployment.md | UpCloud MLflow WORKING |
| skypilot-and-finops-complete-report.md | T4 vs L4 analysis |
| skypilot-compute-offloading-plan.xml | 14 tasks, none implemented |
| upcloud-runpod-skypilot-mlflow-integration-testing.xml | TDD plan, not executed |

## Closed Issues (Infrastructure)

#708 (Hydra cloud configs), #706-705-702 (GCP Pulumi), #692 (RunPod idle),
#689 (MLflow env var), #688 (multi-model RunPod), #687 (GCP account),
#685 (GCP eval), #684 (artifact upload), #681 (SkyPilot Docker providers),
#680 (SAM3 FP16), #678 (S3 AccessDenied), #667 (VesselFM smoke),
#664 (DVC push), #651-650 (GPU benchmark), #642 (Pulumi S3),
#641 (VRAM profiles), #638-637-633 (SkyPilot RunPod smoke)

## Open Issues (Infrastructure)

#717 (cost estimator), #716 (mlruns sync), #715 (sam3_hybrid NaN — RESOLVED),
#714 (atomic checkpoints — RESOLVED), #709 (AWS/Azure recipes),
#690 (Makefile RunPod targets), #686 (spot checkpointing),
#683 (infra timing), #682 (dstack alternative), #679 (zero-infra DevEx),
#627 (RunPod Pulumi), #615 (MLflow nginx+TLS), #611 (HPO barrier)

## What "From Scratch" Means

A new researcher cloning this repo must be able to:
1. `pulumi up` → all GCP infra provisioned
2. `make build-base-gpu && make push-gar` → Docker image in GAR
3. `dvc pull` → training data on local machine
4. `make smoke-test-gcp MODEL=sam3_vanilla` → trains on GCP L4, metrics in MLflow
5. Open MLflow URL → see experiment with finite val_loss
6. `gsutil rsync` → pull mlruns to local for analysis

If ANY step fails, the system is broken.
