---
theme: cloud
doc_count: 41
last_synthesized: "2026-03-22"
implementation_health: mostly_implemented
kg_domains: [cloud, infrastructure]
---

# Theme: Cloud -- SkyPilot, GCP, RunPod, Pulumi

This theme covers the multi-cloud compute layer: SkyPilot as the intercloud broker,
GCP as the production cloud (Pulumi IaC, GCS, Cloud Run MLflow, GAR), RunPod as
the dev environment, provider selection decisions, FinOps cost analysis, spot preemption
recovery, and the infrastructure bottleneck investigations that shaped the architecture.

---

## Key Scientific Insights

### 1. SkyPilot Is an Intercloud Broker, Not IaC

SkyPilot (Yang et al., NSDI'23) is "Slurm for multi-cloud" -- it provisions GPU instances
across providers, manages spot recovery, and auto-selects regions for availability. It is
NOT infrastructure-as-code (Pulumi handles that). The critical constraint: SkyPilot YAML
MUST use `image_id: docker:...` -- bare VM setup scripts are BANNED. Setup blocks are ONLY
for data pull + config, never apt-get or uv sync.

This distinction was violated multiple times early on, leading to fragile VM-based setups
that broke on spot preemption. The metalearning pattern `2026-03-14-skypilot-bare-vm-docker-violation.md`
documents the recurring failure.

**Source:** `skypilot-and-finops-complete-report.md`, `runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md`

### 2. Two-Provider Architecture Is Non-Negotiable

Exactly two cloud providers: RunPod (env/dev) + GCP (staging/prod). Adding a third requires
explicit user authorization. This constraint exists because the project is an open-source
academic platform -- labs override cloud config via Hydra groups, they do not change code.

| Provider | Environment | Role | Data Storage | MLflow |
|----------|-----------|------|-------------|--------|
| RunPod | env (dev) | Quick GPU experiments, entry point for paper readers | Network Volume (from local disk) | File-based |
| GCP | staging + prod | Production runs, paper results, Pulumi IaC | GCS buckets | Cloud Run (optional) |

Lambda Labs was evaluated and rejected due to terrible availability (all 17 regions sold
out simultaneously). UpCloud was a trial that got sunseted. Hetzner remains a dormant
fallback for MLflow hosting (EUR 3.79/month).

**Source:** `cloud-architecture-decisions-2026-03-14.md`, `gcp-vs-lambda-availability.md`

### 3. RunPod Pods ARE Docker Containers

The 8-hour debugging session that produced `runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md`
discovered the fundamental issue: RunPod pods are already Docker containers, making Docker-
in-Docker impossible. This is why RunPod is "dev" (Docker-free runtime) while GCP is
"staging/prod" (Docker mandatory). SkyPilot on RunPod uses `image_id: docker:` to specify
the base image that the pod runs as, not a nested container.

**Source:** `runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md`, `runpod-for-quick-dev-env-use.md`

### 4. T4 Is Banned -- Turing Architecture Lacks BF16

T4 (Turing architecture) has no BF16 support. SAM3's half-precision encoder overflows during
validation (FP16 max = 65504 -> NaN). L4 (Ada Lovelace) supports BF16, is 1.86x faster,
AND 37% cheaper per job. This was discovered during the SAM3 val_loss=NaN investigation and
encoded as a non-negotiable constraint.

**Source:** `sam3-val-loss-final-report.md` (cross-ref from models theme), metalearning `2026-03-15-t4-turing-fp16-nan-ban.md`

### 5. Three Infrastructure Bottlenecks Consume 58% of GPU Test Time

The infrastructure optimization report identified three bottlenecks:
1. **Docker image pull:** 15 minutes from GAR (9 GB image after multi-stage optimization)
2. **RunPod provisioning:** 40+ minute failures (GHCR private registry -> switched to DockerHub)
3. **MLflow upload:** HTTP 500 on 900 MB checkpoints through Cloud Run (`--serve-artifacts` flag)

Solutions: fix MLflow `--no-serve-artifacts` (5 min), add zstd compression (1 hr), evaluate
GKE Image Streaming for deeper gains. The 9 GB image size is the irreducible floor after
multi-stage optimization (CUDA runtime + PyTorch + MONAI + deps).

**Source:** `docker-pull-runpod-provisioning-mlflow-cloud-run-multipart-upload-report.md`

### 6. GCP Stack Is Fully Automated via Pulumi

The GCP production stack is created by a single `pulumi up`:
- Cloud SQL PostgreSQL (MLflow + Optuna backends)
- 3 GCS buckets (mlflow-artifacts, dvc-data, checkpoints -- checkpoints bucket deprecated)
- GAR Docker registry (europe-north1)
- Cloud Run MLflow server (optional)
- Service accounts (skypilot-training, mlflow-server)
- IAM bindings

The entire stack was verified working as of 2026-03-15. The Pulumi guide covers
academic vs startup deployment scenarios and multi-cloud extensibility.

**Source:** `pulumi-iac-implementation-guide.md`, `gcp-setup-tutorial.md`, `skypilot-pulumi-gcp-runpod-mlflow-dvc-docker-registry-summary.md`

### 7. Spot Preemption Recovery Requires Application-Level Checkpointing

SkyPilot managed jobs (`sky jobs launch`) provide infrastructure recovery (new VM in
another region) but NOT checkpoint state management. The application must: detect resume
vs fresh start, find the latest checkpoint, load state and resume from correct epoch.
GCS bucket persistence means checkpoints survive preemption. The `check_resume_state_task()`
in train_flow.py handles this.

**Source:** `skypilot-spot-preemption-checkpoint-research-report.md`, `skypilot-spot-resume.md`

### 8. Docker Registry Proximity Matters

DockerHub for RunPod (public, zero-auth pull, fastest). GAR for GCP (same-region as
GCS/Cloud Run, ADC auth). GHCR caused 40+ minute STARTING hangs on RunPod due to
private registry authentication overhead. Registry choice is per-environment via
Hydra config groups.

**Source:** `docker-pull-runpod-provisioning-mlflow-cloud-run-multipart-upload-report.md`

### 9. 4th Pass Factorial Burned $6.30 on 5 Failure Modes

The 4th pass GCP launch attempt encountered: (1) `sky` binary not on PATH, (2) `job_recovery`
field unsupported in SkyPilot v1.0, (3) DVC pull failing on `data/processed/minivess`,
(4) job stuck STARTING for 2.5 hours, (5) no batch monitoring detected failures for 2+ hours.
All would have been caught by local SkyPilot YAML validation tests (Issue #908).

**Source:** `run-debug-factorial-experiment-report-4th-pass-failure.md` (cross-ref from evaluation theme)

### 10. Ralph Monitor Loop for Automated Cloud Failure Diagnosis

The ralph_monitor.py implements a monitor->diagnose->fix->relaunch loop for SkyPilot jobs.
It polls job status, fetches and analyzes logs on failure, categorizes failure types (setup
vs runtime), and outputs structured diagnosis. However, the current version watches ONE job
at a time -- batch monitoring for 32+ concurrent factorial jobs was identified as a gap.

**Source:** `ralph-loop-for-cloud-monitoring.md`, `skypilot-observability-for-factorial-monitor.md`

---

## Architectural Decisions Made

| Decision | Outcome | Source Doc | KG Node |
|----------|---------|-----------|---------|
| Cloud providers | RunPod (dev) + GCP (staging/prod). No third without auth. | cloud-architecture-decisions-2026-03-14.md | cloud.cloud_provider_selection |
| Compute orchestration | SkyPilot intercloud broker (NSDI'23) | skypilot-and-finops-complete-report.md | cloud.compute_orchestration |
| DVC remote | MinIO (local) + GCS (cloud). No AWS S3. | s3-mounting-testing-and-simulation-plan.md | cloud.dvc_remote_strategy |
| IaC tool | Pulumi Python SDK for GCP | pulumi-iac-implementation-guide.md | infrastructure.iac_tool |
| Docker registry | DockerHub (RunPod) + GAR (GCP) + GHCR (CI only) | docker-pull-runpod-provisioning report | infrastructure.docker_registry |
| GPU type | L4 (Ada Lovelace). T4 BANNED. | gcp-vs-lambda-availability.md | -- |
| MLflow hosting | Cloud Run (GCP) + file-based (RunPod) | cloud-tutorial.md | -- |
| GCP region | europe-north1 (Finland) | gcp-setup-tutorial.md | cloud.providers.gcp.region |
| Spot strategy | Managed jobs with auto-recovery | skypilot-spot-preemption-checkpoint-research-report.md | -- |
| FinOps | Per-job cost logging to MLflow | pr1-finops-infrastructure-timing-plan.md | -- |
| Artifact persistence | MLflow artifact store ONLY (no competing mechanisms) | -- | cloud.providers.gcp.gcs_buckets |

---

## Implementation Status

| Document | Type | Status | Key Deliverable |
|----------|------|--------|-----------------|
| cloud-architecture-decisions-2026-03-14.md | document | implemented | Two-provider decision record |
| cloud-tutorial.md | document | reference | Step-by-step cloud setup guide |
| cold-start-prompt-gcp-3rd-debug-run.md | cold_start | executed | 3rd GCP factorial re-run prompt |
| cold-start-prompts-pre-gcp-housekeeping.md | cold_start | executed | 6-PR pre-GCP housekeeping prompts |
| cold-start-track-b-runpod-gcp-qa.md | cold_start | executed | RunPod+GCP remaining QA |
| dev-runpod-doublecheck-code-review.xml | execution_plan | executed | RunPod code review |
| docker-pull-runpod-provisioning-mlflow-cloud-run-multipart-upload-report.md | research_report | reference | 3 bottleneck analysis (12 hypotheses) |
| gcp-phase-b-d-cold-start.xml | execution_plan | executed | GCP phases B-D execution |
| gcp-setup-tutorial.md | document | reference | GCP setup for researchers |
| gcp-spot-with-skypilot-and-pulumi-up-plan.xml | execution_plan | implemented | GCP spot + Pulumi setup |
| gcp-vs-lambda-availability.md | document | reference | Lambda rejected (availability) |
| hetzner-mlflow-plan.md | plan | archived | Hetzner MLflow fallback (EUR 3.79/mo) |
| mlflow-metrics-and-params-double-check-before-large-scale-gcp-work.md | document | executed | MLflow pre-GCP verification |
| oracle-cloud-free.txt | document | archived | Oracle free tier evaluation |
| pr-e-cost-reporting-plan.xml | execution_plan | partial | Cost reporting for factorial |
| pr1-finops-infrastructure-timing-plan.md | plan | partial | FinOps + timing analysis |
| pre-full-gcp-housekeeping-and-qa.xml | execution_plan | executed | Pre-GCP 6-PR housekeeping |
| pre-full-gcp-training-qa-plan.md | plan | executed | Full GCP training QA |
| pre-gcp-master-plan.xml | execution_plan | executed | Master plan for GCP transition |
| pulumi-iac-implementation-guide.md | document | reference | Pulumi deep-dive + deployment scenarios |
| ralph-loop-for-cloud-monitoring.md | document | implemented | Ralph monitor loop design |
| remaining-runpod-gcp-qa-2nd-pass.xml | execution_plan | executed | QA 2nd pass |
| remaining-runpod-gcp-qa-3rd-pass.xml | execution_plan | executed | QA 3rd pass |
| remaining-runpod-gcp-qa-4th-pass.xml | execution_plan | executed | QA 4th pass |
| remaining-runpod-gcp-qa.xml | execution_plan | executed | QA 1st pass |
| runpod-debug-profiling-execution-final-debug-plan-post-alignment.xml | execution_plan | executed | RunPod post-alignment debug |
| runpod-debug-profiling-execution-final-debug-plan.xml | execution_plan | executed | RunPod final debug plan |
| runpod-debug-profiling-execution-final.xml | execution_plan | executed | RunPod execution final |
| runpod-debug-profiling.xml | execution_plan | executed | RunPod debug profiling |
| runpod-dev-verification-plan-for-realz-maybe.xml | execution_plan | executed | RunPod verification (2nd attempt) |
| runpod-dev-verification-plan.xml | execution_plan | executed | RunPod verification (1st attempt) |
| runpod-for-quick-dev-env-use.md | document | implemented | RunPod as dev environment |
| runpod-post-alignment-qa-decisions-2026-03-14.md | document | reference | RunPod QA decisions |
| runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md | document | reference | Provider deep analysis |
| s3-mounting-testing-and-simulation-plan.md | plan | partial | S3 multi-cloud abstraction |
| s3-mounting-testing-user-prompt.md | prompt | reference | S3 plan user prompt |
| skypilot-and-finops-complete-report.md | research_report | reference | SkyPilot architecture + FinOps |
| skypilot-fake-mock-ssh-test-suite-plan.md | plan | planned | Local SkyPilot YAML testing |
| skypilot-pulumi-gcp-runpod-mlflow-dvc-docker-registry-summary.md | document | reference | Infrastructure status audit |
| skypilot-spot-preemption-checkpoint-research-report.md | research_report | reference | Spot recovery mechanisms |
| skypilot-spot-resume.md | document | implemented | Spot resume double-check |

---

## Cross-References

- **Infrastructure theme:** Docker images, Compose files, security hardening
- **Evaluation theme:** Factorial experiment runs on GCP, cost tracking
- **Models theme:** SAM3/VesselFM require cloud GPU (>=16 GB VRAM)
- **Harness theme:** Ralph monitor skill, cold-start prompts
- **KG domains:** `cloud.yaml` (providers, DVC remotes, GCS buckets), `infrastructure.yaml` (gpu_compute, docker_registry)
- **Key metalearning:** `2026-03-16-unauthorized-aws-s3-architecture-migration.md`, `2026-03-14-skypilot-bare-vm-docker-violation.md`, `2026-03-16-runpod-dev-not-primary-recurring-confusion.md`

---

## Constituent Documents

1. `cloud-architecture-decisions-2026-03-14.md`
2. `cloud-tutorial.md`
3. `cold-start-prompt-gcp-3rd-debug-run.md`
4. `cold-start-prompts-pre-gcp-housekeeping.md`
5. `cold-start-track-b-runpod-gcp-qa.md`
6. `dev-runpod-doublecheck-code-review.xml`
7. `docker-pull-runpod-provisioning-mlflow-cloud-run-multipart-upload-report.md`
8. `gcp-phase-b-d-cold-start.xml`
9. `gcp-setup-tutorial.md`
10. `gcp-spot-with-skypilot-and-pulumi-up-plan.xml`
11. `gcp-vs-lambda-availability.md`
12. `hetzner-mlflow-plan.md`
13. `mlflow-metrics-and-params-double-check-before-large-scale-gcp-work.md`
14. `oracle-cloud-free.txt`
15. `pr-e-cost-reporting-plan.xml`
16. `pr1-finops-infrastructure-timing-plan.md`
17. `pre-full-gcp-housekeeping-and-qa.xml`
18. `pre-full-gcp-training-qa-plan.md`
19. `pre-gcp-master-plan.xml`
20. `pulumi-iac-implementation-guide.md`
21. `ralph-loop-for-cloud-monitoring.md`
22. `remaining-runpod-gcp-qa-2nd-pass.xml`
23. `remaining-runpod-gcp-qa-3rd-pass.xml`
24. `remaining-runpod-gcp-qa-4th-pass.xml`
25. `remaining-runpod-gcp-qa.xml`
26. `runpod-debug-profiling-execution-final-debug-plan-post-alignment.xml`
27. `runpod-debug-profiling-execution-final-debug-plan.xml`
28. `runpod-debug-profiling-execution-final.xml`
29. `runpod-debug-profiling.xml`
30. `runpod-dev-verification-plan-for-realz-maybe.xml`
31. `runpod-dev-verification-plan.xml`
32. `runpod-for-quick-dev-env-use.md`
33. `runpod-post-alignment-qa-decisions-2026-03-14.md`
34. `runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md`
35. `s3-mounting-testing-and-simulation-plan.md`
36. `s3-mounting-testing-user-prompt.md`
37. `skypilot-and-finops-complete-report.md`
38. `skypilot-fake-mock-ssh-test-suite-plan.md`
39. `skypilot-pulumi-gcp-runpod-mlflow-dvc-docker-registry-summary.md`
40. `skypilot-spot-preemption-checkpoint-research-report.md`
41. `skypilot-spot-resume.md`
