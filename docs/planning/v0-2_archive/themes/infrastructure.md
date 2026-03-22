---
theme: infrastructure
doc_count: 48
last_synthesized: "2026-03-22"
implementation_health: mostly_implemented
kg_domains: [infrastructure]
---

# Theme: Infrastructure -- Docker, Prefect, DevEx, CI

This theme covers the foundational platform engineering layer: Docker-per-flow isolation,
Prefect orchestration, developer experience automation, MONAI consolidation, security
hardening, profiling/benchmarking, and the overnight execution infrastructure that
validated the full pipeline.

---

## Key Scientific Insights

### 1. Three-Tier Multi-Stage Docker Architecture Is the Execution Model

Docker is NOT optional -- it IS the execution model and the reproducibility guarantee.
The architecture evolved from two separate base images (CPU vs GPU with duplicated deps)
to a clean 3-layer hierarchy:

- **Tier A (GPU):** `nvidia/cuda:12.6.3 -> minivess-base:latest` -- 10 flows (train, post-training, analysis, etc.)
- **Tier B (CPU):** `python:3.13 -> minivess-base-cpu:latest` -- biostatistics flow
- **Tier C (Light):** `python:3.13 -> minivess-base-light:latest` -- dashboard, pipeline

Each base uses 2-stage builder->runner pattern. Flow Dockerfiles are THIN (only COPY, ENV,
CMD) -- never apt-get or uv. This was implemented in full and validated via Docker tiers plan.

**Source:** `docker-base-improvement-plan.md`, `docker-tiers-plan.xml`

### 2. Vision Enforcement: Hard Gates Against Architecture Bypass

The AI assistant bypassed Docker+Prefect architecture 6 times in 6 days to achieve "quick"
model training results. Documentation alone failed because: (a) goal substitution --
when told "train SAM3," the AI optimized for loss reduction, not architecture demonstration;
(b) no hard gates -- nothing PREVENTED `uv run python` from training; (c) the
`_prefect_compat.py` with `PREFECT_DISABLED=1` made bypass trivially easy.

The fix: `_require_docker_context()` in `train_flow.py` enforces Docker at runtime.
Escape hatch: `MINIVESS_ALLOW_HOST=1` for pytest ONLY. The STOP protocol (Source,
Tracking, Outputs, Provenance) was added to CLAUDE.md as Rule #19.

**Source:** `minivess-vision-enforcement-plan.md`, `minivess-vision-enforcement-plan-execution.xml`

### 3. Prefect SQLite Flakiness From Module-Level Env Mutations

Four test files set `PREFECT_DISABLED=1` at module level (`os.environ[...]`), which ran
at import time and conflicted with the session-level `prefect_test_harness()` SQLite server.
Depending on pytest collection order, the env var mutation caused intermittent connection
errors and SQLite lock contention. Fix: removed all module-level mutations; the session-level
harness is sufficient for all test modes.

**Source:** `robustifying-flaky-prefect-sqlite-issues.md`

### 4. Docker Security Hardening: CIS Benchmark + MLSecOps

The security audit synthesized NIST SP 800-190, CIS Docker Benchmark v1.8.0, OWASP Docker
Top 10, NVIDIA CVE-2025-23266 ("NVIDIAScape"), and OpenSSF MLSecOps Whitepaper (2025). Key
implementations: cap_drop ALL + no-new-privileges on all containers, seccomp enforce profiles,
Trivy vulnerability scanning, CycloneDX SBOM generation, SOPS+age secrets management, and
optional MLflow basic-auth + Falco runtime monitoring.

**Source:** `docker-security-hardening-mlsecops-report.md`, `docker-security-hardening-mlsecops-plan.xml`, `docker-security-hardening-mlsecops-plan-2nd-pass.xml`

### 5. MONAI Performance: OOM From Spacingd Resampling Outliers

Volume `mv02.nii.gz` with 4.97 um spacing resampled from (512,512,61) to (2545,2545,305)
= 7.9 GB per volume. Multiple such volumes via CacheDataset + DataLoader workers caused
RSS to reach 43+ GB, triggering Linux OOM killer and terminal crash. Fix: MONAI-native
approach with proper patch-based training (no resampling to isotropic) + adaptive cache
ratio based on available RAM.

**Source:** `monai-performance-optimization-plan.md`

### 6. 17 Docker/Orchestration Bugs Found in First Staging Run

The first full staging run (5 models x 3 folds x 2 epochs x 3 flows) revealed 17 distinct
bugs including: MLflow 3.x YAML folded scalar bug, wrong env var names, missing Docker
network, CDI GPU check, wrong volume prefixes, missing boto3/psycopg2 deps, None MLflow
tag TypeError, and Docker Compose V2 .env discovery bug. All 14 initial fixes applied in
one systematic pass.

**Source:** `docker-improvements-for-debug-training.md`

### 7. Warning Suppression Is a DevEx Requirement

Every Python command in MinIVess produced 3-5 lines of non-actionable warnings (ONNX Runtime
C++ device discovery, cuda.cudart FutureWarning, MetricsReloaded SyntaxWarning). These were
suppressed at entry points (conftest.py, train_flow.py, etc.) following CLAUDE.md DG1.7:
"suppress warnings at entry point, NEVER tell user to 'just ignore'."

**Source:** `warning-logger-level-analysis-and-devex-improvement.md`

### 8. The v0.1-alpha to v2 Modernization Scope

The full modernization plan covers: Python 3.8->3.12+, Poetry->uv, OmegaConf->Hydra-zen,
MONAI 1.2->latest, adding Prefect 3.x (5 flows), MLflow, DVC, SkyPilot, Docker-per-flow,
Optuna HPO, BentoML/ONNX deployment, and an agent layer with Pydantic AI. The project
was explicitly framed as a portfolio piece for Nature Protocols + Medical Image Analysis.

**Source:** `modernize-minivess-mlops-plan.md`

---

## Architectural Decisions Made

| Decision | Outcome | Source Doc | KG Node |
|----------|---------|-----------|---------|
| Container strategy | Docker-per-flow, 12 services, Compose profiles | docker-base-improvement-plan.md | infrastructure.container_strategy |
| Docker architecture | Three-tier multi-stage (GPU/CPU/Light bases) | docker-tiers-plan.xml | infrastructure.docker_architecture |
| CI/CD platform | GitHub Actions + CML (EXPLICITLY DISABLED, credits) | ci-reporting-plan.md | infrastructure.ci_cd_platform |
| Secrets management | Dynaconf + .env.example (single source of truth) | docker-security-hardening-mlsecops-report.md | infrastructure.secrets_management |
| Vision enforcement | _require_docker_context() + STOP protocol | minivess-vision-enforcement-plan.md | -- |
| Prefect execution model | All flows via `prefect deployment run`, never bare Python | prefect-container-production-grade-hardening-plan.xml | -- |
| Security baseline | CIS Benchmark + OWASP Docker Top 10 + seccomp | docker-security-hardening-mlsecops-plan.xml | -- |
| Config framework | Hydra-zen (train) + Dynaconf (deploy) | devex-and-prefect-execution-plan.xml | -- |
| Profiling | torch.profiler integration, YAML-configurable, default ON | profiler-benchmarking-plan.md | -- |
| Package manager | uv ONLY (--all-extras required for dev) | modernize-minivess-mlops-plan.md | -- |

---

## Implementation Status

| Document | Type | Status | Key Deliverable |
|----------|------|--------|-----------------|
| batch-script-executor-monai-and-scripts.sh | script | executed | Overnight batch execution wrapper |
| ci-reporting-plan.md | plan | planned | Bootstrap CI reporting (percentile + BCa) |
| data-prefect-flow-improvement-for-test-datasets-execution.xml | execution_plan | partial | Test dataset Prefect wiring |
| data-prefect-flow-improvement-for-test-datasets.md | document | partial | Flow 1 data engineering gaps |
| devex-and-prefect-execution-plan.xml | execution_plan | implemented | DevEx + Prefect orchestration |
| devex-automation-plan.md | plan | implemented | Single-command multi-env compute |
| docker-base-improvement-plan.md | plan | implemented | 3-layer Docker hierarchy |
| docker-improvements-for-debug-training.md | document | implemented | 17-bug fix catalog from staging run |
| docker-security-hardening-mlsecops-plan-2nd-pass.xml | execution_plan | implemented | Seccomp, SOPS, MLflow auth, Falco |
| docker-security-hardening-mlsecops-plan.xml | execution_plan | implemented | H3/H4 multi-stage, cap_drop, Trivy |
| docker-security-hardening-mlsecops-report.md | research_report | reference | Total quality Docker audit |
| docker-tiers-plan.xml | execution_plan | implemented | Flow-specific base image tiers |
| infrastructure-timing-execution.xml | execution_plan | partial | Shell-level timing + cost logging |
| minivess-vision-enforcement-plan-execution.xml | execution_plan | implemented | Hard gates against architecture bypass |
| minivess-vision-enforcement-plan.md | plan | implemented | STOP protocol + Docker enforcement |
| mlops-practices-report.md | research_report | reference | 19-paper MLOps survey |
| modernize-minivess-mlops-plan-prompt.md | plan | reference | Original user prompt for v2 rewrite |
| modernize-minivess-mlops-plan.md | plan | implemented | Comprehensive v2 modernization plan |
| monai-performance-optimization-plan.md | plan | implemented | OOM fix, MONAI-native patches |
| overnight-child-01-acquisition.xml | execution_plan | executed | Acquisition flow overnight task |
| overnight-child-01.xml | execution_plan | executed | Overnight child plan 1 |
| overnight-child-02-annotation.xml | execution_plan | executed | Annotation flow overnight task |
| overnight-child-02.xml | execution_plan | executed | Overnight child plan 2 |
| overnight-child-03-dashboard.xml | execution_plan | executed | Dashboard flow overnight task |
| overnight-child-03.xml | execution_plan | executed | Overnight child plan 3 |
| overnight-child-04.xml | execution_plan | executed | Overnight child plan 4 |
| overnight-child-debug-configs.xml | execution_plan | executed | Debug config overnight task |
| overnight-child-hydra-bridge.xml | execution_plan | executed | Hydra bridge overnight task |
| overnight-child-monai-eval.xml | execution_plan | executed | MONAI eval overnight task |
| overnight-child-prefect-docker.xml | execution_plan | executed | Prefect Docker overnight task |
| overnight-master-flow0-annotation-dashboard.sh | script | executed | Flow 0 master script |
| overnight-master-plan.sh | script | executed | Master overnight script |
| overnight-master-plan.xml | execution_plan | executed | Master overnight plan |
| overnight-prefect-docker-monai.sh | script | executed | Prefect/Docker/MONAI script |
| overnight-script-consolidation.sh | script | executed | Script consolidation |
| prefect-container-production-grade-hardening-plan.xml | execution_plan | implemented | All 9 flows hardened |
| prefect-docker-optimization-and-monai-consolidation.md | document | implemented | MONAI-first consolidation |
| prefect-mlflow-docker-debug-fixing.xml | execution_plan | implemented | 15-task code cleanup (PR #421) |
| profiler-benchmarking-plan-double-check.md | plan | partial | Profiler double-check |
| profiler-benchmarking-plan-execution.xml | execution_plan | partial | Profiler phases execution |
| profiler-benchmarking-plan.md | plan | partial | GPU/CPU profiling architecture |
| profiler-benchmarking-user-prompt.md | prompt | reference | Original user prompt for profiling |
| r6-remediation-plan.md | plan | implemented | 2nd-pass code review (868->940 tests) |
| robustifying-flaky-prefect-sqlite-issues.md | document | implemented | SQLite flakiness fix |
| root-cause-bug-fixing-plan.xml | execution_plan | implemented | 15-failure batch fix (Rule #23) |
| run-child1-acq.sh | script | executed | Acquisition child runner |
| run-overnight-flow0.sh | script | executed | Flow 0 overnight runner |
| warning-logger-level-analysis-and-devex-improvement.md | document | partial | Warning suppression plan |

---

## Cross-References

- **Evaluation theme:** Biostatistics Flow uses Prefect tasks, Docker isolation
- **Cloud theme:** SkyPilot YAMLs reference Docker images, Docker registry config
- **Models theme:** Model adapters run inside Docker containers via train flow
- **Harness theme:** Overnight runners are Claude harness execution scripts
- **KG domain:** `infrastructure.yaml` -- container_strategy, docker_architecture, ci_cd_platform, secrets_management
- **Key metalearning:** `2026-03-06-standalone-script-antipattern.md`, `2026-03-07-silent-existing-failures.md`, `2026-03-18-whac-a-mole-serial-failure-fixing.md`

---

## Constituent Documents

1. `batch-script-executor-monai-and-scripts.sh`
2. `ci-reporting-plan.md`
3. `data-prefect-flow-improvement-for-test-datasets-execution.xml`
4. `data-prefect-flow-improvement-for-test-datasets.md`
5. `devex-and-prefect-execution-plan.xml`
6. `devex-automation-plan.md`
7. `docker-base-improvement-plan.md`
8. `docker-improvements-for-debug-training.md`
9. `docker-security-hardening-mlsecops-plan-2nd-pass.xml`
10. `docker-security-hardening-mlsecops-plan.xml`
11. `docker-security-hardening-mlsecops-report.md`
12. `docker-tiers-plan.xml`
13. `infrastructure-timing-execution.xml`
14. `minivess-vision-enforcement-plan-execution.xml`
15. `minivess-vision-enforcement-plan.md`
16. `mlops-practices-report.md`
17. `modernize-minivess-mlops-plan-prompt.md`
18. `modernize-minivess-mlops-plan.md`
19. `monai-performance-optimization-plan.md`
20. `overnight-child-01-acquisition.xml`
21. `overnight-child-01.xml`
22. `overnight-child-02-annotation.xml`
23. `overnight-child-02.xml`
24. `overnight-child-03-dashboard.xml`
25. `overnight-child-03.xml`
26. `overnight-child-04.xml`
27. `overnight-child-debug-configs.xml`
28. `overnight-child-hydra-bridge.xml`
29. `overnight-child-monai-eval.xml`
30. `overnight-child-prefect-docker.xml`
31. `overnight-master-flow0-annotation-dashboard.sh`
32. `overnight-master-plan.sh`
33. `overnight-master-plan.xml`
34. `overnight-prefect-docker-monai.sh`
35. `overnight-script-consolidation.sh`
36. `prefect-container-production-grade-hardening-plan.xml`
37. `prefect-docker-optimization-and-monai-consolidation.md`
38. `prefect-mlflow-docker-debug-fixing.xml`
39. `profiler-benchmarking-plan-double-check.md`
40. `profiler-benchmarking-plan-execution.xml`
41. `profiler-benchmarking-plan.md`
42. `profiler-benchmarking-user-prompt.md`
43. `r6-remediation-plan.md`
44. `robustifying-flaky-prefect-sqlite-issues.md`
45. `root-cause-bug-fixing-plan.xml`
46. `run-child1-acq.sh`
47. `run-overnight-flow0.sh`
48. `warning-logger-level-analysis-and-devex-improvement.md`
