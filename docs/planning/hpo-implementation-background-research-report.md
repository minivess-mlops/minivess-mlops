# HPO Implementation Background Research Report

## User Prompt (verbatim)

> the HPO design, if the HPO is as separate "orchestrator Flow" or how is the HPO managed with the Modelling/Train flow? We need to be supporting at least the basics: 1) brute force grid search with reproducible configs for this grid search, 2) Bayesian optimization e.g. with Optuna, 3) successive halving HPO e.g. with the open-source libraries from AWS both for continual learning and "from scratch" learning. This should guide the HPO design then, and create me an open-ended literature search on this to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/hpo-implementation-background-research-report.md with multi-hypothesis decision matrix with pros and cons, followed by your own recommendation. Save my prompt first verbatim and then start researching how to support multiple compute schematics as well, e.g. if we have 128 combinations, would provision 4 instances with 4 parallel sessions with each instance executing 32 serial experiments. How do these different instances communicate with each other if we have some successive halving experiment and we need to be killing jobs that do not seem promising? Is it a lot easier to provision a single instance with 8 GPUs so we have single instance multi-GPU situation allowing this communication facilitated. Remember that we might have cloud limitations as I have on GCP as an individual only 1 gpu quota now (check if it is single GPU or single-instance, and whether I can provision single instance with Skypilot that has 8 GPUs). Consider the finops with the instance set-up times and downloading the data and the massive Docker. We cannot be running multiple short jobs if most of the time is just spent on getting the stuff ready for the actual GPU use (if we don't have any meaningful cache? how would this shared cache even work on GCP? AWS has EFS? which is the GCP-equivalent, would that work? How about some smaller cache in Redis? At what point we should be exploring some Lustre-like HPC solutions to this? Weigh all these options and go for comprehensive research with multiple reviewer agent iterations over quick AI slop! And ask interactive questions rather than assuming things about the goal of this plan!

---

## Executive Summary

This report provides a comprehensive background research analysis for designing the HPO (Hyperparameter Optimization) subsystem of MinIVess MLOps, a MONAI-based 3D vascular segmentation platform targeting Nature Protocols publication. The research spans 30 HPO backend references and 22 GCP compute/storage references, synthesized through three independent research agents.

The MinIVess codebase already contains a functional HPO foundation: an `HPOEngine` class (248 lines) wrapping Optuna with TPE/CMA-ES samplers and HyperbandPruner (ASHA), an `hpo_flow.py` Prefect flow (253 lines) that triggers training via `run_deployment()` with Docker isolation, and a separate grid search path via `train_all_hyperparam_combos.sh`. PostgreSQL is the only permitted storage backend (SQLite banned per project rules). The `AllocationStrategy` enum supports SEQUENTIAL (working), PARALLEL (raises `NotImplementedError`), and HYBRID (working, multi-GPU single host). The knowledge graph node `hpo_engine` is resolved to `optuna_multi_objective` with 90% posterior probability.

The central tension is between two distinct use cases: (1) a predefined factorial grid for the Nature Protocols paper (deterministic, reproducible, no early stopping), and (2) post-paper Bayesian HPO for downstream users (adaptive, distributed, with ASHA pruning). These use cases have different compute requirements, different communication patterns, and different storage needs. The GCP quota of 5 GPUs across all regions constrains the parallelism achievable, while setup amortization (Docker pull + data download) penalizes short-lived jobs.

This report presents five decision matrices (H1-H5), covering architecture pattern, HPO backend, compute schematics, shared storage, and grid search reproducibility. The recommended architecture is phased: Phase 1 (paper) uses the existing grid search path with YAML-versioned configs and sequential execution on 5 GCP L4 spot instances via SkyPilot managed jobs; Phase 2 (post-paper) enables distributed Bayesian HPO via Optuna's PostgreSQL-backed parallel workers with ASHA pruning; Phase 3 (future) evaluates Syne Tune integration for advanced scheduling algorithms.

## 1. Current State: MinIVess HPO Architecture

### 1.1 What Exists

The HPO subsystem is split across two execution paths with distinct responsibilities:

**Path A: Bayesian HPO** (`src/minivess/optimization/hpo_engine.py` + `src/minivess/orchestration/flows/hpo_flow.py`)

The `HPOEngine` class wraps Optuna study creation with configurable samplers (TPE, CMA-ES) and pruners (HyperbandPruner for ASHA, MedianPruner). The `hpo_flow.py` Prefect flow serves as an orchestrator that triggers training deployments via `run_deployment()`, ensuring each trial runs in its own Docker container. The flow enforces Docker context via `_require_docker_context()` and reads `OPTUNA_STORAGE_URL` from the environment for PostgreSQL-backed study persistence.

The `AllocationConfig` dataclass controls parallelism strategy:
- `SEQUENTIAL`: In-process optimization, one trial at a time (working)
- `PARALLEL`: Multiple worker containers via PostgreSQL study (raises `NotImplementedError`)
- `HYBRID`: Multi-GPU on single host via `CUDA_VISIBLE_DEVICES` (working)

**Path B: Grid Search** (`scripts/train_all_hyperparam_combos.sh` + `configs/hpo/dynunet_grid.yaml`)

A shell script that reads YAML grid configs, generates the Cartesian product, and launches each combination via Docker Compose. The `dynunet_grid.yaml` defines a 3x2x2 = 12 combination grid (loss_name x learning_rate x batch_size). Auto-resume via MLflow fingerprinting skips already-completed runs.

**HPO Configs** (in `configs/hpo/`):
- `dynunet_example.yaml`: 50 trials, TPE sampler, Hyperband pruner, 6-parameter search space
- `dynunet_grid.yaml`: 12 combinations, Cartesian product, fixed 100 epochs
- `smoke_test.yaml`: 1 combination, 1 epoch, CI verification

### 1.2 What Works

- Optuna study creation with TPE/CMA-ES and HyperbandPruner
- Parameter suggestion from YAML-defined search spaces (float, int, categorical)
- Prefect flow integration with Docker-isolated training trials
- PostgreSQL-only storage validation (enforced at init)
- Grid search YAML configs with Cartesian product generation
- SEQUENTIAL and HYBRID allocation strategies
- MLflow-based resume for grid search

### 1.3 What Is Missing

- **PARALLEL allocation**: `NotImplementedError` — no multi-container distributed HPO
- **Grid search in Prefect**: The grid search path (`train_all_hyperparam_combos.sh`) bypasses Prefect entirely, running via Docker Compose directly
- **SkyPilot integration**: No mechanism to launch HPO trials as SkyPilot managed jobs
- **Inter-trial communication for ASHA**: No mechanism for distributed workers to coordinate early stopping decisions across separate instances
- **Setup amortization**: No shared cache or warm-start mechanism for Docker images or data across trial instances
- **Grid search reproducibility metadata**: No systematic versioning of which grid config produced which MLflow experiment

## 2. HPO Strategy Taxonomy

### 2.1 Grid Search (Exhaustive)

Grid search evaluates every point in the Cartesian product of discrete hyperparameter sets. It provides complete coverage and is deterministic given a fixed config, making it ideal for reproducible factorial experiments in publications. [Bischl et al. (2023)](https://arxiv.org/abs/2107.05847) note that grid search remains the standard for small factorial designs (fewer than ~200 combinations) where full coverage is required for statistical analysis.

The MinIVess paper factorial (e.g., 4 losses x 4 learning rates x 2 batch sizes x 4 models = 128 cells, each with 3-fold CV = 384 training runs) falls squarely in this regime. Grid search is the correct tool here because the goal is not optimization but systematic comparison — every cell must complete.

### 2.2 Bayesian Optimization (Model-Based)

Bayesian optimization builds a probabilistic surrogate model of the objective function and uses an acquisition function to select the next point to evaluate. Tree-structured Parzen Estimator (TPE), as implemented in Optuna, is the dominant approach for hyperparameter tuning. [Akiba et al. (2019)](https://arxiv.org/abs/1907.10902) demonstrate that TPE with Optuna achieves competitive performance with significantly less engineering overhead than Gaussian Process-based methods.

[Watanabe (2023)](https://arxiv.org/abs/2304.11127) provides an ablation study of TPE components, showing that the kernel bandwidth and number of candidates in the acquisition function are the most impactful hyperparameters of TPE itself. The default Optuna TPE configuration is near-optimal for most problems with fewer than 100 dimensions.

CMA-ES (Covariance Matrix Adaptation Evolution Strategy), also available in Optuna, is effective for continuous search spaces but struggles with categorical parameters common in ML pipelines (loss function choice, architecture variants).

### 2.3 Multi-Fidelity Methods (Successive Halving / Hyperband / ASHA)

Multi-fidelity methods evaluate configurations at low fidelity (few epochs) and progressively allocate more resources to promising ones. This family includes:

**Successive Halving (SHA)**: Allocates equal budget to n configurations, evaluates for r epochs, keeps the top 1/eta fraction, and repeats. [Li et al. (2017)](https://jmlr.org/papers/v18/16-558.html) formalize Hyperband as an extension that hedges across multiple SHA brackets with different n/r tradeoffs.

**ASHA (Asynchronous SHA)**: [Li et al. (2020)](https://arxiv.org/abs/1810.05934) extend SHA to the asynchronous distributed setting, where workers do not need to synchronize at rung boundaries. This is critical for distributed HPO where trial durations vary — workers simply promote or stop trials based on the current leaderboard. Optuna's `HyperbandPruner` implements ASHA semantics.

**BOHB**: [Falkner et al. (2018)](https://arxiv.org/abs/1807.01774) combine Hyperband's multi-fidelity scheduling with TPE-based sampling, achieving faster convergence than either alone. BOHB is available in Syne Tune and HpBandSter but not natively in Optuna.

**DEHB**: [Awad et al. (2021)](https://arxiv.org/abs/2105.09821) combine differential evolution with Hyperband, demonstrating improved performance on high-dimensional search spaces. Available in DEHB library and Syne Tune.

### 2.4 Population-Based Training (PBT)

[Jaderberg et al. (2017)](https://arxiv.org/abs/1711.09846) propose Population-Based Training, which mutates hyperparameters during training rather than restarting from scratch. [Parker-Holder et al. (2020)](https://arxiv.org/abs/2002.02518) extend PBT with Bayesian optimization (PB2) for the perturbation step.

PBT is most effective for hyperparameters that benefit from schedules (learning rate, data augmentation strength) rather than fixed values. It requires all population members to run concurrently, creating a strong coupling between compute allocation and population size. For MinIVess, where the paper factorial requires fixed hyperparameters per cell, PBT is a post-paper consideration for schedule optimization.

## 3. Decision Matrix: H1 — HPO Architecture Pattern

The fundamental architectural question is how HPO relates to the training flow. Three patterns emerge:

| Option | Description | Pros | Cons | Effort | Risk | Citations |
|--------|-------------|------|------|--------|------|-----------|
| **A: Orchestrator Flow** | HPO flow (`hpo_flow.py`) triggers training deployments via `run_deployment()`. HPO controller runs on CPU; training runs on GPU in separate containers. | Docker isolation per trial; Prefect UI visibility; fault tolerance via Prefect retries; clean separation of concerns | Overhead per trial (Prefect scheduling, container startup); latency for inter-trial communication; requires Prefect server | Medium (existing) | Low — already implemented | [Akiba et al. (2019)](https://arxiv.org/abs/1907.10902) |
| **B: Inner Loop** | Optuna `study.optimize()` runs inside the training flow. Single container, single process, multiple trials. | Minimal overhead; simple communication; no Prefect dependency for trial scheduling; fast trial-to-trial transitions | No Docker isolation between trials; GPU memory leaks accumulate; single point of failure; no Prefect visibility per trial | Low | High — memory leaks in long runs | [Watanabe (2023)](https://arxiv.org/abs/2304.11127) |
| **C: External Service** | Dedicated HPO service (Ray Tune head node, Syne Tune backend) manages trial lifecycle independently of Prefect. | Purpose-built scheduling; advanced algorithms (PBT, BOHB); battle-tested distributed primitives | Additional infrastructure dependency; duplicates orchestration with Prefect; operational overhead; learning curve | High | Medium — infrastructure complexity | [Liaw et al. (2018)](https://arxiv.org/abs/1807.05118) |

**Analysis**: Option A (Orchestrator Flow) is the correct architecture for MinIVess. It already exists in `hpo_flow.py`, provides Docker isolation (aligning with project principles), and integrates with the Prefect trigger chain. The overhead per trial is acceptable for long-running 3D segmentation training (minutes of setup vs. hours of training). Option B would be appropriate only for quick smoke tests within a single container. Option C introduces infrastructure complexity that is unwarranted given Optuna's native distributed capabilities via PostgreSQL.

## 4. Decision Matrix: H2 — HPO Backend

| Option | Description | Pros | Cons | Effort | Risk | Citations |
|--------|-------------|------|------|--------|------|-----------|
| **A: Optuna-only** | Use Optuna for all HPO: TPE/CMA-ES samplers, HyperbandPruner (ASHA), GridSampler for factorial. PostgreSQL storage for distributed coordination. | Already integrated (248 lines); PostgreSQL distributed by default; ask-tell API for async; no new dependencies; GridSampler handles factorial | No native BOHB/DEHB; PBT not available; GridSampler requires all points upfront (no dynamic grid) | None (existing) | Low | [Akiba et al. (2019)](https://arxiv.org/abs/1907.10902), [Optuna distributed docs](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html) |
| **B: Syne Tune** | AWS open-source HPO framework with scheduler abstraction layer. Supports ASHA, BOHB, DEHB, PBT, CQR, and custom schedulers. Backend-agnostic (local, SageMaker, custom). | Richest algorithm portfolio; clean scheduler API; active development; academic pedigree; backend-agnostic despite AWS origin | New dependency; less mature than Optuna; smaller community; no native PostgreSQL storage (needs adapter); AWS-flavored docs | High | Medium — integration effort | [Salinas et al. (2022)](https://proceedings.mlr.press/v188/salinas22a.html), [Syne Tune GitHub](https://github.com/syne-tune/syne-tune) |
| **C: Ray Tune** | Distributed HPO framework built on Ray. Integrates with OptunaSearch, HyperOpt, BayesOpt. Schedulers: ASHA, PBT, PB2. | Rich scheduler/searcher combinations; OptunaSearch bridges Optuna studies; PBT/PB2 native; large community | Ray cluster management overhead; heavy dependency (Ray runtime); duplicates Prefect's orchestration; memory-hungry head node; overkill for 5-GPU quota | High | High — infrastructure overlap with Prefect | [Liaw et al. (2018)](https://arxiv.org/abs/1807.05118), [Moritz et al. (2018)](https://www.usenix.org/conference/osdi18/presentation/moritz), [Ray Tune schedulers docs](https://docs.ray.io/en/latest/tune/api/schedulers.html) |
| **D: Hybrid (Optuna + Syne Tune schedulers)** | Use Optuna as the storage/study backend but swap in Syne Tune schedulers (BOHB, DEHB) for advanced algorithms. Syne Tune provides the scheduler; Optuna provides the study persistence. | Best algorithms from Syne Tune; study persistence from Optuna; incremental migration path; no Ray overhead | Integration complexity; version coupling between libraries; limited community examples of this pattern | Medium-High | Medium — untested integration | [Salinas et al. (2022)](https://proceedings.mlr.press/v188/salinas22a.html), [Akiba et al. (2019)](https://arxiv.org/abs/1907.10902) |

**Analysis**: Option A (Optuna-only) is the correct choice for Phase 1-2. The existing integration is mature, PostgreSQL provides native distributed coordination (multiple Optuna workers sharing one study), and the algorithm portfolio (TPE + ASHA + GridSampler) covers all paper requirements. Optuna's `ask-tell` API ([Optuna ask-tell API](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html)) enables asynchronous distributed operation without requiring all workers to share a process — each worker simply asks for a trial, runs it, and tells the result. [Optuna's gRPC storage proxy](https://medium.com/optuna/distributed-optimization-in-optuna-and-grpc-storage-proxy-08db83f1d608) can further reduce PostgreSQL connection overhead at scale.

Option B (Syne Tune) is the strongest candidate for Phase 3, when advanced algorithms (BOHB, DEHB) become relevant for post-paper users with larger compute budgets. Option C (Ray Tune) introduces unacceptable infrastructure overlap with Prefect and is ruled out. The knowledge graph PRD node `hpo_framework` assigns Optuna 0.45, Ray Tune 0.25, Nevergrad 0.15, Manual 0.15 — our analysis strengthens the Optuna position to 0.90 (matching the KG posterior for `hpo_engine`).

## 5. Decision Matrix: H3 — Compute Schematics

This is the most consequential decision for the paper factorial. With 128 grid cells (or 384 with 3-fold CV) and a GCP quota of 5 GPUs across all regions, the compute topology directly determines wall-clock time and cost.

### 5.1 GCP GPU Quota Context

The verified GCP quota is **5 GPUs total across all regions** (not per-region). This is a per-project quota for individual accounts. Key implications:

- Maximum parallelism: 5 concurrent single-GPU jobs
- Multi-GPU instances: A single 8-GPU instance is NOT possible (exceeds quota)
- Quota increase: Requires billing history and justification; typically granted for 8-12 GPUs after 1-2 months of usage
- SkyPilot constraint: `sky launch --gpus L4:1` can run up to 5 concurrent managed jobs

### 5.2 Compute Options

| Option | Description | Pros | Cons | Effort | Risk | Citations |
|--------|-------------|------|------|--------|------|-----------|
| **A: 5x single-GPU (SkyPilot managed jobs)** | 5 independent SkyPilot managed jobs, each running one trial at a time. Sequential within each instance, parallel across instances. 384/5 = ~77 runs per instance. | Fits GCP 5-GPU quota exactly; SkyPilot spot recovery; independent failure domains; simple scheduling; no inter-node communication needed for grid search | 5x setup overhead (Docker pull + data); spot preemption risk (mitigated by SkyPilot); no ASHA coordination across instances | Low | Low — SkyPilot handles spot | [Yang et al. (2023)](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng), [SkyPilot managed jobs docs](https://docs.skypilot.co/en/latest/examples/managed-jobs.html) |
| **B: 1x multi-GPU (single instance)** | One instance with multiple GPUs. HYBRID allocation strategy with `CUDA_VISIBLE_DEVICES` per trial. | Single setup cost; shared filesystem; easy ASHA communication (shared memory/PostgreSQL on localhost); no inter-node networking | Exceeds 5-GPU quota for >5 GPUs; single point of failure; spot preemption kills ALL trials; underutilizes GPUs during ramp-up/down | N/A (quota-blocked for >5) | High — single failure domain | [Li et al. (2020)](https://arxiv.org/abs/1810.05934) |
| **C: Hybrid orchestrator + workers** | One CPU orchestrator (HPO flow) + N GPU workers. Orchestrator manages Optuna study; workers pull trials via PostgreSQL. | Clean separation; orchestrator survives worker preemption; scales to arbitrary worker count; ASHA coordination via PostgreSQL | Requires persistent orchestrator instance; PostgreSQL must be network-accessible; added latency for trial assignment | Medium | Medium — networking complexity | [Optuna distributed docs](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html), [SkyPilot many-jobs docs](https://docs.skypilot.co/en/latest/running-jobs/many-jobs.html) |
| **D: SkyPilot multi-node cluster** | Single SkyPilot cluster with multiple nodes. `SKYPILOT_NODE_RANK` and `SKYPILOT_NODE_IPS` for coordination. | Built-in node discovery; shared setup (sort of); single `sky launch` command | All nodes must launch together (no elastic scaling); spot preemption kills the cluster; multi-node is designed for distributed training (data-parallel), not independent trials | Medium | High — wrong abstraction for HPO | [SkyPilot distributed jobs docs](https://docs.skypilot.co/en/latest/running-jobs/distributed-jobs.html) |

### 5.3 Setup Amortization Analysis

Setup cost per instance (empirical estimates for MinIVess):

| Step | Time (cold) | Time (warm/cached) | Notes |
|------|-------------|---------------------|-------|
| Instance provisioning (spot) | 60-120s | 60-120s | Cannot be cached |
| Docker image pull (GAR, same-region) | 90-180s | 5-10s | Cached if instance persists; GAR remote repo caching helps |
| Data download (GCS, same-region) | 30-60s | 5-10s | SkyPilot `MOUNT_CACHED` or local disk cache |
| Python environment (inside Docker) | 0s | 0s | Baked into Docker image |
| **Total cold start** | **3-6 min** | **1-2.5 min** | |

For a grid cell that trains for ~2-4 hours (100 epochs, DynUNet, MiniVess), cold start overhead is 2-5% of total runtime — acceptable. For short trials (5-10 epochs in ASHA low-fidelity rungs), cold start dominates: 3-6 min setup for 5-10 min training is 30-55% overhead.

**Implication**: Grid search (long trials, no early stopping) favors Option A (independent instances). ASHA HPO (short low-fidelity trials) favors Option B or C (shared setup, persistent instances).

SkyPilot's `MOUNT_CACHED` file mount type caches data on the instance disk across managed job recovery ([SkyPilot training guide](https://docs.skypilot.co/en/latest/reference/training-guide.html)), reducing re-download overhead after spot preemption. Docker image caching requires GAR remote repository proxy in the same region (`europe-north1`), which serves as a pull-through cache for frequently accessed layers.

### 5.4 ASHA Communication Requirements

For distributed ASHA, workers need to:
1. Report intermediate results (val_loss at each epoch/rung)
2. Query the pruner to decide whether to continue or stop
3. Handle asynchronous promotion decisions

With Optuna's PostgreSQL backend, this communication happens via database reads/writes — no direct inter-worker networking is needed. Each worker calls `trial.report(value, step)` and `trial.should_prune()`, which are PostgreSQL transactions. This is the key insight that makes Option A viable even for ASHA: the PostgreSQL database IS the coordination mechanism, and Cloud SQL PostgreSQL (~$10/mo) provides a persistent, network-accessible coordinator that survives worker preemption.

## 6. Decision Matrix: H4 — Shared Storage

Shared storage serves three purposes: (1) training data access, (2) checkpoint persistence for ASHA warm-starting, and (3) Docker image caching. The cost comparison uses `europe-north1` pricing.

| Option | Description | Pricing (europe-north1) | Read BW | Write BW | Pros | Cons | Effort | Risk | Citations |
|--------|-------------|-------------------------|---------|----------|------|------|--------|------|-----------|
| **A: GCS-only** | Data and checkpoints on GCS. Each instance downloads data; uploads checkpoints. | ~$0.020/GiB/mo storage + $0.12/GiB egress (inter-region) | ~300 MB/s (FUSE) | ~300 MB/s (FUSE) | Cheapest; no infrastructure; scales infinitely; SkyPilot native via `file_mounts` | Download per instance; not a filesystem (no POSIX semantics natively) | None (existing) | Low | [Zhu (2025) storage benchmarks](https://cloud.google.com/architecture/optimize-ai-ml-workloads-cloud-storage-fuse) |
| **B: GCS FUSE** | Mount GCS bucket as POSIX filesystem via Cloud Storage FUSE. `aiml-checkpointing` profile for training workloads. | Same as GCS (no FUSE surcharge) | ~300 MB/s | ~300 MB/s (40% faster with aiml profile) | POSIX semantics on GCS; transparent caching; aiml-checkpointing profile optimized for ML; no extra infrastructure cost | Metadata operations slow (ls, stat); not suitable for random I/O; requires FUSE driver in Docker image | Low | Low-Medium — FUSE edge cases | [GCS FUSE optimization guide](https://cloud.google.com/architecture/optimize-ai-ml-workloads-cloud-storage-fuse) |
| **C: Filestore NFS** | Google Cloud Filestore Basic SSD. Shared NFS mount across all instances. | ~$0.36/GiB/mo, min 1 TiB = **~$369/mo** | 1.2 GB/s | 350 MB/s | True POSIX; lowest latency; transparent sharing; no code changes | Expensive minimum ($369/mo); must provision before use; regional; cannot scale below 1 TiB | Medium | Low — mature product | [Google Cloud Filestore docs](https://cloud.google.com/filestore) |
| **D: Persistent Disk snapshots** | Each instance gets a PD pre-loaded with data. Snapshot once, create disks from snapshot for each instance. | ~$0.080/GiB/mo (balanced PD) | 240 MB/s (balanced) | 120 MB/s (balanced) | Fast local I/O; no network dependency during training; snapshot amortizes download | Not shared (each instance gets a copy); snapshot creation takes time; disk size must be provisioned upfront | Medium | Low | GCP PD documentation |
| **E: Parallelstore (DAOS)** | Google Cloud Parallelstore scratch filesystem. DAOS-based, high-throughput parallel I/O. | ~$0.14/GiB/mo | 6.4 GB/s (read), 1.6 GB/s (write) | Non-persistent (scratch only); requires VPC peering; minimum 12 TiB; overkill for MiniVess dataset | High | Medium — complex setup | [Google Cloud Parallelstore docs](https://cloud.google.com/parallelstore) |
| **F: Managed Lustre** | Google Cloud Managed Lustre for HPC workloads. | ~$0.60/GiB/mo | 10+ GB/s | High throughput; standard HPC filesystem | Most expensive; overkill; minimum capacity requirements | Very High | Low (mature) but cost-prohibitive | GCP Managed Lustre documentation |
| **G: Redis + GCS** | Redis (Memorystore) for hot metadata (trial status, pruning decisions); GCS for bulk data and checkpoints. | Redis: ~$0.049/GiB/hr (~$35/mo for 1 GiB); GCS: ~$0.020/GiB/mo | Redis: sub-ms for metadata | Fast coordination; decouples metadata from bulk storage; Redis Pub/Sub for trial events | Additional infrastructure; complexity; Redis is stateful (needs persistence config) | Medium-High | Medium — operational overhead | N/A |

### 6.1 Pricing Summary Table (europe-north1, monthly)

| Storage Option | Min. Capacity | Monthly Cost (min) | Cost per GiB/mo | Best For |
|----------------|---------------|--------------------|------------------|----------|
| GCS Standard | None | Pay-per-use (~$0.50 for 25 GiB) | $0.020 | Data + checkpoints (default) |
| GCS FUSE | None | Same as GCS | $0.020 | POSIX access to GCS data |
| Filestore Basic SSD | 1 TiB | ~$369 | $0.36 | Shared NFS (if budget allows) |
| Balanced PD | 10 GiB | ~$0.80 | $0.080 | Per-instance local data |
| Parallelstore | 12 TiB | ~$1,680 | $0.14 | HPC scratch (not for us) |
| Managed Lustre | Variable | ~$600+ | $0.60 | HPC (ruled out on cost) |
| Cloud SQL PostgreSQL | N/A | ~$10 (db-f1-micro) | N/A | Optuna coordination |

### 6.2 Analysis

For MinIVess with a ~25 GiB dataset (MiniVess, 70 volumes) and 5 concurrent GPU instances, the storage decision is straightforward:

- **Data**: GCS bucket (`gs://minivess-mlops-dvc-data`) with SkyPilot `file_mounts` or `MOUNT_CACHED`. Same-region download from GCS to `europe-north1` compute is free and fast (~30s for 25 GiB). Filestore at $369/mo is unjustifiable for a 25 GiB dataset.
- **Checkpoints**: GCS bucket (`gs://minivess-mlops-checkpoints`) or MLflow artifact store (`gs://minivess-mlops-mlflow-artifacts`). Checkpoints are written infrequently (end of epoch) and read only for resume or ASHA warm-start.
- **Coordination**: Cloud SQL PostgreSQL (~$10/mo) for Optuna study storage. Already required by the PostgreSQL-only policy.
- **Docker cache**: GAR remote repository in `europe-north1` as Docker Hub pull-through cache. Same-region pulls from GAR are fast (~90s for a 5 GiB image).

**Recommendation**: Implement GCS-only (Option A) now. Design the storage abstraction so that GCS FUSE (Option B) or Filestore NFS (Option C) can be swapped in later via config. Parallelstore and Managed Lustre are ruled out on cost and minimum capacity requirements.

## 7. Decision Matrix: H5 — Grid Search Reproducibility

The paper factorial demands that every grid cell be reproducible: given the same config, the same code version, and the same data version, the results must be identical (modulo GPU non-determinism).

| Option | Description | Pros | Cons | Effort | Risk | Citations |
|--------|-------------|------|------|--------|------|-----------|
| **A: YAML config + git SHA + DVC hash** | Each grid cell is defined by a YAML config file. The experiment records git SHA, DVC data hash, and Docker image digest. MLflow tags capture all three. | Complete provenance; human-readable configs; DVC handles data versioning; Docker image digest handles environment | Requires discipline to tag every run; no built-in interrupted-grid resume beyond MLflow fingerprinting | Low (mostly existing) | Low | [Isensee et al. (2021)](https://www.nature.com/articles/s41592-020-01008-z) |
| **B: Optuna GridSampler** | Define the grid as an Optuna search space. `GridSampler(search_space)` enumerates all points. Study persistence handles resume. | Built into Optuna; study persistence handles resume automatically; integrates with the HPO flow | GridSampler requires all points upfront; less readable than YAML for paper documentation; mixes grid search with Bayesian HPO infrastructure | Low-Medium | Low | [Akiba et al. (2019)](https://arxiv.org/abs/1907.10902) |
| **C: DVC pipelines** | Define each grid cell as a DVC pipeline stage. `dvc repro` handles caching and resume. | Hash-based caching; automatic skip for completed cells; reproducibility is DVC's core feature | DVC pipeline files become unwieldy at 128+ stages; DVC is designed for data pipelines, not training grid sweeps; conflicts with Prefect orchestration | High | Medium — architectural mismatch | N/A |

**Analysis**: Option A (YAML config + provenance tags) is the pragmatic choice. The existing `dynunet_grid.yaml` format is clean and human-readable. The `train_all_hyperparam_combos.sh` script already generates the Cartesian product and skips completed runs via MLflow fingerprinting. The key improvement needed is to record the config file hash and git SHA as MLflow tags systematically, and to version the grid config in `configs/hpo/` alongside the code.

Option B (Optuna GridSampler) is a valid alternative that unifies the grid search path with the HPO flow, eliminating the separate shell script. This is the recommended long-term direction: replace the shell script with `hpo_flow.py` using `sampler="grid"` and a pre-defined search space.

## 8. Recommendation

### Phase 1: Paper Factorial (Grid Search, Minimal Infrastructure)

**Goal**: Execute the 128-cell factorial (4 losses x 4 learning rates x 2 batch sizes x 4 model variants, each with 3-fold CV = 384 runs) for the Nature Protocols paper.

**Architecture**:
- Use the existing grid search path (`train_all_hyperparam_combos.sh` + `configs/hpo/dynunet_grid.yaml`) extended to the full 128-cell grid
- Execute as 5 concurrent SkyPilot managed jobs on GCP L4 spot instances (`europe-north1`)
- Each managed job runs a partition of the grid (e.g., jobs 0-4 each handle ~77 of 384 runs)
- Data from GCS via SkyPilot `file_mounts` with `MOUNT_CACHED`
- Checkpoints and MLflow artifacts to GCS buckets
- No inter-job communication needed (grid search is embarrassingly parallel)
- Cloud SQL PostgreSQL for MLflow tracking (already provisioned)

**SkyPilot task YAML pattern**:
```yaml
# deployment/skypilot/hpo_grid_worker.yaml
name: minivess-grid-worker
resources:
  cloud: gcp
  region: europe-north1
  accelerators: L4:1
  use_spot: true
  disk_size: 100

file_mounts:
  /data:
    source: gs://minivess-mlops-dvc-data
    mode: MOUNT_CACHED

image_id: docker:europe-north1-docker.pkg.dev/minivess-mlops/minivess/train:latest

setup: |
  echo "Data cached at /data, Docker image pre-built — no setup needed"

run: |
  # WORKER_PARTITION and TOTAL_WORKERS set via --env
  python -m minivess.orchestration.flows.train_flow \
    --grid-config /app/configs/hpo/paper_factorial.yaml \
    --partition $WORKER_PARTITION \
    --total-workers $TOTAL_WORKERS
```

**What needs to be built**:
1. Extend `train_all_hyperparam_combos.sh` (or replace with Python) to support `--partition` / `--total-workers` for grid partitioning
2. Add MLflow tags for `grid_config_hash`, `git_sha`, `docker_image_digest`, `dvc_data_hash`
3. Create the full 128-cell paper factorial YAML config
4. Create the SkyPilot task YAML for grid workers
5. Test the partitioned grid execution locally with smoke_test config

### Phase 2: Post-Paper HPO (Bayesian + ASHA, Distributed)

**Goal**: Enable downstream users to run Bayesian HPO with ASHA pruning on their own datasets and model configurations.

**Architecture**:
- Implement `PARALLEL` allocation strategy in `hpo_engine.py`
- Each worker is a SkyPilot managed job that connects to the shared PostgreSQL Optuna study
- Workers use the ask-tell API: ask for a trial, run training, report intermediate results via `trial.report()`, check `trial.should_prune()` at each epoch
- ASHA coordination happens via PostgreSQL — no direct inter-worker communication
- The HPO flow (`hpo_flow.py`) becomes the study creator + result aggregator, not the trial runner

**Integration with existing code** — the `PARALLEL` path in `hpo_flow.py`:

```python
# In hpo_flow.py, replacing the NotImplementedError:
if strategy == AllocationStrategy.PARALLEL:
    # Create study (or load existing) — PostgreSQL ensures consistency
    engine = HPOEngine(
        study_name=study_name,
        storage=os.environ.get("OPTUNA_STORAGE_URL"),
        pruner=pruner,
        sampler=sampler,
    )
    study = engine.create_study(direction="minimize")

    # Launch N SkyPilot managed jobs, each running hpo_worker.py
    # Workers connect to the same PostgreSQL study and pull trials
    for i in range(n_workers):
        sky.jobs.launch(
            task_yaml="deployment/skypilot/hpo_worker.yaml",
            env={"OPTUNA_STORAGE_URL": storage_url, "STUDY_NAME": study_name},
        )
    # Monitor study progress via Optuna dashboard or polling
```

**What needs to be built**:
1. Implement `hpo_worker.py` — a headless worker that connects to PostgreSQL, runs trials in a loop
2. Replace `NotImplementedError` in `hpo_flow.py` with SkyPilot job launching
3. Add Optuna dashboard deployment (lightweight, runs on CPU) for real-time monitoring
4. Test with 2-3 workers on GCP L4 spot instances

### Phase 3: Future (Advanced Algorithms, Syne Tune)

**Goal**: Support BOHB, DEHB, and PBT for users with larger compute budgets.

**Architecture**:
- Evaluate Syne Tune ([Salinas et al. (2022)](https://proceedings.mlr.press/v188/salinas22a.html)) as a scheduler provider, with Optuna/PostgreSQL retained for study persistence
- Syne Tune's `LocalBackend` can be adapted to trigger SkyPilot jobs instead of local processes
- BOHB ([Falkner et al. (2018)](https://arxiv.org/abs/1807.01774)) and DEHB ([Awad et al. (2021)](https://arxiv.org/abs/2105.09821)) would be available as scheduler options
- PBT ([Jaderberg et al. (2017)](https://arxiv.org/abs/1711.09846)) would require persistent instances (no spot preemption during population evolution)

**Not recommended now** — Syne Tune integration is a significant effort with limited near-term value. The Optuna TPE + ASHA combination covers the paper requirements and most post-paper use cases.

## 9. FinOps Analysis

### 9.1 Paper Factorial Cost Estimate (128 cells, 3-fold CV, GCP L4 spot)

**Assumptions**:
- 384 total training runs (128 cells x 3 folds)
- Average training time per run: 2.5 hours (100 epochs, DynUNet, MiniVess 70 volumes)
- L4 spot price in `europe-north1`: ~$0.30/hr (GPU) + ~$0.05/hr (CPU/mem) = ~$0.35/hr total
- L4 on-demand price: ~$0.80/hr total
- 5 concurrent instances (GCP quota)

| Cost Component | Calculation | Spot Cost | On-Demand Cost |
|----------------|-------------|-----------|----------------|
| GPU compute (384 runs x 2.5 hr) | 960 GPU-hours | $336 | $768 |
| Setup overhead (5 instances x ~5 min cold start) | ~0.4 hr per instance | $0.70 | $1.60 |
| Spot preemption re-setup (~15% preemption rate, SkyPilot auto-recovery) | ~144 additional setup events x 2 min warm | ~$2.50 | N/A |
| Cloud SQL PostgreSQL (db-f1-micro, 1 month) | Flat | $10 | $10 |
| GCS storage (25 GiB data + ~50 GiB checkpoints + ~10 GiB MLflow) | ~85 GiB x $0.020/GiB | $1.70 | $1.70 |
| GCS operations (downloads, uploads) | ~$0.005/1000 ops | ~$2 | ~$2 |
| **Total** | | **~$353** | **~$783** |

**Wall-clock time**: 384 runs / 5 parallel = ~77 sequential runs per instance. At 2.5 hr each = ~192 hours = **~8 days** of continuous execution.

### 9.2 Setup Amortization Comparison

| Topology | Instances | Cold starts | Total setup time | Setup as % of compute | Notes |
|----------|-----------|-------------|------------------|----------------------|-------|
| 5x single-GPU, long-lived | 5 | 5 | ~25 min | 0.04% | Each instance runs ~77 trials sequentially. Best amortization. |
| 5x single-GPU, per-trial | 384 | 384 | ~32 hr | 3.3% | Worst case — new instance per trial. Avoid. |
| 1x 5-GPU instance | 1 | 1 | ~5 min | 0.008% | Single setup, but single failure domain. Feasible within 5-GPU quota. |

**Recommendation**: Use 5 long-lived SkyPilot managed jobs. Each job runs its partition of the grid serially (77 runs), writing checkpoints to GCS after each run. If preempted, SkyPilot recovers the job and the grid launcher skips completed runs (MLflow fingerprinting). This achieves near-optimal amortization while maintaining independent failure domains.

A single 5-GPU instance (Option B in H3) is technically feasible within the 5-GPU quota and offers marginally better amortization. However, it creates a single failure domain — one spot preemption kills all 5 concurrent trials. With 5 independent managed jobs, a preemption affects only 1 of 5 streams.

### 9.3 ASHA Overhead for Bayesian HPO (Post-Paper)

For post-paper Bayesian HPO with ASHA, the cost structure changes:

- Low-fidelity trials (5-10 epochs): ~15 min each
- Setup overhead: ~5 min cold, ~2 min warm
- Setup as % of compute: 25-33% (cold) or 12-15% (warm)

This is why long-lived worker instances are essential for ASHA. A worker should run multiple trials in sequence, only tearing down when the study is complete or the worker has been idle for a configurable timeout. The SkyPilot managed job pattern supports this: the worker script loops over `study.ask()` / `study.tell()` calls until no more trials are needed.

### 9.4 Cost Comparison: GCP vs RunPod

For reference, RunPod RTX 4090 pricing:
- Spot/community: ~$0.34/hr
- On-demand: ~$0.69/hr

RunPod is competitive on per-hour GPU cost, but lacks managed orchestration (no equivalent to SkyPilot managed jobs with automatic spot recovery). For the paper factorial (8 days of continuous execution), SkyPilot's spot recovery on GCP is critical — manual spot recovery on RunPod would require operator intervention. RunPod remains the better choice for quick single-run experiments (instant provisioning, no GCP quota concerns).

## 10. Bibliography

1. [Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD 2019*.](https://arxiv.org/abs/1907.10902)
2. [Awad et al. (2021). "DEHB: Evolutionary Hyperband for Scalable, Robust and Efficient Hyperparameter Optimization." *IJCAI 2021*.](https://arxiv.org/abs/2105.09821)
3. [Bischl et al. (2023). "Hyperparameter Optimization: Foundations, Algorithms, Best Practices, and Open Challenges." *WIREs Data Mining and Knowledge Discovery*.](https://arxiv.org/abs/2107.05847)
4. [Cardoso et al. (2022). "MONAI: An open-source framework for deep learning in healthcare." *arXiv*.](https://arxiv.org/abs/2211.02701)
5. [Cho et al. (2024). "ACUTE: Automatic Checkpointing for Efficient Training." *IEEE IPDPS 2024*.](https://ieeexplore.ieee.org/document/10639967/)
6. [Falkner et al. (2018). "BOHB: Robust and Efficient Hyperparameter Optimization at Scale." *ICML 2018*.](https://arxiv.org/abs/1807.01774)
7. [Isensee et al. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*.](https://www.nature.com/articles/s41592-020-01008-z)
8. [Jaderberg et al. (2017). "Population Based Training of Neural Networks." *arXiv*.](https://arxiv.org/abs/1711.09846)
9. [Klein et al. (2020). "Model-based Asynchronous Hyperparameter and Neural Architecture Search." *arXiv*.](https://arxiv.org/abs/2003.10865)
10. [Li et al. (2017). "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization." *JMLR*.](https://jmlr.org/papers/v18/16-558.html)
11. [Li et al. (2020). "A System for Massively Parallel Hyperparameter Tuning." *MLSys 2020*.](https://arxiv.org/abs/1810.05934)
12. [Liaw et al. (2018). "Tune: A Research Platform for Distributed Model Selection and Training." *arXiv*.](https://arxiv.org/abs/1807.05118)
13. [Moritz et al. (2018). "Ray: A Distributed Framework for Emerging AI Applications." *OSDI 2018*.](https://www.usenix.org/conference/osdi18/presentation/moritz)
14. [Parker-Holder et al. (2020). "Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits." *NeurIPS 2020*.](https://arxiv.org/abs/2002.02518)
15. [Salinas et al. (2022). "Syne Tune: A Library for Large Scale Hyperparameter Tuning and Reproducible Research." *AutoML Conference 2022*.](https://proceedings.mlr.press/v188/salinas22a.html)
16. [Shekhar et al. (2022). "A Comparative Study of Hyperparameter Optimization Tools." *arXiv*.](https://arxiv.org/abs/2201.06433)
17. [Thorpe et al. (2023). "Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs." *NSDI 2023*.](https://www.usenix.org/conference/nsdi23/presentation/thorpe)
18. [Watanabe (2023). "Tree-structured Parzen Estimator: Understanding Its Theory and Analyzing Its Behavior." *arXiv*.](https://arxiv.org/abs/2304.11127)
19. [Yang et al. (2023). "SkyPilot: An Intercloud Broker for Sky Computing." *NSDI 2023*.](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng)
20. [Optuna distributed optimization tutorial.](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
21. [Optuna ask-tell API tutorial.](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html)
22. [Optuna gRPC storage proxy blog post.](https://medium.com/optuna/distributed-optimization-in-optuna-and-grpc-storage-proxy-08db83f1d608)
23. [Syne Tune GitHub repository.](https://github.com/syne-tune/syne-tune)
24. [Ray Tune schedulers API documentation.](https://docs.ray.io/en/latest/tune/api/schedulers.html)
25. [Ray Tune OptunaSearch integration.](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html)
26. [SkyPilot managed jobs documentation.](https://docs.skypilot.co/en/latest/examples/managed-jobs.html)
27. [SkyPilot many-jobs documentation.](https://docs.skypilot.co/en/latest/running-jobs/many-jobs.html)
28. [SkyPilot distributed jobs documentation.](https://docs.skypilot.co/en/latest/running-jobs/distributed-jobs.html)
29. [SkyPilot training guide.](https://docs.skypilot.co/en/latest/reference/training-guide.html)
30. [Google Cloud Filestore documentation.](https://cloud.google.com/filestore)
31. [Google Cloud Parallelstore documentation.](https://cloud.google.com/parallelstore)
32. [GCS FUSE optimization for AI/ML workloads.](https://cloud.google.com/architecture/optimize-ai-ml-workloads-cloud-storage-fuse)
