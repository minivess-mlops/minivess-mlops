# MinIVess MLOps v2

Model-agnostic biomedical segmentation MLOps platform. Clean rewrite from v0.1-alpha.

## Overarching Principle (TOP-1): Flexible MLOps as MONAI Ecosystem Extension

> **We are building a highly flexible MLOps architecture for multiphoton biomedical imaging
> experiments as an extension to the MONAI ecosystem. Do not constrain expansion.**

This principle governs every architectural decision in this repository. MinIVess is NOT a
bespoke pipeline for one segmentation task — it is a general-purpose MLOps scaffold that
the MONAI ecosystem currently lacks. Expansion must always be possible:

- **New flow topologies** — Parallel flows (Survival Analysis runs independently of
  segmentation), sequential flows (Classification on segmentation masks), and hybrid
  topologies must all be first-class citizens. Never hardcode a fixed DAG.
- **New data modalities** — A new dataset type, a new imaging protocol, or a new
  annotator should add a YAML config, not a code branch.
- **New model families** — Any model that implements the `ModelAdapter` ABC integrates
  automatically. The platform never assumes DynUNet, SAM3, or any specific architecture.
- **New downstream consumers** — Dashboard, Biostatistics, Annotation all read from
  MLflow and the service layer. Adding a new consumer is adding a new flow + adapter,
  not modifying core infrastructure.
- **MONAI-first** — If MONAI has it, use it directly. Zero custom code for things MONAI
  already implements. This repo extends MONAI, never forks it.

**Corollary**: Any design that makes adding a new flow, a new model, or a new dataset
harder than editing one YAML file is a design flaw. Escalate to user before implementing.

---

## Overarching Principle (TOP-2): Zero Manual Work + Reproducibility

> **Automate everything. Nobody should ever manually launch pods, VMs, or instances.
> The highest priorities are excellent DevEx and reproducibility to combat the
> reproducibility crisis in science and ML.**

This is WHY every automation tool in this repo exists:
- **Pulumi** — IaC so nobody manually provisions cloud infrastructure
- **SkyPilot** — Intercloud broker so nobody manually launches GPU jobs
- **Docker** — Containerization so nobody debugs "works on my machine"
- **Prefect** — Orchestration so nobody manually sequences pipeline steps
- **DVC** — Data versioning so nobody manually tracks dataset versions

**Docker is NOT optional.** Docker is the execution model — the reproducibility guarantee.
ALL training, ALL pipeline execution, ALL deployment goes through Docker containers.
The only Docker-free path is `uv run pytest` for fast unit tests. Everything else is Docker.
Suggesting bare-metal execution is suggesting to abandon reproducibility.

**Cloud execution is NEVER manual.** `sky jobs launch task.yaml` — one command. The
SkyPilot YAML + Docker image = fully reproducible cloud execution. No SSH, no manual
setup, no "install these packages on the VM."

See: `.claude/metalearning/2026-03-14-docker-resistance-anti-pattern.md`

---

## Design Goal #1: EXCELLENT DevEx for PhD Researchers

> **MLOps as a scaffold that frees PhD researchers from infrastructure wrangling.**
> Everything automatic by default, everything tweakable by choice.

This is the **first and most important design goal** of the entire repository. Every
feature, every configuration, every automation should be evaluated against this principle.

### Core Principles
1. **Zero-config start** — `just experiment` works out of the box on any machine
2. **Adaptive defaults** — Hardware auto-detection selects batch size, patch size, cache rate
3. **Scientific decisions stay with the researcher** — No default resampling, no implicit
   upsampling. The platform provides knobs, the researcher turns them.
4. **Model-agnostic profiles** — Same `--compute auto` works for DynUNet, SAMv3, SegResNet;
   each model maps hardware budgets differently via `configs/model_profiles/*.yaml`
5. **Dataset-agnostic patches** — Patch sizes constrained by dataset's smallest volume,
   not hardcoded. Pre-training validation ensures patches fit all volumes.
6. **Transparent automation** — Every automatic decision is logged and overridable via YAML
7. **Zero cosmetic noise** — Training output must show only actionable signals. Suppress all
   third-party non-actionable warnings (ONNX Runtime device discovery, MONAI deprecated
   indexing, CUDA cudart FutureWarning) at the script entry point. Users who learn to ignore
   warnings will also ignore real ones. Entry-point suppression pattern:
   `os.environ.setdefault("ORT_LOGGING_LEVEL", "3")` + `warnings.filterwarnings(...)`.
   **NEVER tell the user to "just ignore" a warning.** Fix it or suppress it.
8. **Portfolio-grade code** — Every component demonstrates production ML engineering
8. **Task-agnostic multi-task architecture** — Multi-task learning (auxiliary heads,
   multi-task losses, per-head metrics) is a GENERIC framework configured via YAML,
   NOT hardcoded to any specific set of tasks. The platform supports arbitrary
   auxiliary heads (regression, classification, segmentation) via config. Specific
   tasks (SDF edge detection, centerline extraction, artery/vein classification,
   junction detection, disease classification, survival analysis, etc.) are
   instantiations of the framework, not architectural decisions. **NEVER hardcode
   task-specific logic** into the multi-task infrastructure — all task semantics
   come from config files.
9. **Division of labor via Prefect** — Prefect flows are **required** (not optional),
   separating concerns into 5 persona-based flows even for solo researchers. Each flow
   is independently testable, resumable, cacheable, and uses MLflow as the inter-flow
   contract. The first 4 flows are **core** (always run); the last is **best-effort**
   (runs when resources allow, failure does not block the pipeline):
   - Flow 1: Data Engineering (core)
   - Flow 2: Model Training (core)
   - Flow 3: Model Analysis (core)
   - Flow 4: Deployment (core)
   - Flow 5: Dashboard & Reporting (best-effort — paper figures, Parquet export, drift reports, QA health checks)
   QA was merged into the dashboard health adapter (#342, PR #567).

### Reproducibility via Real-Data E2E Pipeline (Verified 2026-03-02)

The quasi-E2E pipeline has been verified end-to-end with **real data** — not mocks.
All 5 flows produce real artifacts from real MiniVess experiments (70 volumes, 4 losses,
3 folds, 100 epochs). The `PipelineTriggerChain` (in `src/minivess/orchestration/trigger.py`)
runs: data → train(skip) → analyze → deploy → dashboard, producing 35+ verified artifacts.

**Key scripts** (all in `scripts/`):
- `run_full_pipeline.py` — Full trigger chain, all flows in sequence
- `verify_all_artifacts.py` — 73 validation checks (JSON, PNG, Parquet, DuckDB, ONNX)
- `assemble_paper_artifacts.py` — 25 paper-ready figures/tables/data with LaTeX commands
- `tag_champions.py` — Champion tagging from real MLflow experiments
- `generate_real_figures.py` — Figures from real ComparisonTable data
- `export_duckdb_parquet.py` — DuckDB + Parquet from real mlruns

**Artifact locations** (committed in `outputs/`):
- `outputs/analysis/` — Figures (PNG+SVG), comparison tables (MD+TEX)
- `outputs/paper_artifacts/` — Paper-ready assembled artifacts
- `outputs/duckdb/parquet/` — Parquet exports (DuckDB files gitignored — regenerable)
- `outputs/pipeline/trigger_chain_results.json` — Chain status proof

**Gitignored artifacts** (belong in registries, not git):
- `*.onnx` — ONNX models belong in MLflow model registry / BentoML store
- `outputs/**/*.duckdb` — Regenerable via `scripts/export_duckdb_parquet.py`

### Multi-Environment Compute
Everything must work identically on:
- Local workstation (single GPU, limited RAM)
- Intranet / on-prem servers (multi-GPU, team access)
- Ephemeral cloud instances (Docker, mounted drives)
- CI runners (CPU-only, automated)

## Design Goal #2: Platform Engineering for Research

> **Docker-per-flow isolation, SkyPilot compute, Optuna HPO — production infrastructure
> that scales from laptop to multi-cloud without code changes.**

### Architecture Layers
1. **Layer 0: Infrastructure** — Docker Compose (PostgreSQL, MinIO, MLflow, Prefect, Grafana)
2. **Layer 1: GPU Management** — Full GPU for training, NVIDIA MIG for multi-model inference
3. **Layer 2: Training Execution** — Optuna (search) + ASHA (early stopping), PyTorch DDP
4. **Layer 3: Compute Provisioning** — SkyPilot multi-cloud spot instances + on-prem K8s
5. **Layer 4: Workflow Orchestration** — Prefect 3.x Server with Docker workers

### Docker-Per-Flow Isolation
Each of the 6 Prefect flows runs in its own Docker container:
- `deployment/docker/Dockerfile.{base,data,train,analyze,deploy,dashboard}`
- `deployment/docker-compose.flows.yml` — per-flow services
- Flows communicate ONLY through MLflow artifacts + Prefect artifacts (no shared filesystem)
- GPU reservation for training flow via Docker Compose device requests

### Three-Environment Model
| Environment | Docker | Compute | Purpose |
|-------------|--------|---------|---------|
| **dev** | Docker-free | Local GPU | Fast iteration, `uv run pytest` |
| **staging** | Docker Compose | Local GPU in container | Integration testing |
| **prod** | Docker + SkyPilot | Cloud spot / on-prem K8s | Full pipeline |

### SkyPilot = Intercloud Broker for AI Workloads (Non-Negotiable)
SkyPilot is an **intercloud broker** ([Yang et al., NSDI'23](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng))
— a unified system that automatically places and manages AI jobs across heterogeneous
infrastructure. You describe WHAT you need (GPU type, cost constraints), SkyPilot decides
WHERE to run it (which cloud, region, instance type) with automatic spot recovery and
cross-cloud failover. Think of it as **Slurm for the multi-cloud era**, not IaC.

**SkyPilot is NOT Terraform/Pulumi.** IaC provisions specific resources ("create this VM
on AWS us-east-1"). SkyPilot is a workload broker ("I need 1xA100, find the cheapest
option anywhere and handle preemptions"). Different abstraction level entirely.

It is NOT "a RunPod launcher." The SAME SkyPilot YAML must work on:

| Provider | Use Case | SkyPilot Role |
|----------|----------|---------------|
| RunPod | Cloud GPU (spot RTX 4090) | `cloud: runpod` |
| Lambda Labs | Cloud GPU (A100) | `cloud: lambda` |
| AWS | Cloud GPU (spot p4d) | `cloud: aws` |
| GCP | Cloud GPU (preemptible) | `cloud: gcp` |
| Intranet servers | On-prem GPU via SSH | SSH connector |
| Local LAN | Multi-GPU via SSH | SSH connector |

**SkyPilot is ALWAYS used for compute beyond the dev machine.** Never bypass it.

**Docker Execution (MANDATORY):**
- SkyPilot YAML MUST use `image_id: docker:<registry>/<image>:<tag>` — bare VM is BANNED
- Docker image pushed to configurable registry (`$DOCKER_REGISTRY` in `.env.example`):
  GHCR (default), Docker Hub, AWS ECR, GCP GAR, Azure ACR — any OCI-compliant registry
- Push: `make push-registry` (or `make push-ghcr` for GHCR with auto-login)
- **BANNED**: `apt-get install`, `uv sync`, `git clone` in SkyPilot setup scripts.
  All deps belong in the Docker image. SkyPilot setup is ONLY for data pull + config.
- See: `.claude/metalearning/2026-03-14-skypilot-bare-vm-docker-violation.md`

**Key Files:**
- `deployment/skypilot/train_generic.yaml` — spot A100 training
- `deployment/skypilot/train_hpo_sweep.yaml` — parallel HPO trials
- `src/minivess/compute/skypilot_launcher.py` — Python SDK wrapper

**Never suggest bypassing SkyPilot.** "SkyPilot is causing issues" → fix how you USE it.
See: `.claude/metalearning/2026-03-14-skypilot-purpose-misunderstanding.md`

### Optuna + ASHA Hyperparameter Optimization
- `src/minivess/optimization/hpo_engine.py` — HPOEngine with TPE/CmaES + HyperbandPruner
- `src/minivess/optimization/search_space.py` — YAML-driven search space
- `configs/hpo/dynunet_example.yaml` — reference HPO config
- `scripts/run_hpo.py` — CLI entry point

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Project Type** | application (ML pipeline + serving) |
| **Python Version** | 3.12+ |
| **Package Manager** | uv (ONLY) |
| **Linter/Formatter** | ruff |
| **Type Checker** | mypy |
| **Test Framework** | pytest |
| **Config (train)** | Hydra-zen |
| **Config (deploy)** | Dynaconf |
| **ML Framework** | PyTorch + MONAI + TorchIO + TorchMetrics |
| **Serving** | BentoML + ONNX Runtime + Gradio (demo) |
| **Experiment Tracking** | MLflow + DuckDB (analytics) |
| **Data Validation** | Pydantic v2 (schema) + Pandera (DataFrame) + Great Expectations (batch quality) |
| **Label Quality** | Cleanlab + Label Studio |
| **Model Validation** | Deepchecks Vision + WeightWatcher |
| **XAI** | Captum (3D) + SHAP (tabular only) + Quantus (meta-eval) |
| **Calibration** | MAPIE + netcal + Local Temperature Scaling |
| **Data Profiling** | whylogs |
| **LLM Observability** | Langfuse (self-hosted) + Braintrust (eval) + LiteLLM (provider flexibility) |
| **HPO** | Optuna + ASHA (HyperbandPruner) |
| **Cloud Compute** | SkyPilot (multi-cloud spot instances) |
| **GPU Partitioning** | NVIDIA MIG (multi-model inference) |
| **Workflow Orchestration** | Prefect 3.x (required, 5 flows: data, train, analyze, deploy, dashboard) |
| **Agent Orchestration** | Pydantic AI + PrefectAgent (see ADR 0007) |
| **CI/CD** | GitHub Actions + CML (ML-specific PR comments) |
| **Lineage** | OpenLineage (Marquez) |

## Critical Rules

1. **uv ONLY** — Never use pip, conda, poetry, or requirements.txt. Use `uv add`, `uv sync --all-extras`, `uv run`.
   **`--all-extras` is REQUIRED for dev installs.** Without it, PRD-selected tools
   (pandera, hypothesis, great_expectations, whylogs, langgraph, deepchecks, captum,
   quantus, gradio, etc.) are missing and 126+ tests silently skip. Plain `uv sync`
   is BANNED for development — always use `uv sync --all-extras`.
2. **TDD MANDATORY** — All implementation MUST follow the self-learning-iterative-coder skill (`.claude/skills/self-learning-iterative-coder/SKILL.md`). Write failing tests FIRST, then implement. No exceptions.
3. **Library-First (Non-Negotiable)** — Before implementing ANY algorithm, loss function,
   metric, or data processing step, ALWAYS search for existing implementations in established
   libraries (PyTorch, MONAI, scipy, skimage, networkx, gudhi, etc.) or peer-reviewed
   reference repos. Only write custom code when no suitable library exists. For hybrid cases
   (e.g., loss functions), use library code for non-differentiable parts (GT processing)
   and custom code only for the differentiable prediction path. Rationale: library code has
   more eyes, fewer bugs, and better performance than anything we write from scratch.
4. **Pre-commit Required** — All changes must pass pre-commit hooks before commit.
5. **Encoding** — Always specify `encoding='utf-8'` for file operations.
6. **Paths** — Always use `pathlib.Path()`, never string concatenation.
7. **Timezone** — Always use `datetime.now(timezone.utc)`, never `datetime.now()`.
8. **`from __future__ import annotations`** — At the top of every Python file.
9. **Task-Agnostic Architecture (Non-Negotiable)** — This is an MLOps platform, NOT a
   single-use-case repo. NEVER hardcode specific task assumptions (e.g., "SDF + centerline")
   into infrastructure code. Multi-task adapters, losses, metrics, and data pipelines must
   be GENERIC and config-driven. Specific tasks are YAML config instantiations, not code.
   If you find yourself writing `if task_name == "sdf":` in infrastructure code, you are
   doing it wrong. Use registries, config-driven dispatch, and duck typing instead.
10. **Verify Models Beyond Knowledge Cutoff (Non-Negotiable)** — When a model, library,
   or API is near or beyond the training knowledge cutoff, ALWAYS perform a web search
   to verify it exists, get the correct version/API, and confirm the package name BEFORE
   writing any code. If the user provides URLs (arXiv, GitHub, docs), ALWAYS fetch them
   first. **SAMv3 = Meta's Segment Anything Model 3** (github.com/facebookresearch/sam3,
   Nov 2025), NOT SAM2. See `.claude/metalearning/2026-03-02-sam3-implementation-fuckup.md`.
   **SAM3 ALWAYS requires real pretrained weights (ViT-32L, 648M params).** The
   `_StubSam3Encoder` has been permanently removed (2026-03-07). GPU VRAM ≥16 GB is
   enforced at model build time via `sam3_vram_check.py`. See `src/minivess/adapters/CLAUDE.md`.
11. **Plans Are Not Infallible** — When a plan says "ModelX" but CLAUDE.md says "ModelY",
    or the plan contradicts the user's original prompt, STOP and clarify with the user
    before implementing. Cross-reference plans with CLAUDE.md, literature reports, and
    user history. The plan is a derivative; the user's instructions are the source of truth.
12. **Never Confabulate (Non-Negotiable)** — Never construct post-hoc rationalizations
    for naming inconsistencies, knowledge gaps, or ambiguities. If something doesn't add
    up, say "I'm not sure, let me check" and use web search or ask the user. Confident-
    sounding fabrication is the most dangerous failure mode — it wastes hours of work
    and erodes trust. Admitting ignorance costs 5 seconds; confabulation costs hours.
13. **Read Context Before Implementing** — Before implementing any plan, read the
    literature report / research doc that produced the plan. The plan is a derivative;
    the source document has the ground truth. When resuming work from a previous session,
    read the original user prompts / literature reports, not just the generated plan.
14. **Persist All Learnings (Non-Negotiable)** — Terminal output is ephemeral. Every
    corrective insight, self-reflection, or failure analysis MUST be saved to durable
    locations: metalearning docs (`.claude/metalearning/`), CLAUDE.md, and/or memory
    files. If it's not written to a file, it's not learned. Never print self-reflection
    to terminal without also persisting it.
15. **Write Requested Artifacts to Disk** — When the user explicitly asks for a file
    to be saved at a specific path (e.g., "save the XML plan to `docs/planning/foo.xml`"),
    ALWAYS write it to disk using the Write tool. Never leave requested artifacts only
    in conversation context or plan files. If unsure whether a file was requested, re-read
    the user's original prompt.
16. **STRICT BAN: No Regex for Structured Data (Non-Negotiable)** — `import re` and
    regex patterns are BANNED for parsing any structured data: Python source, YAML, JSON,
    log lines, metric names, file paths, config keys. Use proper parsers instead:
    - Python source → `ast.parse()` + `ast.walk()`
    - YAML/TOML/JSON → `yaml.safe_load()`, `tomllib`, `json.loads()`
    - Log lines / metric keys → emit as JSON (JSONL), parse with `json.loads()`
    - String splitting → `str.split()`, `str.rsplit()`, `str.partition()`
    - File paths → `pathlib.Path` attributes (`.stem`, `.suffix`, `.parts`)
    Claude does NOT get to self-assess whether "regex is sufficient". The ban applies
    always. "Regex is sufficient" is itself a banned phrase.
    See: `.claude/metalearning/2026-03-06-regex-ban.md`
17. **NEVER Suggest Standalone Scripts as a Run Path (Non-Negotiable)** — Training,
    evaluation, and pipeline execution MUST go through Prefect flows running in Docker.
    `scripts/*.py` files are migration utilities or one-off tools — they are NEVER a
    supported run path for the pipeline. There is no "dev" environment that bypasses Docker.
    - **WRONG:** `uv run python scripts/train_monitored.py --loss dice_ce`
    - **CORRECT:** `prefect deployment run 'train-flow/default' --params '{"loss": "dice_ce"}'`
    - **CORRECT:** A `.sh` script that wraps a Prefect deployment invocation
    Creating a GitHub issue to "fix this later" does NOT grant permission to keep offering
    the shortcut. The answer to "Prefect flow not yet implemented" is to implement it.
    See: `.claude/metalearning/2026-03-06-standalone-script-antipattern.md`
18. **Explicit Docker Volume Mounts for ALL Artifacts (Non-Negotiable)** — Every input
    and output in a Docker-per-flow run must be explicitly volume-mounted. `/tmp` and
    `tempfile.mkdtemp()` are FORBIDDEN for any artifact that must survive the container.
    Required mounts: `/data` (inputs), `/mlruns` (tracking), `/checkpoints` (model files),
    `/logs` (monitor CSV/JSONL), `/configs` (YAML configs + splits).
    See: `.claude/metalearning/2026-03-06-standalone-script-antipattern.md`
19. **STOP Protocol Before Execution (Non-Negotiable)** — Before launching ANY training
    or pipeline execution, verify ALL four checks:
    - **S**(ource): Running inside Docker container? If not → build/use Docker image first
    - **T**(racking): Prefect orchestration active? If `PREFECT_DISABLED=1` → REJECT
    - **O**(utputs): All artifact paths volume-mounted? If repo-relative → REJECT
    - **P**(rovenance): Reproducible on another machine? If host-env dependent → REJECT
    If ANY check fails, FIX IT — do not bypass. `_require_docker_context()` in
    `train_flow.py` enforces (S) at runtime. Escape hatch: `MINIVESS_ALLOW_HOST=1`
    for pytest ONLY — never in scripts or production.
    See: `docs/planning/minivess-vision-enforcement-plan.md`
20. **Zero Tolerance for Observed Failures (Non-Negotiable)** — Every test failure,
    import error, or warning encountered during a session MUST result in one of:
    (a) **Fixed immediately** if root cause is clear and fix is < 5 minutes
    (b) **GitHub issue created** with root cause, affected files, and priority label
    (c) **Explicitly reported to user** with recommendation
    "Pre-existing" is NOT a valid classification. "Not related to current changes"
    is NOT an excuse to move on. "Separate issue" without actually creating the
    issue within the same response is a lie. The phrase "not related to current
    changes" is BANNED — every failure in this repo was co-authored by Claude Code
    and is therefore Claude Code's responsibility.
    See: `.claude/metalearning/2026-03-07-silent-existing-failures.md`
22. **Single-Source Config via `.env.example` (Non-Negotiable)** — ALL configurable
   values (service URLs, ports, credentials, hostnames, artifact paths) MUST be defined
   in `.env.example` FIRST and ONLY. This file is the project's configuration contract.
   Before writing any configuration value anywhere, CHECK `.env.example`.
   - **If the variable IS in `.env.example`**: reference it as `${VAR_NAME:-default}`
     in Docker Compose / shell, or `os.environ["VAR_NAME"]` in Python (no fallback literal).
   - **If the variable is NOT in `.env.example`**: ADD IT THERE FIRST, then reference it.
   - **`:-default` fallbacks in Compose MUST equal the default shown in `.env.example`.**
     Copy-paste the value — do not invent a new default in the fallback.

   **BANNED patterns** (each is a rule violation):
   - `ENV MLFLOW_TRACKING_URI=...` in any Dockerfile — Dockerfiles set structural paths
     only (`DATA_DIR`, `CHECKPOINT_DIR`); all service URLs come from compose `x-common-env`.
   - `os.environ.get("VAR", "some_hardcoded_value")` in flow files — use
     `resolve_tracking_uri()` for MLflow; for others use `os.environ["VAR"]` and let
     a missing var fail loudly rather than silently use a wrong default.
   - `mlflow_tracking_uri = "http://localhost:5000"` in Dynaconf TOML — MLflow URI is
     consumed directly from the env var by `resolve_tracking_uri()`, not via Dynaconf.
   - Any hardcoded URL/hostname/port in compose files NOT wrapped in `${VAR:-fallback}`.
   - Duplicating a URL in 4 environment TOML files that all have the same value — that
     signals the value belongs in `.env.example`, not in TOML.

   **CORRECT patterns**:
   ```yaml
   # docker-compose.flows.yml
   MLFLOW_TRACKING_URI: http://${MLFLOW_DOCKER_HOST:-minivess-mlflow}:${MLFLOW_PORT:-5000}
   ```
   ```python
   # Python flow files
   from minivess.observability.tracking import resolve_tracking_uri
   tracking_uri = resolve_tracking_uri()  # reads MLFLOW_TRACKING_URI env var
   ```

   See `.env.example` for the complete authoritative list of all configurable values.
   See `tests/v2/unit/test_env_single_source.py` for enforcement checks.

21. **GitHub Actions CI EXPLICITLY DISABLED (Non-Negotiable)** — The user has
    explicitly forbidden automatic GitHub Actions CI. Actions consume credits.
    ALL CI jobs in `ci-v2.yml` are commented out. ALL workflows use
    `workflow_dispatch` only (manual trigger). **NEVER:**
    - Uncomment CI jobs in `.github/workflows/ci-v2.yml`
    - Add `on: push` or `on: pull_request` triggers to ANY workflow
    - Create new workflows with automatic triggers
    - Re-enable the Docker build gate's `pull_request` trigger
    All validation runs LOCALLY via pre-commit hooks and
    `scripts/pr_readiness_check.sh`. Only the user can lift this ban.

## What AI Must NEVER Do (Extended)

- Confabulate explanations for knowledge gaps instead of web-searching
- Follow a plan that contradicts CLAUDE.md or the user's explicit instructions
- Implement models beyond knowledge cutoff without web-searching first
- Print self-reflection or corrective insights only to terminal without persisting to files
- Ignore user-provided URLs (arXiv, GitHub) — always fetch them for context
- Hardcode specific task names (SDF, centerline, etc.) into multi-task infrastructure —
  use config-driven registries. This is an MLOps platform for ALL segmentation research.
- Use `import re` or regex patterns for parsing structured data (Python, YAML, JSON,
  log lines, metric names). Use proper parsers. "Regex is sufficient" is banned.
- Suggest `python scripts/*.py` as a training or pipeline run command — use Prefect flows.
- Use `/tmp` or `tempfile.mkdtemp()` for artifacts that must survive Docker container exit.
- Write an academic citation without a clickable hyperlink. Every citation in any `.md`,
  Issue, or PR MUST be: `[Author et al. (Year). "Title." *Journal*.](URL)`. If no URL
  exists, write `[Full citation — preprint pending]` to make the gap visibly explicit.
  See: `.claude/metalearning/2026-03-09-missing-hyperlinks-academic-references.md`
- Offer a standalone-script shortcut while a GitHub issue to "fix it properly" is open.
- Dismiss test failures as "pre-existing" or "not related to current changes" without
  creating a GitHub issue — every observed failure needs immediate action.
- Say "separate issue" without creating the issue in the same response.
- Uncomment, re-enable, or add ANY automatic GitHub Actions trigger — CI jobs are
  EXPLICITLY DISABLED by the user. Actions consume credits. Only the user can lift this ban.
  ALL validation runs locally. This is not a suggestion — it is an absolute prohibition.
- Set `ENV VAR=hardcoded_value` in a Dockerfile for any service URL, port, or credential
  — these belong in `.env.example` and are injected by docker-compose, not the Dockerfile.
- Use `os.environ.get("MLFLOW_TRACKING_URI", "mlruns")` or any `get(VAR, literal)` pattern
  in flow files — use `resolve_tracking_uri()` or fail loudly on missing env var.
- Define `mlflow_tracking_uri` or any service URL in Dynaconf TOML files — they are read
  directly from env vars; TOML duplication creates a hidden second source of truth.
- Write SkyPilot YAML with bare VM setup scripts (`apt-get install`, `uv sync`, `git clone`
  in setup:). ALL cloud execution MUST use Docker images via `image_id:`. The Docker image
  is pushed to GHCR and contains all deps. SkyPilot setup: is ONLY for data pull + config.
  "But SkyPilot defaults to bare VM" is NOT an excuse — our repo mandate is Docker-only.
  See: `.claude/metalearning/2026-03-14-skypilot-bare-vm-docker-violation.md`
- Dump wall-of-text questions when the user asks for "interactive" input. ALWAYS use the
  `AskUserQuestion` tool for interactive questions — clickable options, not markdown lists.
  Maximum 4 questions per round; ask in iterative batches, not all at once. DevEx applies
  to Claude-to-user interactions too, not just researcher-facing code.
  See: `.claude/metalearning/2026-03-14-wall-of-text-bad-ux.md`

## TDD Workflow (Non-Negotiable)

Every feature, bugfix, or refactor MUST use the self-learning-iterative-coder skill:

```
1. RED:        Write failing tests first     → .claude/skills/.../protocols/red-phase.md
2. GREEN:      Implement minimum code        → .claude/skills/.../protocols/green-phase.md
3. VERIFY:     Run tests + lint + typecheck  → .claude/skills/.../protocols/verify-phase.md
4. FIX:        If failing, targeted fix      → .claude/skills/.../protocols/fix-phase.md
5. CHECKPOINT: Git commit + state            → .claude/skills/.../protocols/checkpoint.md
6. CONVERGE:   All green? Move to next task  → .claude/skills/.../protocols/convergence.md
```

**Activation**: Before starting a multi-task implementation, run the [ACTIVATION-CHECKLIST](.claude/skills/self-learning-iterative-coder/ACTIVATION-CHECKLIST.md).

**Skill reference**: `.claude/skills/self-learning-iterative-coder/SKILL.md`

## Default Loss Function

The default single-model loss is **`cbdice_cldice`** (CbDiceClDiceLoss). This was
determined by the `dynunet_loss_variation_v2` experiment (2026-02-27) which showed:
- `cbdice_cldice` achieves **0.906 clDice** (best topology) with only −5.3% DSC penalty
- `dice_ce` has higher DSC (0.824) but significantly worse topology preservation (0.832 clDice)
- Full results: `docs/results/dynunet_loss_variation_v2_report.md`

When training a single model (not an ablation sweep), always use `cbdice_cldice` unless
the researcher explicitly requests a different loss. For multi-loss experiments, use the
experiment config YAML which specifies the full loss list.

## Quick Commands

```bash
# Install dependencies (--all-extras is REQUIRED for dev — without it, 126 tests silently skip)
uv sync --all-extras

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Full verify (all three gates — uses staging tier)
make test-staging && uv run ruff check src/ tests/ && uv run mypy src/
```

## Test Tiers (Non-Negotiable)

Three tiers, invoked via Makefile. **NEVER run SAM3 model tests in standard CI.**

| Tier | Command | What runs | Target time |
|------|---------|-----------|-------------|
| **Staging** | `make test-staging` | No model loading, no slow, no integration | <3 min |
| **Prod** | `make test-prod` | Everything except GPU instance tests | ~5-10 min |
| **GPU** | `make test-gpu` | SAM3 + GPU-heavy tests in `tests/gpu_instance/` | GPU instance only |

```bash
make test-staging    # PR readiness — fast, no models
make test-prod       # Pre-merge — includes model loading + slow tests
make test-gpu        # RunPod / intranet GPU — SAM3 forward passes
```

**Key rules:**
- `tests/gpu_instance/` is **excluded** from default pytest collection. Standard
  `uv run pytest tests/` never touches these files. Zero noise.
- `@pytest.mark.model_loading` marks tests that instantiate PyTorch models (DynUNet,
  SegResNet, etc.). Excluded from staging tier.
- SAM3 tests are NEVER run on the dev machine or GitHub Actions. They run on external
  GPU instances only. See #564 for the dockerized GPU benchmark plan.
- `@pytest.mark.slow` marks tests that take >30s. Excluded from staging tier.

## Directory Structure (Target v2)

```
minivess-mlops/
├── src/minivess/              # Main package (renamed from src/)
│   ├── adapters/              # ModelAdapter implementations (MONAI, SAMv3, etc.)
│   ├── pipeline/              # Training, inference, evaluation pipelines
│   ├── ensemble/              # Ensembling strategies (soup, voting, conformal)
│   ├── data/                  # Data loading, profiling, DVC integration
│   ├── orchestration/         # Prefect flows + _prefect_compat.py
│   ├── serving/               # BentoML service definitions
│   ├── agents/                # Pydantic AI agent definitions (PrefectAgent wrappers)
│   ├── observability/         # Langfuse + Braintrust integration
│   ├── compliance/            # Audit trails, SaMD lifecycle hooks
│   └── config/                # Hydra-zen + Dynaconf config schemas
├── tests/
│   ├── unit/                  # Fast, isolated tests
│   ├── integration/           # Service integration tests
│   └── e2e/                   # End-to-end pipeline tests
├── configs/                   # Hydra-zen experiment configs
├── deployment/                # Docker, docker-compose, Pulumi IaC
├── docs/                      # Architecture docs, plan, ADRs
├── .claude/                   # Claude Code configuration
│   └── skills/                # TDD skill
└── LEARNINGS.md               # Cross-session accumulated discoveries
```

## What AI Must NEVER Do

- Use pip, conda, poetry, or create requirements.txt
- Write implementation code before tests (violates TDD mandate)
- Claim tests pass without running them ("ghost completions")
- Write placeholder/stub implementations (`pass`, `TODO`, `NotImplementedError`)
- Skip pre-commit hooks
- Hardcode file paths as strings
- Use `datetime.now()` without timezone
- Commit secrets, credentials, or API keys
- Modify files marked with `# AIDEV-IMMUTABLE`
- Push untested changes
- Hardcode specific task names (SDF, centerline, etc.) into multi-task infrastructure —
  use config-driven registries. This is an MLOps platform for ALL segmentation research.

## Observability Stack

| Tool | Role | Deployment |
|------|------|-----------|
| **Prefect 3.x** | Workflow orchestration (5 flows: data, train, analyze, deploy + dashboard best-effort) | Prefect Server (Docker Compose) |
| **SkyPilot** | Multi-cloud GPU compute provisioning (spot instances, failover) | Library + cloud SDK |
| **Optuna** | HPO with ASHA (HyperbandPruner), TPE/CmaES samplers | Library (in-process) |
| **Langfuse** | Production LLM tracing, cost tracking | Self-hosted (Docker Compose) |
| **Braintrust** | Offline evaluation, CI/CD quality gates, AutoEvals | Hybrid deployment (data plane local) |
| **Pydantic AI** | Agent micro-orchestration via PrefectAgent (ADR 0007) | Library (in-process) |
| **LiteLLM** | Unified LLM API, provider flexibility | Library (in-process) |
| **MLflow** | Experiment tracking, model registry | Local Docker Compose |
| **DuckDB** | In-process SQL analytics over MLflow runs | Library (in-process) |
| **Prometheus + Grafana** | Infrastructure metrics | Local Docker Compose |
| **Evidently** | Data/model drift detection | Library + Grafana export |
| **whylogs** | Lightweight data profiling | Library (in-process) |
| **OpenLineage (Marquez)** | Data lineage tracking (IEC 62304) | Local Docker Compose |
| **Deepchecks Vision** | Image data + model validation | Library (in-process) |
| **WeightWatcher** | Spectral model diagnostics | Library (in-process) |
| **CML** | ML-specific CI/CD, auto PR comments | GitHub Actions |
| **Label Studio** | Multi-annotator workflows | Local Docker Compose |

## Key Architecture Decisions

- **Model-agnostic**: All models implement `ModelAdapter` ABC (train/predict/export)
- **MONAI VISTA-3D** is primary 3D segmentation model; SAMv3 is exploratory
- **Local-first**: Docker Compose with zero cloud API tokens for development
- **Docker-per-flow**: Each Prefect flow runs in its own container (no Python import leakage)
- **SkyPilot compute**: Multi-cloud spot instances for training (3-6x cost savings)
- **MIG inference**: NVIDIA MIG partitioning for multi-model serving on single GPU
- **SaMD-principled**: IEC 62304 lifecycle mapping, audit trails, test set lockout
- **Dual config**: Hydra-zen for experiment sweeps, Dynaconf for deployment environments

## MLflow Tracking Architecture

### Backend
- **Local filesystem** backend: `mlruns/` directory (no server required)
- Tracking URI: `mlruns` (resolved to absolute path in code)
- Each run stores params, metrics, tags, artifacts as plain files
- Run lifecycle: FINISHED (success), FAILED (exception), KILLED (abort)

### Experiments
| Experiment | Purpose | Created By |
|-----------|---------|-----------|
| `dynunet_loss_variation_v2` | Training: 4 losses x 3 folds x 100 epochs | `train_monitored.py` |
| `dynunet_half_width_v1` | Training: width ablation (filters/2) | `train_monitored.py` |
| `minivess_evaluation` | Evaluation runs (ensembles + analysis) | `analysis_flow.py` |

### Param Prefixes (standardized naming)
| Prefix | Category | Example |
|--------|----------|---------|
| (none) | Training hyperparams | `learning_rate`, `batch_size`, `training_time_seconds` |
| `arch_` | Model architecture | `arch_filters`, `arch_deep_supervision` |
| `sys_` | System/environment | `sys_python_version`, `sys_gpu_model` |
| `data_` | Dataset metadata | `data_n_volumes`, `data_train_volume_ids` |
| `loss_` | Loss function config | `loss_name`, `loss_weights` |
| `eval_` | Evaluation config | `eval_bootstrap_n`, `eval_ci_level` |
| `upstream_` | Cross-flow links | `upstream_training_run_id` |

NOTE: We use `sys_` (underscore) not `sys/` (slash) to avoid metric naming conflicts.

### Automatic Logging
`ExperimentTracker.start_run()` automatically logs:
- All `TrainingConfig` fields (17 params including weight_decay, warmup_epochs)
- Architecture params from `ModelConfig.architecture_params` (arch_ prefix)
- System info: Python, PyTorch, MONAI, CUDA, cuDNN, OS, GPU, RAM (sys_ prefix)
- Git commit hash, branch, dirty state
- MLflow system metrics (GPU/CPU/memory/disk, 12 metrics at 10s intervals)
- On failure: sets run status to FAILED with error_type tag

### Autolog Decision
`mlflow.pytorch.autolog()` is **NOT used**. It only provides full functionality with
PyTorch Lightning. This project uses vanilla PyTorch + MONAI training loops.
All logging is explicit via `ExperimentTracker`.

### Key Files
| File | Purpose |
|------|---------|
| `src/minivess/observability/tracking.py` | ExperimentTracker class |
| `src/minivess/observability/system_info.py` | System info collection |
| `src/minivess/observability/analytics.py` | DuckDB analytics over runs |
| `src/minivess/pipeline/mlruns_enhancement.py` | Post-hoc tag enhancement |
| `src/minivess/pipeline/champion_tagger.py` | Champion model tagging |
| `src/minivess/pipeline/duckdb_extraction.py` | DuckDB extraction from mlruns |
| `scripts/backfill_mlflow_metadata.py` | Retroactive run update tool |

### Retroactive Updates
Existing runs can be updated via `mlflow.start_run(run_id=existing_id)`.
**Safety rules**: never overwrite existing params (throws), preserve run status,
add `sys_backfill_note` for provenance. See `scripts/backfill_mlflow_metadata.py`.

## PRD System

The project uses a **hierarchical probabilistic PRD** (Bayesian decision network) to
manage open-ended technology decisions. This is an **academic software project** —
the PRD serves as the evidence base for a future peer-reviewed article.

- [docs/planning/prd/README.md](docs/planning/prd/README.md) — PRD navigation and overview
- [docs/planning/prd/llm-context.md](docs/planning/prd/llm-context.md) — AI assistant context
- [docs/planning/prd/bibliography.yaml](docs/planning/prd/bibliography.yaml) — Central bibliography (ALL cited works)
- [docs/planning/hierarchical-prd-planning.md](docs/planning/hierarchical-prd-planning.md) — PRD format blueprint

**PRD-Update Skill**: `.claude/skills/prd-update/SKILL.md` — Operations for maintaining
the PRD (add decisions, update priors, ingest papers, validate).

### Citation Rules (NON-NEGOTIABLE)
1. **Author-year format only** — "Surname et al. (Year)", never numeric [1]
2. **Central bibliography** — All citations in `bibliography.yaml`, decision files reference by `citation_key`
3. **No citation loss** — References are append-only. Pre-commit hook blocks citation removal.
4. **Sub-citations mandatory** — When ingesting a paper, also extract its relevant references
5. **Validation** — `uv run python scripts/validate_prd_citations.py` checks all citation invariants

## Knowledge Graph (Layer 0 Navigator)

The project uses a **5-layer progressive disclosure** knowledge graph for agent-queryable
decision tracking:

- **Layer 0**: [`knowledge-graph/navigator.yaml`](knowledge-graph/navigator.yaml) — Entry point mapping topics to domains
- **Layer 1**: This file (CLAUDE.md) + MEMORY.md
- **Layer 2**: Folder-level CLAUDE.md files (11 domain experts)
- **Layer 3**: [`knowledge-graph/decisions/`](knowledge-graph/decisions/) — 52 PRD decision nodes as YAML
- **Layer 4**: `docs/planning/` (research reports), `.claude/metalearning/` (failure analysis)

Supporting files: [`_network.yaml`](knowledge-graph/_network.yaml) (DAG edges),
[`_schema.yaml`](knowledge-graph/_schema.yaml) (node format),
[`bibliography.yaml`](knowledge-graph/bibliography.yaml) (all citations).

**Agent workflow**: Read `navigator.yaml` FIRST → route to domain file → load decision YAML on demand.

## See Also

- [docs/modernize-minivess-mlops-plan.md](docs/modernize-minivess-mlops-plan.md) — Full modernization plan
- [docs/modernize-minivess-mlops-plan-prompt.md](docs/modernize-minivess-mlops-plan-prompt.md) — Original prompt and Q&A
- [.claude/skills/self-learning-iterative-coder/SKILL.md](.claude/skills/self-learning-iterative-coder/SKILL.md) — TDD skill reference
- [.claude/skills/prd-update/SKILL.md](.claude/skills/prd-update/SKILL.md) — PRD maintenance skill reference
- [knowledge-graph/navigator.yaml](knowledge-graph/navigator.yaml) — Knowledge graph entry point
- ~~wiki/~~ — Deleted (v0.1 legacy, preserved at `v0.1-archive` git tag)
