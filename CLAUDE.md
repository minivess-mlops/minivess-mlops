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

Key constraints: zero-config start, adaptive hardware defaults, model/dataset-agnostic
profiles, transparent automation (logged + overridable via YAML), zero cosmetic noise
(suppress warnings at entry point, NEVER tell user to "just ignore"), task-agnostic
multi-task architecture (NEVER hardcode task names), 5 Prefect flows required (4 core
+ 1 best-effort dashboard). **NEVER tell the user to "just ignore" a warning.**

## Design Goal #2: Platform Engineering for Research

> **Docker-per-flow isolation, SkyPilot compute, Optuna HPO — production infrastructure
> that scales from laptop to multi-cloud without code changes.**

Details: `deployment/CLAUDE.md` (Docker, volumes, GPU, compose files).

### Two-Provider Cloud Architecture (Non-Negotiable)

**EXACTLY two cloud providers.** Adding a third requires explicit user authorization.

| Provider | Environment | Role | Data Storage | MLflow |
|----------|------------|------|--------------|--------|
| **RunPod** | env (dev) | Hacky, optional — for researchers who like RunPod | Data uploaded from local disk to Network Volume | File-based on Network Volume |
| **GCP** | staging + prod | Production-grade, full managed cloud stack | **GCS** (`gs://minivess-mlops-dvc-data`) | Cloud Run (optional) |
| ~~Lambda Labs~~ | archived | No EU availability | — | — |
| ~~UpCloud~~ | sunsetting | Trial period, being replaced by GCP | — | — |

**GCP Project Details** (see `deployment/pulumi/gcp/CLAUDE.md` for full reference):
- **Project**: `minivess-mlops`
- **Region**: `europe-north1` (Finland)
- **GCS Buckets**: `minivess-mlops-dvc-data`, `minivess-mlops-mlflow-artifacts`, `minivess-mlops-checkpoints`
- **Docker Registry**: GAR (`europe-north1-docker.pkg.dev/minivess-mlops/minivess`)
- **IaC**: Pulumi (`deployment/pulumi/gcp/`)
- **Setup guide**: `docs/planning/gcp-setup-tutorial.md`

**RunPod "env" key assumption**: Cannot assume ANY GCP infrastructure exists. Standalone.
Data comes from local disk upload (`make dev-gpu-upload-data`), not from GCS.

**Data flow**:
- **Local → RunPod**: `sky rsync up data/raw/minivess/ → Network Volume`
- **Local → GCP**: `gsutil cp` or `dvc push -r gcs` to GCS bucket
- **RunPod → Local**: `make dev-gpu-sync` (pulls mlruns back)
- **Public data origin**: `s3://minivessdataset` (read-only initial download, NOT a production backend)

### Three-Environment Model
| Environment | Docker | Compute | Data | Purpose |
|-------------|--------|---------|------|---------|
| **local** | Docker-free or Docker Compose | Local GPU (RTX 2070 Super, 8 GB) | MinIO (local) | Fast iteration, `uv run pytest` |
| **env** (RunPod) | Docker image via SkyPilot | RunPod RTX 4090 (24 GB) | Network Volume (upload from local) | Quick GPU experiments |
| **staging/prod** (GCP) | Docker image via SkyPilot | GCP L4/A100 spot | GCS buckets | Production runs, paper results |

**T4 BANNED (Non-Negotiable):** T4 is Turing architecture — no BF16 support.
SAM3's half-precision encoder overflows during validation (FP16 max = 65504 → NaN).
L4 (Ada Lovelace) supports BF16, is 1.86x faster, AND 37% cheaper per job.
See: `.claude/metalearning/2026-03-15-t4-turing-fp16-nan-ban.md`

**Decision tree**: RunPod for quick experiments (cheapest, instant provisioning).
GCP for production runs (same-region infra, spot recovery, Pulumi IaC).

**Local hardware**: RTX 2070 Super (8 GB VRAM) — dev only. Fits DynUNet + SAM3 Vanilla.
SAM3 Hybrid/TopoLoRA and VesselFM require cloud GPU (≥16 GB).

### Zero Hardcoding of Cloud/GPU Config (Non-Negotiable)
This is an **open-source academic repo** used by heterogeneous research labs. NEVER
hardcode cloud providers, GPU types, instance counts, regions, or Docker registries.

- **One lab** may want 1x RTX 4090 on RunPod. **Another** may want 8x A100 on AWS.
- All cloud config must flow through **Hydra config groups** (`configs/cloud/`,
  `configs/registry/`) so labs override via `configs/lab/lab_name.yaml` and users
  override via `configs/user/user_name.yaml` — without touching repo defaults.
- SkyPilot accelerator lists, region priorities, spot preferences, and disk sizes
  are config, not code. A lab with AWS credits should need ZERO code changes.
- Docker registry (GHCR, GAR, ECR, DockerHub) is a config choice, not hardcoded.
- See: `docs/planning/hydra-config-verification-report.md` for the full audit.

### SkyPilot = Intercloud Broker (Non-Negotiable)
SkyPilot = intercloud broker ([Yang et al., NSDI'23](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng))
— Slurm for multi-cloud, NOT IaC. NOT "a RunPod launcher." ALWAYS used for compute
beyond dev machine. NEVER bypass. SkyPilot YAML MUST use `image_id: docker:...` —
bare VM is BANNED. Setup is ONLY for data pull + config (never apt-get, uv sync).
See: `knowledge-graph/domains/cloud.yaml` for full details.

## Quick Reference

Python 3.12+ | uv (ONLY) | ruff + mypy | pytest | Hydra-zen (train) + Dynaconf (deploy)
PyTorch + MONAI + TorchIO | MLflow + DuckDB | Prefect 3.x (5 flows) | SkyPilot (compute)
BentoML + ONNX Runtime | Docker Compose (infra) | Optuna + ASHA (HPO)
Full stack details: `src/minivess/observability/CLAUDE.md`, `deployment/CLAUDE.md`

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
   values MUST be in `.env.example` FIRST. BANNED: hardcoded URLs in Dockerfiles,
   `os.environ.get("VAR", "fallback")` in flow files (use `resolve_tracking_uri()`
   or fail loudly), service URLs in Dynaconf TOML. See `tests/v2/unit/test_env_single_source.py`.

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

## What AI Must NEVER Do

- Confabulate — web-search instead. Hardcode task names, cloud providers, GPU types.
- Use `import re` for structured data. Use pip/conda/poetry. Skip pre-commit hooks.
- Suggest `python scripts/*.py` for training — use Prefect flows in Docker.
- Use `/tmp` for artifacts. Write citations without hyperlinks. Use T4 GPU for SAM3/VesselFM.
- Treat session summaries as authorization for infrastructure changes — ASK the user.
- Add cloud providers beyond RunPod + GCP without explicit authorization.
- Write SkyPilot YAML with bare VM setup (apt-get, uv sync) — Docker image_id ONLY.
- Enable GitHub Actions triggers — CI is EXPLICITLY DISABLED (credits).
- Use `os.environ.get("VAR", "fallback")` — fail loudly or use `resolve_*()`.
- Dump wall-of-text questions — use `AskUserQuestion` tool, max 4 per round.

## TDD Workflow (Non-Negotiable)

RED (tests first) → GREEN (implement) → VERIFY (run all) → FIX → CHECKPOINT → CONVERGE.
Skill: `.claude/skills/self-learning-iterative-coder/SKILL.md`

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

## Datasets

MiniVess (70 vols, primary), DeepVess/TubeNet/VesselNN (external test only).
Splits: 3-fold seed=42, `configs/splits/3fold_seed42.json` (47 train / 23 val).
Default loss: `cbdice_cldice` (CbDiceClDiceLoss).
Full registry: `docs/datasets/README.md` + `src/minivess/data/external_datasets.py`.

## Citation Rules (NON-NEGOTIABLE)

1. **Author-year format only** — "Surname et al. (Year)", never numeric [1]
2. **Central bibliography** — All citations in `bibliography.yaml`, reference by `citation_key`
3. **No citation loss** — References are append-only. Pre-commit hook blocks removal.
4. **Every citation = hyperlink** — `[Author (Year). "Title." *Journal*.](URL)`. No bare text.

## Knowledge Graph

Read [`knowledge-graph/navigator.yaml`](knowledge-graph/navigator.yaml) FIRST -> route to domain -> load on demand.

6-layer architecture ([docs/planning/prd-kg-openspec-architecture.md](docs/planning/prd-kg-openspec-architecture.md)):
- **L0**: `.claude/rules/` + `CLAUDE.md` (Constitution -- invariant rules)
- **L1**: `docs/planning/` + `MEMORY.md` (Hot Context -- current work)
- **L2**: `knowledge-graph/navigator.yaml` (Navigator -- domain routing)
- **L3**: `knowledge-graph/decisions/*.yaml` + `domains/*.yaml` (Evidence -- 65 Bayesian decision nodes)
- **L4**: `openspec/specs/` (Specifications -- GIVEN/WHEN/THEN testable scenarios)
- **L5**: `src/` + `tests/` (Implementation -- actual code)

65 PRD decision nodes across 11 domains. PRD Skill: `.claude/skills/prd-update/SKILL.md`.
OpenSpec: `openspec/` -- spec-driven development via `/opsx:propose`.
