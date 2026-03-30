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

**Three-tier multi-stage Docker builds are MANDATORY.** All base images use 2-stage
builder→runner pattern. See `deployment/CLAUDE.md` for the full hierarchy:
- **Tier A** (GPU): `nvidia/cuda → minivess-base:latest` — 10 flows
- **Tier B** (CPU): `python:3.13 → minivess-base-cpu:latest` — biostatistics
- **Tier C** (Light): `python:3.13 → minivess-base-light:latest` — dashboard, pipeline
Flow Dockerfiles are THIN (only `COPY`, `ENV`, `CMD`) — never `apt-get` or `uv`.
Dockerfile changes MUST preserve the multi-stage builder→runner separation.
See: `docs/planning/docker-base-improvement-plan.md` (status: implemented)

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
- **Region**: `europe-west4` (Netherlands)
- **GCS Buckets**: `minivess-mlops-dvc-data`, `minivess-mlops-mlflow-artifacts`, `minivess-mlops-checkpoints`
- **Docker Registry**: GAR (`europe-west4-docker.pkg.dev/minivess-mlops/minivess`)
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
23. **NEVER Fix Failures Serially (Non-Negotiable)** — When `make test-staging`
    fails, NEVER fix one test, re-run, fix next (whac-a-mole). Instead:
    (1) **GATHER**: Run `--maxfail=200` WITHOUT `-x` to see ALL failures
    (2) **CATEGORIZE**: Group by file, then by root cause (`| uniq -c | sort -rn`)
    (3) **PLAN**: Determine fix strategy per root cause (batch replace, schema fix, etc.)
    (4) **FIX**: All instances of each root cause in one commit
    (5) **VERIFY**: Full suite again without `-x`
    One root cause often explains 50+ failures. Serial fixing wastes 25+ minutes
    on what should be a 10-minute batch operation.
    See: `.claude/metalearning/2026-03-18-whac-a-mole-serial-failure-fixing.md`
24. **Tokens Upfront — Scientific Production-Grade Code (Non-Negotiable)** —
    Spend more tokens reading and understanding BEFORE writing code. The cost of
    sloppy initial code is always higher than the cost of careful initial code:
    - **30% reading / 70% implementing** (not 5% / 95%)
    - Read ALL relevant source files before writing tests
    - Read ALL existing tests before writing new ones
    - Understand the full interface before implementing
    - "I'll just try this and see" without reading context is BANNED
    - One careful implementation pass is cheaper than three sloppy-then-fix passes
    This is a scientific platform — correctness and reproducibility outweigh speed.
    See: `.claude/skills/self-learning-iterative-coder/SKILL.md` Rule #11
25. **Loud Failures, Never Silent Discards (Non-Negotiable)** — Pipeline functions
    MUST raise exceptions or `logger.error()` on invalid/empty input. NEVER silently
    return empty results (`return {}`, `return []`, `return None`). A function that
    receives empty dataloaders must RAISE, not silently skip evaluation.
    - **Critical pipeline functions**: `raise ValueError("dataloaders_dict is empty")`
    - **Optional integrations (Sentry, PostHog)**: `logger.warning("PostHog not configured")`
    - **Stubs awaiting implementation**: `raise NotImplementedError("VesselFM fine-tuning not implemented")`
    - **BANNED**: `if not data: return {}` — this hides broken pipelines for months.
    Every failure point must use `try/except` with `logger.error()` so both humans
    reading logs AND Claude Code monitoring runs understand what is happening.
    See: `.claude/metalearning/2026-03-19-external-test-datasets-never-wired-silent-failure.md`
26. **Greenfield Project — No Legacy, No Deprecation Notices (Non-Negotiable)** — This
    is a greenfield project with zero production users and zero legacy data. NEVER maintain
    backward compatibility. NEVER create migration/normalization layers. NEVER offer "keep
    both formats." NEVER add deprecation notices or "deprecated" comments — DELETE the old
    code entirely. No "keep for reference," no "marked as deprecated," no dual implementations.
    When there are two ways to do the same thing, DELETE one immediately. When changing
    conventions (metric keys, config formats, flow architecture), DELETE the old convention
    entirely. Clean slate always. One way to do each thing.
    See: `.claude/metalearning/2026-03-19-backward-compat-resistance-greenfield-project.md`
27. **Debug Run = Full Production (Non-Negotiable)** — A debug run is the FULL production
    experiment with ONLY 3 differences: (1) fewer epochs, (2) less data, (3) fewer folds.
    EVERYTHING else is identical — all factorial factors, all flows, all zero-shot baselines,
    all logging, all compliance. NEVER propose reducing debug scope without explicit user
    authorization. "Debug" means "run fast to catch bugs" — NOT "run less to avoid bugs."
    See: `.claude/metalearning/2026-03-19-debug-run-is-full-production-no-shortcuts.md`
28. **Zero Silent Skips (Non-Negotiable)** — Every SKIPPED test is a bug hiding as a
    skip. When reporting test results, ALWAYS report the skip count AND the skip reasons.
    "5362 passed, 5 skipped" is NOT "all green" — it is 5 bugs. For each skip:
    - **Module not installed** → install it (uv sync --all-extras) or delete the test if deprecated
    - **Hardware-specific** (CTK, GPU) → acceptable skip, but document WHY in test
    - **Cloud credentials** → auto-skip with clear message, acceptable for local dev
    - **Deprecated library** (langgraph) → DELETE the test, don't skip it
    NEVER silently accept skips. ALWAYS inform the user of skip count and reasons.
    The same applies to warnings — suppress cosmetic warnings at entry point, but
    NEVER suppress warnings that indicate bugs or configuration problems.
    **MANDATORY Skip Investigation Protocol** — Before classifying ANY skip as
    "acceptable" or "hardware-gated", Claude MUST run diagnostic commands:
    (1) `which <tool>` or `dpkg -l | grep <pkg>` — is the tool/package installed?
    (2) `<tool> --version` — is it the right version?
    (3) `find /etc -name <config>` — is the config at a different path?
    (4) If tool IS installed but wrong version → report as FIXABLE, not "acceptable"
    (5) If tool is NOT installed → ask user if we should install it
    Classifying a skip without running diagnostics is the same as confabulating.
    See: `.claude/metalearning/2026-03-21-silent-skip-acceptance-ctk-config-path.md`,
    `.claude/metalearning/2026-03-21-mamba-ssm-silent-skip-cuda-mismatch.md`
22. **Single-Source Config via `.env.example` (Non-Negotiable)** — ALL configurable
   values MUST be in `.env.example` FIRST. BANNED: hardcoded URLs in Dockerfiles,
   `os.environ.get("VAR", "fallback")` in flow files (use `resolve_tracking_uri()`
   or fail loudly), service URLs in Dynaconf TOML. See `tests/v2/unit/test_env_single_source.py`.
29. **ZERO Hardcoded Parameters — Config Is the Only Source (Non-Negotiable)** —
    EVERY numeric parameter that a researcher might want to change MUST come from
    the Hydra-zen / Dynaconf config chain, NEVER from Python defaults. This includes:
    - **Statistical thresholds**: `alpha`, `rope`, `n_bootstrap`, `n_permutations`
    - **Random seeds**: `seed` — ALWAYS from config, never `seed=42` in code
    - **Training hyperparameters**: `learning_rate`, `batch_size`, `max_epochs`
    - **Infrastructure values**: ports, timeouts, retry counts
    The config YAML is the single source of truth. Functions receive these values
    as parameters from the caller (which reads them from config). Default parameter
    values in function signatures are BANNED for any value a researcher might change.
    **Tests**: Must read defaults from the config class (e.g., `BiostatisticsConfig().alpha`),
    NEVER hardcode `0.05` or `42` in assertions. A researcher who changes alpha to 0.01
    in their YAML must not have tests assert against a stale 0.05.
    **Why Claude Code gets this wrong**: LLM training data is full of `alpha=0.05` and
    `seed=42` as "obvious defaults." This is the most insidious anti-pattern because it
    LOOKS correct and WORKS initially — the bug only manifests when a researcher changes
    the config and discovers that code/tests silently ignore their choice.
    See: `.claude/metalearning/2026-03-20-hardcoded-significance-level-antipattern.md`
    Guard: `tests/v2/unit/biostatistics/test_no_hardcoded_alpha.py` (Issue #881)

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

30. **Algorithms Must Match Their Literature Definition (Non-Negotiable)** — Every
    algorithm, loss function, metric, or method MUST be implemented EXACTLY as defined
    in the paper it claims to be. NEVER invent custom definitions for well-known terms.
    - **SWA** means Izmailov et al. (2018) — cyclic LR + `torch.optim.swa_utils.AveragedModel`
      + `update_bn()`. Post-hoc checkpoint averaging is NOT SWA — call it "checkpoint averaging."
    - **SWAG** means Maddox et al. (2019) — requires training-time second-moment collection.
    - **LoRA** means Hu et al. (2021) — must match the paper's layer targeting strategy.
    - If you don't know the exact algorithm, **web-search the paper and read it** before
      naming anything. Giving a well-known algorithm name to a different implementation
      is scientific fraud. See: `.claude/metalearning/2026-03-21-fake-swa-checkpoint-averaging-mislabeled.md`

31. **Zero Improvisation on Declarative Configs (Non-Negotiable)** — Claude Code MUST
    NEVER add, modify, or remove keys from declarative config files (YAML, JSON, TOML)
    unless EXPLICITLY instructed by the user. Declarative configs are SPECIFICATIONS,
    not code to optimize. "Helpful fallbacks", "safety nets", "smart defaults" are ALL
    BANNED in config files. If a config is wrong, TELL THE USER — do not fix it silently.
    This applies to: SkyPilot YAML, Docker Compose YAML, Hydra configs, DVC configs,
    Pulumi configs, `.env.example`, `configs/cloud/*.yaml`, `configs/factorial/*.yaml`.
    **Defense-in-depth (5 layers)**:
    - **L1 (Pre-commit)**: `scripts/validate_yaml_contract.py` checks all SkyPilot YAMLs
      against `configs/cloud/yaml_contract.yaml` golden reference
    - **L2 (Test suite)**: `tests/v2/unit/deployment/test_yaml_contract_enforcement.py`
      validates GPU allowlists, cloud providers, schema drift, cross-file consistency
    - **L3 (Preflight)**: `scripts/preflight_gcp.py` → `check_yaml_contract()` blocks
      launch if YAML violates the contract
    - **L4 (Experiment harness)**: `/experiment-harness` Phase 2 VALIDATE gate reads
      the contract and rejects launches with unauthorized resources
    - **L5 (This rule)**: Claude Code reads this rule and refuses to modify configs
    **Golden contract**: `configs/cloud/yaml_contract.yaml` — SINGLE SOURCE OF TRUTH
    for allowed GPU types, cloud providers, and YAML schema. Update ONLY with explicit
    user authorization.
    **Root cause**: Claude added A100-80GB to `train_factorial.yaml` as a "fallback"
    without user authorization. 34 jobs x A100 = ~$40 instead of ~$8 (5.5x cost).
    See: `.claude/metalearning/2026-03-24-unauthorized-a100-in-skypilot-yaml.md`
    See: `.claude/metalearning/2026-03-24-yaml-is-the-contract-zero-improvisation.md`

32. **Quality Over Speed — No Reactive Rushing (Non-Negotiable)** — On a codebase
    with 6000+ tests, 5 Prefect flows, SkyPilot orchestration, and 75 KG decision
    nodes, EVERY change requires:
    (1) **PLAN** with reviewer agents optimizing the approach
    (2) **IMPLEMENT** via `/self-learning-iterative-coder` (RED→GREEN→VERIFY)
    (3) **REVIEW** via `/simplify` (catch quality issues)
    (4) **VERIFY** with `make test-staging` + `make test-prod` (0 skips, 0 failures)
    (5) **ASK** user before any cloud launch or infrastructure change
    NEVER: rush to launch, commit untested "fixes," generate plans as substitute for
    working code, launch without permission, or react to failures instead of preventing
    them. "Standing by" > "launching broken code." "I need to verify first" > "let me
    just try this real quick."
    See: `.claude/metalearning/2026-03-24-reactive-rushing-instead-of-proactive-quality.md`

33. **Docker+Prefect Execution Is Non-Negotiable — Zero Bypass on staging/prod (Non-Negotiable)** —
    On ANY branch derived from `main` or `prod`, training and pipeline execution MUST go
    through Docker containers orchestrated by Prefect. There are ZERO edge cases:
    - `MINIVESS_ALLOW_HOST=1` is **pytest-only** — NEVER in scripts, suggestions, or AskUserQuestion options
    - `PREFECT_DISABLED=1` is **pytest-only** — NEVER in production, staging, or development runs
    - "Local launcher scripts" that call `training_flow()` directly are BANNED
    - Framing Docker+Prefect as "heavy" or "slower" is BANNED — it IS the execution model
    - The correct local training command is ALWAYS: `docker compose run --shm-size 8g train`
    - All config comes from: `.env` (secrets/URIs) + YAML (experiment params) — never hardcoded
    **Claude's recurring failure**: proposing `MINIVESS_ALLOW_HOST=1` shortcuts because LLM
    training data is saturated with bare-metal `python train.py` patterns. This has been
    documented 7+ times in metalearning. The "quick" path IS the Docker path.
    See: `.claude/metalearning/2026-03-29-local-launcher-hack-proposed-instead-of-docker-prefect.md`
    See: `.claude/metalearning/2026-03-14-docker-resistance-anti-pattern.md`
    See: `.claude/metalearning/2026-03-06-standalone-script-antipattern.md`
    Guard: Issue #971, Issue #972

34. **"Import ≠ Done" — Code Must Be CALLED, DEPLOYED, and OBSERVABLE (Non-Negotiable)** —
    Writing a module and importing it is NOT implementing it. A task is DONE when:
    (1) Code is CALLED in the production code path (not just imported)
    (2) Docker image is REBUILT with the change
    (3) Feature produces OBSERVABLE OUTPUT (logs, heartbeat.json, metrics, dashboard)
    (4) A `docker compose run` invocation DEMONSTRATES the feature working
    (5) AST "import exists" tests are NECESSARY but NOT SUFFICIENT
    **Banned pattern**: Write module → import it → test import exists → mark DONE → move on.
    This creates dead code that passes all tests but provides zero functionality.
    **Required pattern**: Write module → call it from flow → rebuild Docker → run flow →
    verify output is visible in Docker logs / Grafana / Prefect UI → THEN mark DONE.
    **NEVER start training or experiment runs until the observability infrastructure is
    VERIFIED FUNCTIONAL** — not "code exists" but "I can see training metrics updating
    in Grafana, heartbeat.json is being written, Docker healthcheck shows healthy."
    See: `.claude/metalearning/2026-03-30-observability-code-written-but-never-wired-deployed.md`

## What AI Must NEVER Do

- Recommend switching to a fresh session between planning and execution. Context amnesia
  on session switch is a PROVEN failure pattern (14+ metalearning docs). Execute plans in
  the SAME session that created them. Use context compaction if needed, not session breaks.
  See: `.claude/metalearning/2026-03-28-context-amnesia-deferred-deepvess-whac-a-mole.md`
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
- Hardcode `alpha=0.05`, `seed=42`, or ANY parameter that belongs in config.
- Use default parameter values for researcher-configurable numbers — read from config.
- Rename/redefine well-known algorithms — SWA means Izmailov (2018), not "checkpoint averaging."
- Build parallel artifact persistence — MLflow artifact store is THE ONLY mechanism.
  When MLflow breaks, FIX MLflow — don't add file_mounts, rsync, or custom sync hacks.
  See: `.claude/metalearning/2026-03-21-competing-artifact-persistence-mechanisms.md`
- Add, modify, or remove keys from declarative config files without explicit user instruction.
  "Helpful fallbacks" in YAML are BANNED. YAML is the contract — execute AS-IS or report.
  See: `.claude/metalearning/2026-03-24-yaml-is-the-contract-zero-improvisation.md`

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

## Branch Model (Non-Negotiable)

Two protected branches. Both require PRs — no direct pushes.

| Branch | Role | Default? | Test gate | Merge frequency |
|--------|------|----------|-----------|-----------------|
| **`main`** | Staging — daily work | Yes (default) | `make test-staging` (~3 min) | Every PR |
| **`prod`** | Production — verified releases | No | `make test-prod` (~10 min) | Occasional promotion |

**Workflow:**
```
feature-branch → PR → main (staging, fast tests)
                       ↓ (when ready to promote)
                  main → PR → prod (full test suite)
```

- **Feature PRs** always target `main`. Run `make test-staging` before merge.
- **Promotion PRs** (`main → prod`) run `make test-prod`. Only create these when
  you are willing to wait for the full suite. The user decides when to promote.
- **NEVER** create feature PRs targeting `prod` directly.
- See #304 for the original staging/prod split rationale.

## Test Tiers (Non-Negotiable)

Three tiers, invoked via Makefile. **NEVER run SAM3 model tests in standard CI.**

| Tier | Command | What runs | Target time | Branch gate |
|------|---------|-----------|-------------|-------------|
| **Staging** | `make test-staging` | No model loading, no slow, no integration | <3 min | `main` (every PR) |
| **Prod** | `make test-prod` | Everything except GPU instance tests | ~5-10 min | `prod` (promotion PRs) |
| **GPU** | `make test-gpu` | SAM3 + GPU-heavy tests in `tests/gpu_instance/` | GPU instance only | RunPod only |

```bash
make test-staging    # PR readiness — fast, no models (gate for main)
make test-prod       # Full suite — includes model loading + slow (gate for prod)
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

| Dataset | Role | Volumes | Organ | Modality |
|---------|------|---------|-------|----------|
| **MiniVess** | Train/Val (primary) | 70 | Mouse brain cortex | Multiphoton |
| **DeepVess** | External TEST | ~7 | Mouse brain cortex | Multiphoton |
| **VesselNN** | Drift detection ONLY | 12 | Mouse brain | Two-photon |

Splits: 3-fold seed=42, `configs/splits/3fold_seed42.json` (47 train / 23 val).
Default loss: `cbdice_cldice` (CbDiceClDiceLoss).
Full registry: `docs/datasets/README.md` + `src/minivess/data/external_datasets.py`.
KG source of truth: `knowledge-graph/domains/data.yaml::dataset_strategy`.

**TubeNet EXCLUDED (Non-Negotiable)** — Only 1 two-photon volume (mouse olfactory bulb,
different organ from MiniVess cortex). Other 7 TubeNet volumes are HREM/CT/RSOM/OCT-A
(non-multiphoton modalities). Not useful for microvasculature generalization testing.
NEVER re-add TubeNet to test evaluation scope. See: `docs/planning/cold-start-prompt-pre-debug-qa-verification.md` Q31.

**VesselNN is NOT a test dataset** — Reserved for drift detection simulation ONLY.
Data leakage to MiniVess (same PI). Used with synthetic stack generation to test
how distribution shifts are detected. NEVER use for test evaluation.

**Test metric prefix**: `test/deepvess/{metric}` (extensible — adding a new test
dataset = `test/{newdataset}/{metric}`, no code changes needed).

**PLATFORM FRAMING (Non-Negotiable)**: This is a **platform paper** (Nature Protocols),
NOT a SOTA segmentation paper. External test evaluation demonstrates the PLATFORM'S
ability to handle arbitrary test datasets with subgroup analysis and automated
train/test comparison. The actual numbers on DeepVess are standard practice, not a
key finding. The platform capability IS the contribution. NEVER frame scientific
results as the main contribution. See: `.claude/metalearning/2026-03-19-platform-paper-not-sota-science-recurring.md`

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

---

## Knowledge Hub Integration

This repo's knowledge graph and bibliography are indexed in the knowledge-hub DuckDB (5,432 docs, 43M words).
When working in this repo, the `knowledge-hub` MCP server is auto-discovered.

Use the MCP tools for cross-source context:
- `search_knowledge("vascular segmentation loss functions")` — finds biblio papers + Obsidian notes + repo docs
- `search_knowledge("clDice centerline topology")` — cross-reference topology-aware segmentation research
- `search_knowledge("3D medical image reconstruction")` — biomedical imaging context
- `knowledge_hierarchy("Deep 3D")` — browse Obsidian domain tree
- `knowledge_coverage("vascular segmentation")` — find coverage gaps across sources

Biblio path: `/home/petteri/Dropbox/knowledge-hub/repositories/sci-llm-writer/biblio/` (19 domains, 1,183+ papers)
