# MinIVess MLOps v2 — Task Runner

# Default recipe
default:
    @just --list

# Install all dependencies
install:
    uv sync --all-extras

# Run tests
test *ARGS:
    uv run pytest tests/v2/ -x -q {{ARGS}}

# Run unit tests only
test-unit:
    uv run pytest tests/v2/unit/ -x -q

# Run integration tests
test-integration:
    uv run pytest tests/v2/integration/ -x -q

# Lint and format check
lint:
    uv run ruff check src/minivess/ tests/v2/
    uv run ruff format --check src/minivess/ tests/v2/

# Auto-fix lint issues
fix:
    uv run ruff check --fix src/minivess/ tests/v2/
    uv run ruff format src/minivess/ tests/v2/

# Type check
typecheck:
    uv run mypy src/minivess/

# Full verification (all three gates)
verify: lint typecheck test

# Start dev services
dev:
    docker compose -f deployment/docker-compose.yml --profile dev up -d

# Start monitoring services
monitoring:
    docker compose -f deployment/docker-compose.yml --profile monitoring up -d

# Start all services
up:
    docker compose -f deployment/docker-compose.yml --profile full up -d

# Stop all services
down:
    docker compose -f deployment/docker-compose.yml down

# ---------------------------------------------------------------------------
# Prefect Flow Execution (Docker Compose)
# ---------------------------------------------------------------------------
# All flow execution goes through Docker containers.
# See CLAUDE.md Rule #17: training MUST go through Prefect flows in Docker.
# Prerequisites: `just dev` to start PostgreSQL, MinIO, MLflow, Prefect.

FLOWS_COMPOSE := "deployment/docker-compose.flows.yml"

# Flow 0: Data Acquisition
acquisition *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm acquisition {{ARGS}}

# Flow 1: Data Engineering
data *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm data {{ARGS}}

# Flow 2: Model Training (GPU)
train *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm train {{ARGS}}

# Quick debug training (1 epoch, 1 fold)
train-debug:
    docker compose -f {{FLOWS_COMPOSE}} run --rm -e DEBUG=true -e MAX_EPOCHS=1 -e NUM_FOLDS=1 train

# ---------------------------------------------------------------------------
# Debug Recipes — Hydra-zen experiment configs (CLAUDE.md Rule #17)
# All use run_debug.sh which handles preflight, multi-model, flow chaining.
# ---------------------------------------------------------------------------

# Single model, 1 epoch, ~2 min
debug:
    bash scripts/run_debug.sh --experiment debug_single_model

# All 8GB-compatible models, 2 epochs, ~30 min
debug-all-models:
    bash scripts/run_debug.sh --experiment debug_all_models

# Train → post_training → analyze chain, ~15 min
debug-pipeline:
    bash scripts/run_debug.sh --experiment debug_full_pipeline

# 3 losses for ensemble testing, ~15 min
debug-multi-loss:
    bash scripts/run_debug.sh --experiment debug_multi_loss

# MONAI dataloader validation only, ~5 min
debug-data:
    bash scripts/run_debug.sh --experiment debug_data_validation

# Custom debug: just debug EXPERIMENT OVERRIDES (e.g.: just debug-custom debug_single_model "max_epochs=2")
debug-custom EXPERIMENT OVERRIDES="":
    bash scripts/run_debug.sh --experiment {{EXPERIMENT}} --override "{{OVERRIDES}}"

# Flow 2.5: Post-Training (SWA, calibration, conformal)
post-training *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm post_training {{ARGS}}

# Flow 3: Model Analysis
analyze *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm analyze {{ARGS}}

# Flow 4: Deployment (ONNX export, BentoML, promotion)
deploy-flow *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm deploy {{ARGS}}

# Flow 5: Dashboard & Reporting (best-effort)
dashboard *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm dashboard {{ARGS}}

# Flow 6: QA (best-effort — MLflow integrity, ghost run cleanup)
qa *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm qa {{ARGS}}

# Biostatistics (statistical analysis + publication figures)
biostatistics *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm biostatistics {{ARGS}}

# Hyperparameter Optimization (CPU — trials run in separate GPU containers)
hpo *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm hpo {{ARGS}}

# Data Annotation App
annotation *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm annotation {{ARGS}}

# Build all per-flow Docker images
build-flows:
    docker compose -f {{FLOWS_COMPOSE}} build

# Full pipeline (triggers flows via Prefect run_deployment)
pipeline *ARGS:
    docker compose -f {{FLOWS_COMPOSE}} run --rm pipeline {{ARGS}}

# ---------------------------------------------------------------------------
# SkyPilot (Cloud Compute)
# ---------------------------------------------------------------------------

# Launch SkyPilot training job
sky-train CONFIG="deployment/skypilot/train_generic.yaml" *ARGS:
    sky jobs launch {{CONFIG}} {{ARGS}}

# Launch SkyPilot HPO sweep
sky-sweep CONFIG="deployment/skypilot/train_hpo_sweep.yaml" *ARGS:
    sky jobs launch {{CONFIG}} {{ARGS}}

# Show SkyPilot job status
sky-status:
    sky jobs queue

# Clean build artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
