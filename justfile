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

# Train a model (full experiment)
train *ARGS:
    uv run python scripts/train.py {{ARGS}}

# Quick debug training (1 epoch, CPU, small data)
train-debug *ARGS:
    uv run python scripts/train.py --compute cpu --loss dice_ce --debug {{ARGS}}

# Full 3-loss sweep
train-sweep *ARGS:
    uv run python scripts/train.py --compute gpu_low --loss dice_ce,dice_ce_cldice,cbdice {{ARGS}}

# Run serving (placeholder)
serve:
    uv run bentoml serve src/minivess/serving/service.py

# Clean build artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
