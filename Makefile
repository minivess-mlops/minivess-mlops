# MinIVess MLOps — Makefile
# Convenience targets for Docker infrastructure management.
# All training/evaluation goes through Prefect flows, not this Makefile.

.PHONY: init-volumes scan sbom seccomp-audit-train install-trivy help \
       test-staging test-prod test-gpu test-e2e \
       build-base-gpu build-base-cpu build-base-light build-bases requirements-tiers

help:
	@echo "MinIVess MLOps Makefile"
	@echo ""
	@echo "Test Tiers:"
	@echo "  test-staging        Fast tests, no model loading (<3 min)"
	@echo "  test-prod           Full suite except GPU instance tests"
	@echo "  test-gpu            GPU instance tests (SAM3, requires CUDA + weights)"
	@echo "  test-e2e            Full e2e pipeline (Docker + GPU + data, ~60 min)"
	@echo ""
	@echo "Docker Images:"
	@echo "  build-base-gpu      Build minivess-base:latest (CUDA + PyTorch + MONAI)"
	@echo "  build-base-cpu      Build minivess-base-cpu:latest (scipy/pandas/DuckDB)"
	@echo "  build-base-light    Build minivess-base-light:latest (prefect/FastAPI)"
	@echo "  build-bases         Build all 3 base images"
	@echo "  requirements-tiers  Regenerate requirements-{cpu,light}.txt from uv.lock"
	@echo ""
	@echo "Infrastructure:"
	@echo "  init-volumes        Fix Docker named volume ownership (run once after first up)"
	@echo "  scan                Trivy vulnerability scan on all minivess-* images"
	@echo "  sbom                Generate CycloneDX SBOM for minivess-base image"
	@echo "  seccomp-audit-train Run train flow with seccomp audit profile (syscall discovery)"
	@echo "  install-trivy       Install Trivy scanner to /usr/local/bin"
	@echo "  help                Show this help message"

# ---------------------------------------------------------------------------
# Test tiers
# ---------------------------------------------------------------------------
# Staging: no model loading, no integration, no slow tests. Target: <3 min.
test-staging:
	MINIVESS_ALLOW_HOST=1 uv run pytest tests/ -x -q \
	  -m "not model_loading and not slow and not integration" \
	  --ignore=tests/v2/quasi_e2e/ \
	  --timeout=60

# Prod: everything except GPU instance tests. Includes slow + model loading.
# Excludes integration tests (Docker stack not guaranteed).
test-prod:
	MINIVESS_ALLOW_HOST=1 uv run pytest tests/ -x -q \
	  -m "not integration" \
	  --ignore=tests/v2/quasi_e2e/ \
	  --timeout=300

# E2E: Full pipeline via Docker. Requires Docker daemon, GPU, MiniVess data,
# .env configured (from .env.example), Docker images built, minivess-network created.
# Runtime: ~40-60 min. Uses pytest-docker to manage Docker Compose lifecycle.
test-e2e:
	uv run pytest tests/v2/integration/e2e/ -x -q \
	  -m "slow and integration" \
	  --timeout=3600 \
	  -v

# GPU instance: SAM3 tests in tests/gpu_instance/. Requires CUDA + SAM3 weights.
# Run on RunPod / intranet GPU servers, NOT on dev machines.
test-gpu:
	MINIVESS_ALLOW_HOST=1 uv run pytest tests/gpu_instance/ -x -q \
	  -o "collect_ignore_glob=" \
	  --timeout=600

# ---------------------------------------------------------------------------
# Docker base image builds (3-tier hierarchy)
# ---------------------------------------------------------------------------
build-base-gpu:
	DOCKER_BUILDKIT=1 docker build \
	  --build-arg GIT_COMMIT=$$(git rev-parse HEAD) \
	  --build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	  -t minivess-base:latest \
	  -f deployment/docker/Dockerfile.base .

build-base-cpu:
	DOCKER_BUILDKIT=1 docker build \
	  --build-arg GIT_COMMIT=$$(git rev-parse HEAD) \
	  --build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	  -t minivess-base-cpu:latest \
	  -f deployment/docker/Dockerfile.base-cpu .

build-base-light:
	DOCKER_BUILDKIT=1 docker build \
	  --build-arg GIT_COMMIT=$$(git rev-parse HEAD) \
	  --build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	  -t minivess-base-light:latest \
	  -f deployment/docker/Dockerfile.base-light .

build-bases: build-base-gpu build-base-cpu build-base-light

requirements-tiers:
	uv export --frozen --only-group cpu --no-emit-project \
	  --output-file deployment/docker/requirements-cpu.txt
	uv export --frozen --only-group light --no-emit-project \
	  --output-file deployment/docker/requirements-light.txt
	@echo "Regenerated requirements-cpu.txt and requirements-light.txt"

init-volumes:
	@echo "Initializing Docker named volume ownership..."
	docker run --rm --user root \
	  -v deployment_checkpoint_cache:/app/checkpoints \
	  -v deployment_logs_data:/app/logs \
	  -v deployment_mlruns_data:/app/mlruns \
	  -v deployment_data_cache:/app/data \
	  -v deployment_configs_splits:/app/configs/splits \
	  ubuntu:22.04 \
	  chown -R 1000:1000 \
	    /app/checkpoints /app/logs /app/mlruns /app/data /app/configs/splits
	@echo "Volume ownership initialized (UID=1000 = minivess user)"

scan:
	@echo "Scanning MinIVess Docker images with Trivy..."
	@which trivy || (echo "Install Trivy: make install-trivy" && exit 1)
	@for flow in base base-cpu base-light train data analyze deploy dashboard hpo post_training; do \
	  echo "Scanning minivess-$$flow:latest..."; \
	  trivy image --exit-code 0 --severity CRITICAL,HIGH --ignore-unfixed \
	    minivess-$$flow:latest 2>/dev/null || echo "  Image not built: minivess-$$flow"; \
	done

sbom:
	@echo "Generating SBOM for MinIVess base image..."
	@which trivy || (echo "Install Trivy: make install-trivy" && exit 1)
	@mkdir -p sbom/
	@trivy image --format cyclonedx --output sbom/minivess-base-sbom.json \
	  minivess-base:latest
	@echo "SBOM written to sbom/minivess-base-sbom.json"

seccomp-audit-train:
	@echo "Running train flow with seccomp audit profile (syscall discovery)..."
	@echo "NOTE: Requires auditd running on host to capture SCMP_ACT_LOG events."
	docker compose \
	  --env-file .env \
	  -f deployment/docker-compose.flows.yml \
	  run --rm -T \
	  --security-opt seccomp=deployment/seccomp/audit.json \
	  --shm-size 8g \
	  train </dev/null 2>&1 | head -100
	@echo "Check /var/log/audit/audit.log for syscall log entries."

install-trivy:
	@echo "Installing Trivy to /usr/local/bin..."
	curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh \
	  | sh -s -- -b /usr/local/bin
	@echo "Trivy installed: $$(trivy --version)"
