# MinIVess MLOps — Makefile
# Convenience targets for Docker infrastructure management.
# All training/evaluation goes through Prefect flows, not this Makefile.

.PHONY: init-volumes scan sbom seccomp-audit-train install-trivy help \
       test-staging test-prod test-gpu test-e2e \
       build-base-gpu build-base-cpu build-base-light build-bases requirements-tiers \
       build-mamba push-mamba \
       ghcr-login push-ghcr \
       smoke-test-preflight smoke-test-gpu smoke-test-all smoke-test-gcp smoke-test-gcp-all \
       smoke-test-mamba smoke-test-mamba-spot smoke-test-mamba-preflight verify-smoke-test \
       monitor-smoke-test diagnose-last-smoke-test \
       dev-gpu dev-gpu-heavy dev-gpu-stop dev-gpu-ssh \
       dev-gpu-sync dev-gpu-upload-data dev-gpu-cleanup dev-gpu-status

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
	@echo "GHCR (Docker Registry):"
	@echo "  ghcr-login          Login to GitHub Container Registry"
	@echo "  push-ghcr           Tag + push minivess-base:latest to GHCR"
	@echo ""
	@echo "RunPod Dev GPU (Network Volume, file-based MLflow):"
	@echo "  dev-gpu               Launch dev GPU on RunPod (MODEL=dynunet)"
	@echo "  dev-gpu-heavy         Train sam3_hybrid + vesselfm (need >8 GB VRAM)"
	@echo "  dev-gpu-stop          Stop dev GPU pod (preserves state)"
	@echo "  dev-gpu-ssh           SSH into dev GPU pod"
	@echo "  dev-gpu-sync          Pull /opt/vol/mlruns/ back to local mlruns/"
	@echo "  dev-gpu-upload-data   Rsync local data/raw/minivess/ to Network Volume"
	@echo "  dataset-setup         Download + DVC version + RunPod upload (DATASET=minivess)"
	@echo "  dev-gpu-cleanup       Clear volume checkpoints/logs (keeps data+mlruns)"
	@echo "  dev-gpu-status        Show Network Volume usage + pod status"
	@echo ""
	@echo "Docker Images (Mamba variant):"
	@echo "  build-mamba           Build minivess-base:mamba-latest (INSTALL_MAMBA=1, ~30-45 min)"
	@echo "  push-mamba            Push :mamba-latest to GHCR (requires GITHUB_TOKEN)"
	@echo ""
	@echo "RunPod Smoke Tests (Docker, uses :mamba-latest or :latest):"
	@echo "  smoke-test-preflight  Validate env vars + connectivity"
	@echo "  smoke-test-gpu        Launch smoke test on RunPod RTX 4090 (MODEL=sam3_vanilla)"
	@echo "  smoke-test-all        Run all smoke tests on RunPod sequentially"
	@echo "  smoke-test-mamba      Launch MambaVesselNet++ smoke test on RunPod (T10, closes #744)"
	@echo "  smoke-test-mamba-spot Same but spot instance (cheaper, may preempt)"
	@echo "  smoke-test-gcp        Launch smoke test on GCP spot L4 (MODEL=sam3_vanilla)"
	@echo "  smoke-test-gcp-all    Run all smoke tests on GCP spot"
	@echo "  verify-smoke-test     Verify smoke test results (file-based MLflow)"
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
# Staging: gate for PRs → main. No model loading, no slow. Target: <3 min.
test-staging:
	MINIVESS_ALLOW_HOST=1 uv run pytest tests/ -x -q \
	  -m "not model_loading and not slow and not integration and not cloud_mlflow and not skypilot_cloud" \
	  --ignore=tests/v2/quasi_e2e/ \
	  --ignore=tests/v2/cloud/ \
	  --timeout=60

# Prod: gate for promotion PRs main → prod. Full suite, slow + model loading.
# Excludes integration tests (Docker stack not guaranteed).
test-prod:
	MINIVESS_ALLOW_HOST=1 uv run pytest tests/ -x -q \
	  -m "not integration" \
	  --ignore=tests/v2/quasi_e2e/ \
	  --ignore=tests/v2/cloud/ \
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

test-cloud-mlflow:  ## Run cloud MLflow tests (requires MLFLOW_TRACKING_URI set to remote URL)
	uv run pytest tests/v2/cloud/ -m "cloud_mlflow or skypilot_cloud" -v

test-pulumi:  ## Run Pulumi IaC validation tests
	uv run pytest tests/v2/unit/test_pulumi_stack.py -v

# ---------------------------------------------------------------------------
# RunPod Dev GPU (no Docker, deps via uv, RTX 4090/5090)
# ---------------------------------------------------------------------------
# RunPod pods ARE containers — Docker-in-Docker is impossible.
# This is the "dev" environment: direct uv execution, no Prefect.
# For staging/prod (Docker mandatory): use Lambda Labs targets below.

dev-gpu:  ## Launch dev GPU on RunPod RTX 4090/5090 (MODEL=dynunet)
	uv run python scripts/launch_dev_runpod.py --model $(or $(MODEL),dynunet)

dev-gpu-heavy:  ## Train heavy models on RunPod (sam3_hybrid + vesselfm, need >8 GB VRAM)
	@echo "=== Heavy models that CANNOT train on local 8 GB GPU ==="
	@echo "--- sam3_hybrid (~7.5 GB VRAM) ---"
	uv run python scripts/launch_dev_runpod.py --model sam3_hybrid
	@echo "--- vesselfm (~10 GB VRAM) ---"
	uv run python scripts/launch_dev_runpod.py --model vesselfm

dev-gpu-stop:  ## Stop dev GPU pod (preserves state, restart with 'sky start minivess-dev')
	sky stop minivess-dev

dev-gpu-ssh:  ## SSH into dev GPU pod for interactive work
	sky ssh minivess-dev

dev-gpu-sync:  ## Pull /opt/vol/mlruns/ back to local mlruns/ (closes #716)
	sky rsync down minivess-dev:/opt/vol/mlruns/ mlruns/
	@echo "MLflow runs synced to mlruns/"

dev-gpu-upload-data:  ## Rsync local data/raw/minivess/ to Network Volume (first-time setup)
	sky rsync up data/raw/minivess/ minivess-dev:/opt/vol/data/raw/
	@echo "Data uploaded to Network Volume at /opt/vol/data/raw/minivess/"

dataset-setup:  ## One-command dataset workflow: download → DVC version → RunPod upload (DATASET=minivess)
	@echo "=== Dataset Setup: $(or $(DATASET),minivess) ==="
	@echo "Step 1/3: Verify local data exists..."
	@test -d "data/raw/$(or $(DATASET),minivess)/imagesTr" || \
	  (echo "FATAL: data/raw/$(or $(DATASET),minivess)/imagesTr/ not found." && \
	   echo "Download first: uv run python scripts/download_minivess.py" && exit 1)
	@echo "  Found $$(ls data/raw/$(or $(DATASET),minivess)/imagesTr/ | wc -l) files"
	@echo ""
	@echo "Step 2/3: DVC tracking..."
	dvc add data/raw/$(or $(DATASET),minivess)
	@echo "  DVC file: data/raw/$(or $(DATASET),minivess).dvc"
	@echo ""
	@echo "Step 3/3: Upload to RunPod Network Volume..."
	sky rsync up data/raw/$(or $(DATASET),minivess)/ minivess-dev:/opt/vol/data/raw/$(or $(DATASET),minivess)/
	@echo ""
	@echo "=== Dataset $(or $(DATASET),minivess) ready on RunPod ==="

dev-gpu-cleanup:  ## Clear volume checkpoints/logs (keeps data+mlruns)
	sky exec minivess-dev -- "rm -rf /opt/vol/checkpoints/* /opt/vol/logs/*"
	@echo "Checkpoints and logs cleared from Network Volume"

dev-gpu-status:  ## Show Network Volume disk usage + pod status
	@echo "=== SkyPilot cluster status ==="
	sky status minivess-dev 2>/dev/null || echo "  (no active pod)"
	@echo "=== Network Volume disk usage ==="
	sky exec minivess-dev -- "df -h /opt/vol && echo '---' && du -sh /opt/vol/*/ 2>/dev/null" 2>/dev/null || \
	  (echo "  Pod not running — start first: sky start minivess-dev")

# ---------------------------------------------------------------------------
# Docker Images — zstd-compressed builds (20-30% faster pull)
# ---------------------------------------------------------------------------
# zstd decompression is 3-5x faster than gzip (Docker default).
# Requires BuildKit + containerd 1.5+ on target. GAR supports OCI zstd.
# See: docs/planning/docker-pull-runpod-provisioning-mlflow-cloud-run-multipart-upload-report.md

build-base-zstd:  ## Build minivess-base with zstd compression (faster pull)
	docker buildx build \
	  --output type=image,compression=zstd,compression-level=3,oci-mediatypes=true \
	  --build-arg GIT_COMMIT=$$(git rev-parse HEAD) \
	  --build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	  -t $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):latest \
	  -f deployment/docker/Dockerfile.base . \
	  2>&1 | tee /tmp/docker-build-zstd.log

# ---------------------------------------------------------------------------
# Docker Images — Mamba variant (INSTALL_MAMBA=1)
# ---------------------------------------------------------------------------
# Standard :latest image has no mamba-ssm (CUDA compilation is optional).
# :mamba-latest = :latest + mamba-ssm==2.3.1 CUDA extensions (~+2-4 GB image).
# Build once, reuse across all RunPod mamba smoke tests.

build-mamba:  ## Build minivess-base:mamba-latest (INSTALL_MAMBA=1, ~30-45 min)
	DOCKER_BUILDKIT=1 docker build \
	  --build-arg INSTALL_MAMBA=1 \
	  --build-arg GIT_COMMIT=$$(git rev-parse HEAD) \
	  --build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	  -t $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):mamba-latest \
	  -f deployment/docker/Dockerfile.base . \
	  2>&1 | tee /tmp/docker-build-mamba.log
	@echo "=== Verify mamba_ssm import ==="
	docker run --rm $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):mamba-latest \
	  python -c "import mamba_ssm; print(f'mamba_ssm {mamba_ssm.__version__} OK')"

push-mamba: ghcr-login  ## Push :mamba-latest to GHCR (requires GITHUB_TOKEN with write:packages)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):mamba-latest
	@echo "Pushed: $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):mamba-latest"

# ---------------------------------------------------------------------------
# RunPod Smoke Tests (Docker, uses :latest or :mamba-latest)
# ---------------------------------------------------------------------------
# SkyPilot smoke tests using Docker images via image_id: (CLAUDE.md mandate).
# Network Volume provides data cache + file-based MLflow tracking.

smoke-test-preflight:  ## Validate env vars + connectivity before GPU smoke test
	uv run python scripts/validate_smoke_test_env.py

smoke-test-gpu:  ## Launch smoke test on RunPod RTX 4090 (MODEL=sam3_vanilla)
	uv run python scripts/launch_smoke_test.py --model $(or $(MODEL),sam3_vanilla) --cloud runpod

smoke-test-all:  ## Run all smoke tests on RunPod sequentially
	@echo "=== Heavy models (need >8 GB VRAM) ==="
	$(MAKE) smoke-test-gpu MODEL=sam3_hybrid
	$(MAKE) smoke-test-gpu MODEL=vesselfm
	@echo "=== Standard models ==="
	$(MAKE) smoke-test-gpu MODEL=sam3_vanilla
	$(MAKE) smoke-test-gpu MODEL=dynunet

smoke-test-mamba:  ## Launch MambaVesselNet++ smoke test on RunPod (T10, closes #744)
	sky jobs launch deployment/skypilot/smoke_test_mamba.yaml \
	  --env-file .env \
	  -y 2>&1 | tee /tmp/ralph-launch-mamba.log

smoke-test-mamba-spot:  ## Same but spot instance (cheaper, may be preempted)
	sky jobs launch deployment/skypilot/smoke_test_mamba.yaml \
	  --env-file .env \
	  --env USE_SPOT=true \
	  -y 2>&1 | tee /tmp/ralph-launch-mamba-spot.log

smoke-test-mamba-preflight:  ## P0 pre-flight checks for mamba smoke test
	@echo "=== Checking SkyPilot RunPod access ==="
	sky check runpod
	@echo "=== Checking Network Volume ==="
	sky storage ls 2>/dev/null | grep -E "minivess-dev|NAME" || echo "WARNING: minivess-dev volume not found"
	@echo "=== Checking :mamba-latest image ==="
	docker manifest inspect $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):mamba-latest 2>/dev/null | \
	  python3 -c "import json,sys; m=json.load(sys.stdin); print(f'Image OK: {len(m[\"layers\"])} layers')" || \
	  echo "WARNING: mamba-latest not pushed — run: make build-mamba push-mamba"

smoke-test-gcp:  ## Launch smoke test on GCP spot T4/L4 (MODEL=sam3_vanilla)
	uv run python scripts/launch_smoke_test.py --model $(or $(MODEL),sam3_vanilla) --cloud gcp

smoke-test-gcp-all:  ## Run all smoke tests on GCP spot sequentially
	@echo "=== Heavy models (need >8 GB VRAM) ==="
	$(MAKE) smoke-test-gcp MODEL=sam3_hybrid
	$(MAKE) smoke-test-gcp MODEL=vesselfm
	@echo "=== Standard models ==="
	$(MAKE) smoke-test-gcp MODEL=sam3_vanilla
	$(MAKE) smoke-test-gcp MODEL=dynunet

verify-smoke-test:  ## Verify smoke test results on cloud MLflow
	uv run python scripts/verify_smoke_test.py $(or $(MODEL),sam3_vanilla)

monitor-smoke-test:  ## Monitor latest smoke test (Ralph loop, 15s poll)
	uv run python scripts/ralph_monitor.py --latest --poll-interval 15

diagnose-last-smoke-test:  ## Diagnose the most recent failed smoke test
	uv run python scripts/ralph_monitor.py --diagnose-last

# ---------------------------------------------------------------------------
# GHCR (GitHub Container Registry) — push Docker images for SkyPilot
# ---------------------------------------------------------------------------
DOCKER_REGISTRY ?= ghcr.io/petteriteikari
DOCKER_IMAGE_NAME ?= minivess-base
DOCKER_IMAGE_TAG ?= latest
DOCKER_IMAGE_FULL = $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

ghcr-login:  ## Login to GHCR (requires GITHUB_TOKEN with write:packages)
	@echo "Logging in to ghcr.io as $${GHCR_USERNAME:-petteriteikari}..."
	@echo "$${GITHUB_TOKEN}" | docker login ghcr.io -u "$${GHCR_USERNAME:-petteriteikari}" --password-stdin

push-ghcr: ghcr-login push-registry  ## Login to GHCR + push (convenience alias)

push-registry:  ## Tag + push minivess-base to configured DOCKER_REGISTRY
	docker tag minivess-base:latest $(DOCKER_IMAGE_FULL)
	docker push $(DOCKER_IMAGE_FULL)
	@echo "Pushed: $(DOCKER_IMAGE_FULL)"

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

sbom-python:
	@echo "Generating Python-level CycloneDX SBOM from uv environment..."
	@mkdir -p sbom/
	uv run cyclonedx-py environment --output-format json -o sbom/python-sbom.json
	@echo "Python SBOM written to sbom/python-sbom.json"
	uv run cyclonedx-py environment --output-format xml -o sbom/python-sbom.xml
	@echo "Python SBOM (XML) written to sbom/python-sbom.xml"

sbom-scan:
	@echo "Scanning Python SBOM for known vulnerabilities..."
	@which grype || (echo "Install Grype: curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin" && exit 1)
	grype sbom:sbom/python-sbom.json --output table
	@echo "Scan complete."

sbom-all: sbom sbom-python
	@echo "All SBOMs generated (Docker + Python)."

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
