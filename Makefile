# MinIVess MLOps — Makefile
# Convenience targets for Docker infrastructure management.
# All training/evaluation goes through Prefect flows, not this Makefile.

.PHONY: init-volumes scan sbom seccomp-audit-train install-trivy help

help:
	@echo "MinIVess MLOps Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  init-volumes        Fix Docker named volume ownership (run once after first up)"
	@echo "  scan                Trivy vulnerability scan on all minivess-* images"
	@echo "  sbom                Generate CycloneDX SBOM for minivess-base image"
	@echo "  seccomp-audit-train Run train flow with seccomp audit profile (syscall discovery)"
	@echo "  install-trivy       Install Trivy scanner to /usr/local/bin"
	@echo "  help                Show this help message"

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
	@for flow in base train data analyze deploy dashboard qa hpo post_training; do \
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
