# MinIVess MLOps — Makefile
# Convenience targets for Docker infrastructure management.
# All training/evaluation goes through Prefect flows, not this Makefile.

.PHONY: init-volumes scan help

help:
	@echo "MinIVess MLOps Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  init-volumes  Fix Docker named volume ownership (run once after first up)"
	@echo "  scan          Trivy vulnerability scan on all minivess-* images"
	@echo "  help          Show this help message"

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
	@which trivy || (echo "Install Trivy: curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b ~/.local/bin" && exit 1)
	@for flow in base train data analyze deploy dashboard qa hpo post_training; do \
	  echo "Scanning minivess-$$flow:latest..."; \
	  trivy image --exit-code 0 --severity CRITICAL,HIGH --ignore-unfixed \
	    minivess-$$flow:latest 2>/dev/null || echo "  Image not built: minivess-$$flow"; \
	done
