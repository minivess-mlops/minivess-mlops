#!/usr/bin/env bash
# Weekly security scan — run against all built MinIVess flow images.
#
# Usage:
#   bash scripts/weekly_security_scan.sh
#
# Scans all 12 flow images for CRITICAL + HIGH CVEs using Trivy.
# Images that are not built locally are skipped.
# Exit code 1 if any built image has CRITICAL CVEs.
#
# Schedule: run weekly (e.g., cron: 0 2 * * 0)
# Results: printed to stdout; integrate with Prometheus pushgateway or
#          MLflow security_scans experiment for tracking over time.

set -euo pipefail

FLOW_IMAGES=(
  "minivess-train:latest"
  "minivess-data:latest"
  "minivess-analyze:latest"
  "minivess-hpo:latest"
  "minivess-deploy:latest"
  "minivess-dashboard:latest"
  "minivess-acquisition:latest"
  "minivess-annotation:latest"
  "minivess-biostatistics:latest"
  "minivess-post-training:latest"
  "minivess-pipeline:latest"
  "minivess-qa:latest"
  "minivess-mlflow:latest"
  "minivess-base:latest"
)

FAILURES=()
SKIPPED=()
SCANNED=()

echo "=== MinIVess Weekly Security Scan ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

for image in "${FLOW_IMAGES[@]}"; do
  image_name="${image%%:*}"
  if ! docker images --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | grep -q "^${image}$"; then
    SKIPPED+=("$image")
    echo "  SKIP   $image (not built locally)"
    continue
  fi

  echo -n "  SCAN   $image ... "
  if trivy image \
      --exit-code 1 \
      --severity CRITICAL \
      --ignore-unfixed \
      --quiet \
      "$image" 2>/dev/null; then
    SCANNED+=("$image")
    echo "OK"
  else
    FAILURES+=("$image")
    echo "CRITICAL CVEs FOUND"
  fi
done

echo ""
echo "=== Summary ==="
echo "  Scanned: ${#SCANNED[@]}"
echo "  Skipped: ${#SKIPPED[@]} (not built)"
echo "  Failed:  ${#FAILURES[@]}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo ""
  echo "CRITICAL CVEs found in:"
  for img in "${FAILURES[@]}"; do
    echo "  - $img"
  done
  echo ""
  echo "Fix: rebuild with updated base image or uv sync, then re-run."
  exit 1
fi

echo ""
echo "All scanned images: no CRITICAL CVEs."
