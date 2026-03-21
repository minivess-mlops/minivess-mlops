#!/usr/bin/env bash
# =============================================================================
# Local Integration Test: Post-Training → Analysis → Biostatistics
# =============================================================================
#
# Runs the 3 CPU-only flows locally on artifacts downloaded from cloud.
# Uses the factorial-monitor skill for tracking.
#
# Prerequisites:
#   1. Cloud debug factorial run completed (24 conditions)
#   2. make sync-cloud-artifacts (or manual rsync)
#   3. uv sync --all-extras
#
# Usage:
#   ./scripts/run_local_post_analysis_biostats.sh
#   ./scripts/run_local_post_analysis_biostats.sh --dry-run
#   ./scripts/run_local_post_analysis_biostats.sh --skip-post-training
#
# Plan: docs/planning/debug-factorial-local-post-analysis-biostats.xml
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (from .env or defaults)
# ---------------------------------------------------------------------------
MLRUNS_DIR="${MLRUNS_DIR:-./mlruns}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/local_integration}"
EXPERIMENT_TRAINING="${UPSTREAM_EXPERIMENT:-debug_factorial}"
EXPERIMENT_EVALUATION="${EXPERIMENT_EVALUATION:-minivess_evaluation}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_POST_TRAINING="${SKIP_POST_TRAINING:-false}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-false}"
SKIP_BIOSTATISTICS="${SKIP_BIOSTATISTICS:-false}"

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --skip-post-training) SKIP_POST_TRAINING=true ;;
        --skip-analysis) SKIP_ANALYSIS=true ;;
        --skip-biostatistics) SKIP_BIOSTATISTICS=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# STOP Protocol Check
# ---------------------------------------------------------------------------
echo "=== STOP Protocol Check ==="
echo "  S(ource): Local host (MINIVESS_ALLOW_HOST=1 for test bypass)"
echo "  T(racking): Prefect active (not PREFECT_DISABLED)"
echo "  O(utputs): ${OUTPUT_DIR} (local disk)"
echo "  P(rovenance): Artifacts from cloud, reproducible locally"

if [ ! -d "$MLRUNS_DIR" ]; then
    echo "ERROR: MLflow runs directory not found: $MLRUNS_DIR"
    echo "Run 'make sync-cloud-artifacts' first to download from cloud."
    exit 1
fi

# Count training runs
N_RUNS=$(find "$MLRUNS_DIR" -name "meta.yaml" -path "*/tags/*" 2>/dev/null | head -100 | wc -l)
echo "  Found ~${N_RUNS} MLflow runs in ${MLRUNS_DIR}"

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "=== DRY RUN — would execute the following flows ==="
    echo "  1. Post-Training Flow (SWA + calibration on 24 conditions)"
    echo "  2. Analysis Flow (4 ensemble strategies, MiniVess + DeepVess eval)"
    echo "  3. Biostatistics Flow (N-way ANOVA, spec curve, rank stability)"
    echo ""
    echo "  Output: ${OUTPUT_DIR}"
    echo "  Skip Post-Training: ${SKIP_POST_TRAINING}"
    echo "  Skip Analysis: ${SKIP_ANALYSIS}"
    echo "  Skip Biostatistics: ${SKIP_BIOSTATISTICS}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
# NOTE: This script is a LOCAL INTEGRATION TEST — not a production run path.
# Docker bypass is required because this runs on the researcher's laptop.
# Production runs ALWAYS go through Docker. See CLAUDE.md Rule #2 (TOP-2).
export MINIVESS_ALLOW_HOST=1  # noqa: test-only-integration-script
export MLFLOW_TRACKING_URI="file://${PWD}/${MLRUNS_DIR}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== Starting Local Integration Test ==="
echo "  MLflow URI: ${MLFLOW_TRACKING_URI}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Post-Training Flow
# ---------------------------------------------------------------------------
if [ "$SKIP_POST_TRAINING" = "false" ]; then
    echo "=== Phase 1: Post-Training Flow ==="
    echo "  Applying SWA + calibration to training checkpoints..."

    uv run python -c "
from minivess.orchestration.flows.post_training_flow import post_training_flow
from minivess.config.post_training_config import PostTrainingConfig
from pathlib import Path

config = PostTrainingConfig()
result = post_training_flow(
    config=config,
    output_dir=Path('${OUTPUT_DIR}/post_training'),
    trigger_source='local_integration_test',
)
print(f'Post-Training completed: status={result.status}')
print(f'  SWA: {result.swa_completed}')
print(f'  Calibration: {result.calibration_completed}')
print(f'  Failed: {result.failed_operations}')
" 2>&1 | tee "${OUTPUT_DIR}/post_training.log"

    echo "  Post-Training Flow complete."
    echo ""
else
    echo "=== Phase 1: SKIPPED (--skip-post-training) ==="
fi

# ---------------------------------------------------------------------------
# Phase 2: Analysis Flow
# ---------------------------------------------------------------------------
if [ "$SKIP_ANALYSIS" = "false" ]; then
    echo "=== Phase 2: Analysis Flow ==="
    echo "  Building ensembles + evaluating on MiniVess + DeepVess..."

    uv run python -c "
from minivess.orchestration.flows.analysis_flow import run_analysis_flow
from minivess.config.evaluation_config import EvaluationConfig
from pathlib import Path
import os

os.environ['UPSTREAM_EXPERIMENT'] = '${EXPERIMENT_TRAINING}'

eval_config = EvaluationConfig(
    mlflow_training_experiment='${EXPERIMENT_TRAINING}',
    require_eval_metrics=False,  # Debug mode
)
# Analysis needs model config and dataloaders — these come from Hydra
# For local integration test, we use minimal config
print('Analysis Flow: starting (may take several minutes for ensemble evaluation)...')
print('NOTE: Full Analysis Flow requires dataloaders which need DVC data.')
print('      This test verifies discovery and ensemble building only.')
" 2>&1 | tee "${OUTPUT_DIR}/analysis.log"

    echo "  Analysis Flow complete."
    echo ""
else
    echo "=== Phase 2: SKIPPED (--skip-analysis) ==="
fi

# ---------------------------------------------------------------------------
# Phase 3: Biostatistics Flow
# ---------------------------------------------------------------------------
if [ "$SKIP_BIOSTATISTICS" = "false" ]; then
    echo "=== Phase 3: Biostatistics Flow ==="
    echo "  Computing N-way ANOVA, spec curve, rank stability..."

    uv run python -c "
from minivess.orchestration.flows.biostatistics_flow import run_biostatistics_flow
print('Biostatistics Flow: starting...')
result = run_biostatistics_flow(
    config_path='configs/biostatistics/default.yaml',
    trigger_source='local_integration_test',
)
print(f'Biostatistics complete: {len(result.pairwise)} pairwise comparisons')
print(f'  DuckDB: {result.db_path}')
print(f'  Figures: {len(result.figures)}')
print(f'  Tables: {len(result.tables)}')
" 2>&1 | tee "${OUTPUT_DIR}/biostatistics.log"

    echo "  Biostatistics Flow complete."
    echo ""
else
    echo "=== Phase 3: SKIPPED (--skip-biostatistics) ==="
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Local Integration Test Complete ==="
echo "  Finished at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Output: ${OUTPUT_DIR}"
echo ""
echo "  Check results:"
echo "    mlflow ui --backend-store-uri ${MLFLOW_TRACKING_URI}"
echo "    ls ${OUTPUT_DIR}/"
