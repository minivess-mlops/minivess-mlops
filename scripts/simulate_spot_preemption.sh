#!/usr/bin/env bash
# Simulate GCP spot preemption and verify checkpoint resume.
#
# This script:
# 1. Launches a smoke test on GCP spot via SkyPilot
# 2. Waits for training to start (epoch_latest.yaml appears in GCS)
# 3. Simulates maintenance event (preemption)
# 4. Waits for SkyPilot to recover the instance
# 5. Verifies training resumes from the checkpoint
#
# Prerequisites:
# - GCP credentials configured (gcloud auth)
# - .env populated with MLFLOW_GCP_URI, DVC_S3_*, HF_TOKEN
# - sky check gcp passes
#
# Usage:
#   bash scripts/simulate_spot_preemption.sh [MODEL]
#   bash scripts/simulate_spot_preemption.sh sam3_vanilla

set -euo pipefail

MODEL="${1:-sam3_vanilla}"
CLUSTER_NAME="minivess-gcp-smoke"
GCS_CHECKPOINT_BUCKET="${GCS_CHECKPOINT_BUCKET:-minivess-mlops-checkpoints}"
POLL_INTERVAL=30
MAX_WAIT_EPOCHS=300  # seconds to wait for first epoch

echo "=== Spot Preemption Simulation ==="
echo "Model: ${MODEL}"
echo "Cluster: ${CLUSTER_NAME}"
echo "Checkpoint bucket: gs://${GCS_CHECKPOINT_BUCKET}"
echo ""

# Step 1: Launch training
echo "Step 1: Launching training on GCP spot..."
make smoke-test-gcp MODEL="${MODEL}" &
LAUNCH_PID=$!

# Step 2: Wait for training to start
echo "Step 2: Waiting for training to start (checking GCS for checkpoints)..."
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT_EPOCHS ]; do
    # Check if epoch_latest.yaml exists in GCS
    if gsutil ls "gs://${GCS_CHECKPOINT_BUCKET}/fold_0/epoch_latest.yaml" 2>/dev/null; then
        echo "  ✓ Training started — epoch_latest.yaml found in GCS"
        break
    fi
    echo "  Waiting... (${ELAPSED}s / ${MAX_WAIT_EPOCHS}s)"
    sleep $POLL_INTERVAL
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT_EPOCHS ]; then
    echo "  ✗ Timed out waiting for training to start"
    echo "  Check: sky logs ${CLUSTER_NAME}"
    exit 1
fi

# Read current epoch from GCS
echo "  Current state:"
gsutil cat "gs://${GCS_CHECKPOINT_BUCKET}/fold_0/epoch_latest.yaml" 2>/dev/null || echo "  (could not read state)"

# Step 3: Simulate preemption
echo ""
echo "Step 3: Simulating spot preemption..."
INSTANCE=$(sky status ${CLUSTER_NAME} --format json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0].get('handle',{}).get('head_instance_id',''))" 2>/dev/null || echo "")
ZONE=$(sky status ${CLUSTER_NAME} --format json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0].get('handle',{}).get('zone',''))" 2>/dev/null || echo "")

if [ -z "$INSTANCE" ] || [ -z "$ZONE" ]; then
    echo "  ✗ Could not find instance/zone from SkyPilot"
    echo "  Manual preemption: gcloud compute instances simulate-maintenance-event INSTANCE --zone ZONE"
    exit 1
fi

echo "  Instance: ${INSTANCE} (zone: ${ZONE})"
echo "  Sending maintenance event..."
gcloud compute instances simulate-maintenance-event "${INSTANCE}" --zone "${ZONE}"
echo "  ✓ Preemption simulated"

# Step 4: Wait for recovery
echo ""
echo "Step 4: Waiting for SkyPilot to recover..."
sleep 60  # Give SkyPilot time to detect and recover

# Step 5: Verify resume
echo ""
echo "Step 5: Verifying training resumed..."
sleep 120  # Wait for resumed training to write new checkpoint

echo "  Post-recovery state:"
gsutil cat "gs://${GCS_CHECKPOINT_BUCKET}/fold_0/epoch_latest.yaml" 2>/dev/null || echo "  (could not read state)"

echo ""
echo "=== Spot Preemption Simulation Complete ==="
echo "Check MLflow for training run continuity."
echo "Monitor: sky logs ${CLUSTER_NAME}"
