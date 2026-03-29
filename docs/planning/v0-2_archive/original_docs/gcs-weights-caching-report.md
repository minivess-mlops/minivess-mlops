# GCS Pretrained Weight Caching Strategy

**Date**: 2026-03-28
**Status**: PROPOSED (awaiting user authorization before implementation)
**Author**: Iterated LLM Council (5 expert perspectives)
**Affects**: `deployment/skypilot/train_factorial.yaml`, `knowledge-graph/domains/cloud.yaml`, `.env.example`

## Executive Summary

Every SkyPilot factorial job currently downloads pretrained model weights from
HuggingFace during the VM setup phase. SAM3 is ~9 GB, VesselFM is ~2 GB. This
takes 10-15 minutes per job, during which spot VMs are vulnerable to preemption
before any training begins. With 34 factorial conditions, total wasted download
time is 340-510 minutes per full experiment pass.

**Proposed fix**: Upload pretrained weights once to GCS in europe-west4. SkyPilot
setup pulls from same-region GCS instead of HuggingFace. Same-region GCS transfer
is free and completes in ~1-2 minutes. This eliminates the #1 cause of exit code
34 (HF download timeout) job recovery events.

---

## Iterated LLM Council Review

Five expert perspectives reviewed this proposal. Each identified concerns, and
the final strategy addresses all of them.

### Expert 1: Platform Engineer

**Focus**: Architecture correctness, interaction with existing systems.

**Key concern**: The `mlflow_only_artifact_contract` KG invariant
(`knowledge-graph/navigator.yaml` line 191) states that "MLflow artifact store is
THE ONLY persistence mechanism for checkpoints, model files, and experiment
artifacts." Does putting pretrained weights on GCS violate this?

**Resolution**: No. The contract governs **training artifacts** (outputs of this
platform). Pretrained weights are **external inputs** analogous to training data
(which is already on GCS via DVC). The semantic distinction:

| Category | Examples | Persistence Mechanism |
|----------|----------|----------------------|
| **External inputs** | Pretrained weights, training data, config files | GCS (direct or DVC) |
| **Training artifacts** | Checkpoints, metrics, figures, eval results | MLflow artifact store |
| **Experiment metadata** | Run params, tags, hyperparameters | MLflow tracking server |

Pretrained weights never change during training. They are read-only inputs, just
like the DVC-managed MiniVess and DeepVess volumes. The `mlflow_only_artifact_contract`
applies to outputs, not inputs.

**Verdict**: APPROVED. No contract violation. Pretrained weights are inputs, not artifacts.

### Expert 2: Data Engineer

**Focus**: Versioning, data lifecycle, DVC integration.

**Key concern**: Should pretrained weights be DVC-tracked like training data?

**Analysis**:

| Criterion | DVC-tracked | gsutil-managed |
|-----------|------------|----------------|
| Versioning | Automatic (content-hash) | Manual (path conventions) |
| Reproducibility | `dvc pull` gets exact version | Path must be documented |
| Integration effort | Already have DVC + GCS remote | Zero — just gsutil cp |
| Size efficiency | Dedup via content-addressable | No dedup |
| Cache invalidation | `dvc.lock` tracks hashes | Manual |
| Complexity | Adds `.dvc` files for weights | Zero repo changes |

**Recommendation**: Do NOT DVC-track pretrained weights. Rationale:
1. Pretrained weights are immutable upstream artifacts with known checksums
   (HuggingFace provides SHA256 for every file).
2. They change only when the upstream model releases a new version (rare: months/years).
3. DVC adds complexity (lock files, remote sync) for data that changes almost never.
4. Version management is simpler via GCS path conventions: `pretrained/sam3/v1/sam3.pt`.

Instead, use **path-based versioning on GCS** with SHA256 verification in setup.

**Verdict**: APPROVED with path-based versioning. No DVC.

### Expert 3: Cost Engineer

**Focus**: Storage costs, transfer costs, break-even analysis.

**GCS Storage Cost (europe-west4, Standard tier)**:
- SAM3 weights: ~9 GB x $0.023/GB/month = $0.207/month
- VesselFM weights: ~2 GB x $0.023/GB/month = $0.046/month
- **Total storage**: $0.253/month ($3.04/year)

**Transfer Cost**:
- Same-region GCS to GCP VM: **$0.00** (free)
- HuggingFace download: **$0.00** (free from HF) but **~$0.50-2.00/hour of L4 GPU
  time wasted** during the 10-15 min download

**Break-even analysis (per experiment pass)**:
```
Current HF path:
  34 jobs x 12 min average setup download = 408 min = 6.8 hours of GPU idle time
  L4 spot rate: ~$0.24/hour
  Wasted GPU cost: 6.8 x $0.24 = $1.63 per pass

GCS path:
  34 jobs x 1.5 min average GCS pull = 51 min = 0.85 hours of GPU idle time
  Wasted GPU cost: 0.85 x $0.24 = $0.20 per pass

  Savings per pass: $1.43
  Monthly storage cost: $0.25
  Break-even: first pass (saves $1.43, costs $0.25 storage)
```

**Hidden cost savings**:
- Eliminates exit code 34 retries (HF timeout). Each retry re-provisions a VM
  (~5 min) plus re-downloads. A single retry wastes ~$0.20-0.40.
- With `EAGER_NEXT_REGION` recovery, a timeout triggers a new VM in a different
  region, wasting the entire setup time of the killed VM.

**Verdict**: STRONGLY APPROVED. GCS caching pays for itself on the first experiment
pass and saves ~$1.50/pass thereafter.

### Expert 4: Reliability Engineer

**Focus**: Failure modes, fallback paths, spot preemption resilience.

**Current failure mode analysis**:
```
VM provisioned (60s) → Docker pull (2-5 min) → DVC data pull (2-3 min)
→ HF weight download (10-15 min) → Pre-flight checks (30s) → Training starts

Total setup: 15-25 min before training starts
Spot preemption probability in 20 min: ~5-15% (varies by region/time)
```

With GCS caching:
```
VM provisioned (60s) → Docker pull (2-5 min) → DVC data pull (2-3 min)
→ GCS weight pull (1-2 min) → Pre-flight checks (30s) → Training starts

Total setup: 6-11 min before training starts
Spot preemption probability in 10 min: ~2-7% (roughly halved)
```

**Failure mode: GCS weight pull fails**:
- Same-region GCS is 99.95% available (GCS SLA).
- The DVC data pull from the same region already works. If GCS is down, data pull
  fails first (exit 33), and the weight pull never runs.
- **Fallback**: If GCS pull fails, fall back to HuggingFace download. This gives
  two independent paths to weights, improving reliability over the current single path.

**Failure mode: Weights on GCS are corrupted**:
- Mitigated by SHA256 checksum verification after download.
- GCS provides built-in MD5/CRC32C integrity checks on objects.

**Failure mode: Weights on GCS are the wrong version**:
- Mitigated by path-based versioning: `pretrained/sam3/v1/sam3.pt`.
- The version path is in `.env.example` as a config variable, not hardcoded.

**Verdict**: APPROVED. GCS caching strictly improves reliability by reducing the
preemption-vulnerable setup window and adding a fallback path.

### Expert 5: DevEx Engineer

**Focus**: Researcher experience, onboarding, local development.

**Key concern**: Where do weights come from on a new researcher's machine?

**Three execution environments, three weight paths**:

| Environment | Weight Source | Config |
|-------------|-------------|--------|
| **Local (Docker Compose)** | Host filesystem via `MODEL_CACHE_HOST_PATH` volume mount | `.env` |
| **RunPod (env)** | HuggingFace direct (no GCS dependency) | `HF_TOKEN` in `.env` |
| **GCP (staging/prod)** | GCS cache, HF fallback | ADC (automatic) |

**Why NOT bake weights into the Docker image?**
- SAM3 alone is ~9 GB. Current base image is ~8-9 GB.
- Baking weights doubles the image to ~17-18 GB, making Docker pulls 2x slower.
- The Docker image is shared across ALL model families. DynUNet and SegResNet do
  NOT need SAM3 weights. Baking SAM3 into the base image wastes bandwidth for
  non-SAM3 conditions (22 of 34 factorial conditions).
- Weight versions change independently of code. Rebuilding and pushing a 17 GB
  image for a weight update is wasteful.

**Researcher onboarding flow**:
```
New researcher on GCP:
  1. gcloud auth application-default login
  2. Weights are already on GCS (uploaded once by project maintainer)
  3. sky jobs launch train_factorial.yaml → setup pulls from GCS automatically
  No HF account or token needed for GCP path!

New researcher on RunPod:
  1. Set HF_TOKEN in .env (need HuggingFace account for gated models)
  2. Weights download during setup (existing flow, unchanged)

New researcher on local Docker:
  1. First run: weights download to MODEL_CACHE_HOST_PATH via HF
  2. Subsequent runs: cached on host filesystem, no re-download
```

**The GCS cache makes GCP the easiest onboarding path**: no HF account needed,
no token management, no rate limiting, no gated model access issues. Weights are
pre-staged infrastructure.

**Verdict**: APPROVED. GCS caching improves DevEx for the primary (GCP) execution
path without degrading other paths.

---

## Final Strategy (Council Consensus)

### 1. Bucket Selection

**Use the existing `gs://minivess-mlops-checkpoints` bucket** with a `pretrained/`
prefix. Rationale:

- The checkpoints bucket is DEPRECATED for training artifacts (the
  `mlflow_only_artifact_contract` moved artifacts to MLflow). It is currently
  orphaned with no active writers.
- Repurposing it for pretrained weight caching gives it a clear, non-conflicting
  role: **read-only external model weights**.
- No new Pulumi resource needed. No new IAM bindings.
- The bucket is already in europe-west4 (same region as VMs).

**Alternative considered**: A new `gs://minivess-mlops-pretrained` bucket. Rejected
because it requires Pulumi changes, new IAM bindings, and a new `.env.example`
variable for a single-purpose bucket that the existing one can serve.

**Bucket layout**:
```
gs://minivess-mlops-checkpoints/
  pretrained/
    sam3/
      v1/
        sam3.pt            (facebook/sam3 weights, ~9 GB)
        sha256.txt         (checksum for integrity verification)
        source.txt         (HF URL + download date for provenance)
    vesselfm/
      v1/
        vesselFM_base.pt   (bwittmann/vesselFM weights, ~2 GB)
        sha256.txt
        source.txt
```

### 2. Environment Variables

Add to `.env.example`:
```bash
# ─── Pretrained Weight Cache (GCS — optional, GCP-only) ─────────────────
# GCS bucket + prefix for cached pretrained model weights.
# GCP SkyPilot jobs pull from here instead of HuggingFace (same-region = free, ~1-2 min).
# Non-GCP environments (local, RunPod) download from HuggingFace directly.
# The weights are uploaded once by the project maintainer via upload_pretrained.sh.
GCS_PRETRAINED_BUCKET=minivess-mlops-checkpoints
GCS_PRETRAINED_PREFIX=pretrained

# Pretrained weight versions (path component in GCS).
# Update when upstream models release new versions.
SAM3_WEIGHTS_VERSION=v1
VESSELFM_WEIGHTS_VERSION=v1
```

### 3. Setup Script Changes (`train_factorial.yaml`)

Replace the HuggingFace download block (lines 168-203) with a GCS-first,
HF-fallback approach:

```bash
# Pre-cache model weights during setup (don't burn GPU time on download)
# Strategy: GCS cache (same-region, free, ~1-2 min) with HF fallback.
# GCS weights uploaded by project maintainer via scripts/upload_pretrained.sh.
GCS_WEIGHTS_BASE="gs://${GCS_PRETRAINED_BUCKET:-minivess-mlops-checkpoints}/${GCS_PRETRAINED_PREFIX:-pretrained}"
HF_CACHE_DIR="${HF_HOME:-/root/.cache/huggingface}/hub"

download_weights_from_gcs() {
  local model_name="$1"
  local version="$2"
  local filename="$3"
  local gcs_path="${GCS_WEIGHTS_BASE}/${model_name}/${version}/${filename}"
  local dest_dir="${HF_CACHE_DIR}"

  mkdir -p "${dest_dir}"
  echo "Pulling ${model_name} weights from GCS: ${gcs_path}..."
  if timeout 120 gsutil -q cp "${gcs_path}" "${dest_dir}/${filename}"; then
    echo "GCS pull OK: ${filename}"
    # Verify checksum if available
    if timeout 30 gsutil -q cp "${GCS_WEIGHTS_BASE}/${model_name}/${version}/sha256.txt" /tmp/sha256_expected.txt 2>/dev/null; then
      local expected=$(cat /tmp/sha256_expected.txt | awk '{print $1}')
      local actual=$(sha256sum "${dest_dir}/${filename}" | awk '{print $1}')
      if [ "${expected}" = "${actual}" ]; then
        echo "SHA256 verified: ${filename}"
      else
        echo "WARNING: SHA256 mismatch for ${filename}. Expected: ${expected}, Got: ${actual}"
        echo "Falling back to HuggingFace download..."
        rm -f "${dest_dir}/${filename}"
        return 1
      fi
    fi
    return 0
  else
    echo "GCS pull failed. Falling back to HuggingFace..."
    return 1
  fi
}

case "${MODEL_FAMILY}" in
  sam3_vanilla|sam3_topolora|sam3_hybrid)
    echo "Pre-caching SAM3 weights..."
    sam3_version="${SAM3_WEIGHTS_VERSION:-v1}"
    if ! download_weights_from_gcs "sam3" "${sam3_version}" "sam3.pt"; then
      # Fallback: HuggingFace download (existing path)
      hf_dl_ok=false
      for attempt in 1 2 3; do
        if timeout 600 python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'sam3.pt')"; then
          hf_dl_ok=true
          break
        else
          echo "SAM3 weight download attempt ${attempt} failed" && sleep $((30 * attempt))
        fi
      done
      if [ "${hf_dl_ok}" = "false" ]; then
        echo "FATAL: SAM3 weight download failed (GCS + HF both failed)."
        exit 34
      fi
    fi
    ;;
  vesselfm)
    echo "Pre-caching VesselFM weights..."
    vfm_version="${VESSELFM_WEIGHTS_VERSION:-v1}"
    if ! download_weights_from_gcs "vesselfm" "${vfm_version}" "vesselFM_base.pt"; then
      # Fallback: HuggingFace download (existing path)
      hf_dl_ok=false
      for attempt in 1 2 3; do
        if timeout 600 python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bwittmann/vesselFM', 'vesselFM_base.pt')"; then
          hf_dl_ok=true
          break
        else
          echo "VesselFM weight download attempt ${attempt} failed" && sleep $((30 * attempt))
        fi
      done
      if [ "${hf_dl_ok}" = "false" ]; then
        echo "FATAL: VesselFM weight download failed (GCS + HF both failed)."
        exit 34
      fi
    fi
    ;;
esac
```

### 4. One-Time Upload Script

Create `scripts/upload_pretrained.sh`:

```bash
#!/usr/bin/env bash
# Upload pretrained model weights to GCS for caching.
# Run once per model version. Requires: gcloud auth, HF_TOKEN in .env.
#
# Usage:
#   bash scripts/upload_pretrained.sh sam3
#   bash scripts/upload_pretrained.sh vesselfm
#   bash scripts/upload_pretrained.sh all
set -euo pipefail

# Load config from .env
if [ -f .env ]; then
  set -a; source .env; set +a
fi

GCS_BUCKET="${GCS_PRETRAINED_BUCKET:-minivess-mlops-checkpoints}"
GCS_PREFIX="${GCS_PRETRAINED_PREFIX:-pretrained}"
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

upload_sam3() {
  local version="${SAM3_WEIGHTS_VERSION:-v1}"
  local gcs_dest="gs://${GCS_BUCKET}/${GCS_PREFIX}/sam3/${version}"

  echo "=== Uploading SAM3 weights (version: ${version}) ==="

  # Download from HuggingFace
  echo "Downloading sam3.pt from HuggingFace..."
  python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='${TMPDIR}/sam3')
print(f'Downloaded to: {path}')
"

  # Compute SHA256
  sha256sum "${TMPDIR}/sam3/sam3.pt" > "${TMPDIR}/sam3/sha256.txt"
  echo "SHA256: $(cat ${TMPDIR}/sam3/sha256.txt)"

  # Record provenance
  echo "source: https://huggingface.co/facebook/sam3/blob/main/sam3.pt" > "${TMPDIR}/sam3/source.txt"
  echo "downloaded: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${TMPDIR}/sam3/source.txt"
  echo "version: ${version}" >> "${TMPDIR}/sam3/source.txt"

  # Upload to GCS
  echo "Uploading to ${gcs_dest}..."
  gsutil -m cp "${TMPDIR}/sam3/sam3.pt" "${TMPDIR}/sam3/sha256.txt" "${TMPDIR}/sam3/source.txt" "${gcs_dest}/"

  echo "SAM3 upload complete: ${gcs_dest}"
  gsutil ls -l "${gcs_dest}/"
}

upload_vesselfm() {
  local version="${VESSELFM_WEIGHTS_VERSION:-v1}"
  local gcs_dest="gs://${GCS_BUCKET}/${GCS_PREFIX}/vesselfm/${version}"

  echo "=== Uploading VesselFM weights (version: ${version}) ==="

  # Download from HuggingFace
  echo "Downloading vesselFM_base.pt from HuggingFace..."
  python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('bwittmann/vesselFM', 'vesselFM_base.pt', local_dir='${TMPDIR}/vesselfm')
print(f'Downloaded to: {path}')
"

  # Compute SHA256
  sha256sum "${TMPDIR}/vesselfm/vesselFM_base.pt" > "${TMPDIR}/vesselfm/sha256.txt"
  echo "SHA256: $(cat ${TMPDIR}/vesselfm/sha256.txt)"

  # Record provenance
  echo "source: https://huggingface.co/bwittmann/vesselFM/blob/main/vesselFM_base.pt" > "${TMPDIR}/vesselfm/source.txt"
  echo "downloaded: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${TMPDIR}/vesselfm/source.txt"
  echo "version: ${version}" >> "${TMPDIR}/vesselfm/source.txt"

  # Upload to GCS
  echo "Uploading to ${gcs_dest}..."
  gsutil -m cp "${TMPDIR}/vesselfm/vesselFM_base.pt" "${TMPDIR}/vesselfm/sha256.txt" "${TMPDIR}/vesselfm/source.txt" "${gcs_dest}/"

  echo "VesselFM upload complete: ${gcs_dest}"
  gsutil ls -l "${gcs_dest}/"
}

case "${1:-all}" in
  sam3)     upload_sam3 ;;
  vesselfm) upload_vesselfm ;;
  all)      upload_sam3; upload_vesselfm ;;
  *)        echo "Usage: $0 {sam3|vesselfm|all}" && exit 1 ;;
esac

echo ""
echo "=== Verification ==="
echo "Run this to verify GCS contents:"
echo "  gsutil ls -lR gs://${GCS_BUCKET}/${GCS_PREFIX}/"
```

### 5. Verification Script

Create `scripts/verify_pretrained_gcs.sh`:

```bash
#!/usr/bin/env bash
# Verify pretrained weights on GCS match expected checksums.
# Run before experiment passes to ensure weights are intact.
set -euo pipefail

if [ -f .env ]; then
  set -a; source .env; set +a
fi

GCS_BUCKET="${GCS_PRETRAINED_BUCKET:-minivess-mlops-checkpoints}"
GCS_PREFIX="${GCS_PRETRAINED_PREFIX:-pretrained}"
GCS_BASE="gs://${GCS_BUCKET}/${GCS_PREFIX}"

echo "=== Pretrained Weight Verification ==="
echo "Bucket: ${GCS_BASE}"
echo ""

errors=0

verify_model() {
  local model_name="$1"
  local version="$2"
  local filename="$3"
  local gcs_path="${GCS_BASE}/${model_name}/${version}"

  echo "--- ${model_name} (${version}) ---"

  # Check file exists
  if gsutil -q stat "${gcs_path}/${filename}" 2>/dev/null; then
    local size=$(gsutil ls -l "${gcs_path}/${filename}" | awk '{print $1}' | head -1)
    echo "  File: ${filename} (${size} bytes)"
  else
    echo "  ERROR: ${gcs_path}/${filename} NOT FOUND"
    errors=$((errors + 1))
    return
  fi

  # Check checksum file
  if gsutil -q stat "${gcs_path}/sha256.txt" 2>/dev/null; then
    local checksum=$(gsutil cat "${gcs_path}/sha256.txt" | awk '{print $1}')
    echo "  SHA256: ${checksum}"
  else
    echo "  WARNING: sha256.txt not found"
  fi

  # Check provenance
  if gsutil -q stat "${gcs_path}/source.txt" 2>/dev/null; then
    echo "  Provenance:"
    gsutil cat "${gcs_path}/source.txt" | sed 's/^/    /'
  else
    echo "  WARNING: source.txt not found"
  fi
  echo ""
}

verify_model "sam3" "${SAM3_WEIGHTS_VERSION:-v1}" "sam3.pt"
verify_model "vesselfm" "${VESSELFM_WEIGHTS_VERSION:-v1}" "vesselFM_base.pt"

if [ ${errors} -gt 0 ]; then
  echo "FAILED: ${errors} model(s) missing on GCS."
  echo "Run: bash scripts/upload_pretrained.sh all"
  exit 1
else
  echo "All pretrained weights verified on GCS."
fi
```

---

## Phased Implementation Plan

### Phase 1: Upload Weights to GCS (15 minutes, user action)

**Prerequisites**: `gcloud auth application-default login`, `HF_TOKEN` set in `.env`.

```bash
# 1. Upload SAM3 weights (~9 GB, ~5-10 min depending on upload speed)
bash scripts/upload_pretrained.sh sam3

# 2. Upload VesselFM weights (~2 GB, ~1-2 min)
bash scripts/upload_pretrained.sh vesselfm

# 3. Verify uploads
bash scripts/verify_pretrained_gcs.sh

# 4. Quick manual verification
gsutil ls -lR gs://minivess-mlops-checkpoints/pretrained/
```

### Phase 2: Update SkyPilot YAML (code change, requires user authorization)

Modify `deployment/skypilot/train_factorial.yaml`:
1. Add `GCS_PRETRAINED_BUCKET` and `GCS_PRETRAINED_PREFIX` to `envs:` section.
2. Add `SAM3_WEIGHTS_VERSION` and `VESSELFM_WEIGHTS_VERSION` to `envs:` section.
3. Replace the HuggingFace download block (lines 168-203) with the GCS-first
   script from Section 3 above.

**NOTE**: Per CLAUDE.md Rule 31, the SkyPilot YAML is a declarative config.
The changes above are FUNCTIONAL (new download path), not "helpful defaults."
They require explicit user authorization.

### Phase 3: Update Config Files

1. Add `.env.example` variables (Section 2 above).
2. Update `knowledge-graph/domains/cloud.yaml` with the new decision node.
3. Update `deployment/pulumi/gcp/CLAUDE.md` to change the checkpoints bucket
   description from "DEPRECATED" to "Pretrained weight cache."

### Phase 4: Test the New Setup Path

```bash
# Dry run: verify SkyPilot YAML parses correctly
sky jobs launch --dryrun deployment/skypilot/train_factorial.yaml

# Smoke test: run a single DynUNet condition (no pretrained weights needed)
# to verify setup script still works
sky jobs launch deployment/skypilot/train_factorial.yaml \
  --env MODEL_FAMILY=dynunet --env LOSS_NAME=dice_ce --env FOLD_ID=0

# Weight test: run a single SAM3 vanilla condition to verify GCS pull
sky jobs launch deployment/skypilot/train_factorial.yaml \
  --env MODEL_FAMILY=sam3_vanilla --env LOSS_NAME=cbdice_cldice --env FOLD_ID=0
```

### Phase 5: Full Factorial Pass

Once smoke tests confirm GCS caching works, the next experiment pass will use
GCS-cached weights automatically. Monitor for:
- Exit code 34 events (should be near-zero with GCS primary + HF fallback).
- Setup time reduction (expect 6-11 min total vs 15-25 min).
- GCS transfer metrics in Cloud Console.

---

## Decision: Why NOT Bake Weights into Docker Image

| Factor | Baked-in Image | GCS Cache |
|--------|---------------|-----------|
| Image size | ~17-18 GB (+9 GB SAM3) | ~8-9 GB (unchanged) |
| Docker pull time | ~30 min (doubled) | ~15 min (unchanged) |
| Non-SAM3 conditions | 22/34 jobs download 9 GB they never use | Zero waste |
| Weight update cycle | Rebuild + push 17 GB image | `gsutil cp` 9 GB file |
| Registry storage | ~17 GB x N tags | ~9 GB (unchanged image) |
| Model-agnostic principle | Violates TOP-1: base image assumes specific models | Preserves TOP-1 |

**Verdict**: Baking weights into the image violates the model-agnostic principle
(CLAUDE.md TOP-1) and doubles image pull time for all conditions. GCS caching is
strictly superior.

## Decision: Why NOT DVC-Track Pretrained Weights

DVC is designed for data that changes per experiment (splits, augmentation params,
datasets). Pretrained weights are:
- Immutable (upstream model releases are months apart).
- Not experiment-specific (same weights for all SAM3 conditions).
- Already integrity-checked (HuggingFace provides checksums).
- Not generated by this pipeline (external upstream artifacts).

Adding DVC tracking would create `.dvc` pointer files that add repo complexity
for zero reproducibility benefit. Path-based versioning on GCS (`pretrained/sam3/v1/`)
plus SHA256 checksums provides equivalent provenance with zero repo overhead.

## Decision: Repurposing `gs://minivess-mlops-checkpoints`

The checkpoints bucket was created by Pulumi for training checkpoint persistence
via SkyPilot `file_mounts`. This was identified as a competing persistence mechanism
(see metalearning docs below) and the `file_mounts` were removed. The bucket is
currently orphaned.

Repurposing it as a pretrained weight cache:
- Gives the bucket a clear, non-conflicting purpose.
- Avoids creating new Pulumi resources.
- Does NOT violate the `mlflow_only_artifact_contract` because pretrained weights
  are inputs, not training artifacts.
- The `pretrained/` prefix clearly separates cached weights from any legacy
  checkpoint data that may still exist in the bucket.

**Rename consideration**: The bucket name `minivess-mlops-checkpoints` is slightly
misleading for a pretrained weight cache. However, GCS bucket names are globally
unique and changing them requires creating a new bucket, migrating data, and
updating all references. The cost of renaming outweighs the naming clarity benefit.
The `pretrained/` prefix provides sufficient disambiguation.

---

## Interaction with RunPod Environment

RunPod is fully standalone (no GCP dependency). The GCS caching strategy does
NOT affect the RunPod path:

| Environment | Weight Source | Change |
|-------------|-------------|--------|
| GCP (staging/prod) | GCS cache, HF fallback | NEW: GCS primary path |
| RunPod (env) | HuggingFace direct | UNCHANGED |
| Local (Docker Compose) | Host cache via `MODEL_CACHE_HOST_PATH` | UNCHANGED |

The RunPod SkyPilot YAML (`deployment/skypilot/dev_runpod.yaml`) is not modified.
It continues to download weights from HuggingFace, which is the correct standalone
behavior (no GCP assumption).

---

## Future Considerations

### Model Version Updates

When SAM3 or VesselFM releases new weights:
1. Update version in `.env.example` (e.g., `SAM3_WEIGHTS_VERSION=v2`).
2. Run `bash scripts/upload_pretrained.sh sam3` with the new version.
3. Old version remains on GCS (no deletion needed, $0.21/month/version).
4. Factorial YAML picks up new version automatically via env var.

### New Models

When adding a new model family (e.g., a future foundation model):
1. Add upload function to `scripts/upload_pretrained.sh`.
2. Add version variable to `.env.example`.
3. Add case to `train_factorial.yaml` setup block.
4. Run upload script.

### GCS Lifecycle Policy

The checkpoints bucket has a 30-day lifecycle for ARCHIVED objects (set in Pulumi).
This does NOT affect Standard objects. Pretrained weights in Standard storage class
persist indefinitely. No lifecycle change needed.

---

## Files to Create/Modify

| File | Action | What |
|------|--------|------|
| `scripts/upload_pretrained.sh` | CREATE | One-time weight upload to GCS |
| `scripts/verify_pretrained_gcs.sh` | CREATE | Pre-flight weight verification |
| `deployment/skypilot/train_factorial.yaml` | MODIFY | GCS-first weight pull with HF fallback |
| `.env.example` | MODIFY | Add GCS_PRETRAINED_BUCKET, version vars |
| `knowledge-graph/domains/cloud.yaml` | MODIFY | Add weight_caching decision node |
| `deployment/pulumi/gcp/CLAUDE.md` | MODIFY | Update checkpoints bucket description |

## Files that Reference Checkpoints Bucket (for audit)

These files reference `minivess-mlops-checkpoints` and should be reviewed when
the bucket role changes from "deprecated checkpoint store" to "pretrained weight cache":

- `CLAUDE.md` line 99 (GCS Buckets list)
- `.env.example` line 422 (`GCS_CHECKPOINT_BUCKET`)
- `deployment/pulumi/gcp/__main__.py` lines 66-81 (bucket creation)
- `deployment/pulumi/gcp/CLAUDE.md` line 22 (resource table)
- `knowledge-graph/domains/cloud.yaml` line 121 (gcs_buckets)
- `.claude/metalearning/2026-03-24-competing-checkpoint-mount-still-exists.md` line 63
- `.claude/metalearning/2026-03-21-competing-artifact-persistence-mechanisms.md` line 19

## CLAUDE.md Sections Needing Updates (DO NOT MODIFY DIRECTLY)

The following CLAUDE.md sections should be updated after user approval:

1. **Line 99**: GCS Buckets list should note the checkpoints bucket role change:
   `minivess-mlops-checkpoints` (pretrained weight cache, formerly deprecated)

2. **"What AI Must NEVER Do" section**: No change needed. GCS weight caching uses
   existing infrastructure, not a new parallel mechanism.

3. **No new rules needed**: The strategy fits within existing rules (Rule 22 for
   config, Rule 31 for YAML changes requiring authorization).

---

## References

- `deployment/skypilot/train_factorial.yaml` — current HF download (lines 168-203)
- `knowledge-graph/domains/cloud.yaml` — GCS bucket architecture
- `knowledge-graph/navigator.yaml` — `mlflow_only_artifact_contract` invariant (line 191)
- `.claude/metalearning/2026-03-21-competing-artifact-persistence-mechanisms.md`
- `.claude/metalearning/2026-03-24-competing-checkpoint-mount-still-exists.md`
- `deployment/pulumi/gcp/__main__.py` — Pulumi bucket definitions
- `.env.example` — single source of truth for all config
- `deployment/CLAUDE.md` — Docker architecture, `MODEL_CACHE_HOST_PATH`
