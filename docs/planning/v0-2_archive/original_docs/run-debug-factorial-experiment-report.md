# Debug Factorial Experiment — Comprehensive Glitch Report

**Branch**: `test/debug-factorial-run`
**Date**: 2026-03-20
**Total wall-clock time**: ~12 hours (00:00–12:00 UTC)
**Total GCP cost**: ~$2-3 (spot pricing, L4 GPUs)

---

## Project Context (for cold-start LLM)

**MinIVess MLOps** is a model-agnostic biomedical segmentation MLOps platform
extending the MONAI ecosystem. It runs a factorial experiment comparing 6 models
across 3 loss functions and 2 calibration settings on multiphoton brain vessel data.

### Key Architecture

```
Local Machine → run_factorial.sh → SkyPilot → GCP L4 Spot VMs → Docker containers
                                                    ↓
                                            train_flow.py (Prefect)
                                                    ↓
                                            MLflow (Cloud Run) + DVC (GCS)
```

- **Package manager**: `uv` (ONLY — no pip/conda)
- **Cloud**: GCP (staging/prod) + RunPod (dev). Two providers, non-negotiable.
- **Docker**: `europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`
- **Data**: DVC-tracked on GCS (`gs://minivess-mlops-dvc-data`), 70 training volumes
- **MLflow**: Cloud Run (`minivess-mlflow` service), PostgreSQL backend (Cloud SQL)
- **Artifacts**: GCS bucket (`gs://minivess-mlops-mlflow-artifacts`)
- **Checkpoints**: GCS bucket (`gs://minivess-mlops-checkpoints`)
- **GPU**: NVIDIA L4 (Ada Lovelace, BF16, 24 GB). T4 is BANNED (no BF16).

### Experiment Design

4 models × 3 losses × 2 aux_calib = **24 trainable conditions** on fold-0, 2 epochs, half data.
Plus 2 zero-shot baselines = **26 total conditions**.

| Factor | Values |
|--------|--------|
| Models | dynunet, mambavesselnet, sam3_topolora, sam3_hybrid |
| Losses | cbdice_cldice, dice_ce, dice_ce_cldice |
| Aux calibration | true, false |
| Zero-shot | sam3_vanilla (MiniVess), vesselfm (DeepVess) |

### Two-Layer Execution Architecture

```
Layer 1 (Deterministic): scripts/run_factorial.sh
  - Pure sky jobs launch calls in a loop
  - ANY researcher can run this WITHOUT Claude Code

Layer 2 (Monitoring): Claude Code with /factorial-monitor skill
  - Monitors SkyPilot job status, diagnoses failures
  - After training: triggers downstream flows
```

---

## Final Results

| Model | Conditions | SUCCEEDED | FAILED | Job Duration | Root Cause |
|-------|-----------|-----------|--------|-------------|------------|
| dynunet | 6 | **6** | 0 | ~5.5 min | Training works |
| sam3_hybrid | 6 | **5** | 0+1 stuck | ~9.3 min | Training works (1 stuck in spot queue) |
| sam3_topolora | 6 | 0 | **6** | ~2 min (crash) | Glitch #9: LoRA on Conv2d bug |
| mambavesselnet | 6 | 0 | **6** | ~1.9 min (crash) | Glitch #10: mamba-ssm not compiled |
| sam3_vanilla (zero-shot) | 1 | 0 | **1** | ~1.9 min | Glitch #12: max_epochs=0 validation |
| vesselfm (zero-shot) | 1 | 0 | **1** | ~1.9 min | Glitch #12: max_epochs=0 validation |
| **TOTAL** | **26** | **11** | **14+1** | — | — |

**CRITICAL**: All 11 SUCCEEDED jobs have **LOST checkpoints** due to Glitch #8 (P0).
Downstream flows (post-training, analysis, biostatistics, deploy) are ALL BLOCKED.

---

## Phase Timing

| Phase | Start (UTC) | End (UTC) | Duration | Notes |
|-------|-------------|-----------|----------|-------|
| 0: Pre-flight | 00:00 | 00:38 | 38 min | 8 checks, 2 blockers fixed |
| 1: Launch (probes) | 00:38 | 04:08 | 3h 30m | 8 probe iterations, 5 blockers fixed |
| 1: Launch (final) | 04:09 | 04:19 | 10 min | 26 conditions launched with --detach-run |
| 2: Monitoring | 04:19 | 12:00 | ~7h 40m | GCP spot scheduling slow (~60 min/batch) |
| 3-6: Downstream | — | — | — | **BLOCKED by Glitch #8 (P0)** |
| 7: Report | 12:00 | 12:30 | 30 min | This document |

---

## Glitch #1: SkyPilot missing `[gcp]` extra — FIXED

**Phase**: 0 (Pre-flight) | **Severity**: BLOCKER | **Status**: FIXED

### Error

```
$ uv run sky check gcp
ModuleNotFoundError: No module named 'googleapiclient'
```

### Root Cause

`pyproject.toml` line 82 had:
```toml
"skypilot-nightly[runpod]>=1.0.0.dev0",
```
Missing `[gcp]` extra. GCP is one of two required providers (CLAUDE.md).

### Fix Applied

**File**: `pyproject.toml` (lines 82 and 153)
```toml
# BEFORE (both locations):
"skypilot-nightly[runpod]>=1.0.0.dev0",

# AFTER:
"skypilot-nightly[gcp,runpod]>=1.0.0.dev0",
```

### Commands
```bash
uv add "skypilot-nightly[runpod,gcp]>=1.0.0.dev0"
uv run sky api stop; uv run sky api start
uv run sky check gcp  # → enabled [compute, storage]
```

---

## Glitch #2: pyparsing too old for httplib2 — FIXED

**Phase**: 0 (Pre-flight) | **Severity**: BLOCKER | **Status**: FIXED

### Error

```
AttributeError: module 'pyparsing' has no attribute 'DelimitedList'
```

Full chain: `googleapiclient` → `httplib2` → `pyparsing.DelimitedList` (added in 3.1.0)

### Root Cause

`uv.lock` pinned `pyparsing==3.0.9` because `[gcp]` extra was never resolved before.
`pyparsing` 3.0.9 only has `delimitedList` (camelCase), not `DelimitedList` (PascalCase).

### Fix Applied

**File**: `pyproject.toml` line 83
```toml
"pyparsing>=3.1.0",
```

Resolved to `pyparsing==3.3.2`.

---

## Glitch #3: `sky` binary not on system PATH — FIXED

**Phase**: 1 (Launch) | **Severity**: BLOCKER | **Status**: FIXED

### Error

```
./scripts/run_factorial.sh: line 171: sky: command not found
```

### Root Cause

SkyPilot installed in `.venv/bin/sky` (uv-managed), not system `/usr/bin/`.

### Fix Applied

**File**: `scripts/run_factorial.sh` (lines 34-47)
```bash
# ─── Resolve sky binary ─────────────────────────────────────────────────────
if command -v sky &>/dev/null; then
    SKY_CMD="sky"
elif [ -x "${REPO_ROOT}/.venv/bin/sky" ]; then
    SKY_CMD="${REPO_ROOT}/.venv/bin/sky"
else
    echo "ERROR: 'sky' not found in PATH or .venv/bin/sky"
    echo "Install SkyPilot: uv sync --all-extras"
    exit 1
fi
echo "Using sky: ${SKY_CMD}"
```

All `sky jobs launch` calls changed to `"${SKY_CMD}" jobs launch`.
Zero-shot Python subprocess also uses `sky_cmd = '${SKY_CMD}'`.

---

## Glitch #4: L4 GPU not available in europe-north1 — FIXED

**Phase**: 1 (Launch) | **Severity**: BLOCKER | **Status**: FIXED

### Error

```
sky.exceptions.ResourcesUnavailableError: Catalog does not contain any instances
satisfying the request: 1x GCP([Spot], {'L4': 1}, ... region=europe-north1).
```

SkyPilot status: `FAILED_PRECHECKS` for all 24 conditions.

### Root Cause

`train_factorial.yaml` had `region: europe-north1`. GCP L4 GPUs exist ONLY in:
`europe-west1, west2, west3, west4, west6`. Verified:
```bash
gcloud compute accelerator-types list --filter="name=nvidia-l4" | grep europe
```

### Fix Applied

**File**: `deployment/skypilot/train_factorial.yaml` (line 37)
```yaml
# BEFORE:
  region: europe-north1

# AFTER (removed — let SkyPilot auto-select):
  # L4 not available in europe-north1. Prefer Europe-west zones (closest to
  # europe-north1 GAR and GCS). SkyPilot auto-selects cheapest available zone.
```

GAR image (`europe-north1-docker.pkg.dev/...`) is pullable from any GCP region.

**Test updated**: `tests/v2/unit/test_train_factorial_yaml.py`
```python
# BEFORE:
def test_targets_europe_north1(self) -> None:
    assert resources.get("region") == "europe-north1"

# AFTER:
def test_no_hardcoded_region(self) -> None:
    assert "region" not in resources
```

---

## Glitch #5: Docker image missing git + DVC config files — FIXED

**Phase**: 1 (Launch) | **Severity**: BLOCKER | **Status**: FIXED

### Error Sequence (3 iterations)

**Probe 1**: `FAILED_SETUP`
```
ERROR: you are not inside of a DVC repository (checked up to mount point '/')
```
→ Docker image has no `.dvc/` directory.

**Probe 2**: `FAILED_SETUP` (after adding `file_mounts`)
```
bash: git: command not found
```
→ Docker image has no `git` binary.

**Probe 3**: `FAILED_SETUP` (after `dvc init --no-scm`)
```
ERROR: /app is not a git repository
```
→ DVC `pull` requires git even in `--no-scm` mode.

### Root Cause

**File**: `deployment/docker/Dockerfile.base` runner stage only copies:
```dockerfile
COPY --from=builder ... /app/.venv /app/.venv
COPY --from=builder ... /app/src /app/src
COPY --from=builder ... /app/configs /app/configs
```
Missing: `git` binary, `.dvc/config`, `dvc.yaml`, `dvc.lock`.

### Fix Applied

**File**: `deployment/docker/Dockerfile.base` (lines 124, 148-153)

1. Added `git` to apt-get:
```dockerfile
    curl \
    git \       # ← ADDED
    patch \
    sudo \
```

2. Added DVC files:
```dockerfile
COPY --chown=minivess:minivess .dvc/config /app/.dvc/config
COPY --chown=minivess:minivess dvc.yaml /app/dvc.yaml
COPY --chown=minivess:minivess dvc.lock /app/dvc.lock
```

**Docker rebuild + push**:
```bash
DOCKER_BUILDKIT=1 docker build -t minivess-base:latest -f deployment/docker/Dockerfile.base .
docker tag minivess-base:latest europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
docker push europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
```

---

## Glitch #6: DVC partial pull + cp same-file error — FIXED

**Phase**: 1 (Launch) | **Severity**: MEDIUM | **Status**: FIXED

### Error

```
(setup) dvc pull -r gcs
A       data/raw/minivess/
351 files fetched and 350 files added
ERROR: failed to pull data from the cloud - Checkout failed for following targets:
data/processed/minivess
```
Then:
```
cp: 'configs/splits/3fold_seed42.json' and 'configs/splits/splits.json' are the same file
```

### Root Cause

(a) `dvc pull -r gcs` tries ALL pipeline stages including `preprocess` (not on GCS).
(b) Docker image's `configs/splits/` already has both files pointing to same inode.

### Fix Applied

**File**: `deployment/skypilot/train_factorial.yaml` setup section

```bash
# BEFORE:
dvc pull -r gcs || { echo "FATAL..."; exit 1; }
cp configs/splits/3fold_seed42.json configs/splits/splits.json

# AFTER:
dvc pull data/raw/minivess/ -r gcs || {
  echo "WARNING: DVC pull returned non-zero, checking if data arrived..."
}
# Verify data was actually pulled
if [ ! -d "data/raw/minivess/imagesTr" ] || [ "$(ls data/raw/minivess/imagesTr 2>/dev/null | wc -l)" -lt 10 ]; then
  echo "FATAL: Training data not found after DVC pull."
  exit 1
fi

# Use canonical 3-fold splits (may already be same file in Docker image)
if [ "$(readlink -f configs/splits/3fold_seed42.json)" != "$(readlink -f configs/splits/splits.json)" ]; then
  cp configs/splits/3fold_seed42.json configs/splits/splits.json
fi
```

---

## Glitch #7: train_flow.py missing 6 factorial CLI arguments — FIXED

**Phase**: 1 (Launch) | **Severity**: BLOCKER | **Status**: FIXED

### Error

```
train_flow.py: error: unrecognized arguments: --fold 0 --with-aux-calib false
  --max-train-volumes 23 --max-val-volumes 12 --zero-shot false --eval-dataset minivess
```

### Root Cause

`train_flow.py` argparse had 7 arguments. SkyPilot YAML passed 10.

**Missing**: `--fold`, `--with-aux-calib`, `--max-train-volumes`, `--max-val-volumes`,
`--zero-shot`, `--eval-dataset`

**Name mismatches**: `--model` vs `--model-family`, `--loss` vs `--loss-name`,
`--experiment` vs `--experiment-name`

### Fix Applied

**File**: `src/minivess/orchestration/flows/train_flow.py`

1. Added 6 new argparse arguments (lines 1105-1133):
```python
parser.add_argument(
    "--fold", type=int, default=int(os.environ.get("FOLD_ID", "-1")),
    help="Specific fold ID to run (-1 = all folds via num_folds)",
)
parser.add_argument(
    "--with-aux-calib",
    default=os.environ.get("WITH_AUX_CALIB", "false"),
    help="Enable auxiliary calibration loss (true/false)",
)
parser.add_argument(
    "--max-train-volumes", type=int,
    default=int(os.environ.get("MAX_TRAIN_VOLUMES", "0")),
)
parser.add_argument(
    "--max-val-volumes", type=int,
    default=int(os.environ.get("MAX_VAL_VOLUMES", "0")),
)
parser.add_argument(
    "--zero-shot",
    default=os.environ.get("ZERO_SHOT", "false"),
)
parser.add_argument(
    "--eval-dataset",
    default=os.environ.get("EVAL_DATASET", "minivess"),
)
```

2. Added new params to `training_flow()` signature (line 716):
```python
def training_flow(
    *,
    # ... existing params ...
    # Factorial experiment parameters
    fold_id: int | None = None,
    with_aux_calib: bool = False,
    max_train_volumes: int = 0,
    max_val_volumes: int = 0,
    zero_shot: bool = False,
    eval_dataset: str = "minivess",
    **kwargs: Any,
) -> TrainingFlowResult:
```

3. Fixed SkyPilot YAML arg names:
```yaml
# BEFORE:
  --model "${MODEL_FAMILY}" \
  --loss "${LOSS_NAME}" \
  --experiment "${EXPERIMENT_NAME}" \

# AFTER:
  --model-family "${MODEL_FAMILY}" \
  --loss-name "${LOSS_NAME}" \
  --experiment-name "${EXPERIMENT_NAME}" \
```

---

## Glitch #8: CRITICAL P0 — MLflow checkpoint upload 413 on Cloud Run

**Phase**: 2 (Training) | **Severity**: P0-CRITICAL | **Status**: OPEN
**GitHub Issue**: [#878](https://github.com/petteriTeikari/minivess-mlops/issues/878)

### Error

```
requests.exceptions.HTTPError: 413 Client Error: Request Entity Too Large
for url: https://minivess-mlflow-a7w6hliydq-lz.a.run.app/api/2.0/mlflow-artifacts/
  artifacts/10/.../checkpoints/best_val_loss.pth
```

### Full Stack Trace

```python
File "trainer.py", line 842, in fit
    self.tracker.log_artifact(best_path, artifact_path="checkpoints")
File "tracking.py", line 287, in log_artifact
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path or None)
File "mlflow/tracking/fluent.py", line 1533, in log_artifact
    MlflowClient().log_artifact(run_id, local_path, artifact_path)
File "mlflow/tracking/client.py", line 2767, in log_artifact
    self._tracking_client.log_artifact(run_id, local_path, artifact_path)
File "mlflow/tracking/_tracking_service/client.py", line 678, in log_artifact
    artifact_repo.log_artifact(local_path, artifact_path)
File "mlflow/store/artifact/http_artifact_repo.py", line 66, in log_artifact
    augmented_raise_for_status(resp)
File "mlflow/utils/request_utils.py", line 66, in augmented_raise_for_status
    raise HTTPError(...)
requests.exceptions.HTTPError: 413 Client Error: Request Entity Too Large
```

### Root Cause

Google Cloud Run has a **hard, non-configurable 32MB HTTP request body limit**.
MLflow's `HttpArtifactRepository.log_artifact()` uploads the entire file in a
single HTTP POST. When the MLflow server runs on Cloud Run with artifact proxying,
any file > 32MB triggers a 413.

**Current broken workaround**: `MLFLOW_TRACKING_URI: /app/mlruns` (file-based on
ephemeral spot VM). Checkpoints are saved locally but **LOST when the VM terminates**.

### Why This Is P0

ALL downstream flows depend on MLflow artifacts:

```
training_flow → checkpoints → post_training_flow (SWA, calibration)
                            → analysis_flow (evaluation, comparison)
                            → biostatistics_flow (ANOVA, specification curve)
                            → deploy_flow (ONNX export, BentoML)
```

**File**: `src/minivess/orchestration/flow_contract.py` — `find_fold_checkpoints()`:
```python
def find_fold_checkpoints(self, *, parent_run_id: str) -> list[dict[str, Any]]:
    """Return fold checkpoint info from a parent training run's MLflow tags.
    Reads tags of the form ``checkpoint_dir_fold_N`` from *parent_run_id*."""
    client = MlflowClient(tracking_uri=self.tracking_uri)
    run_data = client.get_run(parent_run_id)
    tags = run_data.data.tags
    prefix = "checkpoint_dir_fold_"
    # ... extracts fold_id and checkpoint_dir Path from tags
```

Without checkpoints in MLflow/GCS, this returns empty → all downstream flows skip.

### Checkpoint Save Code

**File**: `src/minivess/pipeline/trainer.py` (lines 839-852)
```python
if self.tracker is not None:
    try:
        self.tracker.log_artifact(
            best_path, artifact_path="checkpoints"
        )
    except Exception:
        logger.warning(
            "Failed to upload checkpoint artifact to MLflow "
            "(checkpoint saved locally at %s)",
            best_path,
            exc_info=True,
        )
```

The `except Exception` catches the 413 and logs a warning, but the checkpoint is only
saved locally at `/app/checkpoints/fold_0/best_val_loss.pth` — on the ephemeral VM.

### MLflow Artifact Upload Code

**File**: `src/minivess/observability/tracking.py` (line 282)
```python
def log_artifact(self, local_path: Path, *, artifact_path: str = "") -> None:
    """Log a file as an MLflow artifact."""
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path or None)
```

This calls MLflow's HTTP artifact repo which does a single POST to Cloud Run.

### Decision Matrix (7 Hypotheses)

| # | Hypothesis | Description | Complexity | Permanent? | Verdict |
|---|-----------|-------------|-----------|-----------|---------|
| **H1** | `--no-serve-artifacts` + direct GCS | Client uploads directly to GCS bucket, bypassing Cloud Run proxy | **LOW** (15 min) | YES | **WINNER — 90% done** |
| H2 | Proxy multipart upload | Use `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD` | N/A | NO | **GCS lacks MultipartUploadMixin** (confirmed MLflow #18275) |
| H3 | GCS signed URL upload | Client gets signed URL, uploads to GCS directly | HIGH (8-12h) | YES | Overengineered given H1 |
| H4 | Dedicated artifact server (GCE/GKE) | Deploy MLflow artifact proxy without 32MB limit | HIGH (12h) | YES | Defeats serverless benefit |
| H5 | Cloud Run HTTP/2 gRPC | Use gRPC streaming for large uploads | VERY HIGH | UNCERTAIN | MLflow doesn't use gRPC |
| **H6** | GCS bucket mount for checkpoints | Mount `gs://minivess-mlops-checkpoints` via SkyPilot `file_mounts` | **LOW-MED** (2-4h) | YES | **Best complement to H1** |
| H7 | Post-training sync script | Sync file-based mlruns to Cloud Run after training | MEDIUM (4-6h) | NO | Current broken approach |

### Why H1 Is 90% Done

**Already implemented components:**

1. **MLflow Cloud Run Dockerfile** (`deployment/docker/Dockerfile.mlflow-gcp`):
```dockerfile
ENTRYPOINT ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--no-serve-artifacts"]    # ← ALREADY bypasses proxy
```

2. **Pulumi Cloud Run deployment** (`deployment/pulumi/gcp/__main__.py` line 247):
```python
{
    "name": "MLFLOW_ARTIFACTS_DESTINATION",
    "value": mlflow_artifacts_bucket.name.apply(lambda n: f"gs://{n}"),
},
```

3. **Smoke test** (`deployment/skypilot/smoke_test_gcp.yaml` line 64):
```yaml
MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}    # ← Points to Cloud Run URL
# MLflow server uses --no-serve-artifacts: client uploads directly to GCS.
```

4. **GCS checkpoint mount** (`deployment/skypilot/smoke_test_gcp.yaml` line 44):
```yaml
file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT
```

**Only remaining work**: Change `train_factorial.yaml` line 57:
```yaml
# CURRENT (broken — file-based on ephemeral VM):
MLFLOW_TRACKING_URI: /app/mlruns

# FIX (direct GCS upload via Cloud Run tracking + --no-serve-artifacts):
MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}
```

And add `file_mounts` for checkpoint persistence:
```yaml
file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT
```

And remove `--env MLFLOW_TRACKING_URI=/app/mlruns` from `run_factorial.sh` (line 193).

### References

- [Cloud Run 32MB limit workaround](https://dev.to/stack-labs/how-to-overcome-cloud-runs-32mb-request-limit-190j)
- [MLflow GCP deployment docs](https://mlflow.org/docs/latest/self-hosting/deploy-to-cloud/gcp/)
- [MLflow #10332](https://github.com/mlflow/mlflow/issues/10332) — 413 on Cloud Run
- [MLflow #18275](https://github.com/mlflow/mlflow/issues/18275) — multipart NOT for GCS
- [aai-institute/gcp-mlflow-cloud-run](https://github.com/aai-institute/gcp-mlflow-cloud-run)
- [MLOps Community: MLflow with Pulumi](https://home.mlops.community/public/blogs/mlflow-on-aws-with-pulumi-a-step-by-step-guide)
- Internal: `.claude/metalearning/2026-03-15-gcp-artifact-upload-success.md`
- Internal: `docs/planning/docker-pull-runpod-provisioning-mlflow-cloud-run-multipart-upload-report.md`

---

## Glitch #9: sam3_topolora — LoRA applied to Conv2d — OPEN

**Phase**: 2 (Training) | **Severity**: HIGH | **Status**: OPEN
**Affected**: 6/24 trainable conditions (all sam3_topolora variants)

### Error

```
TypeError: LoRALinear only supports nn.Linear, got Conv2d.
SAM3 ViT-32L uses only Linear layers in its transformer blocks.
```

### Full Stack Trace

```python
File "sam3_topolora.py", line 172, in __init__
    lora_targets = _apply_lora_to_encoder(
        self.backbone.encoder, rank=config.lora_rank, ...)
File "sam3_topolora.py", line 137, in _apply_lora_to_encoder
    lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
File "sam3_topolora.py", line 68, in __init__
    raise TypeError(msg)
```

### Root Cause

**File**: `src/minivess/adapters/sam3_topolora.py` (lines 113-147)

`_apply_lora_to_encoder()` iterates over ALL encoder modules and wraps both
`nn.Linear` AND `nn.Conv2d` layers:

```python
for name, module in list(encoder.named_modules()):
    # Target Conv2d and Linear layers (excluding tiny layers)
    if isinstance(module, nn.Linear | nn.Conv2d):    # ← BUG: includes Conv2d
        # ...
        lora_layer = LoRALinear(module, ...)    # ← LoRALinear rejects Conv2d
```

But `LoRALinear.__init__()` (line 68) only accepts `nn.Linear`:
```python
if not isinstance(original, nn.Linear):
    raise TypeError(f"LoRALinear only supports nn.Linear, got {type(original).__name__}.")
```

The docstring for `_apply_lora_to_encoder` says "Apply LoRA adapters to Conv2d layers"
which contradicts the `LoRALinear` constraint. The function was written for a stub encoder
that had Conv2d, but real SAM3 ViT-32L has different layer types.

### Fix Strategy

**Option A** (simple): Skip Conv2d in the loop:
```python
# In _apply_lora_to_encoder():
if isinstance(module, nn.Linear):    # ← Remove nn.Conv2d
```

**Option B** (complete): Implement `LoRAConv2d` for Conv2d layers:
```python
class LoRAConv2d(nn.Module):
    """LoRA for Conv2d via 1x1 convolution decomposition."""
    def __init__(self, original: nn.Conv2d, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.original = original
        self.lora_A = nn.Conv2d(original.in_channels, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, original.out_channels, 1, bias=False)
        # ...
```

### Files to Change

- `src/minivess/adapters/sam3_topolora.py` — `_apply_lora_to_encoder()` and optionally `LoRALinear`
- Docker image needs rebuild after fix

---

## Glitch #10: mambavesselnet — mamba-ssm not in Docker image — OPEN

**Phase**: 2 (Training) | **Severity**: HIGH | **Status**: OPEN
**Affected**: 6/24 trainable conditions (all mambavesselnet variants)

### Error

```
RuntimeError: mamba-ssm not installed.
See installation instructions logged above (ERROR level).
```

### Root Cause

**File**: `src/minivess/adapters/model_builder.py` (lines 185-200)
```python
def _require_mamba() -> None:
    if not _mamba_available():
        logger.error(_MAMBA_INSTALL_INSTRUCTIONS)
        msg = "mamba-ssm not installed. See installation instructions logged above."
        raise RuntimeError(msg)
```

`mamba-ssm==2.3.1` requires CUDA compilation (`nvcc`). The Dockerfile.base has a
conditional compilation gate:

**File**: `deployment/docker/Dockerfile.base` (lines 74-90)
```dockerfile
# Phase C (optional): compile mamba-ssm CUDA extensions.
# Gate: ARG INSTALL_MAMBA=0 keeps default build time unchanged
ARG INSTALL_MAMBA=0
RUN if [ "$INSTALL_MAMBA" = "1" ]; then \
    uv sync --frozen --no-dev --extra mamba-ssm; \
fi
```

The current GAR image was built with `INSTALL_MAMBA=0` (default).

### Fix

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg INSTALL_MAMBA=1 \
  -t minivess-base:latest \
  -f deployment/docker/Dockerfile.base .

docker tag minivess-base:latest europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
docker push europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
```

**Note**: This adds ~10-15 min to Docker build time (CUDA compilation) and ~200MB to image size.

---

## Glitch #11: `--detach-run` missing from `sky jobs launch` — FIXED

**Phase**: 1 (Launch) | **Severity**: BLOCKER | **Status**: FIXED

### Symptom

`run_factorial.sh` streamed logs for each job and waited for completion before
launching the next. With 24 conditions at ~15 min each, sequential launch would
take 6+ hours.

### Root Cause

`sky jobs launch` defaults to streaming logs and waiting. Need `--detach-run` flag.

### Fix Applied

**File**: `scripts/run_factorial.sh` (line 196 for trainable, line 281 for zero-shot)

```bash
# Added to both launch commands:
--detach-run \
```

Also added `--env MLFLOW_TRACKING_URI=/app/mlruns` to override `.env` file value
(see Glitch #8 for why this is itself a workaround that needs replacing).

---

## Glitch #12: Zero-shot max_epochs=0 Pydantic validation — OPEN

**Phase**: 2 (Training) | **Severity**: HIGH | **Status**: OPEN
**Affected**: 2/26 conditions (both zero-shot baselines)

### Error

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for TrainingConfig
max_epochs
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0]
```

### Root Cause

**File**: `src/minivess/config/models.py` (line 227)
```python
class TrainingConfig(BaseModel):
    max_epochs: int = Field(default=100, ge=1)    # ← ge=1 rejects 0
```

Zero-shot baselines set `MAX_EPOCHS=0` (no training), but TrainingConfig requires ≥1.

### Fix Strategy

**Option A**: Allow `ge=0` and handle zero-shot at the flow level:
```python
max_epochs: int = Field(default=100, ge=0)
```
Then in `training_flow()`:
```python
if zero_shot and max_epochs == 0:
    # Skip training loop, go directly to evaluation
    ...
```

**Option B**: Create a separate `ZeroShotConfig` that doesn't require `max_epochs`:
```python
class ZeroShotConfig(BaseModel):
    model_family: str
    eval_dataset: str = "minivess"
    # No max_epochs field
```

### Files to Change

- `src/minivess/config/models.py` — `TrainingConfig.max_epochs` constraint
- `src/minivess/orchestration/flows/train_flow.py` — zero-shot flow path

---

## Summary Table

| # | Phase | Severity | Glitch | Root Cause | Fix Type | Status |
|---|-------|----------|--------|------------|----------|--------|
| 1 | 0 | BLOCKER | Missing `[gcp]` extra | pyproject.toml incomplete | Permanent | FIXED |
| 2 | 0 | BLOCKER | pyparsing too old | Lock file pinned 3.0.9 | Permanent | FIXED |
| 3 | 1 | BLOCKER | `sky` not on PATH | Script used bare `sky` | Permanent | FIXED |
| 4 | 1 | BLOCKER | L4 not in europe-north1 | Hardcoded region | Permanent | FIXED |
| 5 | 1 | BLOCKER | Docker no git/DVC | Dockerfile.base incomplete | Permanent | FIXED |
| 6 | 1 | MEDIUM | DVC partial pull + cp | Pull all stages, same inode | Permanent | FIXED |
| 7 | 1 | BLOCKER | train_flow.py 6 missing args | Argparse incomplete | Permanent | FIXED |
| **8** | **2** | **P0** | **MLflow 413 checkpoints** | **Cloud Run 32MB limit** | **H1+H6** | **OPEN [#878](https://github.com/petteriTeikari/minivess-mlops/issues/878)** |
| 9 | 2 | HIGH | LoRA on Conv2d | _apply_lora includes Conv2d | Code fix | OPEN |
| 10 | 2 | HIGH | mamba-ssm missing | INSTALL_MAMBA=0 default | Docker rebuild | OPEN |
| 11 | 1 | BLOCKER | --detach-run missing | Sequential launch | Permanent | FIXED |
| 12 | 2 | HIGH | max_epochs=0 validation | Pydantic ge=1 constraint | Code fix | OPEN |

---

## Lessons for Production Run

### MUST FIX BEFORE PRODUCTION (blockers)

1. **P0 [#878]: MLflow checkpoint persistence** — Implement H1 + H6. 15 min + 2-4 hrs.
2. **sam3_topolora LoRA bug** — Skip Conv2d in `_apply_lora_to_encoder()`. 30 min.
3. **mamba-ssm Docker compilation** — Rebuild with `INSTALL_MAMBA=1`. 20 min.
4. **Zero-shot max_epochs=0** — Allow `ge=0` + handle zero-shot path. 1 hr.

### SHOULD FIX (quality of life)

5. **GCP spot scheduling** — 26 conditions took ~8 hours. Consider quota increase
   or multi-region scheduling.

### VALIDATED (no changes needed for production)

- `run_factorial.sh` with `--detach-run` launches 26 conditions in ~10 min
- DynUNet trains in ~5.5 min on L4 (VRAM: 2.67 GB)
- SAM3 Hybrid trains in ~9.3 min on L4
- SkyPilot spot recovery works (Job 28 recovered from preemption)
- DVC pull from GCS works (350 files, 70 training volumes)
- Prefect flow orchestration works inside Docker on cloud VM
- `MINIVESS_ALLOW_HOST=1` + `PREFECT_DISABLED=1` escape hatches work

### Key File Paths (for LLM context)

| File | Purpose |
|------|---------|
| `deployment/skypilot/train_factorial.yaml` | SkyPilot per-condition YAML (NEEDS: MLFLOW_TRACKING_URI fix, file_mounts) |
| `deployment/skypilot/smoke_test_gcp.yaml` | REFERENCE: working GCS artifact pattern |
| `deployment/docker/Dockerfile.base` | Docker base image (NEEDS: INSTALL_MAMBA=1 for mamba) |
| `deployment/docker/Dockerfile.mlflow-gcp` | MLflow Cloud Run image (already has --no-serve-artifacts) |
| `deployment/pulumi/gcp/__main__.py` | GCP IaC (Cloud Run MLflow with MLFLOW_ARTIFACTS_DESTINATION) |
| `scripts/run_factorial.sh` | Deterministic factorial launch script |
| `configs/experiment/debug_factorial.yaml` | Debug experiment config (4×3×2 + 2 zero-shot) |
| `src/minivess/orchestration/flows/train_flow.py` | Training Prefect flow (argparse + training_flow()) |
| `src/minivess/adapters/sam3_topolora.py` | SAM3 TopoLoRA adapter (LoRA bug) |
| `src/minivess/adapters/model_builder.py` | Model builder (_require_mamba check) |
| `src/minivess/config/models.py` | Pydantic config (TrainingConfig.max_epochs ge=1) |
| `src/minivess/pipeline/trainer.py` | Trainer (checkpoint save + MLflow upload) |
| `src/minivess/observability/tracking.py` | MLflow tracking wrapper (log_artifact) |
| `src/minivess/orchestration/flow_contract.py` | Inter-flow contract (find_fold_checkpoints) |
| `.dvc/config` | DVC remotes (minio local, gcs production) |
