# Prefect/Docker Optimization & MONAI Consolidation Plan

**Date:** 2026-03-08
**Branch structure:** 2 focused branches (see §3)
**Overnight script:** `docs/planning/overnight-prefect-docker-monai.sh`
**Issues addressed:** #434, #503, #504, #505 | **Resolved immediately:** #507 (already passing), #509 (P2 MONAI PR tracker)

---

## 1. High-Level Vision (Non-Negotiable)

This repository is an **add-on to MONAI**, not a separate system. The architecture principle:

> We do the hard work to make 3rd-party models (SAM3, Mamba, VesselFM) conform to MONAI's
> interfaces — we never make MONAI conform to our models.

MONAI modules work **off-the-shelf, zero modification**. This means:

| MONAI Module | Status | Requirement |
|---|---|---|
| `monai.losses` | ✅ Used directly | All losses available via `build_loss_function()` |
| `monai.transforms` | ✅ Used directly | Full Compose pipeline, no custom wrappers |
| `monai.networks` | ✅ DynUNet direct | Custom models (Mamba, VesselFM) wrap via `ModelAdapter` |
| `monai.metrics` | ✅ Partially used | NSD, HD95 use MONAI; extend as needed |
| `monai.data` | ✅ CacheDataset + ThreadDataLoader | Upgrade path: PersistentDataset tracked separately |
| `monai.inferers` | ⚠️ Inconsistent | **THE CORE PROBLEM** — roi_size=(32,32,32) hardcoded in Analysis Flow |

**Reference:** MONAI MLflow integration tutorial this repo extends:
https://github.com/Project-MONAI/MONAILabel/wiki/Experiment-Management-using-MLflow

**MONAI PR candidates tracker:** Issue #509 (P2, reviewed quarterly)

---

## 2. Issues Resolved Before Planning

### #507 — mypy viz errors (CLOSED, already passing)
`uv run mypy src/minivess/pipeline/viz/loss_comparison.py src/minivess/pipeline/viz/generate_all_figures.py --ignore-missing-imports` → **"Success: no issues found"**. Both files correctly use `plt.Axes` / `plt.Figure` via `if TYPE_CHECKING` guards. Issue closed 2026-03-08.

### #509 — MONAI PR contribution tracker (CREATED, P2)
Living issue to catalogue MONAI PR candidates (topology metrics, OME-TIFF reader, VesselFM adapter) and enforce MONAI-friendly architecture going forward.

---

## 3. Branch Structure

```
main
├── fix/prefect-docker-monai-optimization   ← #434, #503, #504 + Dockerfile.base
└── feat/analysis-multi-strategy-eval       ← #505 (scientific fairness — own branch)
```

Both branches are designed to merge to `main` independently. Branch 1 does NOT depend on branch 2.

---

## 4. Branch 1: `fix/prefect-docker-monai-optimization`

**Closes:** #434 (Docker smoke test), #503 (double-logging), #504 (flexible HPO)
**Prerequisite commits:** Dockerfile.base uncommitted changes
**Child plan XML:** `docs/planning/overnight-child-prefect-docker.xml`

### 4.1 Dockerfile.base (commit immediately)

Already correct in the uncommitted diff. Commit first as an atomic unit.

Changes in the diff:
- `ARG UID=1000 / ARG GID=1000` → bind-mount permission fix
- `uv sync --no-dev --no-install-project` (removed `2>/dev/null || true`) → fail-fast on bad deps
- `chmod -R a+rX /home/minivess/.local ... /app/.venv` → rootless `--user X:Y` support
- `umask 0002` in .bashrc → group-readable bind-mount output
- `ENV PATH=/app/.venv/bin:$PATH` → explicit venv on PATH

### 4.2 Rule #22 Compliance (.env.example)

**CRITICAL:** CLAUDE.md Rule #22 requires ALL configurable values to live in `.env.example` first.

Variables missing from `.env.example` that must be added:

```bash
# Container-internal artifact paths (sourced by docker-compose, not hardcoded in Dockerfile)
DATA_DIR=/app/data
CHECKPOINT_DIR=/app/checkpoints
LOGS_DIR=/app/logs
SPLITS_DIR=/app/configs/splits

# HPO
POSTGRES_DB_OPTUNA=optuna
OPTUNA_STORAGE_URL=         # empty = in-memory (sequential); postgresql://... for parallel
REPLICA_INDEX=0             # HPO hybrid mode: CUDA_VISIBLE_DEVICES assignment

# Logging
PREFECT_LOGGING_EXTRA_LOGGERS=minivess
```

After adding to `.env.example`, remove the corresponding `ENV VAR=value` lines from `Dockerfile.train` — they should only exist in `.env.example` + compose `${VAR:-fallback}`.

### 4.3 Issue #503 — Double-Logging

**Root cause:** Prefect 3.x only captures `prefect.*` loggers via `APILogHandler`. Application `minivess.*` loggers are invisible to the Prefect UI.

**Solution (minimal):**

1. Add to `docker-compose.flows.yml` x-common-env:
   ```yaml
   PREFECT_LOGGING_EXTRA_LOGGERS: ${PREFECT_LOGGING_EXTRA_LOGGERS:-minivess}
   ```

2. Create `src/minivess/observability/flow_logging.py`:
   ```python
   def configure_flow_logging(logs_dir: Path, logger_name: str = "minivess") -> None:
       """Add JSONL FileHandler to the named logger for durable structured logs.

       Call once at the top of each Prefect flow entrypoint (NOT inside tasks).
       JSONL format allows json.loads() parsing without regex (CLAUDE.md Rule #16).
       """
   ```

3. Wire `configure_flow_logging(logs_dir=Path(os.environ["LOGS_DIR"]))` at the top of
   `train_flow.py` and `hpo_flow.py`.

**Note:** `docker compose run train` (foreground, not `-d`) already streams stdout to terminal via `PYTHONUNBUFFERED=1`. The only gap is Prefect UI visibility and durable JSONL logs.

### 4.4 Issue #434 — Docker Smoke Test

**Extend** `tests/integration/orchestration/test_docker_training_smoke.py` with:

```
test_train_debug_run_produces_checkpoint   # bind-mount to tmp_path, assert *.pt exists
test_train_debug_run_mlflow_finished       # assert MLflow run status == FINISHED
test_train_debug_run_prefect_active        # assert Prefect context, not compat mode
```

**Key design decisions:**
- Use `pytest.tmp_path` for checkpoint bind-mount (NOT `/tmp` — Rule #18)
- `MLFLOW_TRACKING_URI=mlruns` (filesystem, no server needed) for the test
- Markers: `@pytest.mark.requires_docker`, `@pytest.mark.requires_train_image`, `@pytest.mark.slow`
- Skip gracefully if Docker daemon not running or image not built

**conftest.py additions needed:**
```python
pytest.ini_options markers = [
    "requires_docker: needs Docker daemon running",
    "requires_train_image: needs minivess-train image built",
    "slow: takes > 30s, skip with -m 'not slow'",
    "gpu_required: needs CUDA device",
]
```

### 4.5 Issue #504 — Flexible HPO Allocation

**Three strategies, config-driven via YAML:**

```yaml
# configs/hpo/dynunet_example.yaml — extended allocation block
allocation:
  strategy: sequential        # sequential | parallel | hybrid
  n_containers: 1             # parallel: N workers; hybrid: M GPUs
  trials_per_container: null  # hybrid only: K trials per GPU container
  optuna_storage: null        # null=in-memory; postgresql://... for parallel/hybrid
  gpu_indices: null           # hybrid: list or "auto" (discover via nvidia-smi)
```

**Strategy constraints (enforced in `HPOEngine.from_config()`):**
- `sequential`: `optuna_storage` can be null (in-memory) or SQLite
- `parallel`: `optuna_storage` MUST be `postgresql://...` — raise `ValueError` otherwise
  (SQLite has file-locking failures under concurrent writers)
- `hybrid`: `optuna_storage` MUST be postgresql; GPU assignment via
  `scripts/hpo_worker_entrypoint.sh` setting `CUDA_VISIBLE_DEVICES=$REPLICA_INDEX`

**New docker-compose service:**
```yaml
# docker-compose.flows.yml — hpo-worker for parallel strategy
hpo-worker:
  image: minivess-hpo:latest
  entrypoint: ["bash", "scripts/hpo_worker_entrypoint.sh"]
  environment:
    REPLICA_INDEX: ${REPLICA_INDEX:-0}
    OPTUNA_STORAGE_URL: ${OPTUNA_STORAGE_URL}
  deploy:
    resources:
      reservations:
        devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
```

Launch parallel strategy: `docker compose up --scale hpo-worker=N`

### 4.6 TDD Task Order for Branch 1

```
T1-A  Rule #22 — add missing vars to .env.example (RED: test_env_single_source.py)
T1-B  Rule #22 — remove hardcoded ENVs from Dockerfile.train (GREEN)
T1-C  #503 — test_flow_logging.py RED (JSONL handler, parseable output, no duplicates)
T1-D  #503 — flow_logging.py GREEN + wire into train_flow.py + hpo_flow.py
T1-E  #503 — add PREFECT_LOGGING_EXTRA_LOGGERS to compose x-common-env
T1-F  #434 — extend smoke test: RED (checkpoint, MLflow, Prefect assertions)
T1-G  #434 — implement docker compose run with bind-mount: GREEN
T1-H  #504 — test_hpo_allocation.py RED (AllocationStrategy enum + from_config)
T1-I  #504 — hpo_engine.py GREEN (AllocationStrategy enum + validation)
T1-J  #504 — configs/hpo/dynunet_example.yaml + compose hpo-worker service
T1-K  #504 — hpo_flow.py dispatch to strategy runner
T1-L  #504 — integration smoke: test_hpo_smoke.py sequential single trial
```

---

## 5. Branch 2: `feat/analysis-multi-strategy-eval`

**Closes:** #505 (standardize sliding-window validation — scientific fairness)
**Child plan XML:** `docs/planning/overnight-child-monai-eval.xml`

### 5.1 The Scientific Problem

The Analysis Flow (Flow 3) currently has:
```python
# analysis_flow.py:_evaluate_single_model_on_all() — line ~216
inference_runner = SlidingWindowInferenceRunner(
    roi_size=(32, 32, 32),   # WRONG for all models
    num_classes=2,            # hardcoded
    overlap=0.25,             # hardcoded
)
```

This makes cross-model paper results scientifically invalid. Paper tables that compare DynUNet vs SAM3 vs Mamba are comparing different evaluation conditions.

**Training Flow (Flow 2):** No change. Fast per-model-optimal ROI stays as-is.

### 5.2 Multi-Strategy Evaluation YAML

Add `inference_strategies` to `configs/evaluation/default.yaml`:

```yaml
inference_strategies:
  - name: standard_patch
    description: "Fixed patch across ALL models — use for paper tables"
    roi_size: [128, 128, 16]
    overlap: 0.5
    sw_batch_size: 4
    aggregation_mode: gaussian
    is_primary: true           # Results from this strategy → ComparisonTable + champions

  - name: fast
    description: "Per-model-optimal ROI for quick iteration runs"
    roi_size: per_model        # Sentinel → calls adapter.get_eval_roi_size()
    overlap: 0.25
    sw_batch_size: 4
    aggregation_mode: gaussian
    is_primary: false
```

**`full_volume` strategy** (`roi_size: [512, 512, -1]`) is available but excluded from the default
because it requires ≥ 24 GB VRAM for DynUNet on full MiniVess volumes. Users add it manually.

### 5.3 `InferenceStrategyConfig` Pydantic Schema

Add to `src/minivess/config/evaluation_config.py`:

```python
class InferenceStrategyConfig(BaseModel):
    name: str
    description: str = ""
    roi_size: list[int] | Literal["per_model"]  # -1 in any dim = wildcard (full volume)
    overlap: float = Field(ge=0.0, lt=1.0)
    sw_batch_size: int = Field(ge=1)
    aggregation_mode: Literal["gaussian", "constant"] = "gaussian"
    is_primary: bool = False

# Validation rules (model_validator):
# - Exactly ONE strategy with is_primary=True
# - All names unique
```

### 5.4 `ModelAdapter.get_eval_roi_size()`

Add **concrete (non-abstract)** method to `src/minivess/adapters/base.py`:

```python
def get_eval_roi_size(self) -> tuple[int, int, int]:
    """Per-model-optimal ROI for fast analysis-flow evaluation.
    Used when inference_strategies[i].roi_size == 'per_model'.
    Default (128, 128, 16) is safe for all conv models.
    """
    return (128, 128, 16)
```

**Only `Sam3VanillaAdapter` overrides this** → returns `(512, 512, 3)`.
Rationale: ViT-32L resizes ALL inputs to 1008×1008; fewer z-slices per window = fewer encoder calls.

**No other adapters need overrides.** DynUNet, Mamba, VesselFM all use the default.

### 5.5 `MultiStrategyInferenceRunner`

New file `src/minivess/pipeline/multi_strategy_inference.py`:

```python
class MultiStrategyInferenceRunner:
    """Runs each InferenceStrategyConfig against a model, returns per-strategy metrics.

    - Resolves 'per_model' roi_size via adapter.get_eval_roi_size()
    - Resolves wildcard -1 dimensions from actual volume shape
    - Uses monai.inferers.sliding_window_inference for ALL models (MONAI-first)
    - Handles both ModelAdapter (SegmentationOutput) and plain nn.Module (raw tensor)
    """

    def _resolve_roi_size(self, strategy, model, volume_shape) -> tuple[int, int, int]: ...
    def _make_predictor(self, model) -> Callable: ...
    def run_strategy(self, model, dataloader, strategy) -> dict[str, float]: ...
    def run_all_strategies(self, model, dataloader) -> dict[str, dict[str, float]]: ...
```

**Predictor wrapper pattern** (from `trainer.py:_model_fn`):
```python
def _make_predictor(model):
    def _forward(x):
        out = model(x)
        return out.logits if hasattr(out, "logits") else out
    return _forward
```

### 5.6 MLflow Metric Logging with Strategy Prefixes

- **Primary strategy** metrics logged without prefix: `dsc`, `cldice`, etc.
- **Non-primary** metrics logged with `{strategy_name}/` prefix: `fast/dsc`, `fast/cldice`
- MLflow tag: `eval_inference_strategy_primary = "standard_patch"`
- Key split: `str.partition("/")` — **no regex** (CLAUDE.md Rule #16)

### 5.7 TDD Task Order for Branch 2

```
T2-A  Pydantic: InferenceStrategyConfig — RED (test_inference_strategy_config.py)
T2-B  Pydantic: InferenceStrategyConfig — GREEN (add to evaluation_config.py)
T2-C  YAML: default.yaml inference_strategies block — RED (test_evaluation_yaml_loading.py)
T2-D  YAML: default.yaml — GREEN (add standard_patch + fast strategies)
T2-E  Adapter: get_eval_roi_size() — RED (test_eval_roi_size.py — all 4 adapters)
T2-F  Adapter: base.py + sam3_vanilla.py — GREEN
T2-G  MultiStrategyInferenceRunner — RED (test_multi_strategy_inference.py — 5 tests)
T2-H  MultiStrategyInferenceRunner — GREEN (multi_strategy_inference.py)
T2-I  Analysis Flow: wire MultiStrategyInferenceRunner — RED (test_analysis_flow_multi_strategy.py)
      Includes AST guard: assert literal (32,32,32) no longer in analysis_flow.py
T2-J  Analysis Flow: replace hardcoded SlidingWindowInferenceRunner — GREEN
T2-K  Metric logging: strategy prefix logger — RED (test_strategy_metric_logging.py)
T2-L  Metric logging: strategy_metric_logger.py — GREEN
T2-M  Integration smoke: 2 strategies × DynUNet CPU — RED+GREEN
T2-N  Guard test: AST check no model-family if-branch in analysis_flow.py (Task-Agnostic Rule)
```

---

## 6. MONAI Compatibility Audit (Scope: Both Branches)

### Current State per Module

| Module | Current Usage | Gap / Action |
|---|---|---|
| `monai.losses` | DiceCE, Dice, Focal, HausdorffDT, LogHausdorffDT, clDice | ✅ Factory has all standard losses. P0 issue #474 tracks extended audit. |
| `monai.transforms` | Full Compose pipeline, avoids Spacingd (OOM root cause) | ✅ Correct. No Spacingd is intentional. |
| `monai.networks` | DynUNet (native). Mamba/VesselFM/SAM3 are 3rd-party wrapped. | ✅ Adapters conform to `(B,C,H,W,D)` MONAI convention. |
| `monai.metrics` | NSD, HD95 via MONAI wrappers; topology via MetricsReloaded | ✅ Sufficient for now. |
| `monai.data` | CacheDataset + ThreadDataLoader | ✅ Correct. PersistentDataset upgrade tracked separately. |
| `monai.inferers` | `sliding_window_inference` in trainer + inference.py | ⚠️ Analysis Flow bypasses with hardcoded `(32,32,32)`. **Fixed by Branch 2.** |

### Known Mamba Dimension Risk

`MambaAdapter.forward()` accepts `(B,C,H,W,D)` from MONAI DataLoader but internally uses
`nn.Conv3d` (PyTorch convention: `(B,C,D,H,W)`). If the adapter does not permute before
convolution, all Mamba results are computed on wrong axes. **This must be verified as part
of Branch 2 T2-G (integration smoke)** — a test that asserts Mamba produces correct
output shape `(B,2,H,W,D)` catches this silently.

### MONAI-Compatible Interface Contract for 3rd-Party Adapters

All adapters MUST satisfy:
1. `forward(images: Tensor[B,C,H,W,D]) -> SegmentationOutput` (depth-last, MONAI convention)
2. `get_eval_roi_size() -> tuple[int,int,int]` (concrete with default in base)
3. Compatible with `monai.inferers.sliding_window_inference` as a predictor function
4. Uses `monai.losses.*` losses (no custom loss API required from adapters)

This contract is sufficient today and is MONAI-PR-friendly tomorrow.

---

## 7. Summary: Execution Order

```
IMMEDIATE (same session):
  1. Commit Dockerfile.base → fix/prefect-docker-monai-optimization
  2. Close #507 (already done — mypy passing)
  3. Create #509 (already done — MONAI PR tracker)

BRANCH 1 (fix/prefect-docker-monai-optimization):
  Overnight child plan: docs/planning/overnight-child-prefect-docker.xml
  Tasks T1-A → T1-L (sequential TDD)
  PR → main closes: #434, #503, #504

BRANCH 2 (feat/analysis-multi-strategy-eval):
  Overnight child plan: docs/planning/overnight-child-monai-eval.xml
  Tasks T2-A → T2-N (sequential TDD)
  PR → main closes: #505

FUTURE (tracked via #509):
  - MONAI PR candidates quarterly review
  - Mamba dimension audit (exposed by T2-G integration test)
```

**Overnight orchestration script:** `docs/planning/overnight-prefect-docker-monai.sh`

---

## 8. Reviewer Agent Findings (Preserved)

Parallel specialist review conducted 2026-03-08 by 3 agents:

- **MONAI consolidation specialist** → detailed 8-task TDD plan; identified hardcoded `(32,32,32)` as root cause; flagged Mamba dimension risk; recommended `feat/analysis-multi-strategy-eval` branch name
- **Docker/Prefect/HPO specialist** → identified 4 Rule #22 violations in `.env.example`; designed parallel HPO with PostgreSQL requirement; detailed 12-task TDD plan; confirmed `PREFECT_LOGGING_EXTRA_LOGGERS=minivess` as the Prefect-native solution for #503
- **Structure/exploration agent** → confirmed overnight XML schema; confirmed #507 mypy already passing (verified locally); confirmed `roi_size=(32,32,32)` hardcoded at `analysis_flow.py:_evaluate_single_model_on_all()` line ~216
