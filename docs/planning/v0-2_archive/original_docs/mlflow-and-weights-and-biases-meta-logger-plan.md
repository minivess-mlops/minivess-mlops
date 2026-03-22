# Meta-Logger Plan: MLflow + Weights & Biases + Alternative Backends

**Date**: 2026-03-21
**Priority**: P2 (refactor tracking abstraction), P3 (W&B implementation)
**Status**: Planning only — nothing implemented yet
**Rationale**: Some researchers prefer W&B's real-time training visualization. MLflow
remains the contract layer and source of truth, but W&B (or other backends) can
mirror events for enhanced DevEx.

---

## Design Decisions (from interactive questionnaire)

1. **Dual-write mode**: Config chooses mlflow-only, wandb-only, or mlflow+wandb simultaneously
2. **MLflow is always the inter-flow contract**: `FlowContract`, `find_upstream_run()`,
   checkpoint discovery always use MLflow. W&B is observation-only, never the source of truth.
3. **Comprehensive landscape survey**: Cover W&B, DagsHub, Neptune.ai, ClearML, Comet ML

---

## 1. Experiment Tracking Platform Landscape (2026)

| Platform | Type | Open Source? | MLflow Compat? | Self-Hosted? | Status |
|----------|------|-------------|----------------|-------------|--------|
| **[MLflow](https://mlflow.org/)** | Tracking + registry + serving | Yes (Apache 2.0) | — | Yes (easy) | **Active, Databricks-backed** |
| **[W&B](https://wandb.ai/)** | Tracking + sweeps + reports | Partial (SDK open) | Import only | Yes (enterprise) | **Active, well-funded** |
| **[DagsHub](https://dagshub.com/)** | MLflow + DVC + Label Studio hub | Uses OSS tools | **Full (hosted MLflow)** | No (SaaS only) | **Active, freemium** |
| **[ClearML](https://clear.ml/)** | Auto-logging + orchestration | Yes (Apache 2.0) | Migration scripts | Yes (clearml-server) | **Active, strong OSS** |
| **[Comet ML](https://www.comet.com/)** | Tracking + eval | Partial (SDK commercial) | Via comet-for-mlflow | No (SaaS) | **Active, commercial** |
| **~~[Neptune.ai](https://neptune.ai/)~~** | ~~Metadata store~~ | ~~No~~ | ~~Export tools~~ | ~~No~~ | **SHUTTING DOWN 2026-03-05** |
| **[Aim](https://aimstack.io/)** | Visualization-focused tracker | Yes (Apache 2.0) | No direct compat | Yes | **Active, lightweight** |
| **[SwanLab](https://swanlab.cn/)** | Chinese market experiment tracking | Yes | No | Yes | **Active, regional** |
| **[FastTrackML](https://github.com/G-Research/fasttrackml)** | MLflow-compatible speed-focused server | Yes | **Full (drop-in)** | Yes | Succeeded by mlflow-go |

### Neptune.ai — DO NOT ADOPT

Neptune was acquired by OpenAI in December 2025 (~$400M in stock). The platform is
**shutting down March 5, 2026**. All user data will be irreversibly deleted. Neptune
provides export tools to Parquet for migration to W&B or MLflow.
[Source](https://openai.com/index/openai-to-acquire-neptune/)

### DagsHub — Special Case

DagsHub is NOT an alternative to MLflow — it IS MLflow (hosted). Every DagsHub repo
gets a free MLflow-compatible tracking server. You just set:
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<token>
MLFLOW_TRACKING_PASSWORD=<token>
```
Existing MLflow code works unchanged. This makes DagsHub a **deployment option** for
MLflow, not a separate backend to integrate.

### ClearML — Strongest Open-Source Alternative

ClearML's "auto-magical" approach (2 lines of code capture everything) is compelling
but fundamentally different from MLflow's explicit logging. Migration from MLflow to
ClearML requires rethinking the logging approach. Self-hosted `clearml-server` is
fully open-source (Apache 2.0). Best alternative if MLflow were ever abandoned.

---

## 2. Reference Implementations for Multi-Backend Logging

### 2.1 NVIDIA NeMo-RL LoggerInterface (Best Pattern)

[NeMo-RL Logger Design Doc](https://docs.nvidia.com/nemo/rl/latest/design-docs/logger.html)

Clean ABC with fan-out delegation:

```python
class LoggerInterface(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int,
                    prefix: str = '', step_metric: str | None = None) -> None: ...
    @abstractmethod
    def log_hyperparams(self, params: Mapping[str, Any]) -> None: ...
    @abstractmethod
    def log_histogram(self, histogram: list[Any], step: int, name: str) -> None: ...

class Logger(LoggerInterface):
    """Fan-out to all enabled backends."""
    def __init__(self, cfg: LoggerConfig):
        # Creates: TensorboardLogger, WandbLogger, MLflowLogger, SwanlabLogger
        # based on config flags

    def log_metrics(self, metrics, step, prefix='', step_metric=None):
        for backend in self._backends:
            backend.log_metrics(metrics, step, prefix, step_metric)
```

### 2.2 PyTorch Lightning Multi-Logger (Most Battle-Tested)

[Lightning Logging Docs](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)

```python
trainer = Trainer(logger=[MLFlowLogger(...), WandbLogger(...)])
# self.log("train/loss", loss_value) → dispatches to ALL loggers
```

7 built-in backends. `self.log()` fan-out is transparent. Backend-specific APIs
accessible via `self.logger[0].experiment`.

### 2.3 lightning-hydra-template Config-Driven Selection

[many_loggers.yaml](https://github.com/ashleve/lightning-hydra-template/blob/main/configs/logger/many_loggers.yaml)

```yaml
# configs/logger/many_loggers.yaml
defaults:
  - csv
  # - mlflow        # uncomment to enable
  # - wandb         # uncomment to enable
  - tensorboard
```

Toggle backends by commenting/uncommenting YAML defaults. Each backend has its own
config file. This is the cleanest Hydra-native approach.

### 2.4 Other Implementations

| Implementation | Pattern | URL |
|---------------|---------|-----|
| **Catalyst ILogger** | ABC + dict-based multi-logger | [catalyst-team/catalyst](https://catalyst-team.github.io/catalyst/api/loggers.html) |
| **LightEx MultiLogger** | Wildcard fan-out (`logger.log('*', ...)`) | [ofnote/lightex](https://github.com/ofnote/lightex) |
| **xplogger** | Config-driven LogBook | [shagunsodhani/xplogger](https://github.com/shagunsodhani/xplogger) |
| **NVIDIA FLARE** | Sender/Receiver decoupled pattern | [FLARE docs](https://nvflare.readthedocs.io/en/2.4/programming_guide/experiment_tracking.html) |
| **Anomalib** | Lightning loggers + image logging mixin | [anomalib loggers](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/loggers/index.html) |

---

## 3. Proposed Architecture for MinIVess

### Design Principles

1. **MLflow is always the contract layer** — downstream flows discover upstream runs,
   read checkpoints, and build ensembles via MLflow. This NEVER changes.
2. **Additional backends are mirrors** — they receive the same events but don't affect
   pipeline behavior. If W&B is down, the pipeline continues.
3. **Config-driven** — Hydra-zen config group `configs/tracker/` selects backends.
4. **Zero overhead when disabled** — if only MLflow is enabled, no W&B code is loaded.

### Event Types

```
┌─────────────────────────────────────────────────────────┐
│                    Training Events                       │
├─────────────────────────────────────────────────────────┤
│ Epoch-level:                                             │
│   log_epoch_metrics(metrics, step=epoch)                │
│   log_hyperparams(config_dict)                          │
│                                                          │
│ Run-level:                                               │
│   start_run(name, tags, experiment)                     │
│   end_run(status)                                        │
│   log_artifact(path, artifact_path)                     │
│   set_tag(key, value)                                    │
│                                                          │
│ Artifact-level:                                          │
│   log_checkpoint(path, metadata)                        │
│   log_figure(figure, name)                              │
│   log_table(dataframe, name)                            │
└─────────────────────────────────────────────────────────┘
```

### Class Hierarchy

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

class TrackerBackend(Protocol):
    """Structural subtype for experiment tracking backends.

    Each backend implements these methods. The CompositeTracker
    delegates to all enabled backends.
    """
    def log_metrics(self, metrics: dict[str, float], *, step: int, prefix: str = "") -> None: ...
    def log_hyperparams(self, params: dict[str, Any]) -> None: ...
    def log_artifact(self, local_path: Path, *, artifact_path: str = "") -> None: ...
    def start_run(self, *, run_name: str, tags: dict[str, str], experiment: str) -> str: ...
    def end_run(self, *, status: str = "FINISHED") -> None: ...
    def set_tag(self, key: str, value: str) -> None: ...


class CompositeTracker:
    """Fan-out to multiple backends. MLflow is always first (contract layer)."""

    def __init__(self, backends: list[TrackerBackend]) -> None:
        self._backends = backends

    def log_metrics(self, metrics: dict[str, float], *, step: int, prefix: str = "") -> None:
        for backend in self._backends:
            try:
                backend.log_metrics(metrics, step=step, prefix=prefix)
            except Exception:
                logger.warning("Backend %s failed to log metrics", type(backend).__name__)

    # ... same pattern for all methods


class MLflowBackend:
    """MLflow backend — always the primary/contract layer."""
    # Wraps existing ExperimentTracker from observability/tracking.py


class WandbBackend:
    """Weights & Biases backend — optional mirror."""
    # Uses wandb.init(), wandb.log(), wandb.finish()


class TensorBoardBackend:
    """TensorBoard backend — optional, for local visualization."""
    # Uses torch.utils.tensorboard.SummaryWriter
```

### Hydra-zen Config Groups

```yaml
# configs/tracker/mlflow_only.yaml (default)
tracker:
  backends:
    - type: mlflow
      tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,mlruns}

# configs/tracker/mlflow_wandb.yaml (dual-write)
tracker:
  backends:
    - type: mlflow
      tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,mlruns}
    - type: wandb
      project: ${oc.env:WANDB_PROJECT,minivess}
      entity: ${oc.env:WANDB_ENTITY,null}

# configs/tracker/wandb_only.yaml (W&B only — still creates MLflow for contract)
tracker:
  backends:
    - type: mlflow  # Always present for inter-flow contract
      tracking_uri: mlruns
    - type: wandb
      project: minivess
  primary: wandb  # UI preference, not contract preference
```

### Migration Path

```
Phase 1 (P2): Refactor ExperimentTracker → MLflowBackend
  - Extract TrackerBackend Protocol from existing code
  - Wrap existing tracking.py in MLflowBackend class
  - Create CompositeTracker with single MLflowBackend
  - ALL existing code continues to work unchanged
  - Test: make test-staging passes with zero behavior changes

Phase 2 (P3): Implement WandbBackend
  - Add wandb as optional dependency
  - Implement WandbBackend conforming to TrackerBackend Protocol
  - Add configs/tracker/mlflow_wandb.yaml
  - Test: dual-write produces identical metrics in both systems
```

---

## 4. W&B Advantages for Real-Time Training Visualization

| Feature | MLflow | W&B | Winner |
|---------|--------|-----|--------|
| Real-time training curves | Manual refresh, basic UI | Auto-updating, smooth | **W&B** |
| Hyperparameter sweeps | Manual (Optuna integration) | Built-in sweep agent | **W&B** |
| Collaborative reports | No | Rich markdown + embedded charts | **W&B** |
| Team dashboards | Basic (MLflow server) | Polished workspace | **W&B** |
| Custom panels | No | Vega + Python panels | **W&B** |
| On-prem / air-gapped | Easy (file-based or Docker) | Enterprise only ($$$) | **MLflow** |
| Open-source server | Yes (Apache 2.0) | No (SaaS or enterprise) | **MLflow** |
| Inter-flow contract | Native (mlruns directory) | Not designed for this | **MLflow** |
| DVC integration | Via DagsHub | Via Artifacts (different) | **MLflow** (via DagsHub) |
| Cost | Free | Free tier limited, paid tiers | **MLflow** |

### Why MLflow Stays as Contract

1. **Air-gapped labs**: Some research labs have no internet. MLflow works file-based.
2. **Inter-flow dependency**: Post-training, Analysis, Biostatistics flows discover
   upstream runs via MLflow directory structure. Replicating this in W&B is complex.
3. **Open-source guarantee**: MLflow is Apache 2.0. W&B can change pricing/terms.
4. **DVC integration**: Data versioning goes through DVC → GCS, which connects to
   MLflow artifacts naturally.

### Why W&B as Optional Mirror

1. **Real-time curves**: Researchers want to watch training progress from their phone.
2. **Sweep integration**: W&B sweeps are cleaner than Optuna for HP search.
3. **Reports**: W&B reports are publication-quality with embedded interactive charts.
4. **Community**: Many PhD students are already familiar with W&B.

---

## 5. MONAI-Specific Considerations

MONAI has `MLFlowHandler` for Ignite Engine event attachment but **no unified tracker
abstraction**. Our `TrackerBackend` Protocol would complement MONAI's handler system:

- MONAI handlers attach to Ignite `Events.EPOCH_COMPLETED`
- Our `CompositeTracker.log_epoch_metrics()` is called by the handler
- The handler doesn't know or care which backends are active

This means MONAI users can use our tracker with zero MONAI code changes — the handler
calls the same API, and the tracker fan-outs to whatever backends are configured.

---

## 6. Issues to Create

### P2: Refactor tracking abstraction (post-publication)

**Title**: `refactor: extract TrackerBackend Protocol from ExperimentTracker for multi-backend support`

Scope:
- Define `TrackerBackend` Protocol in `src/minivess/observability/tracker_protocol.py`
- Refactor `ExperimentTracker` → `MLflowBackend` implementing the Protocol
- Create `CompositeTracker` with fan-out delegation
- Wire via Hydra-zen config group `configs/tracker/`
- Zero behavior change — MLflow remains the only enabled backend
- All existing tests pass unchanged

### P3: Implement W&B backend (post-publication, if demand exists)

**Title**: `feat: add Weights & Biases backend for real-time training visualization`

Scope:
- Implement `WandbBackend` conforming to `TrackerBackend` Protocol
- Add `wandb` as optional dependency (`uv add --optional wandb wandb`)
- Create `configs/tracker/mlflow_wandb.yaml`
- Test dual-write produces identical metrics
- Document: W&B is observation-only, MLflow is always the contract layer

---

## References

### Reference Implementations
1. [NeMo-RL Logger Design Doc](https://docs.nvidia.com/nemo/rl/latest/design-docs/logger.html) — cleanest ABC pattern
2. [NeMo-RL Logger API](https://docs.nvidia.com/nemo/rl/latest/apidocs/nemo_rl/nemo_rl.utils.logger.html)
3. [PyTorch Lightning Logging](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) — most battle-tested
4. [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/blob/main/configs/logger/many_loggers.yaml) — config-driven selection
5. [Catalyst ILogger](https://catalyst-team.github.io/catalyst/api/loggers.html)
6. [NVIDIA FLARE Experiment Tracking](https://nvflare.readthedocs.io/en/2.4/programming_guide/experiment_tracking.html)
7. [xplogger](https://github.com/shagunsodhani/xplogger)

### Platforms
8. [MLflow](https://mlflow.org/) — Apache 2.0, Databricks-backed
9. [Weights & Biases](https://wandb.ai/) — Commercial, strong DevEx
10. [DagsHub](https://dagshub.com/) — Hosted MLflow + DVC + Label Studio
11. [ClearML](https://clear.ml/) — Apache 2.0, auto-logging
12. [Comet ML](https://www.comet.com/) — Commercial, MLflow bridge via [comet-for-mlflow](https://github.com/comet-for-mlflow)
13. [~~Neptune.ai~~](https://neptune.ai/) — **SHUTTING DOWN 2026-03-05** ([OpenAI acquisition](https://openai.com/index/openai-to-acquire-neptune/))
14. [Aim](https://aimstack.io/) — Apache 2.0, visualization-focused
15. [FastTrackML](https://github.com/G-Research/fasttrackml) — MLflow-compatible speed server

### MONAI Integration
16. [MONAI MLFlowHandler](https://github.com/Project-MONAI/MONAI/blob/dev/monai/handlers/mlflow_handler.py)
17. [MONAI W&B Discussion](https://github.com/Project-MONAI/MONAI/discussions/3155)
18. [Anomalib Multi-Logger](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/loggers/index.html)
