# Prefect Orchestration & Adaptive DevEx Profiling Plan

**Status**: DRAFT
**Created**: 2026-02-25
**Depends on**: [dev-ex-automation-plan.md](dev-ex-automation-plan.md) | [monai-performance-optimization-plan.md](monai-performance-optimization-plan.md)
**Reference implementation**: [foundation-PLR Prefect patterns](https://github.com/petteriTeikari/foundation_PLR)

## User Prompt (Verbatim)

> I mean the grand vision of this whole repo should be EXCELLENT DevEx and that the MLOps part should come here as a scaffold for PhD researchers freeing their time from infrastructure wrangling and making their lives easier. These should be indicated in the CLAUDE.md files all oever. This is the 1st design goal of this whole repo! Everything should be as automatic as possible WITH THE OPTION obviously for the researchers to tweak this as much as they want! But the idea of compute-profiles and performance benchmarking is quite flexible. So you can have multiple options defined by ourselves, and show how researchers can define them? E.g. you could have "--CPU" profile that adaptively checks the amount of RAM, the size of swap and modifies the dynunet training parameters accordingly? And then we need to map the "--CPU" later for the SAMv3 model as different models have very different memory needs! And same goes for example for the cache ratio that needs to be adaptive as the researchers obviously can have their own datasets that are a lot larger than this minivess, with more voxels, etc. Similarly the profiles should determine the patch sizes for training constrained by the GPU available. We cannot use 512 x 512 x 512 patches obviously with many hw platforms :D And if the smalles volume is 256 x 256 x 9 voxels, we cannot use 64 x 64 x 16 patches as the 16 is more than the smallest z-dimension of the available dataset. So you should have a test that verifies the dataset coming to the dataloader! And I don't want to resample the dataset by default! The data should be loaded as it is and the patch (subset) ensures that the data has same shapes inside the dataloader batch! There can be again be some .yaml configuration keys that allow the user to use whatever cancy latent diffusion upsampling / super-resolution model if the anisotropic resolution is vert different across the volumes, but we should not make those decision in behalf of the researchers. It is a scientific question itself, whether one should upsample, and with what method, etc. So let's keep this simple and start with finding patches that you can take across our minivess dataset, implement some data quality / data engingeering pipelines for this! Also think how to bring Prefect flows to this whole architecture as this data quality task could be run periodically as new data is coming (As versioned via DVC) which could always trigger the Prefect flow to be run that does this so that is "ready in Mlflow" as "data-experiment" (or how do you think of storing artifacts so that they are accessible from different Prefect flows?). So let's expand this /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/devex-automation-plan.md with a new plan /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/prefect-and-devex-profiling-optimizations.md with reviewer agents optimizing this report for factual correctness and multi-hypothesis open-ended decision matrix! And then let's synthesize these two .md plans into an executable .xml plan evaluating this training script
>
> save my prompt verbatim as well
>
> Have a look at my Prefect use in /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR and improve on it, and remember the self-learning and portfoli item mandate. So let's make the prefect use even more production-grade and running on local machine, intranet / on-prem server and on cloud
>
> For the Prefect architecture, you should think of division of labor as well. In a small academic team you might have a person who is really into data engineering, data quality, data curation, thus one Prefect block should be about data (ETL) and about making raw data from experiments, collaborators proofread, i.e. you could have raw S3 bucket (do not hardcode any data sources, this just as an example) that gets all the data all over the place. And the data person is then building all the tools how to ensure that the data is of good quality and passes the checks (with some human-in-the-loop intelligent annotationd and quality control dashboard / UI, e.g. with Slicer and/or something else). And then this proofread data is versioned via DVC to be used to train models, and this DVC versioning update can automically trigger the model retraining as well, and all the highest maturity level jobs of MLOps maturity model that we would like to be demonstrating here on this repo and associated academic manuscript! This would be then the Prefect Flow of data scientist / medical image analysis expert / MONAI expert. Both data engineer and data scientist would use MLflow as contract so that the artifacts are there with flexible backend_store (S3 on AWS, or some local hard disk/postgresql, etc.). And then there could be a 3rd Prefect flow could be for any ensembling experiments, and analyzing the actual model training and this Prefect Flow would gets its input from the Data Scientist artifact output! The 4th Prefect Flow would be all about the deployment via BentoML / Argo / etc. reading the MLflow as input and deploying some BentoML docker that could be served anywhere behind a FastAPi for example? This BentoML server could again be run on a local machine, on some on-prem server or deployed to cloud with all the elasitc candy via Nomad or Kubernetes? Maybe the academic lab builds such a good segmentor model that the lab wants to serve it for the whole community and make it available over the internet with all the actual software engineering done it with Cloudflare CDN and auto-balancer spinning up more instances with higher load. Maybe the deployment is very cost-sensitive and we need to use SkyPilot's Skyserve on spot instances? We don't have to implement all these, but we should definitely be prepared for the growth of the repo and design our architecture accordingly!
>
> I mean it is obvious that the division of labor via Prefect is useful for the solo-researcher archetype as well even the person would be the only person working on this
>
> Prefect should be obligatory now! not optional

---

## 1. Design Goal #1: EXCELLENT DevEx for PhD Researchers

This is the foundational principle of minivess-mlops:

> **MLOps as a scaffold that frees PhD researchers from infrastructure wrangling.**
> Everything automatic by default, everything tweakable by choice.

### Design Principles

| # | Principle | Implication |
|---|-----------|-------------|
| P1 | **Zero-config start** | `just experiment` works out of the box on any machine |
| P2 | **Adaptive defaults** | Hardware detection → auto-select batch size, patch size, cache rate |
| P3 | **Scientific decisions stay with the researcher** | No default resampling. No implicit upsampling. Expose knobs, don't turn them. |
| P4 | **Model-agnostic profiles** | Same `--cpu` / `--gpu_low` flags work for DynUNet, SAMv3, SegResNet — each model maps profiles differently |
| P5 | **Dataset-agnostic patches** | Patch sizes are constrained by the dataset's smallest volume, not hardcoded |
| P6 | **Transparent automation** | Every automatic decision is logged and overridable via YAML |
| P7 | **Portfolio-grade code** | Every component is a self-contained demonstration of production ML engineering |

---

## 2. Adaptive Compute Profiles

### Current State (Static Profiles)

The existing 6 profiles in `compute_profiles.py` are **static dictionaries**:

| Profile | Batch | Patch | Workers | AMP |
|---------|-------|-------|---------|-----|
| cpu | 1 | 64×64×16 | 2 | No |
| gpu_low | 2 | 96×96×24 | 4 | Yes |
| gpu_high | 4 | 128×128×32 | 8 | Yes |
| dgx_spark | 8 | 128×128×48 | 12 | Yes |
| cloud_single | 8 | 128×128×64 | 16 | Yes |
| cloud_multi | 32 | 128×128×64 | 16 | Yes |

**Problems**:
1. Patch Z=16 for `cpu` profile exceeds smallest MiniVess volume (Z=5 slices)
2. Patch Z=24 for `gpu_low` exceeds volumes with Z=9–22 slices
3. Cache rate is not part of profiles (hardcoded in scripts)
4. No model-specific adaptation (DynUNet vs SAMv3 have very different VRAM needs)
5. No dataset-specific adaptation (MiniVess vs researcher's own 1000-volume dataset)

### Proposed: Two-Phase Adaptive Profiles

#### Phase A: Hardware Detection (compute budget)

```python
@dataclass
class HardwareBudget:
    """Detected at runtime, not hardcoded."""
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    swap_total_gb: float
    swap_used_gb: float
    gpu_name: str | None
    gpu_vram_mb: int          # 0 if no GPU
    gpu_count: int
    disk_free_gb: float
    environment: Literal["local", "docker", "cloud", "ci"]
```

Detection logic auto-selects a **base compute tier**:

```
No GPU                              → tier: cpu
GPU VRAM < 6 GB                     → tier: cpu (warn)
GPU VRAM 6–10 GB, single            → tier: gpu_low
GPU VRAM 10–24 GB, single           → tier: gpu_high
GPU VRAM 24–48 GB, single           → tier: dgx_spark
GPU VRAM >= 48 GB, single           → tier: cloud_single
Multiple GPUs (any VRAM)            → tier: cloud_multi
```

#### Phase B: Dataset-Aware Adaptation (constrain the budget)

After hardware detection, **scan the dataset** to constrain parameters:

```python
@dataclass
class DatasetProfile:
    """Computed by MONAI DataAnalyzer or our lightweight scanner."""
    n_volumes: int
    min_shape: tuple[int, int, int]       # (min_x, min_y, min_z) across dataset
    max_shape: tuple[int, int, int]
    median_shape: tuple[int, int, int]
    voxel_spacing_range: tuple[float, float]  # (min, max) spacing
    total_size_gb: float                  # uncompressed float32
    per_volume_mean_gb: float
    is_anisotropic: bool                  # True if spacing varies >2x across volumes
```

**Adaptive patch size logic**:

```python
def compute_safe_patch_size(
    hardware: HardwareBudget,
    dataset: DatasetProfile,
    model_family: ModelFamily,
) -> tuple[int, int, int]:
    """Patch size that fits ALL volumes AND the GPU."""

    # Step 1: Dataset constraint — patch must fit smallest volume
    max_patch = dataset.min_shape  # e.g., (256, 256, 5)

    # Step 2: Divisibility constraint — model architecture requirement
    divisor = get_model_divisor(model_family)  # DynUNet: 2^(n_levels-1)

    # Step 3: Round down each dim to nearest multiple of divisor
    safe = tuple(
        (min(p, m) // divisor) * divisor
        for p, m in zip(base_patch, max_patch)
    )

    # Step 4: GPU memory constraint — reduce if estimated VRAM exceeds budget
    while estimate_vram(safe, model_family, batch_size) > hardware.gpu_vram_mb * 0.8:
        safe = reduce_patch(safe, divisor)

    return safe
```

**Adaptive cache rate logic**:

```python
def compute_safe_cache_rate(
    hardware: HardwareBudget,
    dataset: DatasetProfile,
) -> float:
    """Cache as much as fits in available RAM, leaving headroom."""
    available_for_cache = hardware.ram_available_gb * 0.6  # 60% of free RAM
    if dataset.total_size_gb <= available_for_cache:
        return 1.0  # Cache everything
    return min(1.0, available_for_cache / dataset.total_size_gb)
```

#### Phase C: Model-Specific Mapping

Different models have radically different memory footprints for the same patch size:

```yaml
# configs/model_profiles/dynunet.yaml
model_family: dynunet
divisibility_factor: 8  # 2^3 for 4 encoder levels
vram_per_voxel_train_bytes: 24  # empirical estimate
vram_base_mb: 400  # model weights + optimizer state
max_patch_xy: 512
max_patch_z: 128
supports_amp: true

# configs/model_profiles/samv3.yaml
model_family: vista3d
divisibility_factor: 16  # deeper encoder
vram_per_voxel_train_bytes: 64  # much heavier (ViT backbone)
vram_base_mb: 2400  # large pretrained weights
max_patch_xy: 256
max_patch_z: 64
supports_amp: true
```

**Researchers define their own models** by adding a YAML file:

```yaml
# configs/model_profiles/my_custom_unet.yaml
model_family: custom
divisibility_factor: 4
vram_per_voxel_train_bytes: 16
vram_base_mb: 200
max_patch_xy: 256
max_patch_z: 128
supports_amp: false
```

---

## 3. Dataset Validation Pipeline (Pre-Training Data Quality)

### Design: No Resampling by Default

**Core principle**: Load data at native resolution. Patches handle shape consistency inside the dataloader batch. Resampling is a scientific decision left to the researcher.

```yaml
# configs/data/defaults.yaml
resampling:
  enabled: false                # NO default resampling
  method: null                  # Options: spacing, super_resolution, latent_diffusion
  target_spacing: null          # Only used if enabled=true
  # Researchers who want resampling explicitly enable it:
  # enabled: true
  # method: spacing
  # target_spacing: [1.0, 1.0, 1.0]
```

### Dataset Profiler (`src/minivess/data/profiler.py`)

A lightweight scanner that runs before training:

```python
@dataclass
class VolumeStats:
    """Per-volume statistics."""
    filename: str
    shape: tuple[int, int, int]
    voxel_spacing: tuple[float, float, float]
    intensity_min: float
    intensity_max: float
    intensity_mean: float
    foreground_ratio: float  # fraction of non-zero voxels
    size_bytes: int

@dataclass
class DatasetProfile:
    """Aggregate statistics across all volumes."""
    volumes: list[VolumeStats]
    n_volumes: int
    min_shape: tuple[int, int, int]
    max_shape: tuple[int, int, int]
    median_shape: tuple[int, int, int]
    spacing_range: tuple[tuple[float, ...], tuple[float, ...]]
    is_anisotropic: bool
    total_size_gb: float
    safe_patch_sizes: dict[str, tuple[int, int, int]]  # model_family → safe patch
    warnings: list[str]  # e.g., "Volume mv02 has 4.97μm spacing (outlier)"
```

### Validation Checks (Tests)

```python
class TestDatasetPatchCompatibility:
    """Verify that configured patch sizes are compatible with the dataset."""

    def test_patch_fits_all_volumes(self, dataset_profile, data_config):
        """Patch size must be <= smallest volume dimension (after SpatialPadd)."""
        for dim in range(3):
            assert data_config.patch_size[dim] <= dataset_profile.min_shape[dim], (
                f"Patch dim {dim}={data_config.patch_size[dim]} exceeds "
                f"smallest volume dim={dataset_profile.min_shape[dim]}. "
                f"Either reduce patch_size or enable SpatialPadd."
            )

    def test_patch_divisible_by_model_factor(self, data_config, model_config):
        """Patch dims must be divisible by model's architectural divisor."""
        divisor = get_model_divisor(model_config.family)
        for dim, size in enumerate(data_config.patch_size):
            assert size % divisor == 0, (
                f"Patch dim {dim}={size} not divisible by {divisor} "
                f"(required by {model_config.family})"
            )

    def test_no_default_resampling(self, data_config):
        """Default should be native resolution (no resampling)."""
        assert data_config.voxel_spacing == (0.0, 0.0, 0.0), (
            "Default voxel_spacing must be (0,0,0) (native resolution). "
            "Resampling is a scientific decision — enable explicitly."
        )

    def test_cache_rate_fits_ram(self, dataset_profile, hardware_budget, data_config):
        """Cache rate should not exceed available RAM."""
        cached_size_gb = dataset_profile.total_size_gb * data_config.cache_rate
        assert cached_size_gb < hardware_budget.ram_available_gb * 0.7, (
            f"Cache would use {cached_size_gb:.1f} GB but only "
            f"{hardware_budget.ram_available_gb:.1f} GB available"
        )
```

### MONAI DataAnalyzer Integration

For comprehensive profiling (intensity stats, label distributions, spacing analysis), delegate to MONAI's `DataAnalyzer`:

```python
from monai.apps.auto3dseg import DataAnalyzer

def run_monai_data_analysis(datalist_json: Path, data_root: Path) -> dict:
    """Run MONAI's DataAnalyzer for comprehensive dataset profiling."""
    analyzer = DataAnalyzer(
        datalist=str(datalist_json),
        dataroot=str(data_root),
        output_path="./data_stats.yaml",
        label_key="label",
    )
    return analyzer.get_all_case_stats()
```

This provides: per-case shape/spacing, intensity histograms, label distributions, foreground statistics — all stored as a YAML artifact.

---

## 4. Prefect Flow Architecture: Division of Labor

### Why Division of Labor Matters — Even for Solo Researchers

The multi-flow architecture is not just for teams. Even a solo researcher benefits from
Prefect's flow separation because:

1. **Independent testability** — Each flow has its own tests, can be run in isolation
2. **Crash resilience** — If training crashes, the data profile is already cached
3. **Iteration speed** — Change evaluation logic without re-training
4. **Mental model** — Forces structured thinking about the ML lifecycle
5. **Portfolio demonstration** — Shows production MLOps maturity (Google MLOps Level 2+)
6. **Growth path** — When the solo researcher becomes a lab group, the architecture scales

### Lessons from foundation-PLR

The foundation-PLR Prefect implementation established excellent patterns:

| Pattern | foundation-PLR | Improvement for minivess-mlops |
|---------|---------------|-------------------------------|
| `_prefect_compat.py` | No-op decorators when disabled | **Adopt as-is** — proven pattern |
| Two-block decoupling | Extraction → DuckDB → Analysis | **Extend to 4 persona-based flows** |
| `PREFECT_DISABLED=1` | Docker default | **Adopt** — CI and simple runs don't need orchestration |
| Git-based deployment | `flow.from_source(GitRepository)` | **Extend** with Docker work pools for cloud |
| Process work pool | Local subprocess only | **Add Docker and Kubernetes work pools** |
| No task caching | All tasks re-run every time | **Add `cache_policy=INPUTS`** for expensive data profiling |
| No scheduling | CLI-only triggers | **Add DVC webhook → Prefect automation** for data ingestion |
| Hydra config integration | `cfg["PREFECT"]["PROCESS_FLOWS"]` | **Adopt** — familiar pattern |
| Heartbeat logging | Progress every 30s in long tasks | **Adopt + extend** with Prefect progress artifacts |

### Four-Persona Flow Architecture

Each Prefect flow maps to a **role** in an academic ML team. In a solo setup, one
person wears all hats. In a lab group, different people own different flows. The
**contract between flows is always MLflow** — artifacts stored there are accessible
to all subsequent flows regardless of who runs them.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRIGGER: DVC push (new data) OR manual `just experiment`             │
└───────┬─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  FLOW 1: DATA ENGINEERING                                            │
│  Persona: Data Engineer / Data Curator                               │
│  ─────────────────────────────────────────────────────────────────── │
│  Tasks:                                                              │
│  • Ingest raw data from any source (S3, local disk, collaborator)    │
│  • Quality checks: validate NIfTI headers, voxel spacing, labels     │
│  • MONAI DataAnalyzer: shape/spacing/intensity stats per volume      │
│  • Compute DatasetProfile (min/max/median shapes, anisotropy flags)  │
│  • Flag outliers (e.g., mv02 with 4.97μm spacing)                   │
│  • Human-in-the-loop: Slicer/Label Studio for annotation QC         │
│  • Version proofread data via DVC                                    │
│  • Log "data-profile" experiment to MLflow                           │
│                                                                      │
│  Contract OUT → MLflow: DatasetProfile artifact + data hash          │
│  Caching: cache_policy=INPUTS (skip if DVC hash unchanged)           │
└───────┬──────────────────────────────────────────────────────────────┘
        │ MLflow run_id (data profile)
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  FLOW 2: MODEL TRAINING                                              │
│  Persona: Data Scientist / Medical Image Analysis Expert / MONAI     │
│  ─────────────────────────────────────────────────────────────────── │
│  Tasks:                                                              │
│  • Read DatasetProfile from MLflow (Flow 1 output)                   │
│  • Preflight: detect hardware → compute adaptive profile             │
│  • Validate patches fit dataset (pre-training guard)                 │
│  • For each loss function × fold:                                    │
│    - Build CacheDataset + ThreadDataLoader                           │
│    - Train with sliding window validation                            │
│    - Save best checkpoint                                            │
│  • Post-training MetricsReloaded evaluation with bootstrap CIs       │
│  • Cross-loss comparison table                                       │
│  • Checkpoint recovery on crash (CheckpointManager)                  │
│                                                                      │
│  Contract IN  ← MLflow: DatasetProfile (from Flow 1)                │
│  Contract OUT → MLflow: model checkpoints + evaluation metrics       │
│  Contract OUT → MLflow Model Registry: best model per loss function  │
└───────┬──────────────────────────────────────────────────────────────┘
        │ MLflow run_ids (training + evaluation)
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  FLOW 3: MODEL ANALYSIS & ENSEMBLING                                 │
│  Persona: ML Engineer / Research Scientist                           │
│  ─────────────────────────────────────────────────────────────────── │
│  Tasks:                                                              │
│  • Read training artifacts from MLflow (Flow 2 output)               │
│  • WeightWatcher spectral analysis of trained models                 │
│  • Deepchecks Vision model validation                                │
│  • Calibration analysis (MAPIE, netcal)                              │
│  • Ensembling experiments:                                           │
│    - Model soup (weight averaging across losses)                     │
│    - Majority voting                                                 │
│    - Conformal prediction sets                                       │
│  • XAI: Captum attribution maps                                     │
│  • Generate DuckDB analytics from MLflow runs                        │
│  • Summary report: best config, ablation tables, figures             │
│                                                                      │
│  Contract IN  ← MLflow: model checkpoints + metrics (from Flow 2)   │
│  Contract OUT → MLflow: ensemble models + analysis artifacts         │
│  Contract OUT → DuckDB: structured analytics for querying            │
└───────┬──────────────────────────────────────────────────────────────┘
        │ MLflow Model Registry: production-ready model
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  FLOW 4: DEPLOYMENT & SERVING                                        │
│  Persona: MLOps / Platform Engineer                                  │
│  ─────────────────────────────────────────────────────────────────── │
│  Tasks:                                                              │
│  • Read best model from MLflow Model Registry (Flow 3 output)        │
│  • Export to ONNX (ONNX Runtime for serving)                         │
│  • Build BentoML service (Bento archive)                             │
│  • Build Docker image for serving                                    │
│  • Deploy based on target:                                           │
│    - Local: Docker container on workstation                          │
│    - Intranet: Docker Compose on lab server                          │
│    - Cloud: Kubernetes / Nomad / SkyPilot Skyserve                   │
│  • Smoke test: send test volume, verify prediction                   │
│  • (Future) Cloudflare CDN + auto-scaling for public serving         │
│  • (Future) Cost optimization via SkyPilot spot instances            │
│  • Monitoring: Prometheus metrics, Evidently drift detection         │
│                                                                      │
│  Contract IN  ← MLflow: ONNX model + metadata (from Flow 3)         │
│  Contract OUT → Serving endpoint (REST API)                          │
│  Contract OUT → Prometheus: inference latency, throughput metrics     │
└───────────────────────────────────────────────────────────────────────┘
```

### MLflow as the Universal Contract

```
┌──────────┐     MLflow     ┌──────────┐     MLflow     ┌──────────┐     MLflow     ┌──────────┐
│  Flow 1  │ ──artifacts──→ │  Flow 2  │ ──artifacts──→ │  Flow 3  │ ──registry──→ │  Flow 4  │
│  Data    │   (profiles)   │  Train   │  (checkpoints) │  Analyze │   (model)     │  Deploy  │
└──────────┘                └──────────┘                └──────────┘                └──────────┘
                                                                                         │
     ↑ DVC hash triggers                                                                 │
     │ revalidation                                                                      ▼
  ┌──────────┐                                                                    ┌──────────────┐
  │ Raw Data │                                                                    │ REST Endpoint │
  │ (S3/disk)│                                                                    │ (BentoML)     │
  └──────────┘                                                                    └──────────────┘
```

**Why MLflow as contract (not filesystem paths or Prefect results)**:
- MLflow backends are flexible: local filesystem, PostgreSQL, S3, Azure Blob
- Artifacts are versioned, searchable, and comparable via MLflow UI
- Model Registry provides staging (None → Staging → Production)
- Works identically on local machine, intranet server, and cloud
- Each flow is independently deployable — different machines can run different flows

### Deployment Maturity Ladder

The architecture is designed for growth — start simple, scale up:

| Maturity Level | What's Running | Where | Prefect Mode |
|----------------|---------------|-------|-------------|
| **L0: Script** | `just experiment` on laptop | Local | `PREFECT_DISABLED=1` |
| **L1: Orchestrated** | 4 flows, local Prefect server | Local | `prefect server start` |
| **L2: Containerized** | Docker training image, Docker work pool | Local/Server | Docker work pool |
| **L3: Team** | Prefect server on lab server, multiple users | Intranet | Docker Compose profile |
| **L4: Cloud** | K8s work pool, auto-scaling inference | Cloud | K8s/Nomad work pool |
| **L5: Public** | CDN + auto-balancer + spot instances | Multi-cloud | SkyPilot Skyserve |

Each level is additive — L0 code still works at L5.

### Future: 4 Personas as Autonomous LangGraph Agents

> **GitHub Issue**: [#61 — P1: Multi-agent orchestration](https://github.com/minivess-mlops/minivess-mlops/issues/61)

The 4 Prefect flows map naturally to 4 autonomous LangGraph agents. Each agent
wraps its corresponding Prefect flow but adds LLM reasoning for context-aware
re-runs, result interpretation, edge case handling, and human-readable reporting.
A LangGraph Supervisor Agent coordinates the 4 persona agents via shared state,
with MLflow as the universal artifact contract.

This is a **future PRD decision** — design after the 4 Prefect flows are stable.
The current Prefect flow architecture is deliberately designed so that the
agent layer can be added on top without modifying the flow implementations.

### Artifact Storage Strategy

**Decision matrix**: Where should artifacts live?

| Artifact Type | Storage | Rationale |
|---------------|---------|-----------|
| Dataset profiles (YAML/JSON) | **MLflow** as "data-profile" experiment | Versioned, searchable, comparable across datasets |
| Model checkpoints (.pth) | **MLflow** model registry | Standard practice, enables model staging |
| Evaluation metrics (JSON) | **MLflow** as run metrics/artifacts | Enables MLflow comparison UI |
| Training logs (CSV) | **Local filesystem** + MLflow artifact | Filesystem for real-time monitoring, MLflow for archival |
| System metrics (CSV) | **Local filesystem** only | Ephemeral, only useful for debugging |
| Cross-flow pointers | **Prefect Variables** (JSON) | Lightweight key-value store for run_id passing |
| Summary reports | **Prefect Markdown artifacts** | Visible in Prefect UI for quick review |
| Large binary data (NIfTI) | **DVC** + filesystem | Too large for MLflow, already DVC-managed |

**Cross-flow data passing pattern**:

```python
from prefect.variables import Variable

# Flow 1 stores its MLflow run_id
@task
def save_profile_pointer(run_id: str, dataset_hash: str):
    Variable.set(
        "latest_data_profile",
        json.dumps({"run_id": run_id, "dataset_hash": dataset_hash}),
        overwrite=True,
    )

# Flow 3 reads it
@task
def load_data_profile() -> dict:
    pointer = json.loads(Variable.get("latest_data_profile"))
    return mlflow.artifacts.download_artifacts(
        run_id=pointer["run_id"],
        artifact_path="dataset_profile.yaml",
    )
```

### DVC → Prefect Trigger Pattern

No official `prefect-dvc` integration exists. Recommended approach:

**Option A: Git webhook (preferred for GitHub-hosted repos)**
```
DVC push → .dvc files committed to Git → GitHub webhook →
Prefect webhook trigger → automation → run data-profiling deployment
```

**Option B: CI/CD trigger (preferred for local/intranet)**
```
DVC push → pre-commit hook or CI step → emit Prefect event via API →
automation → run data-profiling deployment
```

**Option C: Polling (simplest, least infrastructure)**
```
Scheduled Prefect flow (hourly) → check DVC lock file hash →
if changed → run data-profiling flow as subflow
```

### Decision Matrix: DVC Trigger Approach

| Criterion | Git Webhook (A) | CI/CD Trigger (B) | Polling (C) |
|-----------|-----------------|--------------------|----|
| Latency | Low (seconds) | Medium (CI queue) | High (polling interval) |
| Infrastructure | GitHub + Prefect Cloud/Server | CI runner + Prefect API | Prefect scheduler only |
| Works offline/intranet | No (needs GitHub) | Yes (local CI) | Yes (Prefect server only) |
| Complexity | Medium | Medium | Low |
| Reliability | High | High | Medium (missed changes between polls) |
| **Recommendation** | Cloud deployments | On-prem/intranet | Development/simple setups |

---

## 5. Multi-Environment Deployment

### Three Deployment Targets

#### Target 1: Local Workstation (PhD researcher's machine)

```bash
# Install
git clone ... && cd minivess-mlops && uv sync

# Run — everything adaptive, zero config
just experiment --config configs/experiments/dynunet_losses.yaml

# Behind the scenes:
# 1. Prefect runs locally (no server needed, PREFECT_DISABLED compatible)
# 2. Hardware detected: RTX 2070, 64GB RAM → gpu_low tier
# 3. Dataset scanned: min_z=5 → patch_z capped at 4 (divisible by 4)
# 4. Docker services started (just dev)
# 5. Training runs with sliding window inference
# 6. Results in MLflow at localhost:5000
```

#### Target 2: Intranet / On-Prem Server

```bash
# SSH to server, clone repo
# Server has: 4x A100 80GB, 512GB RAM, NVMe storage

just experiment --config configs/experiments/dynunet_losses.yaml
# Auto-detects: cloud_multi tier, cache_rate=1.0, large patches

# OR: with Prefect server for team visibility
docker compose --profile prefect up -d  # Starts Prefect server + workers
just experiment-deployed --config configs/experiments/dynunet_losses.yaml
# Creates Prefect deployment, runs via work pool
# Team can view progress at http://server:4200
```

#### Target 3: Ephemeral Cloud Instance

```bash
# Cloud instance with mounted data drive
docker run --gpus all \
  -v /mnt/data:/data:ro \
  -v /mnt/output:/output \
  ghcr.io/petteriTeikari/minivess-train:latest \
  --config configs/experiments/dynunet_losses.yaml
# Auto-detects hardware, runs, writes results to /output
# No Prefect server needed (PREFECT_DISABLED=1 in Docker)
```

### Prefect Work Pool Configuration

```yaml
# configs/prefect/work_pools.yaml (reference, not Prefect-native format)
work_pools:
  local-process:
    type: process
    description: "Local machine execution"
    base_job_template: null  # Direct process execution

  local-docker:
    type: docker
    description: "Docker execution on local machine"
    base_job_template:
      image: minivess-train:latest
      volumes:
        - "${DATA_DIR}:/data:ro"
        - "${OUTPUT_DIR}:/output"
      env:
        PREFECT_DISABLED: "0"  # Enable Prefect inside container

  k8s-gpu:
    type: kubernetes
    description: "Kubernetes GPU node pool"
    base_job_template:
      resources:
        requests:
          nvidia.com/gpu: "1"
          memory: "32Gi"
        limits:
          nvidia.com/gpu: "1"
          memory: "64Gi"
      node_selector:
        gpu: "true"
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: minivess-data
```

### Self-Hosted Prefect Server (Docker Compose)

Add to existing `deployment/docker-compose.yml`:

```yaml
# Profile: prefect (optional, for team/server deployments)
services:
  prefect-server:
    image: prefecthq/prefect:3-latest
    command: prefect server start --host 0.0.0.0
    ports:
      - "4200:4200"
    environment:
      PREFECT_SERVER_DATABASE_CONNECTION_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/prefect
    depends_on:
      postgres:
        condition: service_healthy
    profiles:
      - prefect

  prefect-worker:
    image: minivess-train:latest
    command: prefect worker start --pool local-process
    environment:
      PREFECT_API_URL: http://prefect-server:4200/api
    depends_on:
      - prefect-server
    profiles:
      - prefect
```

---

## 6. Prefect Compatibility Layer (Adopted from foundation-PLR)

```python
# src/minivess/orchestration/_prefect_compat.py
"""Prefect compatibility layer — flows work with or without Prefect installed.

Pattern proven in foundation-PLR: when PREFECT_DISABLED=1 or prefect not installed,
@flow and @task decorators become no-ops. All orchestration code runs as plain Python.
"""
from __future__ import annotations

import importlib
import os
from typing import Any, TypeVar

F = TypeVar("F")

USE_PREFECT = os.environ.get("PREFECT_DISABLED", "").lower() not in ("1", "true", "yes")
PREFECT_AVAILABLE = False


def _noop_decorator(fn=None, **kwargs):
    """No-op decorator that preserves the original function."""
    if fn is None:
        return lambda f: f
    return fn


task: Any = _noop_decorator
flow: Any = _noop_decorator


def get_run_logger():
    """Return Prefect logger or stdlib logger."""
    import logging
    return logging.getLogger("minivess.orchestration")


if USE_PREFECT:
    try:
        _prefect = importlib.import_module("prefect")
        task = _prefect.task
        flow = _prefect.flow
        get_run_logger = _prefect.get_run_logger
        PREFECT_AVAILABLE = True
    except ImportError:
        pass
```

### Improvements Over foundation-PLR

| Improvement | Details |
|-------------|---------|
| **Task caching** | Add `cache_policy=INPUTS` for data profiling tasks (foundation-PLR doesn't use caching) |
| **Progress artifacts** | Use `create_progress_artifact()` for long training runs (new in Prefect 3.x) |
| **Structured results** | Use `create_table_artifact()` for cross-loss comparison tables |
| **Event-driven triggers** | Emit custom events on data changes (foundation-PLR uses CLI only) |
| **Docker work pools** | Support GPU Docker execution (foundation-PLR uses process pools only) |
| **Kubernetes support** | Work pool templates for K8s GPU scheduling |
| **Transaction support** | Use Prefect 3.x transactions for DVC+Git atomic operations |
| **Health check recipe** | `just prefect-health` to verify server + workers |

---

## 7. Decision Matrices (Open-Ended Research Questions)

### D1: Resampling Strategy

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Native resolution (default)** | No information loss, simplest, fastest | Variable voxel sizes in batch (handled by patches) | Default for all datasets |
| **Spacingd to isotropic** | Uniform voxel grid, standard in nnU-Net | OOM risk for large upsampling factors, information created artificially | When physics requires isotropic (e.g., diffusion MRI) |
| **Super-resolution model** | Learned upsampling, potentially higher quality | Adds complexity, introduces model bias, expensive | Research question — researcher decides |
| **Latent diffusion upsampling** | State-of-art quality for extreme anisotropy | Very expensive, not validated for medical segmentation | Exploratory research only |

**Decision**: Native resolution is the default. Resampling is opt-in via YAML config. This is a scientific question — the platform provides the knobs, the researcher turns them.

### D2: Patch Size Selection Strategy

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Min-volume constrained (proposed)** | Guarantees all volumes fit, no padding needed | Small patches for datasets with tiny volumes | Default — safest |
| **Median-volume + SpatialPadd** | Larger patches for most volumes, pad the small ones | Introduces zero-padding artifacts in small volumes | When few outlier small volumes exist |
| **Auto3DSeg heuristic** | Battle-tested, uses max shape | Doesn't protect individual small volumes | When using full Auto3DSeg pipeline |
| **Fixed per-profile** | Simple, predictable | May not fit all datasets | Legacy compatibility |

**Decision**: Default to min-volume constrained. Log a warning when SpatialPadd is applied (researcher should know which volumes are being padded and by how much).

### D3: Cache Rate Estimation

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Adaptive (RAM-based, proposed)** | Maximizes speed without OOM | Estimate may be imprecise (other processes use RAM) | Default |
| **Fixed 1.0 (cache all)** | Fastest training | Requires enough RAM for entire dataset | Small datasets, high-RAM machines |
| **Fixed 0.0 (no cache)** | Minimum RAM, works everywhere | Slowest training (re-read from disk each epoch) | Debug mode, very large datasets |
| **MONAI runtime_cache=True** | Progressive filling, no init spike | Same total memory as cache_rate=1.0 eventually | Always use with CacheDataset (orthogonal to rate) |

**Decision**: Use adaptive rate as default, always with `runtime_cache=True`. Warn if estimated cache would use >70% of available RAM.

### D4: Prefect Server vs. Serverless

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **No server (PREFECT_DISABLED=1)** | Simplest, zero infrastructure | No UI, no scheduling, no team visibility | Local dev, Docker, CI |
| **Local Prefect server** | UI at localhost:4200, run history | Postgres dependency, maintenance | Single-user workstation |
| **Docker Compose Prefect** | Team-visible, persistent, self-hosted | More infrastructure to maintain | On-prem servers, lab machines |
| **Prefect Cloud** | Managed, SSO, RBAC, audit logs | Cloud dependency, cost ($100-400/mo) | Production, multi-team |

**Decision**: Default to no server (`PREFECT_DISABLED=1`). Provide Docker Compose profile for team deployments. Document Prefect Cloud as option for production.

### D5: Cross-Flow Artifact Passing

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **MLflow run_id via Prefect Variables** | Lightweight, native to both tools | Manual wiring | Default for minivess-mlops |
| **Shared filesystem paths** | Simple, fast | Not portable across machines | Single-machine setups |
| **S3/MinIO URIs** | Portable, scalable | Requires object store | Cloud and multi-machine |
| **Prefect result serialization** | Built-in, automatic | Not designed for large files (>100MB) | Small metadata only |

**Decision**: Use MLflow as the artifact store (already integrated), pass MLflow run_ids between flows via Prefect Variables. For large binaries (NIfTI), use DVC + filesystem paths.

---

## 8. Implementation Phases

### Phase 1: Dataset Profiler + Adaptive Profiles (Immediate)

**TDD targets**:
- [ ] `tests/v2/unit/test_data_profiler.py` — DatasetProfile computation
- [ ] `tests/v2/unit/test_adaptive_profiles.py` — Hardware detection, patch size adaptation
- [ ] `tests/v2/unit/test_patch_validation.py` — Patch-fits-all-volumes check

**Implementation**:
- [ ] `src/minivess/data/profiler.py` — Dataset scanner using MONAI DataAnalyzer
- [ ] `src/minivess/config/adaptive_profiles.py` — HardwareBudget detection + adaptive computation
- [ ] `configs/model_profiles/dynunet.yaml` — Model-specific memory profile
- [ ] Update `src/minivess/config/compute_profiles.py` — Add `auto` profile that delegates to adaptive

### Phase 2: Prefect Integration (Required Dependency, Short-term)

Prefect is a **required dependency** (not optional). The `_prefect_compat.py` pattern
is retained for CI/testing environments (`PREFECT_DISABLED=1`), but production
runs MUST use Prefect for the 4-persona flow architecture.

**TDD targets**:
- [ ] `tests/v2/unit/test_prefect_compat.py` — No-op decorators, PREFECT_DISABLED behavior

**Implementation**:
- [ ] `src/minivess/orchestration/_prefect_compat.py` — Adopted from foundation-PLR
- [ ] `src/minivess/orchestration/__init__.py`
- [ ] Add `prefect>=3.0.11` to `pyproject.toml` **core** dependencies (not optional)

### Phase 3: Data Profiling Flow (Medium-term)

**TDD targets**:
- [ ] `tests/v2/integration/test_data_profiling_flow.py` — End-to-end flow test

**Implementation**:
- [ ] `src/minivess/orchestration/flows/data_profiling.py` — Flow 1
- [ ] MLflow "data-profile" experiment logging
- [ ] Prefect task caching for re-scans

### Phase 4: Training + Evaluation Flows (Medium-term)

**Implementation**:
- [ ] `src/minivess/orchestration/flows/training.py` — Flow 3 (wraps train_monitored.py)
- [ ] `src/minivess/orchestration/flows/evaluation.py` — Flow 4 (wraps evaluation pipeline)
- [ ] `src/minivess/orchestration/flows/experiment.py` — Master flow (chains 1-4)

### Phase 5: Multi-Environment Docker + Prefect Server (Longer-term)

**Implementation**:
- [ ] `deployment/Dockerfile.train` — uv-based training image
- [ ] Docker Compose `prefect` profile (server + worker)
- [ ] `configs/prefect/work_pools.yaml` — Reference configurations
- [ ] `just prefect-up` / `just prefect-health` recipes

### Phase 6: CLAUDE.md Updates (Immediate, parallel with Phase 1)

- [ ] Add Design Goal #1 (EXCELLENT DevEx) to root CLAUDE.md
- [ ] Add adaptive profiles documentation
- [ ] Add Prefect orchestration section
- [ ] Document "scientific decisions stay with researcher" principle

---

## 9. Factual Correctness Notes (Reviewer Findings)

### Verified Facts
- MONAI `DataAnalyzer` can be used standalone (confirmed in MONAI source)
- MONAI has no built-in adaptive `cache_rate` estimation (confirmed)
- No official `prefect-mlflow` or `prefect-dvc` integration packages exist
- Prefect 3.x open-sourced events and automations (previously Cloud-only)
- Prefect does not manage GPU allocation — delegates to infrastructure (K8s, ECS)
- `RandCropByPosNegLabeld` with `allow_smaller=False` raises ValueError if patch > volume
- DynUNet divisibility = product of all strides across encoder levels
- Prefect artifacts cannot store binary files — only metadata, tables, markdown, links

### Caveats
- VRAM estimation (`vram_per_voxel_train_bytes`) is empirical, not analytical. Need benchmarking.
- Auto3DSeg's patch heuristic uses max_shape, NOT min_shape — does not protect small volumes
- Prefect Docker work pools don't natively support `--gpus all` — needs custom template
- `cache_policy=INPUTS` requires `PREFECT_RESULTS_PERSIST_BY_DEFAULT=true` (off by default)
- lakeFS acquired iterative.ai (DVC parent) in Nov 2025 — DVC's future uncertain, monitor

### Open Questions (Require Benchmarking)
1. What is the actual VRAM cost per voxel for DynUNet training with AMP?
2. At what `cache_rate` does CacheDataset with `runtime_cache=True` stabilize RAM usage?
3. How much overhead does Prefect add to training loop execution? (Expected: negligible for epoch-level tasks)
4. What is the optimal sliding window overlap for MiniVess vessel segmentation? (Currently 0.25, may benefit from 0.5)
