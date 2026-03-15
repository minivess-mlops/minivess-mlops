# Deployment — Docker Image Anatomy

## Image Size Breakdown: `minivess-base:latest` (21.4 GB)

The GPU base image contains everything needed to train 3D biomedical segmentation
models on cloud GPU instances (RunPod, AWS, GCP) without any runtime installation.

### Why 21.4 GB?

| Component | Size | % | Source |
|-----------|------|---|--------|
| CUDA runtime base | ~4.5 GB | 21% | [`nvidia/cuda:12.6.3-runtime-ubuntu24.04`](https://hub.docker.com/r/nvidia/cuda) |
| PyTorch CUDA wheels (`nvidia/`) | 4.3 GB | 20% | [PyTorch](https://pytorch.org/) bundles cublas, cufft, curand, cusolver, nccl |
| PyTorch framework (`torch/`) | 1.8 GB | 8% | [`torch==2.10.0+cu128`](https://pytorch.org/) |
| [Triton](https://github.com/triton-lang/triton) JIT compiler | 642 MB | 3% | LLVM-based GPU kernel compiler, bundled with PyTorch |
| Medical imaging | ~500 MB | 2% | [SimpleITK](https://simpleitk.org/) (264 MB), [MONAI](https://monai.io/) (15 MB), [nibabel](https://nipy.org/nibabel/), [TorchIO](https://torchio.readthedocs.io/) |
| ML ecosystem | ~500 MB | 2% | [transformers](https://huggingface.co/docs/transformers) (102 MB), [scikit-learn](https://scikit-learn.org/) (48 MB), [statsmodels](https://www.statsmodels.org/) |
| Data & serialization | ~500 MB | 2% | [plotly](https://plotly.com/python/) (196 MB), [pyarrow](https://arrow.apache.org/docs/python/) (149 MB), [pandas](https://pandas.pydata.org/) (76 MB) |
| MLOps tooling | ~250 MB | 1% | [MLflow](https://mlflow.org/) (54 MB), [Prefect](https://www.prefect.io/) (58 MB), [DVC](https://dvc.org/), [Optuna](https://optuna.org/) |
| ONNX | ~150 MB | 1% | [ONNX](https://onnx.ai/) (93 MB) + [ONNX Runtime](https://onnxruntime.ai/) (53 MB) |
| Scientific computing | ~300 MB | 1% | [scipy](https://scipy.org/) (111 MB), [numpy](https://numpy.org/) (42 MB), [llvmlite](https://llvmlite.readthedocs.io/) (162 MB) |
| Validation & monitoring | ~50 MB | <1% | [Evidently](https://www.evidentlyai.com/) (35 MB), [Deepchecks](https://deepchecks.com/) |
| System (Python 3.13, Ubuntu) | ~2.4 GB | 11% | [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) Python 3.13 on Ubuntu 24.04 |
| Filesystem overhead + metadata | ~3.2 GB | 15% | Docker layer unions, ext4 metadata, BuildKit cache |

**The dominant cost**: PyTorch + CUDA account for **52% of the image** (CUDA wheels 4.3 GB
+ PyTorch 1.8 GB + CUDA base 4.5 GB = 10.6 GB). This is normal — a comparable
[NVIDIA NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
is 15-20 GB before adding any application code.

### Three-Tier Base Image Hierarchy

Not everything needs PyTorch + CUDA. The project uses three base images:

```
Tier A (GPU):   nvidia/cuda 12.6 → minivess-base:latest        ~21 GB
                ├─ Used by: train, hpo, post_training, analyze,
                │  deploy, data, annotation, monailabel, acquisition
                └─ Contains: PyTorch, MONAI, CUDA, ALL Python deps

Tier B (CPU):   python:3.13-slim → minivess-base-cpu:latest     ~1.5-2.5 GB
                ├─ Used by: biostatistics
                └─ Contains: scipy, pandas, DuckDB, MLflow (NO PyTorch/CUDA)

Tier C (Light): python:3.13-slim → minivess-base-light:latest   ~1.0-1.5 GB
                ├─ Used by: dashboard, dashboard-api, pipeline
                └─ Contains: Prefect, FastAPI, MLflow (minimal deps)
```

### Multi-Stage Build Architecture

The Dockerfile uses a [multi-stage build](https://docs.docker.com/build/building/multi-stage/)
to keep the production image minimal:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: BUILDER (nvidia/cuda:12.6.3-devel-ubuntu24.04)     │
│                                                             │
│  Has: uv, git, curl, nvcc, CUDA headers, Python 3.13-dev   │
│  Does: uv sync --frozen --no-dev → builds /app/.venv       │
│  DISCARDED after build — never in production image          │
└─────────────────────────────────┬───────────────────────────┘
                                  │ COPY .venv, src/, configs/
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: RUNNER (nvidia/cuda:12.6.3-runtime-ubuntu24.04)    │
│                                                             │
│  Has: Python 3.13, .venv (on PATH), src/, configs/          │
│  NO: uv, git, curl, pip, nvcc, build tools, CUDA headers   │
│  User: minivess (UID 1000, non-root)                        │
│  Security: cap_drop ALL, no-new-privileges                  │
└─────────────────────────────────────────────────────────────┘
```

**What's NOT in the production image** (security hardening):
- No package manager (`uv`, `pip`, `conda`) — can't install new packages at runtime
- No VCS (`git`) — code is baked in, not cloned
- No network tools (`curl`, `wget`) — reduces attack surface
- No CUDA development headers — only runtime libraries
- No build tools (gcc, make) — compilation happened in builder stage

### Rebuilding

```bash
# Rebuild base image (only when pyproject.toml or Dockerfile.base changes):
make build-base-gpu

# Match host UID for bind-mount development:
DOCKER_BUILDKIT=1 docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
  -t minivess-base:latest -f deployment/docker/Dockerfile.base .

# Rebuild CPU and light tiers:
make build-bases

# Push to GHCR (for SkyPilot cloud runs):
make push-ghcr
```

### Registry Size vs Local Size

The GHCR registry stores compressed layers. Typical compression ratio for Python/CUDA
images is ~2-3x, so the ~21 GB local image is approximately **7-10 GB** when pushed
to GHCR. SkyPilot's `image_id: docker:ghcr.io/...` pulls and decompresses automatically.

### Key Dependencies and Their Roles

| Package | Version | Role | Link |
|---------|---------|------|------|
| PyTorch | 2.10.0 | Deep learning framework | [pytorch.org](https://pytorch.org/) |
| MONAI | 1.5.2 | Medical imaging framework (extends PyTorch) | [monai.io](https://monai.io/) |
| SimpleITK | latest | 3D medical image I/O and processing | [simpleitk.org](https://simpleitk.org/) |
| TorchIO | latest | 3D medical image augmentation | [torchio.readthedocs.io](https://torchio.readthedocs.io/) |
| transformers | latest | HuggingFace models (SAM3, VesselFM) | [huggingface.co](https://huggingface.co/docs/transformers) |
| MLflow | 3.x | Experiment tracking + model registry | [mlflow.org](https://mlflow.org/) |
| Prefect | 3.x | Workflow orchestration | [prefect.io](https://www.prefect.io/) |
| DVC | 3.x | Data version control (S3 backend) | [dvc.org](https://dvc.org/) |
| Optuna | latest | Hyperparameter optimization | [optuna.org](https://optuna.org/) |
| ONNX Runtime | latest | Model inference engine | [onnxruntime.ai](https://onnxruntime.ai/) |
| Evidently | latest | Data/model drift detection | [evidentlyai.com](https://www.evidentlyai.com/) |
| Deepchecks | latest | Model validation | [deepchecks.com](https://deepchecks.com/) |
| scipy | latest | Scientific computing | [scipy.org](https://scipy.org/) |
| pandas | latest | Data manipulation | [pandas.pydata.org](https://pandas.pydata.org/) |
| plotly | latest | Interactive visualization | [plotly.com](https://plotly.com/python/) |

See `pyproject.toml` for the complete dependency list with version constraints.
