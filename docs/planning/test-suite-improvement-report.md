# Test Suite Improvement Report — Model-Specific Testing

**Date**: 2026-03-20
**Trigger**: Debug factorial run found 3 model-specific bugs that should have been
caught locally, not on $0.28/hr cloud GPUs.
**Goal**: Design a model-specific test tier that catches construction, LoRA, adapter,
and integration bugs BEFORE cloud runs.

---

## 1. Problem Statement

The debug factorial run (2026-03-20) found 14/26 conditions failing due to bugs
that required ZERO cloud GPU time to diagnose:

| Glitch | Bug Type | Could Local Test Catch It? | VRAM Needed |
|--------|----------|---------------------------|-------------|
| #9: LoRA Conv2d | Type error in model construction | **YES — 0 MB VRAM** | CPU only |
| #10: mamba-ssm missing | Import error | **YES — 0 MB VRAM** | CPU only |
| #12: max_epochs=0 | Pydantic validation | **YES — 0 MB VRAM** | CPU only |

**All 3 bugs are catchable with CPU-only unit tests in < 1 second.**
Total cloud cost wasted on these bugs: ~$1-2 (14 failed VMs × setup time).
Total wall-clock wasted: ~7 hours (spot queue scheduling).

### Current Test Architecture

| Tier | Command | What Runs | Time | Catches Model Bugs? |
|------|---------|-----------|------|---------------------|
| Staging | `make test-staging` | No model loading | <3 min | **NO** — excludes `@model_loading` |
| Prod | `make test-prod` | Includes model loading | ~10 min | Partially — but not adapter construction |
| GPU | `make test-gpu` | SAM3 forward passes | GPU only | Yes — but requires 16+ GB VRAM |

**Gap**: There is no tier that tests model CONSTRUCTION (adapter instantiation,
LoRA application, weight loading) without running a full training loop.

---

## 2. Proposed New Test Tier: `make test-models`

### Design Principles

1. **Fast**: < 30 seconds total. No training loops, no data loading.
2. **CPU-friendly**: Most tests use tiny tensors (< 10 MB). No GPU required.
3. **Conditionally triggered**: Only runs when `src/minivess/adapters/` changes.
4. **Included in staging**: These are fast enough for every PR.
5. **VRAM-tiered**: Optional GPU tests gated by detected VRAM.

### Marker: `@pytest.mark.model_construction`

```python
# conftest.py addition
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "model_construction: Tests model adapter construction (fast, CPU-safe)"
    )
```

### What It Tests

| Test Category | What It Catches | Example from Debug Run | VRAM |
|--------------|-----------------|----------------------|------|
| **Adapter construction** | Import errors, missing deps | Glitch #10 (mamba-ssm) | 0 |
| **LoRA application** | Wrong layer types, rank issues | Glitch #9 (Conv2d) | 0 |
| **Config validation** | Pydantic constraints | Glitch #12 (max_epochs=0) | 0 |
| **Forward pass (tiny)** | Shape mismatches, dtype errors | — | ~50 MB |
| **Gradient flow** | Frozen/unfrozen params, NaN grads | — | ~100 MB |
| **Checkpoint save/load** | Serialization bugs, state dict keys | — | ~50 MB |

### Conditional Triggering

The user wants model tests to only run when adapters change. Two approaches:

**Option A: Makefile target with file-change detection**
```makefile
# Files that trigger model tests
MODEL_FILES := $(shell find src/minivess/adapters/ -name "*.py" -newer .model_test_stamp 2>/dev/null)

test-models:
    uv run pytest tests/v2/unit/adapters/ -v -m "model_construction"
    touch .model_test_stamp

test-models-if-changed:
    @if [ -n "$(MODEL_FILES)" ]; then \
        echo "Model files changed, running model tests..."; \
        $(MAKE) test-models; \
    else \
        echo "No model file changes, skipping model tests."; \
    fi
```

**Option B: Pre-commit hook (only on adapter changes)**
```yaml
# .pre-commit-config.yaml addition
- repo: local
  hooks:
    - id: model-construction-tests
      name: Model construction tests
      entry: uv run pytest tests/v2/unit/adapters/ -v -m "model_construction" --tb=short
      language: system
      files: ^src/minivess/adapters/
      types: [python]
      pass_filenames: false
```

**Recommendation**: Option B (pre-commit hook) is automatic and doesn't require
developer discipline. Option A is useful as a manual Makefile target too.

---

## 3. Test Categories in Detail

### 3.1 Adapter Construction Tests (CPU-only, 0 VRAM)

These test that every registered model adapter can be instantiated without errors.

```python
# tests/v2/unit/adapters/test_adapter_construction.py
import pytest
from minivess.adapters.model_builder import ADAPTER_REGISTRY

# Parametrize over ALL registered adapters
@pytest.mark.model_construction
@pytest.mark.parametrize("model_family", [
    "dynunet", "sam3_vanilla", "sam3_hybrid", "sam3_topolora",
    pytest.param("mambavesselnet", marks=pytest.mark.skipif(
        not _mamba_available(), reason="mamba-ssm not installed (needs nvcc)"
    )),
])
def test_adapter_registered(model_family):
    """Every model family in the factorial config must be in the registry."""
    assert model_family in ADAPTER_REGISTRY

@pytest.mark.model_construction
def test_all_factorial_models_registered():
    """All models in debug_factorial.yaml must be in the adapter registry."""
    import yaml
    config = yaml.safe_load(
        Path("configs/experiment/debug_factorial.yaml").read_text(encoding="utf-8")
    )
    for model in config["factors"]["model_family"]:
        assert model in ADAPTER_REGISTRY, f"{model} not in ADAPTER_REGISTRY"
```

### 3.2 LoRA-Specific Tests (CPU-only)

These would have caught Glitch #9:

```python
# tests/v2/unit/adapters/test_lora_application.py
import pytest
import torch.nn as nn
from minivess.adapters.sam3_topolora import _apply_lora_to_encoder, LoRALinear

@pytest.mark.model_construction
class TestLoRAApplication:
    def test_lora_only_targets_linear(self):
        """LoRA must NOT wrap Conv2d layers (Glitch #9)."""
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.Linear(64, 32),
                )
        encoder = MockEncoder()
        targets = _apply_lora_to_encoder(encoder, rank=8, alpha=16.0, dropout=0.0)

        # Linear layers should be wrapped
        assert any("linear" in t.lower() or "0" in t for t in targets)
        # Conv2d should NOT be wrapped
        for target in targets:
            module = encoder
            for part in target.split("."):
                module = getattr(module, part)
            assert isinstance(module, LoRALinear), f"{target} should be LoRALinear"

    def test_lora_skips_small_layers(self):
        """LoRA should skip layers with fewer features than rank."""
        class TinyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.tiny = nn.Linear(4, 4)
                self.big = nn.Linear(64, 64)
        encoder = TinyEncoder()
        targets = _apply_lora_to_encoder(encoder, rank=8, alpha=16.0, dropout=0.0)
        assert "tiny" not in targets
        assert "big" in targets

    def test_lora_linear_forward_shape(self):
        """LoRALinear must preserve input/output shapes."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8, alpha=16.0, dropout=0.0)
        import torch
        x = torch.randn(2, 64)
        y = lora(x)
        assert y.shape == (2, 32)

    def test_lora_linear_rejects_conv2d(self):
        """LoRALinear must raise TypeError for Conv2d."""
        conv = nn.Conv2d(64, 64, 3)
        with pytest.raises(TypeError, match="only supports nn.Linear"):
            LoRALinear(conv, rank=8, alpha=16.0, dropout=0.0)
```

### 3.3 Config Validation Tests (CPU-only)

Would have caught Glitch #12:

```python
# tests/v2/unit/config/test_training_config_validation.py
import pytest
from pydantic import ValidationError
from minivess.config.models import TrainingConfig

@pytest.mark.model_construction
class TestTrainingConfigValidation:
    def test_accepts_zero_epochs_for_zero_shot(self):
        """max_epochs=0 must be valid for zero-shot evaluation."""
        config = TrainingConfig(max_epochs=0)
        assert config.max_epochs == 0

    def test_accepts_positive_epochs(self):
        config = TrainingConfig(max_epochs=50)
        assert config.max_epochs == 50

    def test_rejects_negative_epochs(self):
        with pytest.raises(ValidationError):
            TrainingConfig(max_epochs=-1)

    def test_rejects_negative_batch_size(self):
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)
```

### 3.4 Forward Pass Tests (VRAM-tiered)

These test actual model inference with tiny inputs. Gated by VRAM detection.

```python
# tests/v2/unit/adapters/test_adapter_forward.py
import pytest
import torch

def _get_gpu_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_mem / (1024**3)

skip_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA GPU available"
)
skip_low_vram = pytest.mark.skipif(
    _get_gpu_vram_gb() < 4.0,
    reason=f"Requires >= 4 GB VRAM (detected {_get_gpu_vram_gb():.1f} GB)"
)

@pytest.mark.model_construction
@skip_no_gpu
@skip_low_vram
@pytest.mark.parametrize("model_family", ["dynunet", "sam3_hybrid"])
def test_forward_pass_tiny_input(model_family):
    """Model produces output with correct shape on tiny input."""
    from minivess.adapters.model_builder import build_adapter
    config = _make_test_config(model_family)
    model = build_adapter(config)
    model.eval()
    x = torch.randn(1, 1, 32, 32, 32, device="cuda")
    with torch.no_grad():
        y = model(x)
    assert y.shape[0] == 1  # batch
    assert y.shape[2:] == x.shape[2:]  # spatial dims preserved
```

### 3.5 Checkpoint Round-Trip Tests (CPU-only)

```python
@pytest.mark.model_construction
def test_checkpoint_save_load_roundtrip(tmp_path):
    """Model state_dict can be saved and reloaded correctly."""
    from minivess.adapters.model_builder import build_adapter
    config = _make_test_config("dynunet")
    model = build_adapter(config)
    path = tmp_path / "test_checkpoint.pth"
    torch.save(model.state_dict(), path)
    model2 = build_adapter(config)
    model2.load_state_dict(torch.load(path, weights_only=True))
    # Verify parameters match
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)
```

---

## 4. VRAM-Tiered Test Strategy

| Tier | VRAM | Marker | What Runs | Models |
|------|------|--------|-----------|--------|
| CPU | 0 GB | `@model_construction` | Construction, LoRA, config, imports | ALL |
| Low GPU | 2 GB | `@vram_2gb` | DynUNet forward, tiny inputs | dynunet |
| Medium GPU | 4 GB | `@vram_4gb` | SAM3 Hybrid forward, small inputs | sam3_hybrid |
| High GPU | 8 GB | `@vram_8gb` | SAM3 TopoLoRA forward, mini-training | sam3_topolora |
| Cloud GPU | 16+ GB | `@gpu_heavy` (existing) | Full SAM3 weights, real training | ALL SAM3 |

**Auto-detection in conftest.py**:
```python
def _detect_vram_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except Exception:
        return 0.0

DETECTED_VRAM_GB = _detect_vram_gb()

def pytest_collection_modifyitems(config, items):
    """Auto-skip tests requiring more VRAM than available."""
    vram_markers = {
        "vram_2gb": 2.0,
        "vram_4gb": 4.0,
        "vram_8gb": 8.0,
    }
    for item in items:
        for marker_name, required_vram in vram_markers.items():
            if marker_name in item.keywords and DETECTED_VRAM_GB < required_vram:
                item.add_marker(pytest.mark.skip(
                    reason=f"Requires >= {required_vram} GB VRAM "
                           f"(detected {DETECTED_VRAM_GB:.1f} GB)"
                ))
```

---

## 5. Conditional Execution Strategy

The user wants model tests to run only when adapters change, not on every commit.

### File-Path Based Triggering

```
src/minivess/adapters/          → triggers: test-models
src/minivess/config/models.py   → triggers: test-models (config validation)
src/minivess/pipeline/trainer.py → triggers: test-models (checkpoint tests)
deployment/docker/Dockerfile.*  → triggers: Docker build tests (existing)
deployment/skypilot/*.yaml      → triggers: SkyPilot YAML tests (existing)
```

### Integration into Existing Tiers

```makefile
# Makefile additions
test-models:
	uv run pytest tests/v2/unit/adapters/ tests/v2/unit/config/ \
	    -v -m "model_construction" --tb=short

test-staging: test-models   # Include model tests in staging (they're fast enough)
```

**Decision**: Since model construction tests are CPU-only and < 5 seconds total,
they should be included in `make test-staging` by default. The pre-commit hook
provides an additional guard specifically when adapter files change.

---

## 6. Beyond Unit Tests: Comprehensive MLOps Testing

### 6.1 Property-Based Testing (Hypothesis)

Use `hypothesis` to generate random model configs and verify invariants:

```python
from hypothesis import given, strategies as st

@given(
    in_channels=st.integers(1, 3),
    num_classes=st.integers(1, 5),
    spatial_dims=st.just(3),
)
def test_dynunet_output_channels_match_num_classes(in_channels, num_classes, spatial_dims):
    """Output channels must always equal num_classes."""
    config = ModelConfig(
        family="dynunet", in_channels=in_channels,
        out_channels=num_classes, spatial_dims=spatial_dims
    )
    model = build_adapter(config)
    # ... verify output shape
```

### 6.2 Data Validation Testing (Pandera / Great Expectations)

```python
def test_training_data_schema():
    """MiniVess training data must have expected shape and dtype."""
    # Verify NIfTI files have shape (H, W, D) with dtype float32
    # Verify labels are binary {0, 1}
    # Verify image/label pairs have matching shapes
```

### 6.3 Pipeline Integration Tests

```python
def test_train_to_posttraining_handoff(tmp_path):
    """Training flow artifacts are discoverable by post-training flow."""
    # 1. Run mini training (1 epoch, synthetic data)
    # 2. Verify MLflow run has checkpoint_dir_fold_0 tag
    # 3. Verify find_fold_checkpoints() returns the checkpoint
    # 4. Verify checkpoint file is loadable
```

### 6.4 Mutation Testing

Use `mutmut` or `cosmic-ray` to verify test quality:

```bash
# Are our LoRA tests actually catching bugs?
mutmut run --paths-to-mutate=src/minivess/adapters/sam3_topolora.py \
           --tests-dir=tests/v2/unit/adapters/
```

### 6.5 Spurious File Detection

Add a test that catches the `file:` directory bug:

```python
def test_no_spurious_directories_in_repo_root():
    """Test runs must not create directories in the repo root."""
    repo_root = Path(__file__).parents[3]
    spurious = [
        p for p in repo_root.iterdir()
        if p.is_dir() and p.name.endswith(":")
    ]
    assert not spurious, f"Spurious directories found: {spurious}"
```

---

## 7. Summary: Test Pyramid for Model Testing

```
                    ┌─────────────┐
                    │  GPU Instance│  make test-gpu (RunPod)
                    │  Full SAM3   │  SAM3 real weights, real training
                    │  16+ GB VRAM │  ~30 min, ~$0.50
                    ├─────────────┤
                 ┌──┤  Local GPU   │  @vram_2gb, @vram_4gb, @vram_8gb
                 │  │  Forward pass│  Real models, tiny inputs
                 │  │  2-8 GB VRAM │  ~30 sec
                 │  ├─────────────┤
              ┌──┤  │  Model Tests │  @model_construction (NEW)
              │  │  │  Construction│  Adapter build, LoRA, config, imports
              │  │  │  CPU only    │  ~5 sec
              │  │  ├─────────────┤
           ┌──┤  │  │  Unit Tests  │  make test-staging (existing)
           │  │  │  │  5400+ tests │  Config, pipeline, observability
           │  │  │  │  No models   │  ~3 min
           └──┴──┴──┴─────────────┘
```

---

## 8. Implementation Plan

| # | Task | Effort | Priority | Catches |
|---|------|--------|----------|---------|
| 1 | Add `@model_construction` marker to conftest.py | 15 min | P0 | — |
| 2 | Write adapter construction tests (parametric) | 1 hr | P0 | Glitch #10 |
| 3 | Write LoRA application tests | 30 min | P0 | Glitch #9 |
| 4 | Write config validation tests | 15 min | P0 | Glitch #12 |
| 5 | Write checkpoint round-trip tests | 30 min | P1 | — |
| 6 | Add VRAM detection to conftest.py | 30 min | P1 | — |
| 7 | Add pre-commit hook for adapter changes | 15 min | P1 | — |
| 8 | Add `make test-models` Makefile target | 10 min | P1 | — |
| 9 | Write forward pass tests (VRAM-tiered) | 1 hr | P2 | Shape bugs |
| 10 | Add spurious directory detection test | 10 min | P2 | file: bug |
| 11 | Property-based testing with Hypothesis | 2 hrs | P3 | Edge cases |
| 12 | Mutation testing setup | 1 hr | P3 | Test quality |

**Total P0+P1**: ~3 hours. **Catches all 3 debug-run bugs locally.**

---

## References

### PyTorch Testing Best Practices
- [PyTorch Official Testing Docs](https://docs.pytorch.org/docs/stable/testing.html)
- [PyCon 2023: Honey I Broke the PyTorch Model](https://github.com/clarahoffmann/pycon-2023-honey-i-broke-the-pytorch-model)
- [PyTorch Wiki: Running and Writing Tests](https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests)
- [CircleCI: Testing PyTorch with pytest](https://circleci.com/blog/testing-pytorch-model-with-pytest/)
- [PyTorch Multi-Device Testing](https://pytorch.org/blog/pt-multidevice-integration/)

### MLOps Testing
- [MLOps Testing KnowledgeBase](../../../KnowledgeBase/MLOps/MLOps%20-%20Testing.md)
- [MLOps Code Quality KnowledgeBase](../../../KnowledgeBase/MLOps/MLOps%20-%20Code%20-%20Notebooks.md)

### MONAI Testing
- MONAI uses `unittest.TestCase` + `torch.testing` for model validation
- `monai.utils.misc.set_determinism()` for reproducible tests
- MONAI CI tests all transforms with tiny random tensors

### Project-Specific
- `docs/planning/run-debug-factorial-experiment-report.md` — 12 glitches found
- `docs/planning/pytorch-model-testing-best-practices-report.md` — Full web research (550 lines)

---

## 9. Research Findings — LoRA-Specific Testing (from Web Research)

Source: TorchTune LoRA tests, HuggingFace PEFT patterns, PyTorch community.

### 9.1 Zero-Init Equivalence Test (Critical for LoRA)

At initialization, LoRA output MUST equal base model output (LoRA_B is zero-initialized):

```python
@pytest.mark.model_construction
def test_lora_zero_init_equivalence():
    """LoRA model at init must produce identical output to base model."""
    base = nn.Linear(64, 32)
    lora = LoRALinear(base, rank=8, alpha=16.0, dropout=0.0)
    x = torch.randn(2, 64)
    with torch.no_grad():
        base_out = base(x)
        lora_out = lora(x)
    torch.testing.assert_close(base_out, lora_out, rtol=1e-5, atol=1e-5)
```

### 9.2 Frozen-vs-Trainable Parameter Verification

```python
@pytest.mark.model_construction
def test_lora_only_adapters_are_trainable():
    """After LoRA application, only LoRA params should require grad."""
    # Build sam3_topolora adapter
    # Count trainable vs frozen
    # Verify: encoder base weights are frozen
    # Verify: LoRA A/B matrices are trainable
    # Verify: decoder is trainable
```

### 9.3 Target Module Validation (Would Have Caught Glitch #9)

```python
@pytest.mark.model_construction
def test_lora_targets_only_valid_module_types():
    """LoRA must only target module types it can handle."""
    # For each target returned by _apply_lora_to_encoder:
    # Verify the wrapped module is an instance LoRALinear accepts
```

### 9.4 Overfit-One-Batch Test (Karpathy's #1 Integration Test)

```python
@pytest.mark.model_construction
@pytest.mark.vram_2gb
def test_overfit_one_batch():
    """Model must overfit a single batch to near-zero loss."""
    model = build_adapter(config)
    x = torch.randn(1, 1, 32, 32, 32)
    y = torch.randint(0, 2, (1, 1, 32, 32, 32)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(50):
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert loss.item() < 0.1, f"Model failed to overfit one batch: loss={loss.item()}"
```

---

## 10. Research Findings — MONAI Testing Infrastructure

Source: MONAI CI, `monai.utils.testing` module.

### MONAI Skip Decorators (reference for our VRAM tiers)

```python
from monai.utils import optional_import, skip_if_no_cuda

# Skip if module not available (like mamba-ssm)
_, has_mamba = optional_import("mamba_ssm")

@unittest.skipUnless(has_mamba, "Requires mamba-ssm")
def test_mambavesselnet_construction(): ...

# Skip if no CUDA
@skip_if_no_cuda
def test_forward_pass_gpu(): ...
```

### MONAI Quick Test Mode

MONAI uses `MONAI_QUICK=1` to run reduced tests. We could adopt similar:

```python
QUICK_MODE = os.environ.get("MINIVESS_QUICK", "") == "1"

@pytest.mark.skipif(QUICK_MODE, reason="Skipped in quick mode")
def test_full_forward_pass(): ...
```

---

## 11. Research Findings — MLOps Testing (from KnowledgeBase)

Source: `/home/petteri/Dropbox/KnowledgeBase/MLOps/MLOps - Testing.md`

### Key Patterns Applicable to MinIVess

1. **ML Test Score** (Breck et al., 2017): Rubric across 8 dimensions. We should
   score ourselves and identify gaps.

2. **Contract Testing for MLflow**: Define contracts for inter-flow communication:
   - Training flow MUST produce: `checkpoint_dir_fold_N` tags, `status=FINISHED`
   - Post-training flow MUST consume: checkpoint files at tagged paths
   - Analysis flow MUST produce: `eval/{model}/{subset}/{metric}` metrics

3. **WeightWatcher**: Spectral analysis for detecting overfitting WITHOUT test data.
   Already in our deps (`weightwatcher>=0.7`). Add post-training check.

4. **Testcontainers**: Use throwaway Docker containers for PostgreSQL + MinIO in
   local CI. Replaces the "Docker Compose must be up" requirement.

5. **Never Mock ML Models**: Test against real (small) data. Our synthetic tensor
   tests are correct — never mock the model forward pass.

6. **Algorithmic Unit Tests**: Create synthetic datasets that test specific model
   properties (e.g., "does the model detect a single bright voxel?").

### Data Validation (Great Expectations / Deepchecks)

```python
def test_minivess_data_schema():
    """MiniVess NIfTI files must have expected properties."""
    # Shape: (H, W, D) where each dim >= 32
    # Dtype: float32 for images, uint8 for labels
    # Labels: binary {0, 1} only
    # Pairs: each imagesTr/XXX.nii.gz has matching labelsTr/XXX.nii.gz
```
- `tests/CLAUDE.md` — Existing test architecture
