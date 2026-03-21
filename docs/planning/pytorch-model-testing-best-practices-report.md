# PyTorch Model Testing Best Practices Report

**Date**: 2026-03-20
**Purpose**: Comprehensive research report on testing patterns for PyTorch models, with focus on
LoRA adapters, MONAI ecosystem, VRAM-tiered execution, and conditional CI triggers.

---

## 1. Component-Level Tests (Construction, Forward Pass, Gradient Flow)

### 1.1 Model Construction / Shape Tests

The most fundamental test verifies that a model can be instantiated and produces
correct output shapes. Shape bugs are among the most common — forgetting a channel
dimension, reshaping incorrectly, or mismatching spatial dimensions between encoder
and decoder.

**Pattern: Output shape validation**
```python
@torch.no_grad()
def test_output_shape(self):
    model = MySegNet(in_channels=1, out_channels=2, spatial_dims=3)
    x = torch.randn(2, 1, 64, 64, 64)  # (B, C, D, H, W)
    out = model(x)
    assert out.shape == (2, 2, 64, 64, 64), f"Expected (2,2,64,64,64), got {out.shape}"
```

**Pattern: Multi-resolution / multi-scale output shape validation**
For models producing auxiliary outputs (deep supervision, multi-task heads):
```python
def test_multi_output_shapes(self):
    model = DeepSupervisionNet(...)
    x = torch.randn(2, 1, 64, 64, 64)
    outputs = model(x)
    assert len(outputs) == 3  # main + 2 deep supervision heads
    for i, out in enumerate(outputs):
        assert out.shape[0] == 2, f"Head {i}: batch dim wrong"
        assert out.shape[1] == num_classes
```

**Source**: Krokotsch (2023), "How to Trust Your Deep Learning Code" — shape tests
are "trivial to write and they can spare you a lot of headaches."

### 1.2 Forward Pass Determinism

Non-deterministic behavior in tests makes it impossible to distinguish bugs from
random fluctuations. All model tests must seed randomness.

**Pattern: Determinism fixture**
```python
@pytest.fixture(autouse=True)
def seed_everything():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
```

### 1.3 Numerical Stability (NaN / Inf Detection)

**Pattern: No NaN or Inf in forward pass**
```python
def test_no_nan_inf_forward(self):
    model = MyModel(...)
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    assert not torch.isnan(out).any(), "NaN detected in forward pass"
    assert not torch.isinf(out).any(), "Inf detected in forward pass"
```

**Pattern: No NaN or Inf after backward pass (gradient check)**
```python
def test_no_nan_gradients(self):
    model = MyModel(...)
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    loss = out.mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
```

**Source**: The `torchtest` library provides `test_nan_vals` and `test_inf_vals`
as built-in checks. SAM3's FP16 overflow on T4 (Turing, no BF16) is a real-world
example of why these tests matter.

### 1.4 Gradient Flow Tests

Verify that ALL trainable parameters receive non-zero gradients. Dead sub-graphs
(parameters disconnected from the loss) are a silent failure mode.

**Pattern: All parameters updated after one step**
```python
def test_all_parameters_receive_gradients(self):
    model = MyModel(...)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    loss = out.mean()
    loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.sum(param.grad ** 2) > 0, f"Zero gradient for {name}"
```

**Source**: Krokotsch (2023) — "detect dead sub-graphs and unused parameters."
This test catches the common bug where a layer is defined in `__init__` but
never used in `forward()`.

### 1.5 Batch Independence Tests

Verify that samples within a batch don't influence each other (except through
batch normalization, which must be tested in eval mode).

**Pattern: Masked backward independence**
```python
def test_batch_independence(self):
    model = MyModel(...)
    model.eval()  # Required if model has BatchNorm
    x = torch.randn(4, 1, 32, 32, 32, requires_grad=True)
    out = model(x)

    # Zero out one sample's output
    mask = torch.ones_like(out)
    mask[0] = 0
    out = out * mask
    out.mean().backward()

    # Sample 0's input gradient should be zero (masked out)
    assert torch.all(x.grad[0] == 0), "Batch contamination: sample 0 has nonzero grad"
    # Other samples should have nonzero gradients
    for i in range(1, 4):
        assert not torch.all(x.grad[i] == 0), f"Sample {i} has zero gradient"
```

**Source**: Krokotsch (2023) — batch normalization in train mode violates this
assumption, so the test must use `model.eval()`.

### 1.6 Device Portability Tests

Verify model produces identical results on CPU and GPU (within floating-point tolerance).

**Pattern: CPU vs GPU output equivalence**
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_device_portability(self):
    model = MyModel(...)
    x = torch.randn(2, 1, 32, 32, 32)

    torch.manual_seed(42)
    out_cpu = model(x)

    model_gpu = model.to("cuda:0")
    torch.manual_seed(42)
    out_gpu = model_gpu(x.to("cuda:0"))

    torch.testing.assert_close(out_cpu, out_gpu.cpu(), rtol=1e-4, atol=1e-4)
```

### 1.7 Output Range Tests

For models with specific output constraints (sigmoid final activation, etc.):

**Pattern: Output bounded in expected range**
```python
def test_output_range(self):
    model = SegmentationModel(...)  # sigmoid output
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    assert out.min() >= 0.0, f"Output below 0: {out.min()}"
    assert out.max() <= 1.0, f"Output above 1: {out.max()}"
```

---

## 2. Integration Tests (Training Loop, Checkpoint Save/Load)

### 2.1 Overfit-One-Batch Test

The single most important integration test: train on one batch for N steps and
verify loss decreases. This catches broken loss functions, optimizer wiring,
and data pipeline issues.

**Pattern: Loss decrease on single batch**
```python
def test_overfit_single_batch(self):
    model = MyModel(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(4, 1, 32, 32, 32)
    y = torch.randint(0, 2, (4, 1, 32, 32, 32))

    initial_loss = None
    for step in range(100):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.1, (
        f"Loss did not decrease sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"
    )
```

**Source**: Karpathy's "Recipe for Training Neural Networks" — overfit a single
batch first. PyTorch Lightning has `Trainer(overfit_batches=1)` built-in.

### 2.2 Checkpoint Save/Load Round-Trip

Verify that saving and loading a checkpoint produces identical model state.

**Pattern: State dict round-trip**
```python
def test_checkpoint_round_trip(self, tmp_path):
    model = MyModel(...)
    x = torch.randn(2, 1, 32, 32, 32)

    # Forward pass before save
    torch.manual_seed(42)
    out_before = model(x)

    # Save checkpoint
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Load into fresh model
    model2 = MyModel(...)
    model2.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # Forward pass after load
    torch.manual_seed(42)
    out_after = model2(x)

    torch.testing.assert_close(out_before, out_after)
```

### 2.3 Optimizer State Checkpoint

For training resumption, optimizer state must also survive round-trips:

```python
def test_optimizer_state_round_trip(self, tmp_path):
    model = MyModel(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train one step
    x = torch.randn(2, 1, 32, 32, 32)
    loss = model(x).mean()
    loss.backward()
    optimizer.step()

    # Save both
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 5,
    }, tmp_path / "checkpoint.tar")

    # Load into fresh instances
    ckpt = torch.load(tmp_path / "checkpoint.tar", weights_only=True)
    model2 = MyModel(...)
    model2.load_state_dict(ckpt["model"])
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    opt2.load_state_dict(ckpt["optimizer"])

    assert ckpt["epoch"] == 5
    # Verify optimizer momentum buffers match
    for p1, p2 in zip(optimizer.state.values(), opt2.state.values()):
        for k in p1:
            if isinstance(p1[k], torch.Tensor):
                torch.testing.assert_close(p1[k], p2[k])
```

### 2.4 TorchScript / ONNX Export Test

Verify model can be exported for deployment:

```python
def test_torchscript_export(self):
    model = MyModel(...)
    model.eval()
    x = torch.randn(1, 1, 64, 64, 64)
    scripted = torch.jit.trace(model, x)
    out_orig = model(x)
    out_script = scripted(x)
    torch.testing.assert_close(out_orig, out_script)
```

**Source**: MONAI's `test_script_save()` and `test_onnx_save()` utilities provide
this pattern natively.

---

## 3. VRAM-Tiered Testing Strategy

### 3.1 The Problem

Model tests have vastly different VRAM requirements. DynUNet fits on 2 GB, but
SAM3 with ViT-32L needs 16+ GB. Running all tests on every machine wastes
resources and causes OOM crashes.

### 3.2 PyTorch's Own Approach

PyTorch's test suite uses `torch.cuda.get_device_properties()` to define
VRAM tiers:

```python
# From pytorch/test/test_cuda.py
TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9
TEST_MEDIUM_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 6e9
```

### 3.3 Recommended Tiered Marker System

```python
# conftest.py
import pytest
import torch

def _vram_gb() -> float:
    """Return total VRAM in GB, or 0.0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

VRAM_GB = _vram_gb()

# Register custom markers
def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "vram_8gb: requires >= 8 GB VRAM")
    config.addinivalue_line("markers", "vram_16gb: requires >= 16 GB VRAM")
    config.addinivalue_line("markers", "vram_24gb: requires >= 24 GB VRAM")

# Auto-skip based on available VRAM
def pytest_collection_modifyitems(config, items):
    vram_markers = {
        "gpu": 0.1,      # any GPU
        "vram_8gb": 8.0,
        "vram_16gb": 16.0,
        "vram_24gb": 24.0,
    }
    for item in items:
        for marker_name, required_gb in vram_markers.items():
            if marker_name in item.keywords:
                if VRAM_GB < required_gb:
                    item.add_marker(pytest.mark.skip(
                        reason=f"Requires {required_gb} GB VRAM, have {VRAM_GB:.1f} GB"
                    ))
```

### 3.4 Applying Tiers to Tests

```python
@pytest.mark.gpu
def test_dynunet_forward():
    """DynUNet fits on any GPU (< 2 GB)."""
    ...

@pytest.mark.vram_8gb
def test_segresnet_forward():
    """SegResNet needs ~4 GB for 3D patches."""
    ...

@pytest.mark.vram_16gb
def test_sam3_forward():
    """SAM3 ViT-32L needs 16+ GB."""
    ...

@pytest.mark.vram_24gb
def test_sam3_hybrid_topological():
    """SAM3 Hybrid + TopoLoRA needs 24 GB."""
    ...
```

### 3.5 Tier Mapping to Environments

| Tier | VRAM | Environment | Marker | Example Models |
|------|------|-------------|--------|----------------|
| CPU  | 0 GB | Any | (none) | Construction, shapes, gradient flow |
| GPU-any | >0 GB | Local dev (RTX 2070 8GB) | `@pytest.mark.gpu` | DynUNet, SegResNet forward |
| 8 GB | >=8 GB | Local dev | `@pytest.mark.vram_8gb` | DynUNet training, SAM3 Vanilla |
| 16 GB | >=16 GB | RunPod RTX 4090 | `@pytest.mark.vram_16gb` | SAM3 ViT-32L full |
| 24 GB | >=24 GB | RunPod RTX 4090 | `@pytest.mark.vram_24gb` | SAM3 Hybrid + TopoLoRA |

### 3.6 GPU Memory Cleanup Between Tests

Long pytest suites leak GPU memory. The proven fix is explicit garbage collection:

```python
# conftest.py
@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Free GPU memory after each test to prevent OOM in long suites."""
    yield
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
```

**Source**: pytest-dev/pytest Discussion #10296 — "calling `gc.collect()` between
tests resolved the memory exhaustion issue." Relying on `__del__()` for GPU cleanup
is an anti-pattern; explicit GC is required.

---

## 4. Conditional Execution (File-Change Triggers, Markers)

### 4.1 pytest Markers for Test Categories

Following HuggingFace Transformers' proven pattern:

```python
# Decorator stacking (HuggingFace pattern)
@pytest.mark.slow           # Skip unless RUN_SLOW=1
@pytest.mark.model_loading  # Skip in staging tier
@pytest.mark.gpu            # Skip if no GPU
def test_sam3_full_training():
    ...
```

**Key rule from HuggingFace**: When stacking decorators, hardware-skip decorators
must be listed LAST (closest to the function definition).

### 4.2 Environment-Variable Controlled Test Execution

Following PyTorch's own pattern:

| Env Variable | Effect |
|-------------|--------|
| `PYTORCH_TEST_WITH_SLOW=1` | Run `@slowTest` decorated tests |
| `RUN_SLOW=1` | HuggingFace: run `@slow` tests |
| `CUDA_VISIBLE_DEVICES=""` | Force CPU-only testing |
| `CUDA_VISIBLE_DEVICES="1"` | Use specific GPU |
| `QUICKTEST=true` | MONAI: skip non-quick tests |

### 4.3 File-Change CI Triggers (dorny/paths-filter)

For CI systems that allow triggers (not applicable to this repo since GH Actions
CI is disabled, but documented for reference):

```yaml
# .github/workflows/model-tests.yml
jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      models: ${{ steps.filter.outputs.models }}
      adapters: ${{ steps.filter.outputs.adapters }}
    steps:
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            models:
              - 'src/minivess/adapters/**'
              - 'src/minivess/models/**'
              - 'configs/model/**'
            adapters:
              - 'src/minivess/adapters/lora/**'
              - 'src/minivess/adapters/topological/**'

  model-tests:
    needs: detect-changes
    if: ${{ needs.detect-changes.outputs.models == 'true' }}
    runs-on: [self-hosted, gpu]
    steps:
      - run: make test-prod  # Only when model code changed
```

### 4.4 Local Equivalent: Makefile Targets

Since GH Actions CI is disabled in this project, local equivalents:

```makefile
test-staging:    # Fast, no models, gate for main
test-prod:       # Full suite including model_loading, gate for prod
test-gpu:        # SAM3 + GPU-heavy, RunPod only
test-models:     # Just model tests (adapter + network tests)
```

---

## 5. LoRA / Adapter-Specific Tests

### 5.1 Zero-Init Property Test

LoRA guarantees that at initialization (B=0), the adapted model produces identical
output to the base model. This is the most critical LoRA-specific test.

**Pattern: LoRA output equals base at init**
```python
def test_lora_zero_init_equivalence(self):
    """At init (B=zeros), LoRA model must produce same output as base."""
    base_model = BaseModel(...)
    lora_model = apply_lora(base_model, rank=8, alpha=16, target_modules=["q_proj", "v_proj"])

    x = torch.randn(2, 1, 64, 64, 64)
    torch.manual_seed(42)
    base_out = base_model(x)
    torch.manual_seed(42)
    lora_out = lora_model(x)

    torch.testing.assert_close(base_out, lora_out, rtol=1e-5, atol=1e-5)
```

### 5.2 Frozen Base Weights Test

Verify that base model weights do NOT change during training, while LoRA
adapter weights DO change.

**Pattern: Frozen params stay frozen, LoRA params update**
```python
def test_lora_freeze_correctness(self):
    """Base weights frozen, LoRA A/B matrices trainable."""
    model = apply_lora(BaseModel(...), rank=8)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    # Store initial values
    base_params_before = {
        name: param.clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }
    lora_params_before = {
        name: param.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Train one step
    x = torch.randn(2, 1, 32, 32, 32)
    loss = model(x).mean()
    loss.backward()
    optimizer.step()

    # Base weights unchanged
    for name, param in model.named_parameters():
        if not param.requires_grad:
            torch.testing.assert_close(
                param, base_params_before[name],
                msg=f"Base weight {name} changed during LoRA training!"
            )

    # LoRA weights changed
    any_changed = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.equal(param, lora_params_before[name]):
                any_changed = True
    assert any_changed, "No LoRA parameters updated during training"
```

**Source**: `torchtest.assert_vars_change()` and `assert_vars_same()` provide
this pattern generically. TorchTune's `test_lora_finetune_distributed.py` tests
this at the recipe level.

### 5.3 LoRA Target Module Validation

Verify that LoRA is applied to the correct module types and names.

**Pattern: LoRA applied only to specified targets**
```python
def test_lora_target_modules(self):
    """LoRA should only be applied to specified target modules."""
    model = apply_lora(
        BaseModel(...),
        target_modules=["q_proj", "v_proj"],
        rank=8,
    )

    lora_modules = [
        name for name, module in model.named_modules()
        if hasattr(module, "lora_A")  # or whatever attribute marks LoRA
    ]

    # All LoRA modules should match target patterns
    for name in lora_modules:
        assert any(target in name for target in ["q_proj", "v_proj"]), (
            f"LoRA applied to non-target module: {name}"
        )

    # At least some targets should have LoRA
    assert len(lora_modules) > 0, "No LoRA modules found"
```

### 5.4 LoRA Rank and Parameter Count Test

```python
def test_lora_parameter_count(self):
    """LoRA should add exactly rank*(in+out) params per target."""
    base_model = BaseModel(...)
    base_params = sum(p.numel() for p in base_model.parameters())

    lora_model = apply_lora(base_model, rank=8, target_modules=["q_proj"])
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    # Total params increased by LoRA overhead
    assert total_params > base_params
    # Trainable params << total params (LoRA efficiency)
    assert trainable_params < total_params * 0.1, (
        f"LoRA trainable ratio too high: {trainable_params/total_params:.2%}"
    )
```

### 5.5 LoRA Weight Merge/Unmerge Test

Post-training, LoRA weights merge into base weights for inference. The merged
model must produce identical outputs.

**Pattern: Merge produces equivalent outputs (TorchTune pattern)**
```python
def test_lora_merge_equivalence(self):
    """Merged model must produce same output as unmerged LoRA model."""
    lora_model = apply_lora(BaseModel(...), rank=8)

    # Simulate some training (make LoRA weights nonzero)
    x = torch.randn(2, 1, 32, 32, 32)
    loss = lora_model(x).mean()
    loss.backward()
    torch.optim.SGD(
        [p for p in lora_model.parameters() if p.requires_grad], lr=0.1
    ).step()

    # Get output from unmerged model
    torch.manual_seed(42)
    out_unmerged = lora_model(x)

    # Merge LoRA into base weights
    merged_model = merge_lora_weights(lora_model)

    # Get output from merged model
    torch.manual_seed(42)
    out_merged = merged_model(x)

    torch.testing.assert_close(out_unmerged, out_merged, rtol=1e-4, atol=1e-4)
```

**Source**: TorchTune's distributed LoRA test confirms "The results of calling
forward on dummy inputs should be the same" between merged and unmerged models.

### 5.6 LoRA Adapter Checkpoint Round-Trip

```python
def test_lora_adapter_save_load(self, tmp_path):
    """Save only adapter weights, load into fresh base model."""
    model = apply_lora(BaseModel(...), rank=8)

    # Train to make LoRA weights nonzero
    ...

    # Save adapter weights only
    adapter_state = {
        name: param for name, param in model.state_dict().items()
        if "lora_" in name
    }
    torch.save(adapter_state, tmp_path / "adapter.pt")

    # Load into fresh base + LoRA
    model2 = apply_lora(BaseModel(...), rank=8)
    adapter_loaded = torch.load(tmp_path / "adapter.pt", weights_only=True)
    model2.load_state_dict(adapter_loaded, strict=False)

    x = torch.randn(2, 1, 32, 32, 32)
    torch.testing.assert_close(model(x), model2(x))
```

### 5.7 Gradient Flow Through Frozen + LoRA Layers

LoRA requires gradient flow through frozen layers to reach the adapter matrices.
This is a subtle failure mode, especially with gradient checkpointing.

**Pattern: Verify LoRA A and B receive gradients**
```python
def test_lora_gradient_flow(self):
    """Gradients must reach LoRA A and B matrices through frozen base."""
    model = apply_lora(BaseModel(...), rank=8)
    x = torch.randn(2, 1, 32, 32, 32)
    out = model(x)
    out.mean().backward()

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.grad is not None, f"No gradient for LoRA param {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for LoRA param {name}"
```

**Source**: Gradient checkpointing + LoRA is a known failure mode. Using forward
hooks to set `requires_grad=True` on frozen embeddings can break during gradient
checkpoint recomputation. The recommended fix: set `requires_grad=True` directly
in the forward method, not via hooks.

---

## 6. MONAI Ecosystem Testing Patterns

### 6.1 MONAI Test Infrastructure Overview

MONAI provides a comprehensive testing framework in `monai.utils.testing`:

**Skip Decorators** (conditional test execution):
| Decorator | Purpose |
|-----------|---------|
| `skip_if_no_cuda` | Skip if no GPU available |
| `skip_if_quick` | Skip if `QUICKTEST=true` env var set |
| `skip_if_no_cpp_extension` | Skip if C++ extensions not compiled |
| `skip_if_windows` / `skip_if_darwin` | Platform-specific skips |
| `SkipIfNoModule("module_name")` | Skip if optional dependency missing |
| `SkipIfBeforePyTorchVersion((major, minor))` | Skip on old PyTorch |
| `SkipIfBeforeComputeCapabilityVersion` | Skip on old GPU compute capability |
| `skip_if_downloading_fails()` | Context manager for download-dependent tests |

**Image Test Case Base Classes**:
| Class | Shape | Type |
|-------|-------|------|
| `NumpyImageTestCase2D` | 128 x 64 | NumPy array |
| `TorchImageTestCase2D` | 128 x 64 | PyTorch tensor |
| `NumpyImageTestCase3D` | 64 x 48 x 80 | NumPy array |
| `TorchImageTestCase3D` | 64 x 48 x 80 | PyTorch tensor |

### 6.2 MONAI Assertion Utilities

```python
from monai.utils.testing import assert_allclose

# Validates type consistency AND numerical closeness
assert_allclose(actual, desired, rtol=1e-5, atol=1e-8)

# Compare model state dicts
from monai.utils.testing import equal_state_dict
assert equal_state_dict(model1.state_dict(), model2.state_dict())
```

### 6.3 MONAI Transform Testing Pattern

MONAI transforms must be tested for:
1. Output shape preservation
2. Value range after transform
3. Determinism (same seed = same output)
4. Inverse transform correctness
5. Dictionary vs array interface consistency

```python
class TestMyTransform(NumpyImageTestCase3D):
    def test_shape(self):
        transform = MyTransform(...)
        result = transform(self.imt)  # self.imt is the 3D test image
        self.assertEqual(result.shape, self.imt.shape)

    def test_value_range(self):
        transform = NormalizeIntensity(...)
        result = transform(self.imt)
        self.assertGreaterEqual(result.min(), -1.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_inverse(self):
        transform = MyInvertibleTransform(...)
        result = transform(self.imt)
        inverted = transform.inverse(result)
        assert_allclose(inverted, self.imt, rtol=1e-4)
```

### 6.4 MONAI Network Testing Pattern

```python
from monai.utils.testing import test_script_save, test_onnx_save

class TestDynUNet(unittest.TestCase):
    def test_forward(self):
        net = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=[3, 3, 3],
            strides=[1, 2, 2],
            upsample_kernel_size=[2, 2],
        )
        x = torch.randn(2, 1, 64, 64, 64)
        out = net(x)
        self.assertEqual(out.shape, (2, 2, 64, 64, 64))

    def test_torchscript(self):
        net = DynUNet(...)
        test_script_save(net, torch.randn(1, 1, 64, 64, 64))

    def test_onnx_export(self):
        net = DynUNet(...)
        test_onnx_save(net, torch.randn(1, 1, 64, 64, 64))
```

### 6.5 MONAI Quick Test Mode

MONAI supports a `QUICKTEST` environment variable for tiered test execution:

```python
from monai.utils.testing import test_is_quick

if test_is_quick():
    # Use smaller data, fewer iterations
    img_size = (32, 32, 32)
    n_epochs = 2
else:
    img_size = (128, 128, 128)
    n_epochs = 100
```

### 6.6 MONAI Distributed Testing

```python
from monai.utils.testing import DistCall, DistTestCase

class TestDistributed(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_distributed_training(self):
        # This test runs on 2 processes
        ...
```

### 6.7 MONAI Hardware Detection

```python
from monai.utils.testing import query_memory, has_cupy, is_tf32_env

# Get idle GPU device IDs via nvidia-smi
idle_gpus = query_memory(threshold=0.9)

# Check TF32 mode (affects numerical precision on Ampere+)
if is_tf32_env():
    # Use relaxed tolerances
    rtol, atol = 1e-3, 1e-3
```

---

## 7. torch.testing Module Reference

### 7.1 assert_close (Primary Comparison Function)

```python
torch.testing.assert_close(
    actual,
    expected,
    *,
    allow_subclasses=True,
    rtol=None,        # Relative tolerance
    atol=None,        # Absolute tolerance
    equal_nan=False,  # Treat NaN as equal
    check_device=True,
    check_dtype=True,
    check_layout=True,
    check_stride=False,
    msg=None,
)
```

Comparison formula: `|actual - expected| <= atol + rtol * |expected|`

**Default tolerances by dtype**:
| dtype | rtol | atol |
|-------|------|------|
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 1.6e-2 | 1e-5 |
| float32 | 1.3e-6 | 1e-5 |
| float64 | 1.3e-6 | 1e-5 |

**Strict equality helper**:
```python
import functools
assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
```

### 7.2 make_tensor (Test Data Generation)

Preferred over `torch.randn` for parameterized tests:
```python
from torch.testing import make_tensor
x = make_tensor((2, 3), device="cpu", dtype=torch.float32)
```

---

## 8. Common Bug Patterns and How to Catch Them

### 8.1 Shape Mismatches
**Bug**: Encoder output spatial dims don't match decoder input
**Test**: Forward pass with known input shape, assert output shape

### 8.2 dtype Issues (FP16/BF16 Overflow)
**Bug**: SAM3's ViT encoder produces values > 65504 in FP16 (T4 Turing, no BF16)
**Test**: Forward pass in target dtype, check for NaN/Inf
```python
def test_no_overflow_bf16(self):
    model = SAM3Encoder(...).to(dtype=torch.bfloat16, device="cuda")
    x = torch.randn(1, 1, 64, 64, 64, dtype=torch.bfloat16, device="cuda")
    out = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
```

### 8.3 LoRA Applied to Wrong Layer Types
**Bug**: LoRA injected into LayerNorm or Conv3d instead of Linear
**Test**: Verify `isinstance` of all LoRA-wrapped modules
```python
def test_lora_only_on_linear(self):
    model = apply_lora(BaseModel(...), rank=8)
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            assert isinstance(module.original_module, nn.Linear), (
                f"LoRA on non-Linear: {name} is {type(module.original_module)}"
            )
```

### 8.4 Dead Parameters (Defined but Not Used)
**Bug**: Layer defined in `__init__` but not called in `forward()`
**Test**: Gradient flow test (Section 1.4) catches this

### 8.5 Device Mismatch
**Bug**: Tensor created on CPU inside a GPU model's forward pass
**Test**: Device portability test (Section 1.6) catches this. Also:
```python
def test_all_outputs_on_same_device(self):
    model = MyModel(...).to("cuda")
    x = torch.randn(2, 1, 32, 32, 32, device="cuda")
    out = model(x)
    assert out.device.type == "cuda"
```

### 8.6 Batch Norm Mode Bugs
**Bug**: Model tested in train mode but deployed in eval mode (different behavior)
**Test**: Compare outputs in both modes
```python
def test_train_vs_eval_mode(self):
    model = MyModel(...)
    x = torch.randn(4, 1, 32, 32, 32)
    model.train()
    out_train = model(x)
    model.eval()
    out_eval = model(x)
    # They should differ if model has BatchNorm/Dropout
    if has_batchnorm_or_dropout(model):
        assert not torch.equal(out_train, out_eval)
```

### 8.7 Non-Reproducible Results
**Bug**: Different outputs on same input due to unseeded randomness
**Test**: Determinism test (Section 1.2) catches this

---

## 9. Testing Tool Reference

| Tool | Status | Purpose |
|------|--------|---------|
| `torch.testing.assert_close` | Active (PyTorch core) | Tensor comparison with tolerances |
| `torch.testing.make_tensor` | Active (PyTorch core) | Parameterized test data generation |
| `torch.autograd.gradcheck` | Active (PyTorch core) | Gradient correctness verification |
| `torchtest` | Maintained (pip) | 5 basic model tests (vars change/same, NaN, Inf, range) |
| `monai.utils.testing` | Active (MONAI) | Skip decorators, image test cases, assert_allclose |
| `weightwatcher` | Active (pip) | Weight distribution analysis without training data |
| `cleanlab` | Active (pip) | Label quality analysis |

---

## 10. Recommended Test Organization for MinIVess

Based on all research, the recommended test structure:

```
tests/v2/
  unit/
    models/
      test_model_construction.py     # Shape, dtype, no NaN (CPU-only, fast)
      test_model_gradient_flow.py    # All params get gradients (CPU-only, fast)
      test_model_batch_independence.py
      test_lora_properties.py        # Zero-init, freeze, target modules (CPU)
      test_lora_gradient_flow.py     # LoRA A/B receive gradients (CPU)
    losses/
      test_loss_shapes.py
      test_loss_known_values.py      # Reference implementation comparison
    transforms/
      test_transform_shapes.py
      test_transform_inverse.py
  integration/
    test_overfit_single_batch.py     # @pytest.mark.model_loading (staging-excluded)
    test_checkpoint_round_trip.py    # @pytest.mark.model_loading
    test_lora_merge_equivalence.py   # @pytest.mark.model_loading
    test_lora_checkpoint_round_trip.py
  gpu_instance/
    test_sam3_forward.py             # @pytest.mark.vram_16gb
    test_sam3_training_step.py       # @pytest.mark.vram_16gb
    test_sam3_topological_lora.py    # @pytest.mark.vram_24gb
```

**Key principles**:
1. CPU tests (shapes, gradients, construction) run everywhere, including staging
2. `model_loading` tests excluded from staging but run in prod
3. GPU instance tests only run on RunPod with explicit markers
4. Every model adapter gets the same test suite (via MixIn base class)
5. Use `torch.testing.assert_close` for all numerical comparisons
6. Autouse fixture for GPU memory cleanup after each test
7. Seed fixture for determinism in all model tests

---

## Sources

### Directly Fetched URLs
- [PyCon 2023: Honey, I Broke the PyTorch Model](https://github.com/clarahoffmann/pycon-2023-honey-i-broke-the-pytorch-model)
- [PyTorch Official Testing Utilities](https://docs.pytorch.org/docs/stable/testing.html)
- [PyTorch Forum: Unit Testing a PyTorch Code](https://discuss.pytorch.org/t/how-to-confirm-that-pytorch-code-is-working-as-intended-unit-testing-a-pytorch-code/16508)
- [PyTorch Wiki: Running and Writing Tests](https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests)
- [CircleCI: Testing PyTorch Model with Pytest](https://circleci.com/blog/testing-pytorch-model-with-pytest/)
- [HuggingFace Transformers Testing Documentation](https://huggingface.co/transformers/v3.4.0/testing.html)
- [torchtest: Unit Testing for PyTorch](https://github.com/suriyadeepan/torchtest)
- [How to Trust Your Deep Learning Code](https://krokotsch.eu/posts/deep-learning-unit-tests/)
- [PyTorch Forum: Mock Torch Device for Unit Testing](https://discuss.pytorch.org/t/mock-torch-device-for-unit-testing/136620)
- [PyTorch Forum: Best Practices Discussion](https://discuss.pytorch.org/t/how-to-write-unittests-best-practices-discussion/13640)
- [dorny/paths-filter GitHub Action](https://github.com/dorny/paths-filter)
- [TorchTune: LoRA Finetune Distributed Tests](https://github.com/pytorch/torchtune/blob/main/tests/recipes/test_lora_finetune_distributed.py)
- [LoRA Implementation Guide](https://mbrenndoerfer.com/writing/lora-implementation-pytorch-peft-guide)
- [pytest Discussion: GPU Memory Management](https://github.com/pytest-dev/pytest/discussions/10296)
- [MONAI Test Directory](https://github.com/Project-MONAI/MONAI/tree/dev/tests)
- [Clearing GPU Memory After PyTorch Training](https://www.geeksforgeeks.org/deep-learning/clearing-gpu-memory-after-pytorch-training-without-kernel-restart/)

### Web Search Results
- [Towards Data Science: Unit Testing in Deep Learning](https://towardsdatascience.com/unit-testing-in-deep-learning-b91d366e4862/)
- [Predibase: Unit Testing ML Code for Gradient Updates](https://predibase.com/blog/unit-testing-machine-learning-code-in-ludwig-and-pytorch-tests-for-gradient)
- [AI Summer: How to Unit Test Deep Learning](https://theaisummer.com/unit-test-deep-learning/)
- [PyTorch Forum: Testing Code for GPUs Without GPU](https://discuss.pytorch.org/t/testing-code-for-gpus-works-without-using-a-gpu/18970)
- [PyTorch Forum: Can Loss Vary When Overfitting Single Batch](https://discuss.pytorch.org/t/can-loss-vary-when-overfitting-a-single-batch/98914)
- [PyTorch Lightning Debugging](https://pytorch-lightning.readthedocs.io/en/1.0.8/debugging.html)
- [PyTorch: Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [scipy PR: Skip Some Torch GPU Tests](https://github.com/scipy/scipy/pull/21564)
- [MONAI GitHub Repository](https://github.com/Project-MONAI/MONAI)
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [Gradient Checkpointing and LoRA](https://github.com/huggingface/peft/issues/1142)
- [HuggingFace PEFT: LoRA Merge/Unmerge](https://github.com/huggingface/peft/issues/2437)
- [TorchTune: Remove Automatic Weight Merging RFC](https://github.com/meta-pytorch/torchtune/issues/2115)
- [minLoRA: Minimal PyTorch LoRA](https://github.com/cccntu/minLoRA)
- [torchtune LoRA Module Documentation](https://pytorch.org/torchtune/0.1/_modules/torchtune/modules/peft/lora.html)
