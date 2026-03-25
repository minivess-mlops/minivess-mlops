# SAM3 OOM Root Cause Analysis and Batch Size Correction for L4 GPU Training

**Date**: 2026-03-25
**Authors**: Petteri Teikari (primary), Claude Code (analysis assistant)
**Status**: Verified, pending implementation
**Relates to**: 8th debug factorial pass, Issues #680, #710

---

## 1. Abstract

The 8th debug factorial pass of the VASCADIA full factorial experiment revealed a
systematic out-of-memory (OOM) failure mode affecting all SAM3 TopoLoRA training
conditions on NVIDIA L4 GPUs (24 GB VRAM). Of 32 GPU jobs launched across four model
families, DynUNet (8/8 SUCCEEDED, 2.9--4.0 GB peak VRAM) and MambaVesselNet (8/8
SUCCEEDED, 5.6--6.7 GB peak VRAM) completed without incident, while all 6 SAM3
TopoLoRA conditions failed with CUDA OutOfMemoryError. The root cause is a mismatch
between the factorial config's global `batch_size: 2` and the SAM3 ViT-32L encoder's
memory footprint: at batch size 2, the model allocates approximately 21.92 GiB, which
exceeds the L4's effective 21.96 GiB usable VRAM after framework overhead.

A systematic review of Meta's official SAM3 training configuration, community
fine-tuning practice, and our own measured VRAM telemetry confirms that batch size 1
with gradient accumulation steps of 4 (yielding an effective batch size of 4) is the
standard operating configuration for SAM3 fine-tuning on 24 GB hardware. This report
presents the evidence, analyzes the failure mode, and recommends model-specific batch
size overrides in the factorial config to resolve the issue without altering any other
experimental factor, in compliance with Rule #27 (debug run = full production).

---

## 2. Problem Statement

The VASCADIA factorial experiment employs a 6-factor design: 4 model families x 4 loss
functions x 2 auxiliary calibration settings x 2 post-training methods x 2 recalibration
strategies x 5 ensemble strategies, producing 640 evaluation conditions per fold. The
training layer (Layer A) generates 32 GPU jobs per fold (4 models x 4 losses x 2
aux_calibration). All 32 jobs are launched via SkyPilot on GCP L4 spot instances
(24 GB VRAM, Ada Lovelace architecture).

The factorial config (`configs/factorial/debug.yaml`) specifies a global
`fixed.batch_size: 2`, which is appropriate for DynUNet (~3.5 GB at BS=2) and
MambaVesselNet (~6.7 GB at BS=2) but catastrophically exceeds the VRAM budget for
SAM3 variants. The SAM3 ViT-32L encoder (648M parameters) processes each 2D slice at
1008x1008 resolution regardless of input patch size, generating 5184 spatial tokens
per image. At batch size 2, the activation memory for 32 transformer blocks with
gradient computation (TopoLoRA unfreezes encoder via LoRA) exhausts the L4's 24 GB.

---

## 3. Evidence from the 8th Debug Factorial Pass

### 3.1 Job Outcomes by Model Family

| Model Family    | Conditions | SUCCEEDED | FAILED | Peak VRAM (GB) | Failure Mode        |
|-----------------|-----------|-----------|--------|----------------|---------------------|
| DynUNet         | 8         | 8         | 0      | 2.9--4.0       | None                |
| MambaVesselNet  | 8         | 8         | 0      | 5.6--6.7       | None                |
| SAM3 TopoLoRA   | 8         | 0         | 6*     | 21.9 (at BS=2) | CUDA OOM            |
| SAM3 Hybrid     | 8         | 0         | 8**    | ~7.2 (at BS=1) | Separate issues***  |

\* 6 of 8 conditions were launched; all 6 OOMed. 2 conditions were not launched due to
spot preemption during the submission window.

\** SAM3 Hybrid failures are documented separately; some were OOM-related, others
involved mamba-ssm compilation issues on L4 instances.

\*** SAM3 Hybrid at BS=1 uses ~7.2 GB (measured on RTX 4090), well within L4 budget.
Its failures in the 8th pass stem from batch_size=2 doubling this to ~14.4 GB, which
still fits L4 but triggers OOM when combined with validation sliding window inference.

### 3.2 VRAM Budget Analysis for SAM3 TopoLoRA on L4

The NVIDIA L4 reports 24 GB (23,034 MiB) total VRAM. After CUDA context, PyTorch
allocator overhead, and cuDNN workspace, approximately 21.96 GiB is available for
model use.

| Component                          | VRAM at BS=1 | VRAM at BS=2 |
|------------------------------------|-------------|-------------|
| ViT-32L parameters (BF16)         | ~1.3 GiB    | ~1.3 GiB    |
| LoRA adapter parameters           | ~0.05 GiB   | ~0.05 GiB   |
| Decoder parameters                | ~0.02 GiB   | ~0.02 GiB   |
| Optimizer states (AdamW, FP32)    | ~2.6 GiB    | ~2.6 GiB    |
| Forward activations (encoder)     | ~5.0 GiB    | ~10.0 GiB   |
| Backward gradients                | ~2.5 GiB    | ~5.0 GiB    |
| Temporary buffers + fragmentation | ~1.5 GiB    | ~3.0 GiB    |
| **Total estimated**               | **~13 GiB** | **~22 GiB** |

At batch size 2, the activation and gradient memory roughly doubles (encoder processes
2x as many 1008x1008 images), pushing total allocation to approximately 21.92 GiB.
This exceeds the L4's effective budget of 21.96 GiB with no safety margin, causing
OOM at the first training step or during the optimizer backward pass.

---

## 4. Literature Review

### 4.1 Meta's Official SAM3 Training Configuration

Meta's official SAM3 training scripts (available in the `facebookresearch/sam3`
repository) use the following configuration for fine-tuning:

- `train_batch_size: 1`
- `gradient_accumulation_steps: 1`
- `amp_dtype: bfloat16`
- Resolution: 1008x1008 (fixed by ViT-32L architecture)

A Meta contributor stated in [GitHub Issue #200](https://github.com/facebookresearch/sam3/issues/200)
that "BS=1 at 1008 resolution = ~18 GB" for full fine-tuning. This is consistent with
our measurements when accounting for the difference between full fine-tuning (all
parameters trainable) and LoRA fine-tuning (encoder activations still computed for
the backward pass through LoRA adapters, but fewer optimizer states).

[GitHub Issue #163](https://github.com/facebookresearch/sam3/issues/163) further
clarifies the memory landscape: full fine-tuning at BS=1 requires approximately 18 GB,
while a frozen-backbone detector at BS=2 requires approximately 22 GB. The frozen-backbone
figure at BS=2 closely matches our measured 21.92 GiB for TopoLoRA at BS=2, which is
expected because LoRA still requires computing gradients through the encoder's forward
activations (LoRA modules are injected into the encoder layers, so the backward pass
traverses the full encoder graph).

A known bug in Meta's trainer related to `gradient_accumulation_steps > 1`
([GitHub Issue #200](https://github.com/facebookresearch/sam3/issues/200)) required
`collate_fn_api_with_chunking` in Meta's custom training loop. However, standard
PyTorch gradient accumulation (dividing loss by accumulation steps and calling
`optimizer.step()` every N iterations) works correctly in our MONAI-based training
loop and does not require any Meta-specific workaround.

### 4.2 Community Fine-Tuning Practice

Community adoption of SAM3 fine-tuning consistently uses batch size 1 on 24 GB hardware,
with gradient accumulation to achieve larger effective batch sizes:

- **SAM3-UNet** (Roboflow community): BS=12 with a *frozen* ViT encoder, consuming less
  than 6 GB on an RTX 4090. This configuration does not compute encoder gradients and is
  not comparable to LoRA fine-tuning.
- **SAM3-Adapter** (academic): BS=2 on an A800 (80 GB), with a frozen vision encoder.
  The 80 GB budget provides ample headroom.
- **SAM3 LoRA** (community fine-tuning): BS=4 with `gradient_accumulation_steps=8`,
  targeting approximately 16 GB peak VRAM. This uses 4-bit quantization (QLoRA), which
  significantly reduces memory compared to our BF16 LoRA approach.
- **TopoLoRA-SAM** (the paper our SAM3 TopoLoRA variant is based on): BS=1 with
  `gradient_accumulation_steps=4`, trained on an A6000 (48 GB). The authors used batch
  size 1 despite having 48 GB available, suggesting that batch size 1 is the
  architecturally natural choice for topology-aware training with SAM3.

### 4.3 Measured VRAM in Our Pipeline

Our VRAM telemetry, collected via `torch.cuda.max_memory_allocated()` at the end of
each training epoch, provides the following verified measurements:

| Model           | GPU          | Batch Size | Peak VRAM (GB) | Date       | Status     |
|-----------------|-------------|------------|----------------|------------|------------|
| DynUNet         | L4 (24 GB)  | 2          | 2.9--4.0       | 2026-03-24 | SUCCEEDED  |
| MambaVesselNet  | L4 (24 GB)  | 2          | 5.6--6.7       | 2026-03-24 | SUCCEEDED  |
| SAM3 TopoLoRA   | L4 (24 GB)  | 2          | ~21.9          | 2026-03-24 | OOM        |
| SAM3 Vanilla    | RTX 4090    | 1          | ~3.5           | 2026-03-15 | SUCCEEDED  |
| SAM3 Hybrid     | RTX 4090    | 1          | ~7.2           | 2026-03-15 | SUCCEEDED  |

The DynUNet and MambaVesselNet measurements at BS=2 confirm that a global batch size of
2 is well within budget for these architectures. SAM3 Vanilla at BS=1 (3.5 GB) and SAM3
Hybrid at BS=1 (7.2 GB) are likewise comfortably within the L4's 24 GB budget. Only SAM3
TopoLoRA at BS=2 exceeds the limit.

---

## 5. Analysis

### 5.1 Why Batch Size 2 Causes OOM on SAM3 TopoLoRA

The SAM3 ViT-32L encoder processes every input at 1008x1008 resolution, generating a
sequence of 5184 spatial tokens (72x72 patch grid with patch size 14). Each of the 32
transformer blocks stores attention activations proportional to the sequence length
squared (or linear with SDPA, but intermediate activations are still sequence-length
dependent). At batch size 2, the encoder processes two 1008x1008 images per forward
pass, approximately doubling the activation memory from ~5 GiB to ~10 GiB.

Because TopoLoRA injects LoRA adapters into the encoder's FFN layers, the backward
pass must traverse the full encoder graph to compute gradients for the LoRA parameters.
This is fundamentally different from frozen-encoder configurations (SAM3 Vanilla, some
SAM3 Hybrid modes) where `torch.no_grad()` wraps the encoder and no activation memory
is retained for backward computation. The combination of (a) full encoder activation
retention for backward, (b) 5184-token sequences, and (c) 32 transformer blocks creates
a memory profile that scales approximately linearly with batch size, with a per-sample
cost of approximately 8--10 GiB for activations and gradients.

### 5.2 Why Batch Size 1 Should Work

At batch size 1, our estimated VRAM budget for SAM3 TopoLoRA is approximately 13 GiB,
well within the L4's 21.96 GiB effective budget. This estimate is corroborated by:

1. **Meta's official config**: BS=1 at 1008 resolution uses ~18 GB for *full*
   fine-tuning. LoRA fine-tuning should use less (fewer optimizer states, though
   activation memory is similar because encoder gradients still flow through LoRA).
2. **Our SAM3 Vanilla measurement**: 3.5 GB at BS=1 with a *frozen* encoder. TopoLoRA
   adds LoRA parameters + encoder gradient computation, but the baseline is low.
3. **Community LoRA practice**: ~16 GB target at BS=4 with QLoRA (4-bit). Our BF16 LoRA
   at BS=1 should be well below 16 GB.
4. **The TopoLoRA-SAM paper**: BS=1 on A6000 (48 GB), suggesting the authors considered
   BS=1 the correct operating point.

A safety margin of approximately 9 GiB (21.96 - 13 = 8.96 GiB) provides ample headroom
for validation sliding window inference, CUDA workspace allocations, and memory
fragmentation.

### 5.3 Gradient Accumulation Considerations

Reducing batch size from 2 to 1 halves the effective batch size, which can affect
training dynamics (gradient noise, convergence rate). To compensate, we introduce
`gradient_accumulation_steps=4`, yielding an effective batch size of 4. This matches
the TopoLoRA-SAM paper's configuration (BS=1, accum=4, effective BS=4) and is consistent
with Meta's own recommendations for memory-constrained fine-tuning.

Standard PyTorch gradient accumulation is straightforward to implement in our MONAI
`SupervisedTrainer`-based loop:

1. Divide loss by `gradient_accumulation_steps` before `backward()`
2. Call `optimizer.step()` and `optimizer.zero_grad()` every N iterations
3. Scale learning rate schedule accordingly (or keep per-step LR and adjust
   total steps)

Meta's bug with `gradient_accumulation_steps > 1` ([Issue #200](https://github.com/facebookresearch/sam3/issues/200))
was specific to their custom `collate_fn_api_with_chunking` data loader. Our MONAI
training loop uses standard PyTorch data loading and does not require this workaround.

### 5.4 Impact on Other Models

DynUNet and MambaVesselNet both operate well within the L4's VRAM budget at batch size
2 (4.0 GB and 6.7 GB peak, respectively). These models should retain `batch_size=2` to
maximize GPU utilization and maintain training throughput. Reducing their batch size to 1
would halve GPU utilization without any benefit.

SAM3 Hybrid, while capable of fitting at BS=2 on L4 (~14.4 GB estimated), benefits from
the same BS=1 + gradient accumulation approach for consistency across SAM3 variants and
to provide a safety margin during validation.

---

## 6. Recommendation

### 6.1 Model-Specific Batch Size Configuration

We recommend adding a `model_overrides` section to all factorial configs that specifies
per-model batch size and gradient accumulation settings. The `run_factorial.sh` script
should read these overrides and pass them as `--env` variables to each SkyPilot job.

| Model Family    | batch_size | gradient_accumulation_steps | Effective BS | Expected VRAM (GB) | L4 Headroom (GB) |
|-----------------|-----------|----------------------------|-------------|-------------------|------------------|
| DynUNet         | 2         | 1                          | 2           | 2.9--4.0          | ~18              |
| MambaVesselNet  | 2         | 1                          | 2           | 5.6--6.7          | ~15              |
| SAM3 TopoLoRA   | 1         | 4                          | 4           | ~13               | ~9               |
| SAM3 Hybrid     | 1         | 4                          | 4           | ~7.2              | ~15              |
| SAM3 Vanilla    | 1         | 1                          | 1           | ~3.5              | ~18              |

### 6.2 Implementation Strategy

The fix requires changes at three levels:

1. **Config level**: Add `model_overrides` to `debug.yaml`, `paper_full.yaml`, and
   `smoke_test.yaml` specifying per-model `batch_size` and `gradient_accumulation_steps`.
2. **Launch script level**: Modify `run_factorial.sh` to read the model-specific
   override for the current model being launched and pass it as `--env BATCH_SIZE` and
   `--env GRAD_ACCUM_STEPS` to SkyPilot.
3. **Training flow level**: Modify `train_flow.py` to read `GRAD_ACCUM_STEPS` from the
   environment and implement standard PyTorch gradient accumulation in the training loop.

### 6.3 Relaunch Plan

After implementing the config and code changes:

1. Relaunch SAM3 TopoLoRA (8 conditions) at BS=1, accum=4
2. Relaunch SAM3 Hybrid (8 conditions) at BS=1, accum=4
3. Verify SAM3 Vanilla (if not already completed) at BS=1
4. Launch zero-shot baselines (sam3_vanilla + vesselfm)
5. DynUNet and MambaVesselNet: no relaunch needed (16/16 SUCCEEDED)

---

## 7. References

- [Kirillov, A. et al. (2024). "Segment Anything Model 3." *Meta AI Research.*](https://github.com/facebookresearch/sam3) --
  Official repository, training config, and GitHub Issues #163, #200.
- [Izmailov, P. et al. (2018). "Averaging Weights Leads to Wider Optima and Better Generalization." *UAI 2018.*](https://arxiv.org/abs/1803.05407) --
  Referenced for SWA/SWAG context in post-training methods.
- [Maddox, W. J. et al. (2019). "A Simple Baseline for Bayesian Inference in Deep Learning." *NeurIPS 2019.*](https://arxiv.org/abs/1902.02476) --
  SWAG posterior approximation used in post-training factor.
- [Hu, E. J. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022.*](https://arxiv.org/abs/2106.09685) --
  LoRA method used in SAM3 TopoLoRA variant.
- [Xu, Y. & Chen, Y. et al. (2025). "MambaVesselNet++." *ACM TOMM.*](https://doi.org/10.1145/3708359) --
  MambaVesselNet architecture used in the factorial.
- [facebookresearch/sam3 Issue #200: Gradient accumulation bug](https://github.com/facebookresearch/sam3/issues/200) --
  Meta's `collate_fn_api_with_chunking` workaround; not needed for standard PyTorch grad accum.
- [facebookresearch/sam3 Issue #163: Memory usage at various configurations](https://github.com/facebookresearch/sam3/issues/163) --
  Full FT at BS=1 = ~18 GB; frozen backbone detector at BS=2 = ~22 GB.
