# Weight Averaging Methods: SWA, SWAG, Multi-SWA/SWAG, PSWA, LAWA, Model Soups, Checkpoint Averaging

**Date**: 2026-03-21
**Context**: MinIVess MLOps v2 — post-training pipeline methods
**Triggered by**: Discovery that our "SWA" implementation was mislabeled checkpoint averaging
**Metalearning**: `.claude/metalearning/2026-03-21-fake-swa-checkpoint-averaging-mislabeled.md`

---

## 1. SWA — Stochastic Weight Averaging

**Paper**: [Izmailov et al. (2018). "Averaging Weights Leads to Wider Optima and Better Generalization." UAI.](https://arxiv.org/abs/1803.05407)
**Code**: [github.com/timgaripov/swa](https://github.com/timgaripov/swa)
**PyTorch**: `torch.optim.swa_utils` (stable since PyTorch 1.6)

### Algorithm

SWA is a **two-phase training** procedure:

1. **Phase 1 (conventional SGD)**: Train with standard decaying LR for some portion of
   the budget (e.g., 75% of epochs).
2. **Phase 2 (SWA collection)**: Resume training with a **cyclic or high-constant LR**,
   maintaining a running average of weights via `AveragedModel`.
3. **Post-training**: Recalibrate batch normalization statistics via `update_bn()`.

The key insight: the cyclic/constant LR in Phase 2 causes the optimizer to traverse
diverse points in weight space (rather than converging to a single minimum). Averaging
these traversed points produces a model in a **flatter region** of the loss landscape
with better generalization.

### Critical quote from the original paper (Section 3.2)

> "The pretrained model w-hat can be trained with the conventional training procedure
> for **full training budget** or reduced number of epochs (e.g. 0.75B)."

This explicitly confirms: you CAN take a fully trained model and run SWA as additional
epochs. Post-training SWA is legitimate.

### Critical quote from the PyTorch blog (by Izmailov)

> "Even if you have **already trained** your model, it's easy to realize the benefits
> of SWA by running SWA for a **small number of epochs** starting with a **pre-trained model**."

### PyTorch implementation

```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

for epoch in range(swa_start, total_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(batch["image"]), batch["label"])
        loss.backward()
        optimizer.step()
    swa_model.update_parameters(model)
    swa_scheduler.step()

update_bn(train_loader, swa_model)  # Recalibrate BN
```

### Typical hyperparameters

| Parameter | CIFAR (VGG/WRN) | ImageNet (ResNet) |
|-----------|----------------|-------------------|
| `swa_start` | 75% of epochs | 75% of epochs |
| `swa_lr` | 0.01 (constant) | 0.001-1e-5 (cyclic) |
| Cycle length | 1 epoch | 1 epoch |
| BN update | 1 pass over training data | 1 pass |

### Effect on MinIVess

SWA fits naturally into our **post-training flow** (Flow 2.5):
- Load fully-trained checkpoint from `train_flow`
- Resume training with `SWALR` for N additional epochs
- Requires: training data (DataLoaders), model architecture, optimizer
- Produces: single improved checkpoint

---

## 2. SWAG — SWA-Gaussian (Stochastic Weight Averaging Gaussian)

**Paper**: [Maddox et al. (2019). "A Simple Baseline for Bayesian Deep Learning." NeurIPS.](https://arxiv.org/abs/1902.02476)
**Code**: [github.com/wjmaddox/swa_gaussian](https://github.com/wjmaddox/swa_gaussian)

### Algorithm

SWAG extends SWA by additionally tracking the **second moments** of weights during
the SWA collection phase, constructing a low-rank-plus-diagonal Gaussian approximation
to the posterior:

1. Same Phase 1 + Phase 2 as SWA
2. Additionally collect: running mean, running squared mean, and a low-rank deviation matrix
3. At inference: **sample** from the Gaussian posterior to get multiple models
4. Average predictions across samples for uncertainty quantification

```python
# During SWA phase (Phase 2):
for epoch in range(swa_start, total_epochs):
    # ... train as in SWA ...
    swag_model.collect_model(model)  # Track mean, sq_mean, deviations

# At inference:
for i in range(n_samples):
    swag_model.sample()  # Draw from Gaussian
    pred = swag_model(x)
    predictions.append(pred)
ensemble_pred = torch.stack(predictions).mean(0)
```

### Key difference from SWA

- SWA produces **1 model** (the average)
- SWAG produces a **distribution over models** — sample K models at inference
- SWAG provides **calibrated uncertainty estimates** (critical for medical imaging)

### Why SWAG for MinIVess (publication gate)

For a Nature Protocols paper on biomedical segmentation:
- Uncertainty quantification is clinically relevant
- SWAG's posterior approximation enables predictive uncertainty maps
- Single training run → multiple model samples → calibrated confidence

---

## 3. Multi-SWA and Multi-SWAG

**Paper**: [Wilson & Izmailov (2020). "Bayesian Deep Learning and a Probabilistic Perspective of Generalization."](https://arxiv.org/abs/2002.08791)
**Code**: [github.com/izmailovpavel/understandingbdl](https://github.com/izmailovpavel/understandingbdl)

### Algorithm

Train M independent SWA (or SWAG) models with different:
- Random seeds
- Hyperparameter settings (LR, weight decay)
- Data ordering

Then ensemble their predictions (for Multi-SWA) or combine their posterior samples
(for Multi-SWAG).

### Multi-SWAG specifically

Each of M SWAG models produces K samples → M×K total models averaged. This provides
the best-calibrated uncertainty in the Wilson & Izmailov experiments.

### Typical usage

```bash
# From understandingbdl repo:
python run_swag.py --epochs=300 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --seed=1
python run_swag.py --epochs=300 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --seed=2
python run_swag.py --epochs=300 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --seed=3
# → 3 independent SWAG models, ensemble at inference
```

---

## 4. PSWA — Periodic SWA

**Paper**: [Guo et al. (2022). "SWA Revisited: Towards a Better Understanding." arXiv:2201.00519.](https://arxiv.org/abs/2201.00519)
**Code**: [github.com/ZJLAB-AMMI/PSWA](https://github.com/ZJLAB-AMMI/PSWA)

### Key insight

> "SWA starts **after a converged SGD** (namely the backbone SGD), which outputs
> a local optimum."

PSWA applies SWA **multiple times sequentially**. After each SWA phase produces an
averaged model, that model becomes the starting point for another round of SGD + SWA.

Finding: "Following an SGD process with insufficient convergence, running SWA more
times leads to continual incremental benefits."

---

## 5. LAWA — Latest Weight Averaging

**Paper**: [Kaddour (2022). "Stop Wasting My Time! Saving Days of ImageNet and BERT Training." NeurIPS HITY.](https://arxiv.org/abs/2209.14981)
**Code**: [github.com/JeanKaddour/LAWA](https://github.com/JeanKaddour/LAWA)

### Algorithm

Online algorithm that maintains a **rolling average of the k latest checkpoints**
during training. Unlike SWA, no cyclic LR — just average the last k snapshots
at any point during normal training.

```python
# During normal training:
if epoch % save_interval == 0:
    checkpoints.append(model.state_dict())
    if len(checkpoints) > k:
        checkpoints.pop(0)
    lawa_model = average(checkpoints[-k:])
```

### Key difference from SWA

- No modified LR schedule
- Rolling window (only last k), not all collected snapshots
- Can be applied during training with zero overhead
- Complementary to SWA (can combine both)

---

## 6. Model Soups

**Paper**: [Wortsman et al. (2022). "Model Soups: Averaging Weights of Multiple Fine-Tuned Models." ICML.](https://arxiv.org/abs/2203.05482)

### Algorithm

Average weights from **independently trained models** with different hyperparameters.
Variants:

- **Uniform soup**: Average all models equally
- **Greedy soup**: Iteratively add models that improve validation accuracy
- **Learned soup**: Learn mixing coefficients

### Key difference from SWA

- Models are trained **independently** (different hyperparameters, possibly different data)
- No cyclic LR or modified training procedure
- Pure post-hoc weight-space averaging
- Works best when models share the same pre-trained initialization (e.g., fine-tuned from CLIP)

---

## 7. Checkpoint Averaging (What Our Code Actually Does)

**Reference**: [timm `avg_checkpoints.py`](https://github.com/huggingface/pytorch-image-models/blob/main/avg_checkpoints.py)

### Algorithm

Load N saved checkpoint files from a single training run, compute arithmetic mean of
their state dicts. No training, no gradient computation, no LR schedule.

```python
state_dicts = [torch.load(p) for p in checkpoint_paths]
averaged = {k: sum(sd[k] for sd in state_dicts) / len(state_dicts) for k in state_dicts[0]}
```

### Our specific variant (MinIVess)

Our training saves 7 checkpoints per run, each the best according to a different metric:
- `best_val_loss.pth`
- `best_val_dice.pth`
- `best_val_cldice.pth`
- `best_val_f1_foreground.pth`
- `best_val_compound_masd_cldice.pth`
- `best_val_masd.pth`
- `last.pth`

Averaging these produces weights very similar to any single checkpoint because they all
come from the same training trajectory (typically from nearby epochs near convergence).
The debug run confirmed: <0.001 DSC difference between averaged and single checkpoint.

### Distinction from real SWA

| Property | Checkpoint Averaging | Real SWA |
|----------|---------------------|----------|
| Training required | No | Yes (additional epochs) |
| LR schedule | None | Cyclic or constant |
| Gradient computation | None | Full backward passes |
| Weight diversity | Low (same trajectory) | High (cyclic LR explores basins) |
| BN recalibration | Not needed | Required (`update_bn()`) |
| Typical improvement | <0.1% | 0.5-2% |

---

## 8. Related Methods

### SeWA — Selective Weight Average

**Paper**: [Wang et al. (2025). "SeWA: Selective Weight Average via Probabilistic Masking." arXiv:2502.10119.](https://arxiv.org/abs/2502.10119)

Learns which checkpoints to include in the average using Gumbel-Softmax masking.

### TWA — Trainable Weight Averaging

**Paper**: [Li et al. (2023). "Trainable Weight Averaging." ICLR.](https://arxiv.org/abs/2205.13104)
**Code**: [github.com/nblt/TWA](https://github.com/nblt/TWA)

Learns optimal weighting coefficients for historical checkpoints rather than uniform averaging.

### EMA — Exponential Moving Average

Standard technique: maintain `theta_ema = alpha * theta_ema + (1-alpha) * theta` during
training. Used by default in many frameworks (e.g., timm's `ModelEmaV3`). Distinct from
SWA in that EMA uses exponential weighting with a decay factor, not uniform averaging
with cyclic LR.

---

## 9. SWA in Medical Image Segmentation

### Skin lesion segmentation
[Jahanifar et al. (2019). "Efficient Skin Lesion Segmentation Using Separable-Unet with Stochastic Weight Averaging."](https://pubmed.ncbi.nlm.nih.gov/31416556/)
Computer Methods and Programs in Biomedicine. Used real SWA during training. Dice 93.03% on ISIC 2016.

### Brain tumor segmentation (BraTS 2020)
[Henry et al. (2021). "Brain Tumor Segmentation with Self-ensembled, Deeply-Supervised 3D U-Net Neural Networks."](https://arxiv.org/abs/2011.01045)
Top-10 BraTS 2020 result. Used SWA during training of multiple 3D U-Net models.

### nnU-Net
Does NOT use SWA. Uses 5-fold cross-validation with softmax output ensembling at
inference time instead. This is an output-space ensemble, not weight-space averaging.

---

## 10. Decision for MinIVess Publication

### Post-training method for full factorial design

**Selected: SWAG** (Maddox et al. 2019)

Rationale:
- Provides uncertainty quantification (clinically relevant for vessel segmentation)
- Uses `torch.optim.swa_utils` (library-first, Rule #3)
- Post-training application: load trained checkpoint → resume with SWALR + track moments
- Single training run → K posterior samples → calibrated predictions
- Well-cited (NeurIPS 2019, 1000+ citations)

### Factorial factor update

```yaml
post_training:
  method:
    - none                # Baseline: use training checkpoint as-is
    - swag                # Maddox et al. 2019: SWA + Gaussian posterior
```

Checkpoint averaging is **removed** from the publication factorial design:
- Produces negligible improvement (<0.001 DSC)
- Not a real algorithm with literature backing for this use case
- Kept in codebase as utility (renamed properly) but not as a factorial factor

### Implementation location

SWAG requires **resumed training** (forward + backward passes). It belongs in the
post-training flow but needs DataLoaders and optimizer access:

```
train_flow → checkpoint
                ↓
post_training_flow → load checkpoint + training data
                     resume training with SWALR for N epochs
                     track first/second moments (SWAG)
                     update_bn()
                     save SWAG model
```

---

## References

1. [Izmailov et al. (2018). "Averaging Weights Leads to Wider Optima and Better Generalization." UAI.](https://arxiv.org/abs/1803.05407)
2. [Maddox et al. (2019). "A Simple Baseline for Bayesian Deep Learning." NeurIPS.](https://arxiv.org/abs/1902.02476)
3. [Wilson & Izmailov (2020). "Bayesian Deep Learning and a Probabilistic Perspective of Generalization."](https://arxiv.org/abs/2002.08791)
4. [Guo et al. (2022). "SWA Revisited: Towards a Better Understanding."](https://arxiv.org/abs/2201.00519)
5. [Kaddour (2022). "Stop Wasting My Time! Saving Days of ImageNet and BERT Training." NeurIPS HITY.](https://arxiv.org/abs/2209.14981)
6. [Wortsman et al. (2022). "Model Soups: Averaging Weights of Multiple Fine-Tuned Models." ICML.](https://arxiv.org/abs/2203.05482)
7. [Wang et al. (2025). "SeWA: Selective Weight Average via Probabilistic Masking."](https://arxiv.org/abs/2502.10119)
8. [Li et al. (2023). "Trainable Weight Averaging." ICLR.](https://arxiv.org/abs/2205.13104)
9. ["When, Where and Why to Average Weights?" (2025).](https://arxiv.org/abs/2502.06761)
10. [PyTorch SWA blog by Izmailov.](https://pytorch.org/blog/pytorch-1-6-now-includes-stochastic-weight-averaging/)
11. [github.com/timgaripov/swa](https://github.com/timgaripov/swa)
12. [github.com/izmailovpavel/understandingbdl](https://github.com/izmailovpavel/understandingbdl)
13. [github.com/wjmaddox/swa_gaussian](https://github.com/wjmaddox/swa_gaussian)
14. [github.com/ZJLAB-AMMI/PSWA](https://github.com/ZJLAB-AMMI/PSWA)
15. [github.com/JeanKaddour/LAWA](https://github.com/JeanKaddour/LAWA)
