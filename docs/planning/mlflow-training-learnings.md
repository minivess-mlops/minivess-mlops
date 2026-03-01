# MLflow Training Learnings — Half-Width DynUNet v1

**Date:** 2026-02-28
**Experiment:** `dynunet_half_width_v1`
**Config:** `configs/experiments/dynunet_half_width.yaml`
**Hardware:** NVIDIA RTX 2070 SUPER (8 GB), 70 volumes, 4 losses x 3 folds x 100 epochs

## Purpose

Assess whether the half-width DynUNet (filters=[16,32,64,128] vs full [32,64,128,256])
achieves adequate segmentation quality with significantly reduced compute requirements.
Simultaneously validate the comprehensive MLflow logging improvements from PR #109.

---

## Pre-Training Checklist

- [x] Data present: 70 volumes in `data/raw/minivess/`
- [x] GPU: RTX 2070 SUPER, 8192 MiB VRAM
- [x] Config: `dynunet_half_width.yaml` with 4 losses, 3 folds, 100 epochs
- [x] MLflow enhancements: system_info, DVC provenance, dataset metadata, system metrics
- [ ] Training launched
- [ ] Training completed

---

## MLflow Logging Observations

### During Training

*(To be updated during/after training)*

### Post-Training

*(To be updated after training completes)*

---

## Training Results

*(To be updated after training completes)*

---

## Comparison: Half-Width vs Full-Width

| Metric | Full (32,64,128,256) | Half (16,32,64,128) | Delta |
|--------|---------------------|---------------------|-------|
| Parameters | ~TBD | ~TBD | |
| Peak VRAM | ~5410 MB | ~TBD | |
| Training time | ~25h | ~TBD | |
| val_dice (best) | TBD | TBD | |
| val_cldice (best) | TBD | TBD | |
| val_compound_masd_cldice | TBD | TBD | |

---

## Lessons Learned

*(To be updated)*
