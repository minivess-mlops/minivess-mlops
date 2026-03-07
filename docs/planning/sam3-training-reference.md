# SAMv3 Training Reference

## Quick Start

```bash
# Full training: 100 epochs on all 3 variants
./scripts/train_sam3_all_variants.sh

# Quick test: 1 epoch each (smoke test)
./scripts/train_sam3_all_variants.sh --debug

# 50 epochs with explicit GPU profile
./scripts/train_sam3_all_variants.sh --epochs 50 --compute gpu_low
```

## Epochs: 50 vs 100?

**Recommendation: 100 epochs**

### Evidence from MiniVess experiments:
- **DynUNet (2026-02-25):** All loss variants converged cleanly by epoch 100
- **Training cost:** 100 epochs ≈ 4–6 hours on A100 GPU per model × 3 variants = 12–18 hours total
- **Metric stability:** Val metrics stabilized by epoch 80, plateau by epoch 100

### When to use 50 epochs:
- **Quick prototyping** — early architectural decisions or loss tuning
- **Resource-constrained** — limited GPU hours, laptop testing
- **Resume capability** — train 50, checkpoint, extend to 100 later

### When to use 100 epochs:
- **Final evaluation** — paper results, champion selection
- **Baseline establishment** — SAMv3 vs DynUNet comparison
- **Archive runs** — reproducible experiments for publication

## The 3 SAMv3 Variants

| Variant | Architecture | Trainable Params | VRAM | Expected DSC | Go/No-Go Gate |
|---------|-------------|------------------|------|--------------|---------------|
| **sam3_vanilla** | Frozen ViT-32L + decoder | ~3–5M | ~5 GB | 0.35–0.55 | DSC ≥ 0.10 |
| **sam3_topolora** | ViT + LoRA (r=16) + topology loss | ~1–2M | ~6 GB | +10–20% clDice | clDice Δ ≥ 2% |
| **sam3_hybrid** | Frozen ViT + DynUNet + fusion | ~25–30M | ~7.5 GB | Best SAM variant | DSC_V3 > DSC_V2 |

### Key differences:
- **V1 (vanilla):** Minimal SAM adaptation — tests raw foundation model performance
- **V2 (topolora):** Parameter-efficient fine-tuning with topology awareness
- **V3 (hybrid):** Strongest SAM variant — fusion with 3D context

## Compute Profiles

```bash
# Auto-detect optimal profile for your GPU
./scripts/train_sam3_all_variants.sh --compute auto

# Explicit profiles
--compute gpu_low    # 1 GPU, batch_size=1, mixed precision (8GB safe)
--compute gpu_mid    # 1 GPU, batch_size=2, AMP (24GB Vram)
--compute gpu_full   # 1 GPU, batch_size=4, AMP (optimal 40GB+)
--compute cpu        # CPU-only (not recommended, very slow)
```

## Typical Training Progress

**Epoch 0–10:** Loss decreases sharply, metrics noisy
**Epoch 10–40:** Metrics stabilize, good progress on validation
**Epoch 40–80:** Diminishing returns, slow improvement
**Epoch 80–100:** Plateau, occasionally minor overfitting on training set

### Early stopping candidates (if time-constrained):
- **Epoch 50:** ~85–90% of final metric performance
- **Epoch 75:** ~95% of final performance

## Outputs

```
logs/sam3_variants/
├── sam3_vanilla/
│   ├── logs/
│   │   ├── training.log        # Detailed training log
│   │   ├── metrics.json        # Per-epoch metrics
│   │   └── checkpoints/        # Periodic saves
│   └── mlruns/                 # MLflow run artifacts
├── sam3_topolora/
│   └── ...
└── sam3_hybrid/
    └── ...
```

## Next Steps After Training

```bash
# 1. Inspect metrics from MLflow
uv run python scripts/generate_comparison_table.py

# 2. Compare SAMv3 vs DynUNet baseline
uv run python scripts/compare_models.py \
    --baseline dynunet_cbdice_cldice \
    --variants sam3_vanilla sam3_topolora sam3_hybrid

# 3. Tag champions (best per metric)
uv run python scripts/tag_champions.py --experiment sam3_evaluation

# 4. Export for paper/analysis
uv run python scripts/export_duckdb_parquet.py
```

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size or use lower compute profile
./scripts/train_sam3_all_variants.sh --compute gpu_low

# For sam3_hybrid specifically, AMP is mandatory
# (already enabled in gpu_low profile)
```

### Training stalls / slow progress
```bash
# Check GPU utilization
nvidia-smi dmon -s pcum

# Verify dataset is fully loaded (check first epoch time)
tail -f logs/sam3_variants/sam3_vanilla/logs/training.log
```

### Resume after crash
```bash
# Resume from last checkpoint
./scripts/train_sam3_all_variants.sh --resume
```

## References

- **SAM3 Paper:** Ravi et al. (2025). "SAM 3." arXiv:2511.16719
- **TopoLoRA-SAM:** Xiang et al. arXiv:2601.02273
- **nnSAM pattern:** Li et al. (2025)
- **MiniVess baseline:** DynUNet with cbdice_cldice (CLAUDE.md)
