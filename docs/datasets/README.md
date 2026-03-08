# MinIVess MLOps — Datasets

**Single authoritative reference** for all datasets used in this project.
Code registry: `src/minivess/data/external_datasets.py`
Split definitions: `configs/splits/3fold_seed42.json`

---

## Dataset Summary

| Dataset | Role | Volumes | Modality | Spacing (μm) | License |
|---------|------|---------|----------|--------------|---------|
| **MiniVess** | Primary training | 70 | 2PM mouse brain vasculature | 0.31–4.97 (XY~1.0, Z~1.5) | CC BY-NC-SA |
| **DeepVess** | External test | ~6 sub-vols | Multi-photon mouse brain | 1.00×1.00×1.70 | eCommons-educational |
| **TubeNet 2PM** | External test | ~2 sub-vols | Two-photon mouse brain | 0.20×0.46×5.20 | CC-BY-4.0 ✓ |
| **VesselNN** | External test | 12 | Two-photon mouse brain | 0.50×0.50×1.00 | MIT ✓ |

---

## 1. MiniVess (Primary Training Dataset)

**Citation**: Poon et al. (2023). "A dataset of rodent cerebrovasculature from in vivo
multiphoton fluorescence microscopy imaging." *Scientific Data* 10:141.
DOI: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8)

**EBRAINS Dataset ID**: `bf268b89-1420-476b-b428-b85a913eb523`
**Direct URL**: https://search.kg.ebrains.eu/instances/bf268b89-1420-476b-b428-b85a913eb523
**EBRAINS API**: `https://data-proxy.ebrains.eu/api/v1/datasets/bf268b89-1420-476b-b428-b85a913eb523`

### Properties
- **Volumes**: 70 two-photon fluorescence microscopy volumes
- **Anatomy**: Mouse cerebral microvasculature (cortex + hippocampus)
- **XY dimensions**: 512×512 pixels
- **Z slices**: 5–110 (median ~22)
- **Voxel spacing**: XY ~0.31–1.23 μm, Z ~0.5–4.97 μm (highly variable)
- **Known outlier**: `mv02` has Z-spacing of 4.97 μm — causes OOM with `Spacingd(pixdim=(1,1,1))`
  - **Fix**: `voxel_spacing=(0,0,0)` in config disables Spacingd entirely
- **Compressed size**: ~984 MB ZIP
- **Uncompressed**: ~4.5 GB

### Cross-Validation Splits
- **Strategy**: 3-fold, stratified by volume ID, seed=42
- **Definition file**: `configs/splits/3fold_seed42.json` (committed to repo — deterministic)
- **Sizes**: 47 train + 23 val per fold
- **Test set**: None from MiniVess — use external datasets for generalization

### Download
```bash
# Primary method (EBRAINS API):
uv run python scripts/download_minivess.py

# With local ZIP:
uv run python scripts/download_minivess.py --zip-path dataset_local/d-bf268b89-....zip

# DVC pull (if remote configured):
dvc pull data/minivess.dvc
```

### DVC Tracking
- **DVC file**: `data/minivess.dvc` (211 files, 984 MB, md5-tracked)
- **HuggingFace Hub remote**: `hf://datasets/minivess/minivess-data`
- **Local remote**: `configs/dvc/remotes.yaml`

### On-Disk Layout (after `download_minivess.py`)
```
data/raw/minivess/
├── imagesTr/   mv01.nii.gz … mv70.nii.gz  (70 images)
├── labelsTr/   mv01.nii.gz … mv70.nii.gz  (70 binary masks)
└── metadata/   mv01.json  … mv70.json     (70 JSON metadata)
```

---

## 2. DeepVess (External Test Dataset)

**Citation**: Haft-Javaherian et al. (2019). "Deep convolutional neural networks for
segmenting 3D in vivo multiphoton images of vasculature in Alzheimer disease mouse models."
*PLOS ONE* 14(3): e0213539. DOI: [10.1371/journal.pone.0213539](https://doi.org/10.1371/journal.pone.0213539)

**Source URL**: https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3
**License**: eCommons-educational (`license_verified=False` — verify before redistribution)

### Properties
- **Volumes**: 1 large volume → decomposed into ~6 sub-volumes for inference
- **Anatomy**: Mouse brain vasculature (Alzheimer model)
- **Voxel spacing**: 1.00 × 1.00 × 1.70 μm
- **Purpose**: External generalization test — never used in training

### Download
```python
from minivess.data.external_datasets import download_external_dataset
download_external_dataset("deepvess", output_dir=Path("data/external/deepvess"))
```

---

## 3. TubeNet 2PM (External Test Dataset)

**Citation**: Holroyd et al. (2025). "tUbeNet: a deep learning tool for 3D vessel
segmentation." *Biology Methods and Protocols* 10(1): bpaf009.
DOI: [10.1093/biomethods/bpaf009](https://doi.org/10.1093/biomethods/bpaf009)

**Source URL**: https://rdr.ucl.ac.uk/articles/dataset/3D_Microvascular_Image_Data_and_Labels_for_Machine_Learning/25715604
**License**: CC-BY-4.0 ✓ (verified)

### Properties
- **Volumes**: 1 large volume → decomposed into ~2 sub-volumes
- **Anatomy**: Mouse brain microvasculature (two-photon)
- **Voxel spacing**: 0.20 × 0.46 × 5.20 μm (highly anisotropic Z)
- **Purpose**: External generalization test — never used in training

---

## 4. VesselNN (External Test Dataset)

**Citation**: Teikari et al. (2016). "Deep learning convolutional networks for multiphoton
microscopy vasculature segmentation." arXiv:1606.02382.
DOI: [10.48550/arXiv.1606.02382](https://doi.org/10.48550/arXiv.1606.02382)

**Source URL**: https://github.com/petteriTeikari/vesselNN
**License**: MIT ✓ (verified)

### Properties
- **Volumes**: 12 two-photon microscopy volumes
- **Anatomy**: Mouse brain vasculature
- **Voxel spacing**: 0.50 × 0.50 × 1.00 μm
- **Purpose**: External generalization test — never used in training

---

## Data Leakage Warning

**VesselFM** (`bwittmann/vesselFM`, CVPR 2025) was pre-trained on 17 datasets including
MiniVess (1/17). **Evaluating VesselFM on MiniVess constitutes data leakage** — results
should be reported with this caveat. The experiment config tags this:

```yaml
# configs/experiment/vesselfm_finetune.yaml
tags:
  data_leakage: pretrained_includes_minivess
```

External test datasets (DeepVess, TubeNet 2PM, VesselNN) are **not** in VesselFM's
pre-training set and can be used for clean zero-shot evaluation.

---

## Dataset Selection Rationale

Only **multiphoton / two-photon microscopy** datasets of mouse brain vasculature are
included. Excluded: light-sheet, EM, CT, MRA, OCTA, human vascular data. This ensures
modality-matched generalization testing consistent with the paper's scope.

---

## Future / Planned (not yet implemented)

- **Real-time acquisition integration**: See issue #328 — discussion of adaptive
  acquisition platforms referenced in the paper Discussion section.
- **Automated DVC versioning**: New volumes from the annotation workflow (#330)
  auto-committed with DVC on approval.
- **HuggingFace Hub push**: `dvc push --remote hf` after any dataset update.
