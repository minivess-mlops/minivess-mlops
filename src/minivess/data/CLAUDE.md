# Data Module — AI Context

## Dataset Registry (always read this first)

Full dataset documentation: `docs/datasets/README.md`
Dataset summary is also in top-level `CLAUDE.md` under "Datasets".

**Datasets used:**
| Name | Role | Key file |
|------|------|----------|
| MiniVess | Primary training (70 vols) | `scripts/download_minivess.py` |
| DeepVess | External test only | `external_datasets.py::EXTERNAL_DATASETS["deepvess"]` |
| TubeNet 2PM | External test only | `external_datasets.py::EXTERNAL_DATASETS["tubenet_2pm"]` |
| VesselNN | External test only | `external_datasets.py::EXTERNAL_DATASETS["vesselnn"]` |

## Key Files

| File | Purpose |
|------|---------|
| `external_datasets.py` | **Authoritative registry** for DeepVess/TubeNet/VesselNN. `EXTERNAL_DATASETS` dict + download helpers |
| `loader.py` | `discover_nifti_pairs()` — supports EBRAINS layout (raw/seg) and Medical Decathlon (imagesTr/labelsTr) |
| `test_datasets.py` | `DatasetRegistry`, `DatasetEntry`, `DatasetSubset` for hierarchical evaluation |
| `debug_dataset.py` | Synthetic NIfTI generator for fast unit tests (no real data required) |
| `versioning.py` | DVC version tagging, change detection |
| `drift_synthetic.py` | Synthetic drift injection for dashboard testing |
| `acquisition_simulator.py` | Simulates time-series acquisition with configurable drift schedules |

## MiniVess-Specific Facts

- **Splits**: `configs/splits/3fold_seed42.json` — do NOT re-generate at runtime
- **mv02 outlier**: Z-spacing 4.97 μm → OOM with `Spacingd(pixdim=(1,1,1))`; always use `voxel_spacing=(0,0,0)` to disable
- **On-disk layout**: `data/raw/minivess/{imagesTr,labelsTr,metadata}/`
- **DVC**: `data/minivess.dvc` tracks 211 files

## MONAI Dimension Order

All adapters and data functions use MONAI's `(B, C, H, W, D)` convention — depth is LAST.
Never permute to `(B, C, D, H, W)` (PyTorch default). This affects all sliding-window inference.

## What AI Must NEVER Do

- Hardcode dataset names or paths — use `external_datasets.py::EXTERNAL_DATASETS`
- Assume only MiniVess exists — always check `EXTERNAL_DATASETS` for the full list
- Use `Spacingd(pixdim=(1,1,1))` without accounting for mv02 outlier
- Download datasets with `tempfile.mkdtemp()` — use `tmp_path` in tests, volume-mounted paths in Docker
