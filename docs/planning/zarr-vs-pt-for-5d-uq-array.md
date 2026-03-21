---
title: "Multi-Hypothesis Decision Matrix: 5D UQ Map Storage Format"
status: active
created: "2026-03-21"
reviewers:
  - factual-correctness: "All compression ratios cite published benchmarks"
  - option-breadth: "6 formats compared across 12 criteria"
  - visualization: "3D rendering support in ImageJ/Fiji/3D Slicer/Napari"
---

# Storage Format for 5D Uncertainty Maps — Decision Matrix

## Context

The analysis flow produces 5D uncertainty maps (B, 1, D, H, W) from deep ensemble
predictions (Lakshminarayanan et al., 2017): total predictive entropy, aleatoric
uncertainty, and epistemic uncertainty (mutual information).

**Reference tensor**: `(1, 1, 256, 256, 256)` float32 = **64.0 MB raw**
**Production scale**: ~70 volumes × 3 maps = 210 tensors × 64 MB = **~13.4 GB raw**

The current implementation uses `torch.save()` (.pt) — this was an unauthorized
choice that needs to be corrected. This report evaluates 6 alternatives.

---

## Candidate Formats

### 1. torch.save (.pt) — Current (Wrong) Choice

| Criterion | Assessment |
|-----------|------------|
| **Compression** | None built-in. File ≈ raw tensor size (~64 MB). |
| **Random access** | None. Full file must be loaded via pickle. |
| **Language support** | Python-only (pickle). No R, Julia, MATLAB. |
| **Metadata** | Arbitrary Python objects via pickle — security risk. |
| **Thread safety** | Not thread-safe. Holds GIL. |
| **Security** | **CRITICAL**: Pickle allows arbitrary code execution. Malicious .pt files have been used to deploy RATs via Hugging Face ([Snyk](https://snyk.io/articles/python-pickle-poisoning-and-backdooring-pth-files/)). |
| **3D Visualization** | None. No viewer supports .pt. |
| **Verdict** | Unsuitable. Security risk, no compression, no interop. |

### 2. Zarr (zarr-python)

| Criterion | Assessment |
|-----------|------------|
| **Compression** | Best-in-class. Blosc2 with sub-codecs (LZ4, Zstd, Snappy), shuffle/bitshuffle/bytedelta filters. Zarr v3 default: Zstandard. |
| **Compression ratios** | Microscopy benchmarks ([PMC9900847](https://pmc.ncbi.nlm.nih.gov/articles/PMC9900847/)): Blosc-Zstd-shuffle: **2.7x ± 0.8**. ERA5 float32 climate data: bytedelta+Zstd: **5.7x**. Uncertainty maps (spatially smooth) likely **3-6x**. |
| **Est. size (64 MB raw)** | **~11-21 MB** |
| **Random access** | First-class. Each chunk is a separate file. `(1,1,64,64,64)` chunking = 64 independent chunks. Single-slice reads touch only relevant chunks. |
| **Language support** | Python, R ([zarr CRAN](https://cran.r-project.org/package=zarr), [Rarr Bioconductor](https://github.com/grimbough/Rarr)), Julia ([Zarr.jl](https://github.com/JuliaIO/Zarr.jl)), MATLAB ([MathWorks](https://github.com/mathworks/MATLAB-support-for-Zarr-files)), C/C++, Rust, JavaScript. |
| **Metadata** | Rich JSON at array and group levels. Full xarray/CF-convention integration (units, axis names, coordinates). |
| **Thread safety** | Thread-safe reads. Chunk-aligned writes need no synchronization. |
| **Cloud-native** | Yes — directory of files, directly addressable on GCS/S3. Each chunk is an independent HTTP object. |
| **Biomedical standard** | **OME-NGFF** ([Moore et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9980008/)) is the community-endorsed next-gen bioimaging format. |
| **3D Visualization** | **Napari**: Native OME-Zarr support via `napari-ome-zarr` plugin — direct volumetric rendering with contrast/opacity controls. **Fiji/ImageJ**: Via `OME Zarr` reader (MoBIE plugin or n5-zarr bridge). **3D Slicer**: Not native, but `nifti-zarr` bridge spec exists ([neuroscales/nifti-zarr](https://github.com/neuroscales/nifti-zarr)). **QuPath**: Native OME-Zarr support for whole-slide images. |
| **Verdict** | **Strong recommendation.** Best compression, cloud-native, chunked access, growing biomedical adoption. |

### 3. HDF5 (h5py)

| Criterion | Assessment |
|-----------|------------|
| **Compression** | Good. Built-in gzip/szip. Third-party: Blosc2, LZ4, Zstd via hdf5plugin. |
| **Compression ratios** | gzip+shuffle on structured data: **2-4x**. With Blosc2: comparable to Zarr. |
| **Est. size (64 MB raw)** | **~16-32 MB** |
| **Random access** | Yes, via chunked datasets with B-tree indexing. All chunks in single monolith file. |
| **Language support** | **Best-in-class.** Python (h5py), R (rhdf5, hdf5r), Julia (HDF5.jl), MATLAB (`h5read` native), C/C++, Fortran, Java. MATLAB .mat v7.3 files ARE HDF5. |
| **Metadata** | Arbitrary key-value attributes on datasets/groups. Hierarchical structure. |
| **Thread safety** | Weak. Global library lock. SWMR mode available but restricted. |
| **Cloud-native** | No — single monolith file. |
| **3D Visualization** | **Fiji/ImageJ**: Native HDF5 reading via `HDF5 Vibez` or `BigDataViewer`. **3D Slicer**: Via `SlicerHDF5` extension. **ParaView**: Native HDF5/XDMF support for volume rendering. **Napari**: Via `napari-hdf5` plugin. |
| **Verdict** | Solid for local/HPC. Loses to Zarr on cloud-native and concurrent writes. |

### 4. NIfTI (.nii.gz) via nibabel

| Criterion | Assessment |
|-----------|------------|
| **Compression** | gzip only. Applied to entire file as sequential stream. |
| **Compression ratios** | Neuroimaging volumes: **3-8x** (large zero-background regions compress well). |
| **Est. size (64 MB raw)** | **~8-21 MB** |
| **Random access** | **None.** gzip is sequential — reading one slice decompresses from the start. |
| **Language support** | Python (nibabel), R (RNifti), Julia (NIfTI.jl), MATLAB (`niftiread` native), FSL, FreeSurfer, SPM, ANTs, ITK. |
| **Metadata** | Fixed 348-byte header (NIfTI-1). Affine transform (sform/qform), voxel dimensions, datatype. 80-char description field. No arbitrary key-value. |
| **Thread safety** | N/A — each file is independent. |
| **Cloud-native** | No. |
| **3D Visualization** | **GOLD STANDARD for 3D viewers.** **3D Slicer**: Native — loads directly, volume rendering with GPU raycasting, overlay uncertainty on anatomy. **Fiji/ImageJ**: Native via NIfTI plugin. **ITK-SNAP**: Native — interactive 3D segmentation with overlay. **FSLeyes**: Native — designed for NIfTI, supports overlay of multiple volumes (anatomy + uncertainty). **FreeSurfer FreeView**: Native. **ParaView**: Via SimpleITK/VTK. **MONAI**: `LoadImage` natively reads NIfTI. |
| **5D handling** | NIfTI header supports up to 7 dimensions via `dim[0..7]`. For 5D data, viewers typically present it as a 3D volume with a "4th dimension slider" (time or channel). **FSLeyes** and **3D Slicer** both support browsing through extra dimensions. However, most viewers expect 3D or 4D (3D+time), so a pure 3D uncertainty volume is the most viewer-friendly format. |
| **Best practice** | Save each uncertainty component as a separate 3D NIfTI: `vol_042_total.nii.gz`, `vol_042_aleatoric.nii.gz`, `vol_042_epistemic.nii.gz`. This lets a neuroimaging researcher open them as overlays in any viewer without custom scripts. The affine transform in the header ensures spatial alignment. |
| **Verdict** | **Best for human visualization.** Every neuroimaging tool reads NIfTI. Must be the export format for collaborator-facing deliverables. Poor for pipeline-internal storage (no chunking, no metadata). |

### 5. NumPy .npz (savez_compressed)

| Criterion | Assessment |
|-----------|------------|
| **Compression** | ZIP+deflate only. Level 6 default. |
| **Compression ratios** | Float arrays: **1.4-4x**. Poor for random data, better for smooth. |
| **Est. size (64 MB raw)** | **~16-45 MB** |
| **Random access** | None. Per-array access possible but each array fully decompressed. |
| **Language support** | Python-only. No R, Julia, MATLAB. |
| **Metadata** | None. |
| **3D Visualization** | None. No viewer supports .npz. |
| **Verdict** | Throwaway intermediates only. Not suitable for a platform. |

### 6. safetensors (Hugging Face)

| Criterion | Assessment |
|-----------|------------|
| **Compression** | **None by design.** Raw binary for zero-copy mmap loading. |
| **Est. size (64 MB raw)** | **~64 MB** |
| **Random access** | Per-tensor (by name), not per-slice. |
| **Language support** | Python, Rust. No R, Julia, MATLAB. |
| **Security** | Excellent — no arbitrary code execution. [Audited](https://huggingface.co/blog/safetensors-security-audit). |
| **3D Visualization** | None. |
| **Verdict** | Excellent for model weights. Wrong tool for scientific data arrays. |

---

## Decision Matrix

| Criterion (weight) | torch.save | Zarr | HDF5 | NIfTI | .npz | safetensors |
|---------------------|-----------|------|------|-------|------|-------------|
| Compression (high) | ❌ 1x | ✅ 3-6x | ⚠️ 2-4x | ⚠️ 3-8x | ⚠️ 1.4-4x | ❌ 1x |
| Chunked access (high) | ❌ | ✅ first-class | ⚠️ monolith | ❌ | ❌ | ⚠️ per-tensor |
| R/Julia/MATLAB (med) | ❌ | ✅ all three | ✅ all three | ✅ all three | ❌ | ❌ |
| Rich metadata (med) | ❌ pickle | ✅ JSON+CF | ✅ attrs | ⚠️ fixed hdr | ❌ | ⚠️ string map |
| Cloud-native (high) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 3D visualization (high) | ❌ | ⚠️ Napari/Fiji | ⚠️ Fiji/Slicer | ✅ **all viewers** | ❌ | ❌ |
| Security (med) | ❌ CRITICAL | ✅ | ✅ | ✅ | ✅ | ✅ |
| Thread safety (low) | ❌ | ✅ | ⚠️ | N/A | ❌ | ⚠️ |
| Biomedical standard (med) | ❌ | ✅ OME-NGFF | ⚠️ legacy | ✅ neuroimaging | ❌ | ❌ |
| MONAI integration (med) | ⚠️ torch | ⚠️ numpy bridge | ⚠️ h5py | ✅ native | ⚠️ numpy | ❌ |

---

## Recommendation: Dual-Format Strategy

### Primary (pipeline-internal): **Zarr** with OME-NGFF metadata

```python
import zarr

store = zarr.DirectoryStore("outputs/uncertainty/vol_042.zarr")
root = zarr.group(store)
root.create_dataset(
    "total_uncertainty",
    data=total.numpy(),
    chunks=(1, 1, 64, 64, 64),
    compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE),
)
root.attrs["volume_id"] = "vol_042"
root.attrs["uncertainty_method"] = "deep_ensemble"
root.attrs["n_members"] = 4
```

- **~4x compression**: 64 MB → ~16 MB per component
- **Cloud-ready**: Directly addressable on GCS
- **Chunked**: Dashboard reads single slices without loading full volume
- **Metadata**: JSON attrs for provenance tracking

### Secondary (collaborator export): **NIfTI (.nii.gz)**

```python
import nibabel as nib

# Save as separate 3D volumes — most viewer-friendly
nib.save(
    nib.Nifti1Image(total_uncertainty[0, 0].numpy(), affine=vol_affine),
    "outputs/nifti/vol_042_total_uncertainty.nii.gz",
)
```

- **Every viewer reads it**: 3D Slicer, FSLeyes, ITK-SNAP, Fiji, FreeSurfer
- **Overlay support**: Load anatomy + uncertainty as separate layers
- **Affine-aligned**: Spatial coordinates preserved from source volume
- **Best for**: Paper figures, collaborator review, QC visualization

### Retire: **torch.save (.pt)**

- Pickle security risk
- No compression, no chunking, no metadata
- Python-only — no cross-language access

---

## Visualization Workflow

For a researcher who wants to visualize total predictive uncertainty overlaid on
a vessel segmentation:

1. **Quick local (Napari)**: Load `.zarr` directly → volumetric rendering with
   opacity mapped to uncertainty magnitude
2. **Clinical-style (3D Slicer)**: Load `.nii.gz` → volume rendering with
   colormap transfer function → overlay on anatomy volume
3. **Batch QC (Fiji/ImageJ)**: Load `.nii.gz` → montage max-intensity projections
   of uncertainty across all volumes
4. **FSLeyes**: Load anatomy + uncertainty as overlay → slice-by-slice review
   with synchronized cursors

**Key insight**: 3D viewers universally prefer individual 3D volumes (not 5D tensors).
Saving total/aleatoric/epistemic as **separate .nii.gz files** is the most
viewer-friendly approach. The 5D batch dimension (B) should be flattened to
per-volume files.

---

## Dependencies

```toml
# pyproject.toml — already available or trivial to add
zarr = ">=3.0"      # chunked array storage
nibabel = ">=5.0"   # NIfTI I/O (already in project for MONAI)
blosc2 = ">=2.0"    # compression codec
```

Both `nibabel` and `zarr` are pure Python with no CUDA dependency.

---

## Sources

- [Zarr Performance](https://zarr.readthedocs.io/en/latest/user-guide/performance/)
- [Compression of Microscopy Data (PMC9900847)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9900847/)
- [OME-NGFF Specification (Nature Methods, 2021)](https://www.nature.com/articles/s41592-021-01326-w)
- [OME-Zarr Cloud-Optimized (PMC9980008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9980008/)
- [Bytedelta Filter Benchmarks (Blosc.org)](https://blosc.org/posts/bytedelta-enhance-compression-toolset/)
- [Safetensors Security Audit](https://huggingface.co/blog/safetensors-security-audit)
- [Pickle Poisoning (Snyk)](https://snyk.io/articles/python-pickle-poisoning-and-backdooring-pth-files/)
- [NIfTI File Format (Brainder)](https://brainder.org/2012/09/23/the-nifti-file-format/)
- [nifti-zarr Bridge (neuroscales)](https://github.com/neuroscales/nifti-zarr)
