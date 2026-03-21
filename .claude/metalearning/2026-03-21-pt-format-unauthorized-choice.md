# Metalearning: Unauthorized .pt Format Choice for 5D UQ Maps (2026-03-21)

## What Happened

When implementing UncertaintyStoragePolicy (#887), Claude chose `torch.save()` (.pt)
for saving 5D uncertainty maps without:
1. Asking the user what format they wanted
2. Researching alternatives (zarr, HDF5, NIfTI, mmap)
3. Considering disk space — .pt stores uncompressed PyTorch tensors

This is a storage format decision that affects the entire analysis pipeline, disk
usage (~50MB/vol uncompressed), and downstream consumers (dashboard, biostatistics).
Claude made it unilaterally based on "it's the simplest thing that works."

## Why This Was Wrong

- **Rule #29**: ZERO Hardcoded Parameters — storage format is a researcher choice
- **Session summaries ≠ authorization** — the issue description said "full 5D maps"
  but never specified the format
- **Library-first (Rule #3)**: zarr, HDF5, and NIfTI are purpose-built for N-D arrays.
  `.pt` is for serializing PyTorch model state, not for data arrays.
- **TOP-2 (Reproducibility)**: .pt files are tied to PyTorch version and pickle.
  zarr/HDF5 are language-agnostic and self-describing.

## The Cost

- `.pt` uses Python pickle — security risk, version fragility
- No compression — 50MB/vol vs ~5-15MB/vol with zarr (blosc)
- No chunking — can't read a single slice without loading the full volume
- No metadata — just raw tensors in a dict, no units, no coordinate info
- Dashboard/R/Julia consumers can't read .pt without PyTorch

## Rule for Future Sessions

**Before choosing a storage format for data arrays:**
1. ASK the user what format they prefer
2. Research alternatives (zarr, HDF5, NIfTI, Parquet for tabular)
3. Consider downstream consumers — who reads this data?
4. Consider compression, chunking, and language-agnosticism
5. `.pt` is for MODEL state dicts ONLY — never for data arrays

"Simplest thing that works" is not a valid justification for infrastructure decisions
that affect reproducibility and interoperability.
