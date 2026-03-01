---
title: MinIVess MLOps Dashboard
---

# MinIVess MLOps Dashboard

Interactive dashboard for the MinIVess biomedical segmentation MLOps platform.

## Overview

This dashboard provides interactive visualizations of:

- **Loss function comparison** — Box plots and forest plots across 4 loss functions
- **Training curves** — Epoch-by-epoch convergence with uncertainty bands
- **External generalization** — Performance on DeepVess and tUbeNet 2PM datasets

## Data Source

All data is loaded from DuckDB-WASM Parquet files exported by the analysis flow.

```js
const db = await DuckDBClient.of({
  runs: FileAttachment("data/runs.parquet"),
  eval_metrics: FileAttachment("data/eval_metrics.parquet"),
});
```

## Quick Links

- [Loss Comparison](/loss-comparison)
- [Training Curves](/training-curves)
- [External Generalization](/external-generalization)
