# Metalearning: MetricsReloaded Decision Not Traced to Specific Metric Selection

**Date:** 2026-03-17
**Severity:** HIGH — asked user which metrics to use when the answer was in the repo
**Trigger:** User said "Did you see the Metrics Reloaded report? Is that explicitly
mentioned in our kg?"

---

## What Happened

1. Claude asked "which metrics for ANOVA?" with 4 options including "all 8 metrics"
2. The answer was already in the repo:
   - `docs/MetricsReloaded.html` — the actual MetricsReloaded questionnaire output
   - `knowledge-graph/decisions/L3-technology/primary_metrics.yaml` — resolved to metricsreloaded_full
   - `knowledge-graph/manuscript/methods.yaml` line 118: "Primary: DSC, clDice, MASD"
3. But the KG does NOT explicitly say:
   - **clDice and MASD are the TRUSTED metrics** (recommended by MetricsReloaded)
   - **DSC (Dice) is included as a FOIL** — commonly used but potentially misleading
   - The scientific narrative: "see how poorly Dice ranks models compared to topology-aware metrics"

## Root Cause

The KG decision node `primary_metrics.yaml` says "metricsreloaded_full" but does NOT:
1. List the specific metrics the MetricsReloaded report recommended
2. Distinguish trusted metrics (clDice, MASD) from commonly-used-but-misleading ones (Dice)
3. Link to the actual report file (`docs/MetricsReloaded.html`)
4. Explain the scientific RATIONALE for the 3-metric selection

## Fix

Update `primary_metrics.yaml` to explicitly document:
- **Trusted** (MetricsReloaded-recommended): clDice, MASD
- **Foil** (commonly used, potentially misleading): DSC/Dice
- **Optional** (available but not primary): HD95, ASSD, NSD, BE_0, BE_1, junction_F1
- Link to `docs/MetricsReloaded.html`

## Rule

**Before asking about metrics, read `primary_metrics.yaml` + `MetricsReloaded.html`.**
The MetricsReloaded questionnaire IS the ground truth for metric selection.
