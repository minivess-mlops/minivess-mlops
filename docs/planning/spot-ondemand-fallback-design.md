# Spot/On-Demand Fallback Design

## Decision

SAM3 models (sam3_hybrid, sam3_topolora) use **on-demand** instances (`use_spot: false`).
DynUNet and MambaVesselNet use **spot** instances (`use_spot: true`, default).

## Recommendation

Per-model spot override implemented in `configs/factorial/debug.yaml` and
`configs/factorial/paper_full.yaml` via `model_overrides.{model}.use_spot`.

## Cost Comparison

| Scenario | Spot ($) | On-Demand ($) | Breakeven |
|----------|----------|---------------|-----------|
| Debug (34 jobs, L4) | ~$6.85 | ~$14.50 | 80% preemption rate |
| Production (96 jobs, L4) | ~$19.35 | ~$40.80 | 80% preemption rate |

SAM3 models hit 80% preemption rate on 25+ min jobs, making on-demand cheaper
AND faster (no re-setup cycles). DynUNet/MambaVesselNet complete in <10 min,
so spot saves 60-91% with minimal preemption risk.

## Implementation

- `model_overrides.sam3_hybrid.use_spot: false` in factorial configs
- `model_overrides.sam3_topolora.use_spot: false` in factorial configs
- `scripts/run_factorial.sh` reads `use_spot` per-model and passes to SkyPilot

See: `.claude/metalearning/2026-03-24-unauthorized-a100-in-skypilot-yaml.md`
See: `docs/planning/v0-2_archive/original_docs/gpu-instances-finops-report.md`
