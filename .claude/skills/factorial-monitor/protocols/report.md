# Protocol: Report

## When to Generate

- All jobs SUCCEEDED (happy path)
- Max cycles exhausted (Rule F4)
- User requests early termination

## Report Structure

```
═══════════════════════════════════════════════════════════
  FACTORIAL EXPERIMENT REPORT
  Experiment: experiment_2026-03-19_debug
  Config: configs/experiment/debug_factorial.yaml
═══════════════════════════════════════════════════════════

RESULTS
  Total conditions: 24
  Succeeded: 22 (91.7%)
  Failed: 2 (8.3%)
  Fix-relaunch cycles: 1

COST
  Cycle 0: $21.84 (24 jobs)
  Cycle 1:  $3.90 (6 jobs)
  Total:   $25.74

FACTORIAL GRID
  ┌─────────┬──────────┬──────────┬──────────┐
  │         │ fold-0   │ fold-1   │ fold-2   │
  ├─────────┼──────────┼──────────┼──────────┤
  │ dice_ce │ ✓ dynunet│ ✓ dynunet│ ✓ dynunet│
  │         │ ✓ sam3   │ ✓ sam3   │ ✗ sam3   │
  ├─────────┼──────────┼──────────┼──────────┤
  │ cbdice  │ ✓ dynunet│ ✓ dynunet│ ✓ dynunet│
  │         │ ✓ sam3   │ ✓ sam3   │ ✗ sam3   │
  └─────────┴──────────┴──────────┴──────────┘

FAILURES (if any)
  Root cause: OOM (2 jobs, sam3 × fold-2)
  Fix attempted: Reduced patch_size to 96
  Result: Still failing — SAM3 + fold-2 exceeds L4 VRAM
  Recommendation: Use A100 for SAM3 fold-2 or reduce val set

ISSUES CREATED
  #873: SAM3 fold-2 OOM on L4 — requires A100 or patch reduction

═══════════════════════════════════════════════════════════
```

## Output Files

Save to `outputs/factorial_run_<experiment_id>.jsonl`:
- One JSON object per job with all manifest fields
- Aggregated summary as last line

## GitHub Issues

For each UNRECOVERABLE failure after max cycles:
- compose_with: issue-creator
- Include: root cause, affected conditions, fixes attempted, cost incurred
- Link to the manifest file and relevant logs

## MLflow Integration

If jobs logged to MLflow, include:
- Experiment name and run IDs for all succeeded jobs
- Comparison table of key metrics (loss, dice score) across conditions
- Link to MLflow UI for detailed analysis
