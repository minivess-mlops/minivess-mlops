# Protocol: Diagnose

## Prerequisites

- ALL factorial jobs are in terminal state (Rule F1)
- Each failed job has a preliminary FailureInfo from ralph_monitor.analyze_logs()

## Aggregation (Rule F2)

Group ALL failures by root cause category:

```
Root Cause Report
─────────────────
Category: DVC_NO_GIT (4 jobs)
  Jobs: [103, 107, 111, 115]
  Conditions: sam3/cbdice/fold-0, sam3/cbdice/fold-1, sam3/cldice/fold-0, sam3/cldice/fold-1
  Root cause: .dvc not initialized in container
  Fix strategy: Add dvc init to Docker entrypoint
  Auto-fixable: YES
  Confidence: HIGH

Category: OOM (2 jobs)
  Jobs: [109, 113]
  Conditions: sam3/cbdice/fold-2, sam3/cldice/fold-2
  Root cause: GPU VRAM exhausted at fold-2 (more data)
  Fix strategy: Reduce batch_size or patch_size for SAM3
  Auto-fixable: NO (requires config decision)
  Confidence: MEDIUM

Summary: 18/24 SUCCEEDED, 6 FAILED
  Root causes: DVC_NO_GIT (4), OOM (2)
```

## Pattern Detection

Look for patterns across failures:
- **Model-specific**: "All SAM3 jobs failed" → model configuration issue
- **Fold-specific**: "All fold-2 failed" → data size issue
- **Loss-specific**: "All cbdice failed" → loss function bug
- **Universal**: "All jobs failed" → infrastructure issue (Docker, DVC, env vars)

## Categorization Rules

For each root cause, classify:
- **AUTO-FIXABLE**: Infrastructure issues with clear fixes (DVC_NO_GIT, DISK_FULL, ENV_VAR_LITERAL)
- **CONFIG-FIXABLE**: Requires parameter changes (OOM → reduce batch_size)
- **CODE-FIXABLE**: Requires code changes (bug in loss function, import error)
- **UNRECOVERABLE**: Cannot be fixed without user decision (DATA_MISSING, REGISTRY_AUTH)

## "Transient" Classification

"Probably transient" is BANNED unless:
- The failure is a confirmed spot preemption (SkyPilot exit code 24)
- An identical job with identical code succeeded in a different region
- The error is a known cloud provider intermittent (specific API timeout patterns)

## Transition to FIX

If any failures exist, present the aggregated report and transition to FIX.
If 0 failures, skip directly to REPORT.
