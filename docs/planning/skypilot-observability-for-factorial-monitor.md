# SkyPilot Observability for /factorial-monitor — Upgrade Plan

**Date**: 2026-03-22
**Status**: PLANNED
**Trigger**: 8 FAILED_SETUP jobs went undetected for 2+ hours during 4th pass launch

---

## Problem

The current monitoring approach failed catastrophically:
- `run_factorial.sh` submits jobs and prints to stdout (which nobody watches)
- `ralph_monitor.py` watches ONE job at a time (not 32 concurrent jobs)
- No alerting on FAILED_SETUP — failures discovered hours later via manual `sky jobs queue`
- No job-to-condition mapping — raw job IDs don't tell which model×loss×calib failed

## What SkyPilot v1.0 Provides

| Feature | Available? | How |
|---------|-----------|-----|
| Web dashboard | **Yes** (v0.10+) | `sky dashboard` |
| JSON job status | **Yes** | `sky jobs queue -o json` |
| Python SDK polling | **Yes** | `sky.jobs.queue_v2(refresh=True)` |
| GPU metrics | K8s only | DCGM-Exporter + Prometheus (not for GCP VMs) |
| Notifications | **No** | Must be custom (Slack webhook, desktop notify) |
| Log streaming | **Yes** | `sky jobs logs --no-follow JOB_ID` |

## Recommended Architecture

```
┌─────────────────────────────────────────────────┐
│           /factorial-monitor skill               │
├─────────────────────────────────────────────────┤
│  1. LAUNCH: run_factorial.sh                     │
│     └── Write outputs/{ts}_job_map.json          │
│         {job_name → {model, loss, calib, fold}}  │
│                                                  │
│  2. MONITOR LOOP (30s interval):                 │
│     ├── sky jobs queue -o json                   │
│     ├── Diff against previous poll               │
│     ├── State transitions:                       │
│     │   ├── → FAILED_SETUP: diagnose + ALERT     │
│     │   ├── → FAILED: diagnose + ALERT           │
│     │   ├── → SUCCEEDED: log + update progress   │
│     │   └── → RECOVERING: log preemption         │
│     └── Print factorial progress matrix          │
│                                                  │
│  3. ALERTS:                                      │
│     ├── notify-send (desktop, immediate)         │
│     ├── JSONL log (durable)                      │
│     └── [Optional] Slack webhook                 │
│                                                  │
│  4. DASHBOARD: sky dashboard (browser)           │
│     └── Shows logs, status, cluster info         │
└─────────────────────────────────────────────────┘
```

## Key Improvements

1. **Replace table parsing with `sky jobs queue -o json`** — structured, reliable
2. **Batch monitoring** — watch ALL 32 jobs in one loop, not one at a time
3. **Job-to-condition mapping** — `run_factorial.sh` writes JSON mapping file
4. **Desktop notifications** — `notify-send` on FAILED_SETUP within 30s
5. **Factorial progress matrix** — visual grid showing model×loss×calib status
6. **`sky dashboard`** — launch alongside for log streaming and cluster view

## Cross-References

- `docs/planning/dvc-test-suite-improvement.xml` — preflight prevents failures
- `docs/planning/skypilot-fake-mock-ssh-test-suite-plan.md` — local YAML validation
- `src/minivess/compute/ralph_monitor.py` — existing monitor (single-job only)
- Issue #907: Cloud GPU pipeline gaps
- Issue #908: Local SkyPilot test suite
