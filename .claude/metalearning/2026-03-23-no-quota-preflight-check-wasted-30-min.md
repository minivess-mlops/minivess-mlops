# Metalearning: No GCP Quota Preflight Check — 30 Min Wasted on Zone-Hopping

**Date**: 2026-03-23
**Severity**: HIGH — wasted 30+ minutes watching SkyPilot bounce between zones
**Category**: Missing preflight validation, reactive instead of proactive

## What Happened

After pinning the SkyPilot controller to `europe-north1` (correct region for our infra),
the controller failed to provision because `CPUS_PER_VM_FAMILY` quota for `n4` is 0 in
that region. SkyPilot retried within the same region (europe-north1-a, -b, -c) with the
same quota error, then gave up. Meanwhile, 30+ minutes passed while Claude Code polled
`sky jobs queue` without investigating WHY the controller wasn't starting.

## Root Cause

1. **No quota validation in preflight**: `preflight_gcp.py` checks 10 things (Docker,
   DVC, env vars, controller cloud match) but NONE check whether GCP actually has
   sufficient CPU/GPU quota to provision the requested VMs.

2. **Reactive monitoring instead of proactive validation**: Claude waited for SkyPilot
   to fail, then read the error. Should have checked quota BEFORE launching.

3. **GCP quota is per-region, per-VM-family**: `n4-standard-4` has 0 quota in
   europe-north1 but works in us-central1. This is a new GCP project with minimal
   default quotas.

## Prevention

1. **Add preflight check #11: GCP CPU quota in target region**:
   ```bash
   gcloud compute regions describe europe-north1 --format="value(quotas.filter(metric:CPUS).limit)"
   ```
   Assert limit >= 4 (for controller) + the number needed for GPU VMs.

2. **Add preflight check #12: GCP GPU quota in target region**:
   Check L4 GPU quota in the region where jobs will run.

3. **These checks should also be in the cloud test suite** — a test that queries
   GCP quotas and asserts minimum thresholds for the expected workload.

## See Also

- `.claude/metalearning/2026-03-23-skypilot-controller-on-wrong-cloud.md`
- Issue #913 (launch bottleneck)
