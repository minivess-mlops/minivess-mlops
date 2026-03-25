# Metalearning: SkyPilot Implementation Trust Deficit

**Date**: 2026-03-24
**Severity**: CRITICAL — systemic pattern of incorrect SkyPilot implementation
**Category**: Repeated implementation errors, insufficient documentation verification

## Pattern: Claude Code Keeps Getting SkyPilot Wrong

Across the 5th and 6th factorial passes, Claude Code made at least 6 distinct
SkyPilot-related errors, each costing time and cloud credits:

### Error 1: Controller on wrong cloud (RunPod for GCP jobs)
- `~/.sky/config.yaml` had `cloud: runpod` from dev phase
- Caused 36 min/submission + RunPod outage killed 25/34 jobs
- Root cause: Never verified controller placement for production

### Error 2: No region pin on controller
- `cloud: gcp` without region → SkyPilot zone-hopped across ALL of US
- 30+ min wasted watching controller bounce between zones
- Root cause: Didn't read SkyPilot docs on `infra:` format

### Error 3: Unauthorized A100 in accelerators
- Added `A100-80GB: 1` as "helpful fallback" without user authorization
- 5.5x cost risk, violated YAML-as-contract principle
- Root cause: "Optimization instinct" overriding declarative config integrity

### Error 4: `job_recovery` ban test was WRONG
- Test banned `job_recovery` field claiming "removed in SkyPilot v1.0"
- Field IS supported — it's a critical resilience feature
- Root cause: Assumed from a comment, never verified against actual SkyPilot API

### Error 5: sync_sky_config.py wrote `cloud:` conflicting with `.sky.yaml` `infra:`
- SkyPilot v1.0 rejects both `cloud` and `infra` simultaneously
- ALL 32 jobs LAUNCH_FAILED
- Root cause: Didn't test sync script against actual SkyPilot validation

### Error 6: n4 quota = 0 in europe-north1
- Pinned controller to europe-north1 without checking quota
- Controller bounced for 30+ min before discovering quota issue
- Root cause: No preflight check for CPU quota in target region

## Why This Keeps Happening

1. **Claude never reads SkyPilot docs BEFORE implementing**. Every error above would
   have been caught by reading the relevant doc page first. Claude reads docs AFTER
   failures, not before implementation.

2. **Claude trusts comments over code**. "job_recovery removed in v1.0" was a COMMENT
   in our YAML, not verified against the actual SkyPilot Python API.

3. **Claude doesn't test SkyPilot changes locally**. Every change to `.sky.yaml`,
   `train_factorial.yaml`, or `sync_sky_config.py` should be verified with
   `sky jobs launch --dryrun` or `sky.Task.from_yaml()` BEFORE committing.

4. **No integration test for the SkyPilot → GCP → training chain**. All our SkyPilot
   tests are YAML structure tests. None test actual `sky jobs launch` behavior.

## Prevention Rules

1. **BEFORE any SkyPilot config change**: Read the relevant docs page. Cite it.
2. **BEFORE committing SkyPilot YAML changes**: Verify with `sky.Task.from_yaml()`
3. **NEVER add GPU types, clouds, or resources** without explicit user instruction
4. **NEVER trust comments about API behavior**: Verify against actual code/docs
5. **Test sync_sky_config.py output** against SkyPilot validation before using it
6. **Check GCP quotas programmatically** before pinning regions

## Documentation Sources (MUST READ before SkyPilot changes)

- Config: https://docs.skypilot.co/en/stable/reference/config.html
- Config sources: https://docs.skypilot.co/en/stable/reference/config-sources.html
- Auto-failover: https://docs.skypilot.co/en/stable/examples/auto-failover.html
- Managed jobs: https://docs.skypilot.co/en/stable/examples/managed-jobs.html
- Job recovery: https://docs.skypilot.co/en/stable/reference/yaml-spec.html

## See Also

- `2026-03-23-skypilot-controller-on-wrong-cloud.md`
- `2026-03-23-no-quota-preflight-check-wasted-30-min.md`
- `2026-03-24-unauthorized-a100-in-skypilot-yaml.md`
- `2026-03-24-yaml-is-the-contract-zero-improvisation.md`
