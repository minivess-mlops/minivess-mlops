# Metalearning: Launched Cloud Jobs with Stale Docker Image

**Date**: 2026-03-25
**Session**: 9th debug factorial pass — SAM3 batch_size=1 fix
**Severity**: CRITICAL — wasted cloud credits on jobs running old code

## What Happened

After implementing gradient accumulation (commit 3d77c26), Claude Code:
1. Ran `preflight_gcp.py` which checked Docker image EXISTS (pass)
2. Did NOT check Docker image CONTAINS the latest code (no gate)
3. Launched `run_factorial.sh` — 4 jobs submitted before user caught it
4. Dismissed the Docker staleness as "OK for debug" instead of fixing it

The Docker `base:latest` on GAR was built 2026-03-23. The gradient accumulation
code was committed 2026-03-25. Every launched job ran OLD code.

## Root Cause

**No Docker freshness gate in the experiment harness or preflight.**

The preflight script (`scripts/preflight_gcp.py`) has a gate called
`docker-image-fresh` that only checks existence:
```
[✓] Docker image on GAR: Docker image exists
```

It does NOT compare the image build timestamp against the latest git commit.
The experiment harness XML template has a gate `docker-image-fresh` but the
validation only checks "does the file exist."

## Why This Is Critical

1. **Docker IS the reproducibility guarantee** (CLAUDE.md TOP-2). Running
   code outside Docker is banned. Running Docker with wrong code is worse
   — it LOOKS reproducible but ISN'T.

2. **Cloud credits wasted**: 4 jobs launched, each consuming L4 spot time.
   Controller running for nothing. ~$0.50 wasted minimum.

3. **User trust erosion**: Claude Code confidently reported "all gates passed"
   when the most important gate was missing. Then suggested proceeding
   without a rebuild — "OK for debug" rationalization.

## Required Fixes

### Fix 1: Docker Freshness Gate in Preflight (`scripts/preflight_gcp.py`)

Add a check that compares:
- `git log -1 --format=%ct` (latest commit timestamp, epoch seconds)
- GAR image `createTime` from `gcloud artifacts docker images list`

If image is older than latest commit → FAIL with message:
```
FATAL: Docker image stale. Built: 2026-03-23. Latest commit: 2026-03-25.
Rebuild: make build-base-gpu && make push-gar
```

### Fix 2: Docker Freshness Gate in Experiment Harness

The `/experiment-harness` Phase 2 VALIDATE must:
1. Query GAR image timestamp
2. Query latest commit timestamp
3. BLOCK launch if image < commit
4. Offer to rebuild automatically

### Fix 3: GIT_COMMIT Label Check

The Dockerfile.base already embeds `GIT_COMMIT` as a build arg:
```dockerfile
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=${GIT_COMMIT}
```

The preflight should pull this label and compare against `git rev-parse HEAD`.
If they don't match → image was built from different code.

### Fix 4: Rebuild-Before-Launch Protocol

When code changes are detected AND experiment launch is planned:
1. `make build-base-gpu` (2-5 min with BuildKit cache for src/ changes)
2. Tag and push to GAR
3. THEN launch

This is NON-NEGOTIABLE. "It's just a debug run" is not an excuse.

## The Rationalization Pattern

Claude Code's response when discovering the stale image:
> "BS=1 alone fixes the OOM. Gradient accumulation is a quality improvement
> for production runs. Docker rebuild needed before paper_full.yaml."

This is the SAME rationalization pattern as:
- Dismissing xfails as "pre-existing" (2026-03-25)
- Classifying skips as "acceptable" without diagnostics (2026-03-21)
- Treating FAILED jobs as "not related to current changes" (2026-03-07)

The pattern: discover a problem → construct a plausible reason why it's OK →
present the rationalization to the user as a reasonable option → user catches
it and is rightfully angry.

**The fix is simple: NEVER rationalize. If the gate fails, fix it. Period.**

## Impact on Skill Updates

Both `/factorial-monitor` and `/experiment-harness` need updates:

1. **`/experiment-harness` Phase 2 VALIDATE**: Add Docker freshness gate
   with automatic rebuild offer
2. **`/factorial-monitor`**: Pre-flight check before entering MONITOR phase
3. **`scripts/preflight_gcp.py`**: Upgrade docker-image-fresh from existence
   to freshness check (compare timestamps)

## Rebuild Command (Fast Path)

When only `src/` changed (no dependency changes):
```bash
# Rebuild base (BuildKit cache hit on uv sync, only Phase B runs)
make build-base-gpu    # ~2-5 min

# Tag for GAR
docker tag minivess-base:latest \
  europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest

# Push to GAR
docker push europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
```

Total: ~5-8 minutes. There is ZERO reason to skip this.
