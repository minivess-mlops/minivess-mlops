# 2026-03-14 — SkyPilot Bare VM Execution Violates Docker-Only Mandate

## Severity: CRITICAL

## What Happened

Claude Code designed and deployed a SkyPilot GPU smoke test (`smoke_test_gpu.yaml`)
that installs everything from scratch on a bare RunPod VM:
- `curl` to install uv
- `uv python install 3.13`
- `git clone` the repo
- `apt-get install` build tools + libboost
- `uv sync --all-extras` (~500 packages)
- `dvc pull` training data

This approach:
1. **Violates Rule #17** (STOP Protocol): Code executed outside Docker container
2. **Violates Rule #18**: No explicit volume mounts — artifacts on ephemeral VM disk
3. **Violates Rule #19**: STOP protocol's S(ource) check — not running in Docker
4. **Violates Design Goal #2**: Docker-per-flow isolation is the architecture
5. **Wastes GPU credits**: ~10-15 min of paid RTX 4090 time on setup alone
6. **Caused disk_size failure**: 40 GB disk filled with deps, torch.save failed

## Root Cause

Claude Code treated SkyPilot's default execution model (bare VM shell scripts)
as acceptable without questioning whether it aligned with the repo's Docker mandate.
SkyPilot DOES support Docker execution via `image_id: docker:<image>`, but this
was never considered. The entire debug plan focused on fixing bare-VM setup scripts
instead of questioning the fundamental approach.

## Why This Wasn't Caught Earlier

1. SkyPilot docs default to bare-VM examples — Claude followed convention, not mandate
2. The CLAUDE.md Docker rules (Rules #17-19) are about training execution, but don't
   explicitly mention SkyPilot/cloud compute
3. The knowledge graph `navigator.yaml` doesn't route "cloud compute" queries to
   Docker infrastructure rules
4. Claude spent multiple sessions debugging setup scripts without once asking:
   "Should we be running setup scripts at all?"

## Correct Architecture

```
WRONG (what we did):
  sky jobs launch → bare RunPod VM → install everything → run train_flow.py

CORRECT (what we should do):
  1. Push minivess-base:latest to GHCR (GitHub Container Registry)
  2. SkyPilot YAML: image_id: docker:ghcr.io/petteriTeikari/minivess-base:latest
  3. setup: only DVC pull + splits copy (~30 sec, no install)
  4. run: uv run python -m minivess.orchestration.flows.train_flow

ALTERNATIVE (even better):
  1. Push purpose-built smoke-test image with data baked in
  2. SkyPilot YAML: image_id: docker:ghcr.io/petteriTeikari/minivess-smoke:latest
  3. setup: empty (everything pre-built)
  4. run: just train_flow.py
```

## What Must Change

1. **Navigator + CLAUDE.md**: "cloud compute" must explicitly route to Docker mandate
2. **Knowledge graph**: Add decision node for "cloud execution = Docker image, not bare VM"
3. **SkyPilot YAML**: Must use `image_id: docker:<image>` — bare VM setup is BANNED
4. **Docker registry**: Set up GHCR push workflow (user previously asked, was ignored)
5. **Regression test**: Verify smoke_test_gpu.yaml has `image_id` key

## Anti-Pattern Identified

**"Infrastructure Inertia"**: When debugging a failing approach, Claude focused
entirely on making the broken approach work (fix setup scripts) instead of
questioning whether the approach itself was correct. Each fix (uv install, apt-get,
Python version, DVC targeted pull) reinforced commitment to the wrong path.

This is the sunk-cost fallacy applied to code: "We've already fixed 6 issues in
this setup script, let's fix the 7th" instead of "Why are we running setup scripts
on a GPU VM at all?"

## Resolution

- [ ] Create GHCR push workflow (manual trigger, respects CI ban)
- [ ] Rebuild smoke_test_gpu.yaml with `image_id: docker:...`
- [ ] Add Docker mandate to cloud compute section of CLAUDE.md
- [ ] Add cloud → Docker routing in navigator.yaml
- [ ] Add regression test: SkyPilot YAMLs must have `image_id`
