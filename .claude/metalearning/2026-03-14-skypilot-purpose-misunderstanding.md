# 2026-03-14 — Failure to Understand SkyPilot's Role in This Repo

## Severity: CRITICAL (Architectural Misunderstanding)

## What Happened

When the Docker-only violation was discovered (see `2026-03-14-skypilot-bare-vm-docker-violation.md`),
Claude's initial reaction implied "SkyPilot defaults to bare VM, so maybe we should bypass SkyPilot."
This reveals a fundamental failure to understand WHY SkyPilot is in this repo.

## Why SkyPilot Exists in This Repo

SkyPilot is the **cloud-agnostic compute orchestrator** — the equivalent of Pulumi for IaC.
It abstracts away provider-specific APIs so the SAME YAML works on:
- RunPod (current)
- Lambda Labs
- AWS (EC2 spot)
- GCP (Preemptible VMs)
- Azure (Spot VMs)
- Local intranet servers (via SSH)
- On-prem K8s clusters

**The whole point is provider portability.** Suggesting to bypass SkyPilot for RunPod-native
Docker is like suggesting to bypass Pulumi and manually configure servers. It defeats the
entire purpose of the abstraction layer.

## The Correct Mental Model

```
WRONG mental model (what Claude had):
  SkyPilot = "a way to launch VMs on RunPod"
  Docker = "an alternative to SkyPilot"
  → Led to: bare VM scripts, then "maybe bypass SkyPilot for Docker"

CORRECT mental model:
  SkyPilot = "cloud-agnostic compute orchestrator (like Pulumi for IaC)"
  Docker = "execution environment WITHIN SkyPilot"
  → Leads to: SkyPilot launches Docker containers on ANY cloud

ANALOGY:
  SkyPilot : GPU compute = Pulumi : cloud infrastructure
  Docker : SkyPilot = containers : Pulumi
  RunPod : SkyPilot = AWS : Pulumi (just one provider, not THE provider)
```

## What Must Change

1. **SkyPilot + Docker = ALWAYS together**. SkyPilot YAMLs use `image_id: docker:<image>`.
   SkyPilot orchestrates WHERE the Docker container runs. Docker defines WHAT runs.

2. **SkyPilot scope is broader than cloud**. It should be used for:
   - Cloud GPU training (RunPod, Lambda Labs, AWS, GCP)
   - Intranet GPU servers (via SSH connector)
   - Local LAN multi-GPU (via SSH)
   - Any compute that's not the dev machine itself

3. **Never suggest bypassing SkyPilot.** Just like never suggesting "skip Pulumi, use
   AWS console manually." The abstraction layer IS the value.

4. **CLAUDE.md must clarify** that SkyPilot's role is cloud-agnostic orchestration,
   not "a RunPod launcher."

## Anti-Pattern: Tool Reduction Under Pressure

When debugging fails, Claude's instinct is to reduce complexity by removing tools:
"SkyPilot is causing issues → maybe bypass SkyPilot." This is wrong.

The correct response to a tool failing is to USE IT CORRECTLY, not to bypass it.
SkyPilot supports Docker via `image_id`. The fix is using that feature, not
abandoning the tool.

This is the same anti-pattern as "Prefect is complex → run scripts directly" or
"Docker is complex → run on host." The complexity is the VALUE — it provides
portability, reproducibility, and isolation.

## Resolution

- [x] Metalearning doc created (this file)
- [ ] Update CLAUDE.md SkyPilot section to clarify cloud-agnostic purpose
- [ ] Add SkyPilot scope diagram to knowledge graph
- [ ] Ensure SkyPilot YAML uses `image_id: docker:<image>` (GHCR)
- [ ] Add SSH connector setup for intranet servers
- [ ] Add regression test: SkyPilot YAML must have `image_id` key
