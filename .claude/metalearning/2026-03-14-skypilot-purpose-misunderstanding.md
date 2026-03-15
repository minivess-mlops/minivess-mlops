# 2026-03-14 — Failure to Understand SkyPilot's Role in This Repo

## Severity: CRITICAL (Architectural Misunderstanding)

## What Happened

When the Docker-only violation was discovered (see `2026-03-14-skypilot-bare-vm-docker-violation.md`),
Claude's initial reaction implied "SkyPilot defaults to bare VM, so maybe we should bypass SkyPilot."
This reveals a fundamental failure to understand WHY SkyPilot is in this repo.

Then Claude compounded the error by writing a wrong analogy: "SkyPilot is like Pulumi for IaC."
The user explicitly corrected this — SkyPilot is NOT Infrastructure as Code. SkyPilot and
Pulumi/Terraform operate at completely different abstraction levels.

## What SkyPilot Actually Is

SkyPilot is an **intercloud broker** — a term coined in the
[NSDI'23 paper](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng)
by Yang et al. at UC Berkeley.

Official self-description: *"A system to run, manage, and scale AI workloads on any
AI infrastructure."*

**The key distinction from IaC:**
- **Terraform/Pulumi** = "Create this specific VM on AWS us-east-1 with this AMI."
  You tell it EXACTLY what resource to provision. It manages resource lifecycle.
- **SkyPilot** = "I need 1×A100 for 8 hours, cheapest option." It AUTOMATICALLY decides
  which cloud, region, and instance type to use. It handles placement optimization,
  spot preemption recovery, and cross-cloud failover.

**Better analogies:**
- SkyPilot is like **Slurm for the multi-cloud era** — you submit a job, it places it
- SkyPilot is like **a travel aggregator (Kayak)** — you say "cheapest flight A→B",
  it searches across all airlines
- The NSDI paper explicitly differentiates from Terraform: *"Terraform provisions and
  manages resources on different clouds, but requires the usage of provider-specific
  APIs, and also does not handle job placement."*

SkyPilot combines your infrastructure (K8s clusters, Slurm clusters, cloud VMs, SSH
machines) into a **unified compute pool** optimized for AI workloads.

## Why SkyPilot Exists in This Repo

This repo's highest priorities are **excellent DevEx** and **reproducibility**. Nobody
should manually launch pods, VMs, or instances. SkyPilot automates this:

1. **Zero manual work** — `sky jobs launch task.yaml` handles everything
2. **Provider portability** — same YAML works on RunPod, Lambda, AWS, GCP, SSH
3. **Cost optimization** — automatic spot instances with preemption recovery
4. **Reproducibility** — declarative YAML = reproducible compute environment

The SAME SkyPilot YAML works on:
- RunPod (current: cloud GPU spot RTX 4090)
- Lambda Labs (A100)
- AWS (p4d spot)
- GCP (preemptible)
- Intranet servers (SSH connector)
- Local LAN multi-GPU (SSH)

**The whole point is that the researcher never thinks about infrastructure.**

## The Root Failure: Wrong Analogy

Claude wrote "SkyPilot is like Pulumi for IaC" without web-searching what SkyPilot
actually calls itself. This is a violation of CLAUDE.md Rule #12 (Never Confabulate)
and Rule #10 (Verify Beyond Knowledge Cutoff).

The correct approach was:
1. Web-search SkyPilot's official docs and NSDI paper
2. Read their self-description
3. Use THEIR terminology, not fabricated analogies

## Anti-Pattern: Tool Reduction Under Pressure

When debugging fails, Claude's instinct is to reduce complexity by removing tools:
"SkyPilot is causing issues → maybe bypass SkyPilot." This is wrong.

The correct response to a tool failing is to USE IT CORRECTLY, not to bypass it.
SkyPilot supports Docker via `image_id`. The fix is using that feature, not
abandoning the tool.

This is the same anti-pattern as "Prefect is complex → run scripts directly" or
"Docker is complex → run on host." The complexity is the VALUE — it provides
portability, reproducibility, and isolation. These tools exist because the repo's
#1 priority is excellent DevEx and reproducibility.

## Resolution

- [x] Metalearning doc created (this file)
- [x] Wrong Pulumi analogy corrected in CLAUDE.md, navigator.yaml, infrastructure.yaml
- [x] Correct terminology: "intercloud broker" (NSDI'23)
- [ ] Add SSH connector setup for intranet servers
- [ ] Add regression test: SkyPilot YAML must have `image_id` key
