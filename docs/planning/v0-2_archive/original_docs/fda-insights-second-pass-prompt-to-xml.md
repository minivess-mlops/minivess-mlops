# FDA Insights Second Pass: Prompt to XML Execution Plan

**Created**: 2026-03-18
**Purpose**: Verbatim user prompt that drives the XML execution plan creation

---

## User Prompt (verbatim)

Let's then continue in a new branch fix/fda-readiness-regops-openlineage-medmlops-pccp-qsmr-sbom-cybersecurity-2nd-pass and create an executable plan to be implemented with our self-learning TDD Skill based on the recent research-reports:

- `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/fda-insights-second-pass.md`
- `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md`
- `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/openlineage-marquez-iec62304-report.md`

PR #833 merged to main: `docs/planning/fda-insights-second-pass.md`

Report covers 11 sections across 7 major themes:

1. SBOM — We can't easily describe ours yet. Gap: no CycloneDX/SPDX generation. Fix: cyclonedx-bom + grype in Makefile.
2. Cybersecurity/SecOps/MLSecOps — QMSR (Feb 2026) makes cybersecurity a QMS requirement, not just product design. Our CI/CD IS "production controls" — we just need to document it as such. 14 common AINN deficiencies mapped against our stack.
3. QMSR readiness — Mapped all QMSR requirements against our stack. Key gap: we don't document our CI/CD pipeline as QMSR production controls. Most other areas adequate.
4. PCCP alignment — Our factorial design (4x3x2x3x2x5) IS a PCCP template. K252366 (a2z-Unified-Triage) is the exact blueprint — same architecture pattern (factorial variants validated on sequestered test data).
5. MLOps Maturity Level 4 — Currently at Level 2 (pipeline automation). Gap: drift detection exists but not connected to automated retraining trigger. Roadmap to Level 4 with locked→adaptive pattern via PCCP.
6. Multi-site data collection — Opt-in architecture for NEUROVEX installations. PostHog (UX), Sentry (errors), Grafana (drift), Marquez (lineage), Flower (federated learning). All opt-in with anonymization gate.
7. Regulatory convergence — SaMD, SOC2, EU DPP, UAD 3.6 all converging on the same stack. We have 80% of it. Missing 20%: SBOM, vuln monitoring, threat model, audit wiring.

20+ library recommendations across 4 priority tiers, from immediate (cyclonedx-bom) to long-term (flower).

Most important post-June 2025 developments (these could be added to a future revision):

1. FDA Sept 30, 2025 Request for Public Comment on measuring real-world performance of AI devices — explicitly names data drift, concept drift, model drift as threats. Our drift simulation flow is a direct regulatory asset.
2. Harrison.ai petition (Oct 2025) — 34-page petition to exempt radiology AI from 510(k). Comment period closed Feb 27, 2026. If FDA doesn't deny by mid-April, it takes effect. Could make PCCP obsolete for CADx/CADe product codes.
3. CycloneDX v7.2.2 now supports uv virtual environments natively — perfect for our stack.
4. MedMLOps framework (de Almeida et al., 2025, European Radiology) — four pillars specific to medical imaging: availability, continuous monitoring/validation/(re)training, privacy, ease of use. Maps directly to our Prefect flow topology.
5. MITRE ATLAS framework for adversarial AI threats — the medical AI security taxonomy we should reference.
6. PostHog SOC 2 Type II certified (May 2025) + HIPAA-ready with BAA — confirms it's viable for our opt-in telemetry architecture.
7. EU Digital Omnibus Proposal (Feb 2026) — AI Act requirements for medical AI should be applied within existing conformity assessment, not separate certification.

And let's create the plan to `docs/planning/fda-insights-second-pass-executable.xml` which will improve our existing implementation especially in regard to the OpenLineage use and how the system is not being truly wired in Flows!

Note that the .md reports are rather comprehensive, and we should be evaluating what makes sense for repo (examine and update knowledge-graph accordingly) as for example the "UX stack with Sentry, Posthog, CoPilotKit, Intercom/Crisp, etc." can be deferred (Open a P2 Issue on this), or the real implementation can be deferred but we should create some bare minimum stubs definitely for Sentry and Posthog!

The data annotation Flow (or module) and the dashboard (that could be made interactive with excellent UX/UI) should have copilotkit (ag-ui) installed along with webmcp on the interface (and potentially a2ui from google as well (see e.g. /home/petteri/Dropbox/github-personal/music-attribution-scaffold for a reference how to have bare minimum agentic UI, and remember also that we have Pydantic AI micro-orchestration stubs (or should be at least) along with the Prefect macro-orchestration, see `docs/planning/langgraph-agents-plan.md` and `docs/planning/langgraph-agents-plan-advanced.md`, and remember to update the kg with the "agentic stack info" with the existing Pydantic AI stubs and the to-be-added ag-ui, webmcp, copilotkit that actually demonstrate some agentic behavior, Open a P1 issue on this).

The federated learning feature is not yet on the "publication gate" pathway, but you should open a P2 issue on thinking what options we have and what should we have a look more in the future, compare FLARE from NVIDIA, MONAI FL (https://monai.readthedocs.io/en/1.3.2/fl.html) and the Flower does not seem useful for us?

This is very important for us — the MedMLOps framework (de Almeida et al., 2025, European Radiology) four pillars: Availability, Continuous monitoring/validation/(re)training, Patient privacy/data protection, Ease of use. You should definitely update the kg for these insights, and iterate with reviewer agents adding this to the high-level requirements of our MLOps pipeline as we should definitely build upon the existing academic practices and build something new and bring that novelty in the manuscript so that peer reviewers accept our submission!

The SBOM, MLSecOps/Cybersecurity, PCCP and QMSR readiness are all important! Examine how to update the kg accordingly and what new libraries we should implement, and what stubs to implement, or is it enough to make our implementation better?

And we should have the https://github.com/intuitem/ciso-assistant-community installed to this repo as CI to be run on Github Actions which we commented out, right? How does the CycloneDX compare to this? And for example we could open a Github Issue with P2 that talks about buying a Oneleet subscription over Vanta's security theatre and close this issue to be buried with the other issues to the Closed Issues ;) (way too expensive for an academic project but could be needed if someone started to use this stack for commercial reasons and started kicking out all non-commercial libraries from this?

So P0 issue would be to wire OpenLineages for real, executable P1 issue would be the cyclonedx-bom + grype and what else to the ones already listed? Examine the .md deeply and optimize the XML plan for execution as we have done the background research already and the XML should be about how to implement everything with self-learning TDD Skill, what issues to open, whether we need Ralph Loop Skill for monitoring the infrastructure development, etc.

---

## Decisions from Prompt

### Priority Mapping

| Priority | What | Action |
|----------|------|--------|
| **P0** | Wire OpenLineage into all Prefect flows for real | Executable TDD tasks in XML plan |
| **P1** | CycloneDX SBOM + Grype vulnerability scanning | Executable TDD tasks in XML plan |
| **P1** | Agentic stack: CopilotKit/AG-UI + WebMCP + Pydantic AI stubs | GitHub Issue |
| **P2** | UX monitoring: Sentry + PostHog stubs (bare minimum) | GitHub Issue + minimal stubs |
| **P2** | Federated learning: NVIDIA FLARE vs MONAI FL comparison | GitHub Issue |
| **P2** | CISO Assistant Community for compliance CI | GitHub Issue |
| **P2** | Oneleet vs Vanta commercial compliance discussion | GitHub Issue (create + close) |

### KG Updates Needed
- MedMLOps 4-pillar framework → high-level requirements
- Agentic stack (AG-UI, WebMCP, CopilotKit, Pydantic AI) → architecture domain
- SBOM resolution → `cyclonedx_plus_grype`
- CISO Assistant → operations domain
