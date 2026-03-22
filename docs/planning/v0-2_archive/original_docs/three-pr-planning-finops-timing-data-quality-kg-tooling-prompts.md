# Three-PR Planning Session: FinOps/Timing, Data Quality, KG/Tooling

**Created**: 2026-03-17
**Purpose**: Planning session for PR-1, PR-2, PR-3 with interactive Q&A

## Original User Prompt (verbatim)

> Create me first in parallel plans for those PR#1-3, so that each of those plans must be saved to its own .md plan about how to implement the PR with both our self-learning Skill and ralph loop for instrastructure monitoring Skill. Save them to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning and optimize each plan with reviewer agents, and ask me rather too many questions than too little so we can make sure that we align with our visions! Make sure though to read the CLAUDE.md harness system and our knowledge graph (/home/petteri/Dropbox/github-personal/minivess-mlops/knowledge-graph) comprehensively before starting to help you asking good questions and things are confusing either in the knowledge graph or in issues how to implement something and avoid burdening me with trivial questions that should be auto-discoverable and queryable yourself: /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-16-ralph-loop-not-invoked.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-16-runpod-dev-not-primary-recurring-confusion.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-16-level4-mandate-never-negotiate.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-16-infrastructure-scaffold-not-shortcuts.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-16-asking-humans-cloud-state-queries.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-16-kg-exists-read-it-first.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-16-asking-humans-what-tools-should-detect.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-15-sota-framing-violation-platform-not-race.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-15-ralph-loop-not-auto-activated.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-15-pulumi-passphrase-bypass-antipattern.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-14-failed-to-follow-gcp-instructions.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-14-skypilot-purpose-misunderstanding.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-14-poor-repo-vision-understanding.md. So use the interactive question format with multiple answers that I can easily select and write free text if I need, is this clear? Zero assumptions in your plan and do not do major level decisions without consulting with me. You should more handle the low-level implementations when we are aligned with our high-level architecture visions which should be rather good already in knowledge graph. This is now a dual mandate also to improve our knowledge graph and you can first explore it first and identify knowledge gaps in it, as in nodes missing or poorly communicated nodes that could serve as prerequisite step for planning any PRs themselves. And the PR planning and the questions should jointly improve the knowledge graph as our Q&A should compound to the improved institutional knowledge. Save my prompt now verbatim to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/three-pr-planning-finops-timing-data-quality-kg-tooling-prmpts.md with all the questions and answers also saved verbatim! Is this clear?

---

## Q&A Session Log

### Round 1 (Architecture Decisions)

**Q1: PR-1 FinOps — How should MLflow cost metrics reach Grafana?**
- Options: Custom Prometheus exporter / Grafana SQL datasource / JSONL+Loki / MLflow UI only
- **User clarification (before answering)**: "Prometheus exporting should happen from BentoML, not from MLflow"
- **Reformulated Q1**: Should cost metrics be pushed at training time via custom Prometheus counter, inference-time from MLflow, both, or via DuckDB→Grafana SQL?
- **Answer**: Training-time custom Prometheus counter via BentoML

**User architectural clarification (verbatim)**:
> "MLflow as contract" is our vision so the training is done with MLflow that the experiment
> tracking is for with MLflow model and artifact registry. MLflow is used "internally" for
> Evals and Biostatistics Flows, but in the Deploy flow, BentoML fetches the "champion model"
> and deploys it so it is available for any Gradio demo or more refined batch processing API
> and also the annotation module. The MLflow serving at the moment is just for internal purposes

**Q2: PR-1 scope — Should #751 (Docker pull optimization) stay in PR-1?**
- **Answer**: Move to separate PR. PR-1 stays focused: #683, #747, #717, #735.

**Q3: PR-2 — When a validation gate fails, what should happen?**
- Options: Halt+override / Configurable per-gate / Warn+continue
- **Answer**: Configurable per-gate (each gate has severity in Dynaconf: error/warning/info)

**Q4: PR-3 KG — _network.yaml empty, 13 nodes missing, Claim C10 "Bayesian priors"**
- **User clarification (verbatim)**:
> The Bayesian KG was the high-level vision that we have developed in the probabilistic PRD:
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/prd which we tried to connect
> with the KG (kg as the deterministic winner probability) but it seems the information is
> not there how kg, probabilistic PRD and our desire for spec-driven development with OpenSpec
> is connected? This has to be significantly improved apparently if you could not figure out
> this from the kg

**User architectural clarification #2 (verbatim)**:
> "MLflow as contract" is our vision so the training is done with MLflow that the experiment
> tracking is for with MLflow model and artifact registry. MLflow is used "internally" for
> Evals and Biostatistics Flows, but in the Deploy flow, BentoML fetches the "champion model"
> and deploys it so it is available for any Gradio demo or more refined batch processing API
> and also the annotation module. The MLflow serving at the moment is just for internal purposes

**Analysis after reading PRD**: The PRD→KG→OpenSpec pipeline IS architecturally sound:
- PRD has 71 Bayesian decision nodes with prior probabilities + conditional tables
- KG domains materialize "resolved" decisions (posterior=1.0 = deterministic winner)
- OpenSpec operationalizes decisions via GIVEN/WHEN/THEN specs
- Manuscript layer ties claims to evidence chains
- **What's missing**: Integration documentation explaining this triple connection

### Round 2 (Implementation Design)

**Q5: PR-1 — Should Prometheus cost gauges live in a shared module?**
- **Answer**: Shared module (src/minivess/observability/prometheus_metrics.py). Reusable by any flow.

**Q6: PR-1 — Should finops cost aggregation extend RunAnalytics or be standalone?**
- **Answer**: Extend RunAnalytics. Add cost_by_model_family(), cost_trends(), break_even_analysis() as methods.

**Q7: PR-2 — Should DeepChecks 3D→2D adapter be in PR-2 or deferred?**
- **Answer**: Include in PR-2. Level 4 mandate: implement ALL viable methods. Stubs already exist.

**Q8: PR-3 — Which artifact to document PRD→KG→OpenSpec connection?**
- **Answer**: Architecture doc + navigator update. Create docs/planning/prd-kg-openspec-integration-architecture.md with Mermaid diagram. Update navigator.yaml with 5-layer knowledge architecture.

### Round 3 (Implementation Details)

**Q9: PR-1 — MLflow cleanup: Prefect task or standalone script?**
- **Answer**: Prefect maintenance task. Add cleanup_stale_runs_task() to analysis_flow or new maintenance_flow.

**Q10: PR-2 — Where should per-gate severity config live?**
- **Answer**: Extend settings.toml. DATA_QUALITY_GATE_PANDERA='error', DATA_QUALITY_GATE_GE='warning', etc.

**Q11: PR-3 — How many KG nodes to create?**
- **Answer**: All 13 missing + fix 7 incomplete (20 total). PR-3 IS the KG improvement PR.

**Q12: Cross-cutting — Ralph-loop in which plans?**
- **Answer**: PR-1 plan only. PR-2/PR-3 are local TDD. PR-4 (future) will be ralph-loop heavy.

---

## Decision Summary (all 12 answers)

| # | Decision | Answer |
|---|----------|--------|
| Q1 | MLflow→Grafana path | BentoML Prometheus (training-time custom counter) |
| Q2 | PR-1 scope (#751) | Move to separate PR |
| Q3 | Gate failure policy | Configurable per-gate via Dynaconf |
| Q4 | KG Bayesian claims | PRD→KG→OpenSpec integration doc needed |
| Q5 | Metrics module location | Shared prometheus_metrics.py |
| Q6 | FinOps analytics design | Extend RunAnalytics |
| Q7 | DeepChecks scope | Include in PR-2 |
| Q8 | Integration doc format | Architecture doc + navigator update + Mermaid |
| Q9 | Cleanup design | Prefect maintenance task |
| Q10 | Gate config location | Extend settings.toml |
| Q11 | KG node count | All 13 missing + 7 incomplete = 20 |
| Q12 | Ralph-loop scope | PR-1 only |
