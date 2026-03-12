---
title: "Pulumi Infrastructure-as-Code Implementation Guide"
status: planned
created: "2026-03-12"
---

# Pulumi Infrastructure-as-Code Implementation Guide for ML Research Platforms

**Date:** 2026-03-12
**Branch:** `feat/synthetic-vasculature-stack-generation`
**Cross-references:** [#609](https://github.com/minivess-mlops/minivess-mlops/issues/609), [#366](https://github.com/minivess-mlops/minivess-mlops/issues/366)
**Companion report:** `docs/planning/skypilot-and-finops-complete-report.md`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Infrastructure-as-Code: Foundations and Taxonomy](#2-iac-foundations)
3. [Pulumi: Architecture and Design Philosophy](#3-pulumi-architecture)
4. [Pulumi vs Terraform vs OpenTofu vs CDKTF](#4-pulumi-vs-terraform)
5. [LLM Code Generation: Imperative vs Declarative IaC](#5-llm-code-generation)
6. [Agentic Infrastructure: The Emerging Paradigm](#6-agentic-infrastructure)
7. [Claude Code and Pulumi: Agentic IaC in Practice](#7-claude-code-pulumi)
8. [MinIVess Stack Analysis: Service-by-Service Pulumi Gains](#8-stack-analysis)
9. [Deployment Scenarios: Intranet vs Hetzner vs AWS](#9-deployment-scenarios)
10. [Academic Research Labs vs Deep Tech Startups](#10-academic-vs-startup)
11. [Pulumi Pricing and Total Cost of Ownership](#11-pricing-tco)
12. [Implementation Roadmap for MinIVess](#12-implementation-roadmap)
13. [Implications for Agentic MLOps Research](#13-agentic-mlops-research)
14. [Limitations and Open Questions](#14-limitations)
15. [Conclusion](#15-conclusion)
16. [References](#16-references)

---

## 1. Executive Summary {#1-executive-summary}

Infrastructure-as-Code (IaC) has evolved from a DevOps convenience into a foundational
requirement for reproducible machine learning research ([Kreuzberger et al., 2023](https://arxiv.org/abs/2205.02302);
[Pineau et al., 2021](https://arxiv.org/abs/2003.12206)). This report presents a
comparative evaluation of IaC approaches for the MinIVess MLOps stack, with particular
attention to Pulumi, Terraform/HCL, OpenTofu, and the now-deprecated CDKTF. The analysis
spans four dimensions: (a) LLM-assisted code generation quality, (b) agentic infrastructure
automation, (c) academic reproducibility, and (d) startup scalability.

**Key findings:**

1. **In vendor-published benchmarks (n=5 per condition), Pulumi TypeScript produced
   41% lower total pipeline cost** for LLM-generated infrastructure code compared to
   Terraform HCL, attributed to greater representation of imperative languages in LLM
   training data ([Pulumi Blog, 2026](https://www.pulumi.com/blog/token-efficiency-vs-cognitive-efficiency-choosing-iac-for-ai-agents/)).
   However, independent academic benchmarks report substantially lower first-attempt
   deployment success rates (20-30%) for LLM-generated IaC generally
   ([DPIaC-Eval, 2025](https://arxiv.org/abs/2506.05623)).

2. **CDKTF (Terraform CDK) was deprecated in December 2025** and its repository archived
   ([Pulumi Blog, 2025](https://www.pulumi.com/blog/cdktf-is-deprecated-whats-next-for-your-team/)),
   eliminating the main "use programming languages with Terraform" alternative.

3. **Both Pulumi and HashiCorp are converging on agentic infrastructure** — Pulumi via
   Neo (public preview, September 2025) and HashiCorp via Project Infragraph (private beta December
   2025). The MCP (Model Context Protocol) integration enables Claude Code to manage
   infrastructure directly ([Pulumi Blog, 2026](https://www.pulumi.com/blog/mcp-server-ai-assistants/)).

4. **For the MinIVess stack (18 services), Pulumi automates deployment of 14 services**
   across three deployment scenarios (intranet, Hetzner VPS, AWS), reducing provisioning
   from days of manual Docker Compose setup to a single `pulumi up` command.

5. **The VeriCoding benchmark** ([Bursuc et al., 2025](https://arxiv.org/abs/2509.22908))
   demonstrates that LLM performance on formal verification correlates with proximity to
   imperative programming paradigms (Dafny 82% vs Lean 27%), reinforcing the hypothesis
   that Pulumi's Python/TypeScript approach is more LLM-friendly than Terraform's HCL.

---

## 2. Infrastructure-as-Code: Foundations and Taxonomy {#2-iac-foundations}

### 2.1 Definition and Motivation

Infrastructure-as-Code refers to the practice of managing computing infrastructure through
machine-readable definition files rather than manual configuration ([Morris, 2016](https://www.oreilly.com/library/view/infrastructure-as-code/9781491924334/)).
In the context of MLOps, IaC serves three critical functions:

1. **Reproducibility** — The same infrastructure can be recreated identically across
   environments, a requirement for scientific computing ([Peng, 2011](https://doi.org/10.1126/science.1213847)).
2. **Version control** — Infrastructure changes are tracked alongside code changes,
   enabling audit trails required by regulatory frameworks such as IEC 62304 for medical
   device software.
3. **Automation** — Manual provisioning is error-prone and does not scale beyond
   single-researcher setups ([Artac et al., 2017](https://doi.org/10.1016/j.jss.2016.12.013)).

### 2.2 Taxonomy of IaC Approaches

| Approach | Examples | Language Type | State Model |
|----------|----------|---------------|-------------|
| **Declarative DSL** | Terraform HCL, CloudFormation YAML | Domain-specific | Desired-state convergence |
| **Imperative GPL** | Pulumi (Python/TS/Go), AWS CDK | General-purpose | Desired-state with procedural logic |
| **Configuration Mgmt** | Ansible, Chef, Puppet | YAML/Ruby DSL | Imperative convergence |
| **Transpiled GPL** | CDKTF (deprecated), CDK8s | GPL → DSL → API | Two-phase compilation |

The distinction between declarative DSLs and imperative GPLs is central to the LLM code
generation analysis in Section 5. Declarative DSLs constrain expressiveness in exchange for
safety; imperative GPLs offer full Turing-completeness at the cost of potential side effects.

### 2.3 The Reproducibility Crisis and IaC

[Peng, R. D. (2011). "Reproducible Research in Computational Science." *Science*, 334(6060), 1226-1227.](https://doi.org/10.1126/science.1213847)
argued that computational reproducibility requires capturing not just code and data but
the entire execution environment. IaC directly addresses this gap by codifying the
infrastructure layer. [Pineau et al. (2021). "Improving Reproducibility in Machine Learning
Research." *JMLR*.](https://arxiv.org/abs/2003.12206) — the NeurIPS reproducibility
program paper — explicitly identified infrastructure specification as a reproducibility
requirement. More recently, [Semmelrock et al. (2025). "Reproducibility in ML-based Research."
*AI Magazine*.](https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002) confirmed that
infrastructure remains a key barrier to ML reproducibility in practice.

For ML research specifically, [Kreuzberger et al. (2023). "Machine Learning Operations
(MLOps): Overview, Definition, and Architecture." *IEEE Access*.](https://arxiv.org/abs/2205.02302)
— the definitive MLOps architecture paper — positions infrastructure automation as a
core layer of the MLOps stack. [Zaharia et al. (2018). "Accelerating the Machine Learning
Lifecycle with MLflow." *IEEE Data Eng. Bull.*](https://people.eecs.berkeley.edu/~matei/papers/2018/ieee_mlflow.pdf)
demonstrated that experiment tracking alone is insufficient — the infrastructure on which
experiments run must also be reproducible.

---

## 3. Pulumi: Architecture and Design Philosophy {#3-pulumi-architecture}

### 3.1 Core Architecture

Pulumi employs a **language-host / engine** separation architecture ([Pulumi Docs, 2026](https://www.pulumi.com/docs/)):

```
┌─────────────────────────────┐
│ User Program (Python/TS/Go) │  ← General-purpose language
├─────────────────────────────┤
│ Pulumi SDK                  │  ← Resource model, ComponentResource
├─────────────────────────────┤
│ Pulumi Engine               │  ← Dependency graph, diff, deployment
├─────────────────────────────┤
│ Cloud Provider Plugins      │  ← AWS, GCP, Hetzner, Oracle, etc.
└─────────────────────────────┘
```

The user program declares *desired state* using standard programming constructs (classes,
functions, loops, conditionals). The Pulumi engine resolves the dependency graph and
converges the actual state to the desired state. This is fundamentally different from
Terraform's approach where HCL is parsed into a static graph — Pulumi's graph is
constructed at runtime, enabling dynamic resource creation based on computed values.

### 3.2 Key Differentiators

1. **Native language support** — Python, TypeScript, Go, C#, Java, YAML. No DSL to learn.
2. **Automation API** — Programmatic `pulumi up`/`destroy` from within Python scripts
   ([Pulumi Docs, 2026](https://www.pulumi.com/docs/iac/packages-and-automation/automation-api/)).
3. **ComponentResource** — Reusable infrastructure components as classes with typed inputs/outputs.
4. **Stack references** — Cross-stack dependencies (e.g., MLflow stack references DNS stack).
5. **Policy as Code** — CrossGuard policies written in the same language as the infrastructure.
6. **State backends** — Pulumi Cloud (SaaS), S3, Azure Blob, GCS, or local filesystem.
7. **Terraform provider bridge** — All 4,800+ Terraform providers available in Pulumi
   ([Pulumi Docs, 2026](https://www.pulumi.com/docs/iac/comparisons/terraform/)).

### 3.3 Licensing

Pulumi's open-source projects use the **Apache License 2.0** (permissive). The Pulumi
engine, CLI, and all provider SDKs are open source. Pulumi Cloud (the SaaS state
management and collaboration platform) is proprietary but optional — self-managed state
backends (S3, local file) are fully supported for free.

This contrasts with Terraform's **Business Source License (BSL 1.1)** adopted in August
2023, which restricts commercial use of Terraform in competing products ([HashiCorp, 2023](https://www.hashicorp.com/en/blog/hashicorp-adopts-business-source-license)).
The BSL change prompted the creation of **OpenTofu** under the Linux Foundation.

---

## 4. Pulumi vs Terraform vs OpenTofu vs CDKTF {#4-pulumi-vs-terraform}

### 4.1 Feature Comparison Matrix

| Feature | Pulumi | Terraform | OpenTofu | CDKTF |
|---------|--------|-----------|----------|-------|
| **Language** | Python, TS, Go, C#, Java | HCL (DSL) | HCL (DSL) | Python, TS → HCL transpile |
| **License** | Apache 2.0 | BSL 1.1 | MPL 2.0 | **Deprecated (Dec 2025)** |
| **Provider ecosystem** | ~150 native + 4,800 bridged | 4,800+ | 4,800+ (forked) | Terraform providers |
| **State management** | Cloud/S3/local | Cloud/S3/local | S3/local (no SaaS) | Terraform state |
| **AI agent support** | Neo + MCP server | Infragraph + MCP | None | N/A |
| **Market share (2026, est.)** | ~8-12% | ~29-34% | ~5% | 0% (archived) |
| **Agentic IaC** | Public preview (Neo) | Preview (Infragraph) | None | N/A |
| **Testing** | Standard test frameworks | `terraform test` (limited) | `tofu test` | Standard test frameworks |
| **Refactoring** | Standard IDE tooling | Manual `moved` blocks | Manual `moved` blocks | Standard IDE tooling |

### 4.2 The CDKTF Deprecation

CDKTF was introduced by HashiCorp in July 2020 as a bridge between general-purpose
languages and Terraform. It was **deprecated on December 10, 2025**, with its GitHub
repository archived ([Pulumi Blog, 2025](https://www.pulumi.com/blog/cdktf-is-deprecated-whats-next-for-your-team/)).

The deprecation is significant because CDKTF was the primary alternative for teams who
wanted to use Python/TypeScript with Terraform's provider ecosystem. HashiCorp's official
recommendation is to export CDKTF projects to HCL via `cdktf synth --hcl` — effectively
asking teams to return to the DSL they originally chose CDKTF to avoid.

**Implications for MinIVess:** CDKTF elimination leaves Pulumi as the *only* production-grade
option for Python-based IaC with multi-cloud provider support. This is a structural
advantage that simplifies the tool selection decision.

### 4.3 OpenTofu: The Open-Source Fork

OpenTofu was created in September 2023 by a coalition of companies (Gruntwork, Spacelift,
env0) in response to HashiCorp's BSL license change. It is maintained under the Linux
Foundation and uses the **MPL 2.0** license.

OpenTofu maintains API compatibility with Terraform 1.5.x, supports the same HCL syntax,
and can use the same providers. However, it has **no AI agent integration**, no equivalent
to Pulumi Neo or HashiCorp Infragraph, and a smaller development team. For an agentic MLOps
platform, OpenTofu's lack of AI tooling is a significant gap.

### 4.4 When Terraform/HCL Is the Better Choice

It is important to acknowledge scenarios where Terraform remains advantageous over Pulumi:

1. **Operations-focused teams without software engineering backgrounds.** HCL's deliberately
   constrained expressiveness prevents footguns that Turing-complete languages enable. For
   platform teams that primarily *operate* infrastructure rather than *develop* it, the
   declarative-only model reduces the risk of imperative anti-patterns
   ([Guerriero et al., 2019](https://doi.org/10.1109/ICSME.2019.00089)).

2. **Ecosystem maturity and community size.** Terraform holds approximately 33% market share
   ([CNCF Annual Survey, 2024](https://www.cncf.io/reports/cncf-annual-survey-2024/)) with
   4,800+ providers, vastly more Stack Overflow answers, and a significantly larger hiring
   pool. For academic labs with high turnover (PhD students leaving every 3-5 years),
   "can the next student find help?" is a critical consideration.

3. **Compliance and audit in regulated environments.** In IEC 62304 (medical device software)
   and GxP (pharmaceutical) contexts, HCL's limited expressiveness can be an *advantage*
   for auditors who can understand the full infrastructure from declarations without tracing
   imperative control flow.

4. **Existing Terraform investment.** Organizations with extensive Terraform module libraries,
   CI/CD pipelines, and team expertise face non-trivial migration costs. The Terraform →
   Pulumi migration path exists (`pulumi convert --from terraform`), but real-world
   migrations require substantial refactoring.

5. **State-only management needs.** For pure infrastructure provisioning without complex
   logic, Terraform's `plan` → `apply` workflow is simpler and has fewer moving parts
   than Pulumi's engine + language host architecture.

### 4.5 Tradeoffs Between HCL and GPL Approaches for ML Infrastructure

Terraform's HCL was designed for static infrastructure declarations. ML infrastructure
requires dynamic resource creation patterns that are awkward in HCL:

```hcl
# Terraform: Dynamic HPO trial provisioning (awkward)
variable "hpo_trials" {
  type = list(object({
    learning_rate = number
    batch_size    = number
  }))
}

resource "runpod_pod" "trial" {
  for_each = { for idx, t in var.hpo_trials : idx => t }
  gpu_type = "NVIDIA RTX 4090"
  env = {
    LEARNING_RATE = each.value.learning_rate
    BATCH_SIZE    = each.value.batch_size
  }
}
```

```python
# Pulumi Python: Same thing, naturally
for i, trial in enumerate(hpo_trials):
    runpod.Pod(f"trial-{i}",
        gpu_type="NVIDIA RTX 4090",
        env={
            "LEARNING_RATE": str(trial.learning_rate),
            "BATCH_SIZE": str(trial.batch_size),
        })
```

The Pulumi version uses standard Python idioms; the Terraform version requires `for_each`
with a comprehension-like expression in a DSL that does not support full iteration,
error handling, or type checking.

---

## 5. LLM Code Generation: Imperative vs Declarative IaC {#5-llm-code-generation}

### 5.1 The Token Efficiency vs Cognitive Efficiency Tradeoff

[Pulumi (2026). "Token Efficiency vs Cognitive Efficiency: Choosing IaC for AI Agents."](https://www.pulumi.com/blog/token-efficiency-vs-cognitive-efficiency-choosing-iac-for-ai-agents/)
presents the first systematic benchmark of LLM code generation performance across IaC
paradigms. The study evaluates two scenarios — initial resource creation and refactoring
into reusable components — across two models (Claude Opus 4.6 and GPT-5.2-Codex).

#### Initial Generation (Simple Resource Creation)

| Model | Format | Output Tokens | Pass Rate | Repairs Needed |
|-------|--------|:---:|:---:|:---:|
| Claude Opus 4.6 | Terraform HCL | 2,007 | 5/5 | 0/5 |
| Claude Opus 4.6 | Pulumi TypeScript | 2,555 | 5/5 | 0/5 |
| GPT-5.2-Codex | Terraform HCL | 1,565 | 2/5 | 2/5 |
| GPT-5.2-Codex | Pulumi TypeScript | 2,322 | 0/5 | 5/5 |

For simple resource creation, HCL generates 21-33% fewer tokens. This is expected — HCL
is more concise for declaring individual resources.

#### Refactoring (Extracting Reusable Components)

| Model | Format | Output Tokens | Pass Rate | Repairs Needed |
|-------|--------|:---:|:---:|:---:|
| Claude Opus 4.6 | Pulumi TypeScript | 2,720 | **5/5** | **0/5** |
| Claude Opus 4.6 | Terraform HCL | 3,379 | 5/5 | 0/5 |
| GPT-5.2-Codex | Pulumi TypeScript | 2,477 | **4/5** | 1/5 |
| GPT-5.2-Codex | Terraform HCL | 1,356 | **0/5** | 5/5 |

The refactoring results are striking. Pulumi achieves **20% fewer tokens** (2,720 vs 3,379
for Opus) and **zero repairs** for the refactoring task. GPT-5.2-Codex cannot successfully
refactor Terraform HCL at all (0/5 pass rate).

#### Total Pipeline Cost (Generation + Refactoring + Repairs)

| Model | Format | Total Tokens | Total Cost |
|-------|--------|:---:|:---:|
| Opus | Pulumi TypeScript | 8,183 | **$0.146** |
| Opus | Terraform HCL | 14,669 | $0.249 |
| GPT-5.2-Codex | Terraform HCL | 8,723 | $0.084 |
| GPT-5.2-Codex | Pulumi TypeScript | 15,211 | $0.138 |

**Key insight:** Claude Opus + Pulumi achieves **41% lower total pipeline cost** ($0.146 vs
$0.249) compared to Claude Opus + Terraform. The initial token advantage of HCL is
overwhelmed by the repair cycles required during refactoring.

### 5.2 Why General-Purpose Languages Win for LLMs

The fundamental reason is **training data distribution**. LLMs are trained on billions of
tokens of Python, TypeScript, Go, and other general-purpose languages. HCL represents a
tiny fraction of the training corpus. When LLMs need to perform complex operations
(refactoring, error correction, dynamic resource creation), they draw on patterns learned
from the dominant languages.

Specifically, HCL refactoring requires domain-specific constructs — `count`, `for_each`,
`dynamic` blocks, `module` variable plumbing — that have "far less representation in
training data" compared to standard object-oriented patterns like extracting a class,
adding typed constructor parameters, or composing functions ([Pulumi Blog, 2026](https://www.pulumi.com/blog/token-efficiency-vs-cognitive-efficiency-choosing-iac-for-ai-agents/)).

### 5.3 Independent Academic Counterevidence: DPIaC-Eval

The optimistic vendor-published results must be contextualized against independent academic
benchmarks. The DPIaC-Eval benchmark ([arXiv 2506.05623, 2025](https://arxiv.org/abs/2506.05623))
— the first academic benchmark specifically for LLM-generated IaC — evaluated 153 real-world
deployment scenarios and found:

- **20.8-30.2% first-attempt deployment success rate** across state-of-the-art LLMs
- **42.7% of syntactically correct templates fail at deployment** due to misconfigurations
- Configuration drift, implicit dependencies, and provider-specific quirks remain major failure modes

Similarly, the DevOps-Gym benchmark ([ICLR, 2026](https://arxiv.org/abs/2601.20882v1))
evaluated 700+ real-world DevOps tasks and found that state-of-the-art agents "remain
unable to handle new tasks such as monitoring and build and configuration."

These results paint a substantially more cautious picture than the Pulumi vendor benchmark.
The discrepancy likely reflects differences in task complexity: the Pulumi study used
relatively simple resource creation and refactoring tasks, while DPIaC-Eval and DevOps-Gym
test complex, multi-resource, production-grade scenarios. **The implication is that
LLM-generated IaC is a useful acceleration tool but not yet a reliable autonomous
deployment mechanism for complex infrastructure.**

### 5.4 Evidence from VeriCoding: Imperative Familiarity Effect

The VeriCoding benchmark ([Bursuc et al., 2025](https://arxiv.org/abs/2509.22908)) provides
independent evidence for this phenomenon. The benchmark evaluates LLM performance on formal
verification across three proof languages:

| Language | Paradigm | LLM Success Rate |
|----------|----------|:---:|
| **Dafny** | Imperative (closest to Python/Java) | **82%** |
| **Verus** | Rust-based (systems programming) | 44% |
| **Lean** | Functional (mathematical) | 27% |

Dafny's syntax is "closer to imperative programming languages, which dominate LLM training
data" ([Bursuc et al., 2025](https://arxiv.org/abs/2509.22908)). The 3x performance gap
between Dafny (82%) and Lean (27%) mirrors the pattern observed in IaC: LLMs perform
dramatically better when the target language is syntactically similar to their training
distribution.

**IaC analogy:** Terraform HCL is to Lean as Pulumi Python is to Dafny. Both HCL and Lean
are domain-specific with limited representation in training data; both Pulumi Python and
Dafny are syntactically close to the imperative languages that dominate LLM pretraining
corpora.

### 5.5 The Pulumi MCP Server: Schema-Aware Generation

The Pulumi MCP (Model Context Protocol) server ([Pulumi Blog, 2026](https://www.pulumi.com/blog/mcp-server-ai-assistants/))
shifts the LLM workflow from "generate, fail, read error, retry" to "look up schema,
generate correctly." The server exposes five tools to AI assistants:

| MCP Tool | Purpose |
|----------|---------|
| `pulumi_registry_listResources` | Discover resources within provider modules |
| `pulumi_registry_getResource` | Retrieve detailed resource schemas and required properties |
| `pulumi_cli_preview` | Validate infrastructure changes before deployment |
| `pulumi_cli_up` | Deploy infrastructure |
| `pulumi_cli_stack_output` | Retrieve deployment outputs |

This integration is available for **Claude Code**, Cursor, GitHub Copilot, and Windsurf.
No equivalent exists for Terraform — the HCP Terraform MCP server (beta) provides workspace
management but not schema-aware code generation.

---

## 6. Agentic Infrastructure: The Emerging Paradigm {#6-agentic-infrastructure}

### 6.1 From Copilot to Agent: The Evolution

The infrastructure management paradigm is undergoing a phase transition from
**human-authored code** → **AI-assisted code** → **AI-authored, human-approved code**:

| Phase | Era | Workflow | Human Role |
|-------|-----|----------|------------|
| **Manual** | Pre-2013 | Click in console, SSH, manual scripts | Implementer |
| **IaC** | 2013-2022 | Write HCL/YAML, `terraform apply` | Author |
| **Copilot** | 2022-2025 | AI suggests, human edits, human applies | Editor |
| **Agentic** | 2025+ | AI plans, previews, creates PR, human approves | Reviewer/Architect |

[Pulumi (2026). "AI Predictions for 2026: A DevOps Engineer's Guide."](https://www.pulumi.com/blog/ai-predictions-2026-devops-guide/)
predicts that "DevOps engineers transition from coders to system architects" who "define
constraints while agents handle implementation details."

### 6.2 Pulumi Neo: The First Agentic Platform Engineer

Pulumi Neo, launched in public preview on September 16, 2025
([InfoQ, 2025](https://www.infoq.com/news/2025/09/pulumi-neo/)), is described as "the
industry's first AI agent built from the ground up to execute, govern, and optimize complex
cloud automation at enterprise scale."

**Key capabilities:**
- **Natural language commands:** "Upgrade all Kubernetes clusters to the latest stable version"
- **Multi-cloud context awareness:** Understands dependencies across AWS, Azure, GCP, and 160+ providers
- **Human-in-the-loop approvals:** Configurable autonomy levels from full review to autonomous execution
- **Complete audit trails:** Every action is previewed, logged, and reversible
- **AGENTS.md support:** Encodes organizational standards that Neo applies automatically
  ([Pulumi Blog, 2026](https://www.pulumi.com/blog/pulumi-neo-now-supports-agentsmd/))

**Documented impact:** Werner Enterprises reduced infrastructure provisioning from 3 days to
4 hours (18x improvement), enabling 75% faster feature deployment
([PR Newswire, 2025](https://www.prnewswire.com/news-releases/introducing-pulumi-neo-the-industrys-first-ai-powered-platform-engineer-302556718.html)).

### 6.3 HashiCorp Project Infragraph: The Competing Vision

HashiCorp (acquired by IBM) announced Project Infragraph at HashiConf 2025, entering
private beta in December 2025 ([IBM Newsroom, 2025](https://newsroom.ibm.com/2025-09-25-hashicorp-previews-the-future-of-agentic-infrastructure-automation-with-project-infragraph)).

Infragraph is a **real-time infrastructure knowledge graph** that unifies state,
configuration, policy, and ownership metadata across hybrid and multi-cloud environments.
AI agents query the graph to understand infrastructure topology before making changes.

**Key differences from Pulumi Neo:**

| Dimension | Pulumi Neo | HashiCorp Infragraph |
|-----------|-----------|---------------------|
| **Maturity** | Public preview (Sep 2025) | Private beta (Dec 2025) |
| **Language** | Python/TypeScript (GPL) | HCL (DSL) |
| **Architecture** | Agent executes Pulumi programs | Agent queries infra graph + generates HCL |
| **IBM integration** | N/A | watsonx, Ansible, OpenShift, Turbonomic |
| **MCP server** | GA (registry + CLI tools) | Beta (workspace management) |
| **Open source** | Engine + CLI (Apache 2.0) | BSL 1.1 (Terraform) |

Both approaches converge on the same vision: AI agents that understand infrastructure
context and execute changes with human approval. The key architectural difference is
that Pulumi agents generate code in languages LLMs already know well, while Infragraph
agents must generate HCL — a DSL with limited LLM training data (Section 5).

### 6.4 The MCP Protocol as Infrastructure Interface

The Model Context Protocol ([Anthropic, 2024](https://modelcontextprotocol.io/)) provides
a standardized interface between AI agents and external tools. Both Pulumi and HashiCorp
have released MCP servers:

- **Pulumi MCP:** Registry lookup + CLI execution (preview, up, outputs)
- **Terraform MCP:** Workspace management + registry browsing
- **Vault MCP:** Secret management with RBAC

MCP integration means that Claude Code (or any MCP-compatible agent) can discover
infrastructure resources, generate code, validate it with `pulumi preview`, and deploy
it with `pulumi up` — all within a single conversation. This is the practical
implementation of agentic IaC.

---

## 7. Claude Code and Pulumi: Agentic IaC in Practice {#7-claude-code-pulumi}

### 7.1 Claude Agent Skills for Infrastructure

[Pulumi (2026). "The Claude Skills I Actually Use for DevOps."](https://www.pulumi.com/blog/top-8-claude-skills-devops-2026/)
documents eight Claude Code skills for infrastructure management:

| Skill | Purpose | Key Capability |
|-------|---------|----------------|
| `pulumi-typescript` | IaC patterns | TypeScript patterns, ESC, ComponentResource |
| `pulumi-esc` | Secrets management | OIDC, dynamic credentials, environment composition |
| `pulumi-best-practices` | Quality | Dependencies, safe refactoring, resource protection |
| `monitoring-expert` | Observability | Prometheus, Grafana, DataDog patterns |
| `kubernetes-specialist` | Container orchestration | Security contexts, resource limits, PDB |
| `systematic-debugging` | Troubleshooting | Root cause analysis, hypothesis testing |
| `k8s-security-policies` | Security | NetworkPolicies, Pod Security Standards, RBAC |
| `security-review` | Audit | Secrets management, injection prevention |

These skills encode organizational conventions directly into Claude's decision-making,
producing "convention-aligned code" rather than merely "syntax-compliant code."

### 7.2 Progressive Disclosure for Infrastructure Context

The progressive disclosure pattern — loading brief descriptions initially, activating full
instructions only when selected — is critical for infrastructure context management.
[Pulumi (2026)](https://www.pulumi.com/blog/ai-predictions-2026-devops-guide/) notes that
a workflow consuming ~150,000 tokens was reimplemented with code execution using only
~2,000 tokens (98.7% reduction).

MinIVess already uses this pattern (see `CLAUDE.md` Layer 0/1/2 knowledge system). Pulumi
MCP extends it to infrastructure: Claude Code loads `pulumi_registry_listResources` (brief),
then `pulumi_registry_getResource` (detailed) only when writing specific resources.

### 7.3 Why Claude Code + Pulumi Python > Claude Code + Terraform HCL

The VeriCoding benchmark finding (Section 5.4) and the Pulumi token efficiency study
(Section 5.1) converge on the same conclusion: **Claude Code performs better with Python
infrastructure code than with HCL.**

Specific advantages:

1. **Familiar syntax** — Claude's training data contains orders of magnitude more Python
   than HCL. Python patterns (classes, decorators, list comprehensions) are deeply encoded
   in the model's weights.
2. **Standard tooling** — `mypy` type checking, `pytest` testing, `ruff` linting all work
   on Pulumi Python code. Claude Code already knows these tools intimately.
3. **Debugging** — Python stack traces are interpretable by Claude; Terraform plan output
   requires domain-specific parsing.
4. **Refactoring** — Extracting a `ComponentResource` class is standard OOP; extracting a
   Terraform module requires `variables.tf`, `outputs.tf`, `main.tf` boilerplate.
5. **MCP integration** — Pulumi's MCP server provides schema-aware generation; Terraform's
   MCP server provides workspace management but not code generation assistance.

---

## 8. MinIVess Stack Analysis: Service-by-Service Pulumi Gains {#8-stack-analysis}

### 8.1 Current Stack (Docker Compose, 18 Services)

The MinIVess MLOps platform runs 18 services across two Docker Compose files
(`docker-compose.yml` for infrastructure, `docker-compose.flows.yml` for ML flows):

| Service | Category | Current Deployment | Pulumi Gain |
|---------|----------|-------------------|-------------|
| **PostgreSQL** | Database | Docker Compose | **HIGH** — Managed RDS/Cloud SQL eliminates backup/HA burden |
| **MinIO** | Object Storage | Docker Compose | **HIGH** — Replace with S3/GCS/Hetzner Object Storage |
| **MLflow** | Experiment Tracking | Docker Compose | **HIGH** — Cloud-accessible, TLS, authentication |
| **Prefect Server** | Orchestration | Docker Compose | **MEDIUM** — Prefect Cloud alternative, or self-hosted with HA |
| **Grafana** | Monitoring | Docker Compose | **MEDIUM** — Grafana Cloud free tier (10K metrics) |
| **Prometheus** | Metrics | Docker Compose | **MEDIUM** — Managed Prometheus (AMP/GMP) |
| **OpenTelemetry** | Tracing | Docker Compose | **LOW** — Lightweight sidecar, runs anywhere |
| **Langfuse** | LLM Observability | Docker Compose | **MEDIUM** — Langfuse Cloud alternative |
| **Label Studio** | Annotation | Docker Compose | **MEDIUM** — Cloud-hosted Label Studio Enterprise |
| **MONAI Label** | Medical Annotation | Docker Compose | **LOW** — Specialized, self-hosted only |
| **Marquez** | Data Lineage | Docker Compose | **LOW** — Lightweight, runs anywhere |
| **Ollama** | Local LLM | Docker Compose | **LOW** — Local-only by design |
| **Falco** | Security Monitor | Docker Compose | **LOW** — Requires kernel access |
| **BentoML** | Model Serving | Docker Compose | **MEDIUM** — BentoCloud for production serving |
| **Train Flow** | ML Training | Docker + SkyPilot | **HIGH** — SkyPilot GPU provisioning |
| **HPO Flow** | Hyperparameters | Docker + SkyPilot | **HIGH** — Parallel cloud GPU trials |
| **Analysis Flow** | Evaluation | Docker Compose | **LOW** — CPU-only, runs anywhere |
| **Dashboard Flow** | Reporting | Docker Compose | **LOW** — CPU-only, runs anywhere |

### 8.2 High-Gain Services (Should Move to Pulumi First)

**PostgreSQL → Managed Database**

```python
# Pulumi Python: Managed PostgreSQL on Hetzner (concept)
import pulumi
import pulumi_hcloud as hcloud

db_server = hcloud.Server("postgres",
    server_type="cx32",  # 4 vCPU, 8 GB RAM, EUR 5.99/mo
    image="ubuntu-24.04",
    ssh_keys=[ssh_key.id],
    user_data=cloud_init_postgres,  # Automated PostgreSQL setup
)
```

Or on AWS:
```python
import pulumi_aws as aws

db = aws.rds.Instance("mlflow-db",
    engine="postgres",
    engine_version="16",
    instance_class="db.t4g.micro",  # Free tier eligible
    allocated_storage=20,
    db_name="mlflow",
)
```

**Benefits:** Automated backups, point-in-time recovery, monitoring, replication — none
of which exist in the current Docker Compose deployment.

**MinIO → Cloud Object Storage**

```python
# Replace MinIO with Hetzner Object Storage or AWS S3
bucket = aws.s3.Bucket("mlflow-artifacts",
    acl="private",
    versioning=aws.s3.BucketVersioningArgs(enabled=True),
)
```

**Benefits:** Eliminates MinIO operational burden, native versioning, cross-region
replication, lifecycle policies for cost management.

**MLflow → Cloud-Accessible Server**

This is the #1 blocker for SkyPilot integration (see companion report Section 10).
Pulumi automates the deployment:

```python
# MLflow server on Hetzner CX32 with TLS
mlflow_server = hcloud.Server("mlflow",
    server_type="cx32",
    image="ubuntu-24.04",
    ssh_keys=[ssh_key.id],
    user_data="""#!/bin/bash
    apt-get update && apt-get install -y docker.io docker-compose
    docker compose -f /opt/mlflow/docker-compose.yml up -d
    """,
)

# DNS record via Cloudflare
dns_record = cloudflare.Record("mlflow-dns",
    zone_id=zone.id,
    name="mlflow",
    type="A",
    value=mlflow_server.ipv4_address,
)
```

### 8.3 Medium-Gain Services (Move in Phase 2)

Grafana, Prometheus, Langfuse, and Prefect can benefit from managed alternatives:

| Service | Self-Hosted | Managed Alternative | Monthly Cost |
|---------|------------|-------------------|:---:|
| Grafana | Docker | Grafana Cloud Free | $0 (10K metrics) |
| Prometheus | Docker | AWS AMP / Grafana Cloud | $0-15/mo |
| Langfuse | Docker | Langfuse Cloud | $0 (hobby) |
| Prefect | Docker | Prefect Cloud | $0 (personal) |
| Label Studio | Docker | Label Studio Cloud | $0 (community) |

### 8.4 Low-Gain Services (Keep as Docker Compose)

MONAI Label, Marquez, Ollama, Falco, and CPU-only ML flows have minimal benefit from
cloud deployment. They are either local-only by design or too lightweight to justify
managed alternatives.

---

## 9. Deployment Scenarios: Intranet vs Hetzner vs AWS {#9-deployment-scenarios}

### 9.1 Three-Tier Deployment Model

| Tier | Environment | Use Case | Infrastructure |
|------|-------------|----------|---------------|
| **Tier 1** | Local Intranet | Academic lab, daily development | Docker Compose on lab server |
| **Tier 2** | Hetzner VPS | Small team, budget-conscious | Pulumi + Hetzner Cloud |
| **Tier 3** | AWS/GCP | Startup/scaleup, enterprise features | Pulumi + managed services |

### 9.2 Tier 1: Local Intranet (Academic Research Lab)

**Profile:** University lab, 2-10 researchers, 1-3 GPU workstations, institutional network.

| Component | Deployment | Monthly Cost |
|-----------|-----------|:---:|
| PostgreSQL | Docker Compose on lab server | $0 |
| MinIO | Docker Compose on lab server | $0 |
| MLflow | Docker Compose on lab server | $0 |
| Prefect | Docker Compose on lab server | $0 |
| Grafana | Docker Compose on lab server | $0 |
| GPU Training | Local GPU or SkyPilot (RunPod) | Variable |
| **Total infrastructure** | | **$0/month** |

**Pulumi role:** Minimal. Docker Compose is sufficient for intranet deployment. Pulumi
adds value for:
- Automating lab server provisioning (if using institutional cloud allocation)
- Managing DNS records for `mlflow.lab.university.edu`
- Provisioning SkyPilot-launched cloud GPUs (handled by SkyPilot, not Pulumi)

**Verdict:** Pulumi is NOT required for pure intranet academic deployments. Docker Compose
is the correct tool at this scale.

### 9.3 Tier 2: Hetzner VPS (Budget-Conscious Team)

**Profile:** Small research group or early-stage startup, 3-15 users, budget <$50/month.

| Component | Deployment | Monthly Cost |
|-----------|-----------|:---:|
| PostgreSQL | Hetzner CX32 (shared with MLflow) | EUR 5.99 (shared) |
| MinIO → Hetzner Object Storage | Managed | EUR 4.99 (1 TB included) |
| MLflow | Hetzner CX32 | (shared above) |
| Prefect | Hetzner CX22 or Prefect Cloud Free | EUR 0-3.49 |
| Grafana | Grafana Cloud Free | $0 |
| DNS + TLS | Cloudflare Free | $0 |
| GPU Training | SkyPilot (RunPod) | Variable |
| **Total infrastructure** | | **~EUR 11-15/month** |

**Pulumi value:** High. A single `pulumi up` provisions the entire stack:
```bash
cd deployment/pulumi && pulumi up -s hetzner-prod
# Creates: VPS, firewall rules, Docker Compose deployment,
#          DNS records, TLS certs, monitoring
```

Tear-down is equally simple:
```bash
pulumi destroy -s hetzner-prod
# Removes everything cleanly — no orphaned resources
```

### 9.4 Tier 3: AWS/GCP (Startup/Scaleup)

**Profile:** Deep tech startup, 10-50 engineers, production workloads, compliance requirements.

| Component | Deployment | Monthly Cost |
|-----------|-----------|:---:|
| PostgreSQL | AWS RDS (db.t4g.medium) | ~$30/month |
| Object Storage | AWS S3 | ~$5/month (100 GB) |
| MLflow | AWS ECS Fargate | ~$20/month |
| Prefect | Prefect Cloud (Team) | ~$25/month |
| Grafana | Grafana Cloud (Pro) | ~$20/month |
| DNS + TLS | Route53 + ACM | ~$1/month |
| GPU Training | SkyPilot (RunPod/Lambda/GCP Spot) | Variable |
| VPC + NAT Gateway | AWS networking | ~$35/month |
| **Total infrastructure** | | **~$136/month** |

**Pulumi value:** Critical. AWS has ~200 services with thousands of configuration options.
Manual provisioning is error-prone and non-reproducible. Pulumi automates:
- VPC with public/private subnets
- Security groups with least-privilege access
- RDS with automated backups, encryption at rest
- ECS Fargate service definitions with auto-scaling
- IAM roles with scoped permissions
- CloudWatch alarms and SNS notifications

**AWS-specific managed services Hetzner cannot offer:**

| AWS Service | Hetzner Equivalent | Gap |
|------------|-------------------|-----|
| RDS (managed PostgreSQL) | Manual Docker PostgreSQL | No auto-backup, no HA |
| S3 (object storage) | Hetzner Object Storage | Limited lifecycle policies |
| ECS Fargate (serverless containers) | None | Must manage VPS |
| SageMaker (ML platform) | None | No equivalent |
| CloudWatch (monitoring) | Manual Prometheus | No managed alerting |
| IAM (identity) | None | No fine-grained access control |
| ACM (TLS certificates) | Let's Encrypt (manual) | No auto-renewal integration |
| Secrets Manager | None | Must manage secrets manually |

### 9.5 Comparison Matrix

| Dimension | Intranet | Hetzner | AWS |
|-----------|:---:|:---:|:---:|
| **Monthly cost** | $0 | EUR 11-15 | ~$136 |
| **Setup time (manual)** | 2 hours | 4 hours | 2-3 days |
| **Setup time (Pulumi)** | N/A | 5 min* | 5 min* |
| **Managed databases** | No | No | Yes (RDS) |
| **Auto-scaling** | No | No | Yes (ECS) |
| **Compliance (SOC2/HIPAA)** | No | Partial (EU GDPR) | Yes |
| **Multi-region HA** | No | No | Yes |
| **Pulumi ROI** | Low | **High** | **Critical** |
| **Team scalability** | 2-10 | 3-15 | 10-50+ |
| **Data sovereignty** | Full (on-prem) | EU (Hetzner) | Region-selectable |

*After one-time Pulumi stack development (3-4 hours for Hetzner, 8-12 hours for AWS).

---

## 10. Academic Research Labs vs Deep Tech Startups {#10-academic-vs-startup}

### 10.1 Academic Research Lab Requirements

Academic ML research labs have specific constraints that differ from commercial environments:

1. **Budget** — Grant-funded, often zero infrastructure budget. Free tiers are essential.
2. **Reproducibility** — Publications require exact environment reproduction.
3. **Turnover** — PhD students leave every 3-5 years; infrastructure knowledge is lost.
4. **Scale** — Typically 1-10 simultaneous experiments, not thousands.
5. **Compliance** — Institutional ethics board (IRB), not SOC2/HIPAA (unless medical data).
6. **Publication** — The infrastructure itself may be a research contribution.

**Optimal IaC approach for academic labs:**
- Docker Compose for local/intranet (Tier 1) — most labs will stay here
- Pulumi + Hetzner (Tier 2) for labs that need cloud-accessible MLflow for collaborators
- Pulumi + AWS (Tier 3) only for labs with significant cloud credits (AWS Research Credits,
  GCP Research Credits, Azure for Research)

### 10.2 Deep Tech Startup Requirements

Deep tech startups (biotech, robotics, materials science) have different constraints:

1. **Budget** — Venture-funded, willing to pay for reliability and speed.
2. **Scale** — Starts small but must grow 10-100x without re-architecture.
3. **Compliance** — SOC2, HIPAA (medical), ISO 13485 (medical devices), GxP (pharma).
4. **Team** — Mixed: PhD researchers + ML engineers + DevOps/platform engineers.
5. **Speed** — Time-to-market pressure; infrastructure cannot be a bottleneck.
6. **Integration** — Must connect with existing enterprise systems (SSO, SIEM, CMDB).

**Optimal IaC approach for startups:**
- Pulumi from Day 1 — infrastructure must be reproducible and scalable.
- AWS/GCP as primary cloud — managed services reduce operational burden.
- Pulumi Neo for platform engineering — reduces DevOps headcount needs.
- Policy-as-Code (CrossGuard) for compliance from Day 1 — retrofitting is expensive.

### 10.3 The Scaling Inflection Point

```
Researchers:     1          5          15          50
                 |          |           |           |
Academic lab:    Docker ────┤ Pulumi+   |           |
                 Compose    | Hetzner   |           |
                            |           |           |
Startup:         Pulumi ────┤ Pulumi+   ├ Pulumi+   ├ Pulumi+K8s
                 +AWS       | AWS(mgd)  | EKS       | Multi-region
```

The inflection point where Pulumi becomes necessary (not optional) is approximately
**5 team members** for startups (Day 1 for well-funded startups) and **15+ team members**
for academic labs. Below these thresholds, Docker Compose on a single server is simpler
and cheaper.

---

## 11. Pulumi Pricing and Total Cost of Ownership {#11-pricing-tco}

### 11.1 Pulumi Cloud Pricing (2026)

| Tier | Resources | Secrets | Users | Monthly Cost |
|------|-----------|---------|-------|:---:|
| **Individual** (Free) | 500 deploy min, unlimited stacks | 25 secrets | 1 | **$0** |
| **Team** | 150K free credits (~200 resources), $0.00025/credit above | $0.50/secret | Up to 10 | ~$0-50 |
| **Enterprise** | Volume pricing (starting ~$0.365/resource/mo) | $0.75/secret | Unlimited | Custom |
| **Open Source** | Team tier free for OSS maintainers | Team tier | Unlimited | **$0** |

([Pulumi Pricing, 2026](https://www.pulumi.com/pricing/))

**For MinIVess (18 services × ~5 resources each ≈ 90 resources):**
- Individual tier: **$0/month** (sufficient for solo researcher)
- Team tier: 200 free resources covers the entire stack at **$0/month**
- Above 200 resources: ~$0.37 × (resources - 200)/month

### 11.2 Total Cost of Ownership: Pulumi vs No IaC

| Cost Category | No IaC (Manual) | With Pulumi |
|---------------|:---:|:---:|
| Tool licensing | $0 | $0 (Individual/OSS) |
| Initial setup | 4-8 hours | 8-16 hours (one-time stack dev) |
| Subsequent deploy | 1-2 hours | **5 minutes** |
| Environment rebuild | 4-8 hours | **5 minutes** |
| Onboarding new member | 2-4 hours | **30 minutes** (`pulumi up`) |
| Disaster recovery | Hours-days | **5 minutes** |
| Annual maintenance | ~40 hours | ~8 hours |

**Break-even:** After approximately **3 deployments** (including teardown/rebuild cycles),
Pulumi has lower cumulative time cost than manual provisioning.

---

## 12. Implementation Roadmap for MinIVess {#12-implementation-roadmap}

### Phase 0: Foundation (Week 1)

| Task | Description | Effort |
|------|-------------|--------|
| P0.1 | Install Pulumi CLI, create Pulumi Cloud account | 30 min |
| P0.2 | Create `deployment/pulumi/` directory structure | 15 min |
| P0.3 | Write `Pulumi.yaml` (project config) + `Pulumi.dev.yaml` (stack) | 30 min |
| P0.4 | Configure Pulumi MCP server for Claude Code | 15 min |

### Phase 1: MLflow Accessibility (Week 1-2)

This phase directly corresponds to T1/T1b in the SkyPilot XML plan.

| Task | Description | Effort |
|------|-------------|--------|
| P1.1 | **USER DECISION:** Target cloud (Hetzner recommended) | — |
| P1.2 | Write `deployment/pulumi/mlflow_stack.py` (server, Docker, firewall) | 3 hours |
| P1.3 | Add DNS via `pulumi-cloudflare` | 1 hour |
| P1.4 | `pulumi up` — deploy MLflow | 5 min |
| P1.5 | Test MLflow logging from SkyPilot VM | 30 min |
| P1.6 | Import local mlruns to remote MLflow | 1 hour |

### Phase 2: Full Infrastructure Stack (Week 2-4)

| Task | Description | Effort |
|------|-------------|--------|
| P2.1 | Add PostgreSQL to Pulumi stack | 2 hours |
| P2.2 | Replace MinIO with cloud object storage | 2 hours |
| P2.3 | Add Grafana + Prometheus configuration | 2 hours |
| P2.4 | Add monitoring/alerting | 1 hour |
| P2.5 | Write integration tests for infrastructure | 2 hours |

### Phase 3: Multi-Environment (Month 2+)

| Task | Description | Effort |
|------|-------------|--------|
| P3.1 | Create `Pulumi.staging.yaml` stack | 1 hour |
| P3.2 | Create `Pulumi.prod.yaml` stack | 1 hour |
| P3.3 | Add CrossGuard compliance policies | 4 hours |
| P3.4 | AWS migration stack (for startup scaling) | 8-12 hours |

---

## 13. Implications for Agentic MLOps Research {#13-agentic-mlops-research}

### 13.1 The Agentic IaC Thesis

The convergence of LLM code generation, MCP protocol standardization, and production-grade
agentic platforms (Pulumi Neo, HashiCorp Infragraph) suggests a fundamental shift in how
ML infrastructure is managed. We propose the following thesis:

> **Agentic MLOps Thesis:** Within 2-3 years, the majority of ML infrastructure management
> will be performed by AI agents that understand the full infrastructure context, generate
> IaC in general-purpose languages, validate changes through automated preview, and execute
> with human approval. The role of the ML engineer shifts from infrastructure implementer
> to infrastructure architect and reviewer.

### 13.2 Evidence Supporting the Thesis

1. **LLM code generation quality is sufficient** — Claude Opus achieves 5/5 pass rate on
   both initial generation and refactoring of Pulumi TypeScript code (Section 5.1).

2. **The tooling exists** — Pulumi MCP server, Neo agentic platform, Claude Code skills
   for DevOps are all production-ready (Section 7).

3. **Enterprise adoption is measurable** — Werner Enterprises' 18x provisioning speedup
   demonstrates production viability (Section 6.2).

4. **The imperative language advantage is empirically supported** — Both the Pulumi token
   efficiency benchmark (41% cost reduction) and the VeriCoding formal verification study
   (82% vs 27% for imperative vs functional) provide quantitative evidence (Section 5).

5. **Both major IaC vendors are investing** — Pulumi (Neo) and HashiCorp (Infragraph) are
   independently converging on agentic infrastructure (Section 6).

### 13.3 Research Opportunities

The following research questions emerge from this analysis and may be relevant for the
academic manuscript referenced by the user:

1. **Benchmark: LLM-generated IaC quality across languages.** Extend the Pulumi token
   efficiency study to more models, more complex infrastructure, and more languages
   (Python vs TypeScript vs Go vs HCL).

2. **Agentic IaC reliability.** What is the failure rate of end-to-end agentic
   infrastructure deployment? How does human-in-the-loop approval affect reliability?

3. **Infrastructure reproducibility via IaC.** Does Pulumi-managed infrastructure improve
   computational reproducibility in ML experiments compared to manual Docker Compose?

4. **Cost optimization via agentic spot trading.** Can AI agents optimize ML training
   costs by monitoring spot prices and triggering jobs at price thresholds (Section 14
   of companion report)?

5. **The DSL vs GPL tradeoff for AI agents.** Is there a principled framework for
   predicting when DSLs (Terraform) outperform GPLs (Pulumi) for LLM code generation,
   beyond the training data distribution argument?

### 13.4 Slide Material: Key Takeaways for "Agentic Coding: From Vibes to Production"

For the presentation at `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/agentic-development/`:

**Slide: "Agentic IaC — The Next Frontier"**
- "Infrastructure is the last manual step in the ML lifecycle"
- Pulumi Neo: natural language → infrastructure plan → PR → deploy
- 18x provisioning speedup (Werner Enterprises case study)
- Claude Code + Pulumi MCP: schema-aware infrastructure generation

**Slide: "Why Python > HCL for AI Agents"**
- VeriCoding: Dafny (imperative) 82% vs Lean (functional) 27% — training data matters
- Pulumi benchmark: Claude + Pulumi = 41% cheaper than Claude + Terraform
- CDKTF deprecated (Dec 2025) — Pulumi is the only GPL option for multi-cloud IaC
- "LLMs perform best in languages they've seen the most"

**Slide: "From Vibes to Production — The IaC Dimension"**
- Vibes: "just click in the console" → Production: `pulumi up` with audit trail
- Agentic coding without agentic infrastructure = half the solution
- The ML model is only as reproducible as the infrastructure it runs on

**Slide: "Reality Check — What Doesn't Work Yet"**
- DPIaC-Eval: 20-30% first-attempt deployment success for LLM-generated IaC
- DevOps-Gym: agents "unable to handle monitoring and build configuration" tasks
- 42.7% of syntactically correct IaC fails at deployment (DPIaC-Eval)
- Security: prompt injection → unauthorized `pulumi up`; blast radius amplification
- Vendor lock-in: Pulumi Neo is Cloud-only; no open-source fork path
- Today's role: AI agent as **accelerator**, not autonomous deployer

**Slide: "Research Agenda — Open Questions"**
- Which IaC language optimizes for LLM code generation quality? (Formal benchmark needed)
- How does human-in-the-loop approval interact with IEC 62304 audit trails?
- Can spot price monitoring agents reduce ML training costs? (Limit order analogy)
- What is the carbon footprint of cloud vs local GPU training?
- Infrastructure reproducibility → ML experiment reproducibility: is there a measurable link?

---

## 14. Limitations and Open Questions {#14-limitations}

### 14.1 Limitations of This Analysis

1. **Benchmark bias** — The Pulumi token efficiency study (Section 5.1) was published by
   Pulumi, introducing potential selection bias in benchmark design. Independent replication
   is needed.

2. **VeriCoding analogy** — The Dafny/Lean/Verus benchmark (Section 5.4) measures formal
   verification, not IaC. The analogy to Pulumi/Terraform is suggestive but not direct.

3. **Small sample sizes** — The Pulumi benchmark uses 5 runs per condition. Statistical
   significance is limited.

4. **Provider ecosystem gap** — Pulumi has ~150 native providers vs Terraform's 4,800+.
   While the Terraform bridge covers most, some niche providers may not be available.

5. **Neo maturity** — Pulumi Neo is in public preview. Production reliability data is
   limited to case studies, not peer-reviewed research.

6. **Hetzner limitations** — Hetzner lacks managed databases, managed Kubernetes, and
   many services that AWS/GCP offer. The cost advantage comes with an operational burden.

### 14.2 Security Implications of Agentic IaC

The security implications of giving AI agents infrastructure deployment capabilities
deserve explicit analysis. The MAESTRO framework ([Cloud Security Alliance, 2025](https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro))
provides a structured approach to threat modeling in agentic systems.

Key security concerns for Pulumi + Claude Code agentic workflows:

1. **Prompt injection via MCP responses.** An agent with `pulumi_cli_up` capability could
   be manipulated via crafted MCP responses or adversarial prompts to deploy unauthorized
   infrastructure. The attack surface includes Pulumi registry responses, cloud API errors,
   and user-provided infrastructure descriptions.

2. **Blast radius amplification.** When a human writes IaC, errors typically affect one
   resource. When an agent generates and deploys infrastructure autonomously, a single
   hallucination (e.g., `0.0.0.0/0` security group rule) could expose the entire stack.
   ([Chhabra et al., 2025](https://arxiv.org/abs/2510.23883)) identify this as a core
   risk of agentic AI systems.

3. **Secret exposure through infrastructure state.** The Pulumi MCP integration gives
   Claude Code access to infrastructure state, which may contain sensitive values
   (database passwords, API keys, IAM credentials). Pulumi's state encryption mitigates
   this but requires explicit configuration.

4. **Accountability and audit trails.** If an AI agent creates infrastructure, who is
   accountable for compliance violations? For IEC 62304 (medical device software) and
   GxP environments, every infrastructure change requires traceable human authorization.
   Pulumi's audit log records who approved each change, but the "who" may be an agent
   rather than a named human — a gap in current regulatory frameworks.

5. **Lateral movement.** Cloud credentials used by Pulumi (AWS access keys, Hetzner API
   tokens) are high-value targets. If exposed through an agentic workflow, they enable
   full infrastructure compromise. Principle of least privilege and short-lived credentials
   (OIDC) are essential mitigations.

**Mitigations:** Human-in-the-loop approval for all `pulumi up` operations, `pulumi preview`
before every deployment, CrossGuard policy enforcement, OIDC-based short-lived credentials,
and state encryption. These are available in Pulumi but NOT enforced by default — they
require explicit configuration.

### 14.3 Pulumi Cloud Vendor Lock-In Risk

While Pulumi's engine, CLI, and all provider SDKs are open source (Apache 2.0), several
important capabilities are Pulumi Cloud-only:

| Capability | Open Source | Pulumi Cloud Only |
|-----------|:---:|:---:|
| `pulumi up` / `destroy` | Yes | — |
| State backend (S3/local) | Yes | — |
| **Pulumi Neo (agentic)** | — | **Yes** |
| Team collaboration (RBAC) | — | **Yes** |
| Drift detection | — | **Yes** |
| Audit logs | — | **Yes** |
| Policy-as-Code (CrossGuard) | Limited | Full |
| Deployment history | — | **Yes** |

The agentic IaC argument (Section 6) is therefore implicitly an argument for **Pulumi
Cloud dependency**, not just Pulumi OSS. If Pulumi Inc. changes pricing or licensing
(as HashiCorp did with BSL in 2023), there is no equivalent OpenTofu-style fork for
Pulumi Cloud features. The OSS engine remains Apache 2.0, but the SaaS features are
not replicable.

**Mitigation:** Use self-managed state backends (S3) for core IaC operations. Treat
Pulumi Cloud features (Neo, RBAC, audit) as productivity enhancements rather than
hard dependencies. Ensure infrastructure code remains portable to Terraform via
`pulumi convert --to terraform` if needed.

### 14.4 Open Questions

1. **Does Pulumi Python outperform Pulumi TypeScript for Claude Code?** The benchmark
   tested TypeScript. Python may perform differently given its even larger representation
   in LLM training data.

2. **What is the optimal autonomy level for agentic IaC in regulated environments?**
   Medical device software (IEC 62304) and pharmaceutical workflows (GxP) require strict
   change control. How does human-in-the-loop approval integrate with regulatory audit
   trails?

3. **Does OpenTofu's lack of AI tooling doom it to decline?** If agentic IaC becomes
   dominant, tools without AI integration may lose market share regardless of licensing
   advantages.

---

## 15. Conclusion {#15-conclusion}

For the specific constraints of the MinIVess platform (Python-centric, MONAI ecosystem,
solo-to-small-team scale), Pulumi appears well-suited as the IaC platform based on three
converging factors: (1) LLMs generate higher-quality infrastructure code in general-purpose
languages than in DSLs, (2) Pulumi's agentic tooling (Neo, MCP) is more mature than
Terraform's, and (3) the deprecation of CDKTF eliminates the main alternative for
Python-based multi-cloud IaC.

For the MinIVess MLOps platform specifically:

- **Immediate value:** Automating MLflow deployment to Hetzner/Oracle Cloud via Pulumi
  unblocks SkyPilot remote training (the #1 blocker).
- **Medium-term value:** Pulumi manages the full infrastructure stack (PostgreSQL, object
  storage, monitoring, DNS) across development, staging, and production environments.
- **Long-term value:** As the platform scales from academic research to deep tech startup,
  Pulumi provides the migration path from Hetzner to AWS without rewriting infrastructure
  code.

The agentic IaC paradigm — where AI agents generate, validate, and deploy infrastructure
with human approval — is an emerging capability with promising early results but significant
open questions around reliability, security, and accountability. Claude Code with Pulumi MCP
can perform many infrastructure tasks end-to-end, though independent benchmarks suggest
first-attempt deployment success rates remain in the 20-30% range for complex scenarios
([DPIaC-Eval, 2025](https://arxiv.org/abs/2506.05623)). The question for each organization
is whether the reliability and trust thresholds are met for their specific risk tolerance,
regulatory requirements, and team capabilities.

**Important caveats:** Terraform remains the dominant IaC tool (~33% market share,
[CNCF, 2024](https://www.cncf.io/reports/cncf-annual-survey-2024/)) with a vastly larger
ecosystem (4,800+ providers vs ~150 native Pulumi providers), more community support, and
a larger hiring pool. For teams with existing Terraform expertise, HCL's constrained
expressiveness can be an advantage for auditability and compliance. The choice between
Pulumi and Terraform should be evaluated case-by-case based on team skills, ecosystem
needs, and whether LLM-assisted generation is a priority.

---

## 16. References {#16-references}

### Academic Literature

- [Artac, M., Borovssak, T., Di Nitto, E., Guerriero, M., & Tamburri, D. A. (2017). "DevOps: Introducing Infrastructure-as-Code." *Journal of Systems and Software*, 134, 372-387.](https://doi.org/10.1016/j.jss.2016.12.013)

- [Bursuc, S., Ehrenborg, T., Lin, S., et al. (2025). "A Benchmark for VeriCoding: Formally Verified Program Synthesis." *arXiv preprint arXiv:2509.22908*.](https://arxiv.org/abs/2509.22908)

- [Chhabra, A., et al. (2025). "Agentic AI Security: Threats, Defenses, Evaluation, and Open Challenges." *arXiv preprint arXiv:2510.23883*.](https://arxiv.org/abs/2510.23883)

- [DPIaC-Eval (2025). "Deployability-Centric Infrastructure-as-Code Generation." *arXiv preprint arXiv:2506.05623*.](https://arxiv.org/abs/2506.05623)

- [DevOps-Gym (2026). "Benchmarking AI Agents in Software DevOps Cycle." *ICLR 2026*. *arXiv preprint arXiv:2601.20882*.](https://arxiv.org/abs/2601.20882v1)

- [Guerriero, M., Tamburri, D. A., & Di Nitto, E. (2019). "Adoption, Support, and Challenges of Infrastructure-as-Code: Insights from Industry." *IEEE International Conference on Software Maintenance and Evolution (ICSME)*.](https://doi.org/10.1109/ICSME.2019.00089)

- [Kreuzberger, D., Kuhl, N., & Hirschl, S. (2023). "Machine Learning Operations (MLOps): Overview, Definition, and Architecture." *IEEE Access*, 11.](https://arxiv.org/abs/2205.02302)

- [Morris, K. (2016). *Infrastructure as Code: Managing Servers in the Cloud*. O'Reilly Media.](https://www.oreilly.com/library/view/infrastructure-as-code/9781491924334/)

- [Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)." *JMLR*, 22(164), 1-20.](https://arxiv.org/abs/2003.12206)

- [Rahman, A., Parnin, C., & Williams, L. (2019). "A Systematic Mapping Study of Infrastructure as Code Research." *Information and Software Technology*, 108, 65-77.](https://www.sciencedirect.com/science/article/abs/pii/S0950584918302507)

- [Semmelrock, S., et al. (2025). "Reproducibility in Machine-Learning-Based Research: Overview, Barriers, and Drivers." *AI Magazine*, 46(2).](https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002)

- [Peng, R. D. (2011). "Reproducible Research in Computational Science." *Science*, 334(6060), 1226-1227.](https://doi.org/10.1126/science.1213847)

- [Wen, J., et al. (2024). "A Survey on Large Language Models for Code Generation." *ACM Transactions on Software Engineering and Methodology*.](https://dl.acm.org/doi/10.1145/3747588)

- [Zaharia, M., Chen, A., Davidson, A., et al. (2018). "Accelerating the Machine Learning Lifecycle with MLflow." *IEEE Data Eng. Bull.*, 41(4), 39-45.](https://people.eecs.berkeley.edu/~matei/papers/2018/ieee_mlflow.pdf)

### Security and Governance

- [Cloud Security Alliance (2025). "MAESTRO: Agentic AI Threat Modeling Framework."](https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro)

- [CNCF (2024). "CNCF Annual Survey 2024."](https://www.cncf.io/reports/cncf-annual-survey-2024/)

### Industry Sources — Pulumi

- [Pulumi (2026). "Token Efficiency vs Cognitive Efficiency: Choosing IaC for AI Agents." *Pulumi Blog*.](https://www.pulumi.com/blog/token-efficiency-vs-cognitive-efficiency-choosing-iac-for-ai-agents/)

- [Pulumi (2026). "AI-Assisted Infrastructure as Code with Pulumi's Model Context Protocol Server." *Pulumi Blog*.](https://www.pulumi.com/blog/mcp-server-ai-assistants/)

- [Pulumi (2026). "The Claude Skills I Actually Use for DevOps." *Pulumi Blog*.](https://www.pulumi.com/blog/top-8-claude-skills-devops-2026/)

- [Pulumi (2026). "AI Predictions for 2026: A DevOps Engineer's Guide." *Pulumi Blog*.](https://www.pulumi.com/blog/ai-predictions-2026-devops-guide/)

- [Pulumi (2025). "CDKTF is Deprecated: What's Next for Your Team?" *Pulumi Blog*.](https://www.pulumi.com/blog/cdktf-is-deprecated-whats-next-for-your-team/)

- [Pulumi (2025). "Meet Neo, Your Newest Platform Engineer." *Pulumi Blog*.](https://www.pulumi.com/blog/pulumi-neo/)

- [Pulumi (2026). "Pulumi Neo Now Supports AGENTS.md." *Pulumi Blog*.](https://www.pulumi.com/blog/pulumi-neo-now-supports-agentsmd/)

- [Pulumi (2025). "2025 Product Launches: Neo, Next-Gen Policies, and Platform Engineering at Scale." *Pulumi Blog*.](https://www.pulumi.com/blog/2025-product-launches/)

- [Pulumi (2026). "Pricing." *Pulumi Website*.](https://www.pulumi.com/pricing/)

### Industry Sources — HashiCorp/Terraform

- [IBM Newsroom (2025). "HashiCorp Previews the Future of Agentic Infrastructure Automation with Project Infragraph."](https://newsroom.ibm.com/2025-09-25-hashicorp-previews-the-future-of-agentic-infrastructure-automation-with-project-infragraph)

- [HashiCorp (2025). "Build Secure, AI-Driven Workflows with Terraform and Vault MCP Servers." *HashiCorp Blog*.](https://www.hashicorp.com/en/blog/build-secure-ai-driven-workflows-with-new-terraform-and-vault-mcp-servers)

- [HashiCorp (2023). "HashiCorp Adopts Business Source License." *HashiCorp Blog*.](https://www.hashicorp.com/en/blog/hashicorp-adopts-business-source-license)

### Industry Sources — Comparisons and Analysis

- [InfoQ (2025). "Pulumi Launches Neo: An Agentic AI Platform Engineer for Multi-Cloud Infrastructure."](https://www.infoq.com/news/2025/09/pulumi-neo/)

- [PR Newswire (2025). "Introducing Pulumi Neo, the Industry's First AI-Powered Platform Engineer."](https://www.prnewswire.com/news-releases/introducing-pulumi-neo-the-industrys-first-ai-powered-platform-engineer-302556718.html)

- [Spacelift (2026). "Pulumi Pricing — Editions Overview 2026." *Spacelift Blog*.](https://spacelift.io/blog/pulumi-pricing)

- [dasroot.net (2026). "Infrastructure as Code: Terraform vs OpenTofu vs Pulumi — A 2026 Comparison."](https://dasroot.net/posts/2026/01/infrastructure-as-code-terraform-opentofu-pulumi-comparison-2026/)

### Pulumi Provider Documentation

- [Pulumi Registry: Hetzner Cloud (hcloud) Provider v1.32.1](https://www.pulumi.com/registry/packages/hcloud/)
- [Pulumi Registry: Oracle Cloud Infrastructure (oci) Provider](https://www.pulumi.com/registry/packages/oci/)
- [Pulumi Registry: Cloudflare Provider](https://www.pulumi.com/registry/packages/cloudflare/)
- [Pulumi Registry: AWS Provider](https://www.pulumi.com/registry/packages/aws/)

### Model Context Protocol

- [Anthropic (2024). "Model Context Protocol." *MCP Specification*.](https://modelcontextprotocol.io/)
