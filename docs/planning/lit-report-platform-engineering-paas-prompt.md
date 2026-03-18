# R9 Platform Engineering PaaS — Original Prompt (Verbatim)

```
And then we actually has still one literature research report to be done on this which is more
like P2 priority report on the kg, and as high-level vision for this repo maybe happening in
1-2 years if someone is interested in working this. This is PaaS and platform engineering
approach to come up with some highly flexible platform to be built on top of MONAI as
easy-to-use platform or infrastructure-as-platform type that can be both very easy to use by
the researchers who want to do science without having to figure out how the underlying
behind-the-scenes infrastructure provisioning is happening (wrapping Skypilot, Pulumi, Docker,
and everything). As said this is not a high priority theme, but we should a bit plan things so
that we don't go against this vision in this repo, but we don't need to be implementing too
much in practice then. You seemed to have this at Anthropic going on? ;) Any insights from
this? :P https://x.com/AprilNEA/status/2034209430158619084 [...]
```

## The Antspace/Baku Discovery (AprilNEA, March 2026)

Reverse engineering of Claude Code's `environment-manager` binary revealed:

- **Binary**: `/usr/local/bin/environment-runner` — 27MB Go binary, not stripped, with debug info
- **Source**: `github.com/anthropics/anthropic/api-go/environment-manager/`
- **Two deploy targets**: Vercel (public) and **Antspace** (completely undocumented)
- **Antspace protocol**: POST create deployment → upload dist.tar.gz → stream NDJSON status (packaging → uploading → building → deploying → deployed)
- **Powers "Baku"** — Anthropic's internal web app builder on claude.ai
- **The pattern**: Natural language → Claude builds app → Deploys to Antspace
- **Infrastructure**: Firecracker microVMs (KVM hypervisor), Linux kernel 6.18.5

### Internal Package Structure (from objdump)
```
cmd/                  — CLI commands (cobra)
internal/
  api/               — API client (session ingress, CCR backend)
  auth/              — GitHub app token provider
  claude/            — Claude Code install, upgrade, config, execution
  config/            — Session mode (new/resume/resume-cached/setup-only)
  envtype/
    anthropic/       — Anthropic-hosted environment setup
    byoc/            — Bring Your Own Container environment
  gitproxy/          — Git credential proxy
  mcp/
    servers/codesign/  — Code signing MCP server
    servers/supabase/  — Supabase integration MCP server
  orchestrator/      — Poll loop, hooks, whoami
  podmonitor/        — Lease manager (k8s pod monitoring)
  sandbox/           — Sandbox-runtime config, install, wrapper
  tunnel/
    actions/deploy/  — Deploy to Vercel + "Antspace"
    actions/snapshot/ — File snapshots
```

### Key Dependencies
- `github.com/anthropics/anthropic/api-go` — Internal Anthropic Go SDK
- `github.com/mark3labs/mcp-go` v0.37.0 — Model Context Protocol
- `go.opentelemetry.io/otel` v1.39.0 — Tracing (OTLP/HTTP export)
- `google.golang.org/grpc` v1.79.0 — gRPC client
- `github.com/DataDog/datadog-go` v5 — Metrics reporting

### Strategic Implication for NEUROVEX

This is the emerging AI-native PaaS pattern: **the LLM IS the interface to infrastructure**.
Instead of researchers learning SkyPilot YAML, Docker Compose, and Prefect deployment commands,
they describe their experiment in natural language and the platform provisions everything.

For NEUROVEX, this means:
1. The current stack (SkyPilot + Pulumi + Docker + Prefect + MLflow) becomes the
   "backend runtime" — invisible to the researcher
2. A Pydantic AI agent (or CopilotKit UI) becomes the "frontend" — the researcher
   says "train DynUNet with cbdice_cldice loss on 3 folds" and everything happens
3. The knowledge graph provides the constraint system — the agent reads KG posteriors
   to suggest sensible defaults and prevent invalid configurations
4. OpenLineage provides the audit trail — every action the agent takes is logged

This is a P2 vision (1-2 years) but the architectural decisions we make TODAY must
not prevent this evolution. Specifically:
- Config-driven everything (Hydra-zen) → agent can compose configs
- Docker-per-flow isolation → agent can provision containers
- SkyPilot YAML → agent can generate and submit job specs
- MLflow tracking → agent can read results and suggest next steps
