# ADR-0004: Local-First Observability Stack

## Status

Accepted

## Date

2026-02-23

## Context

The v0.1-alpha codebase relied on a single EC2-hosted MLflow instance for experiment tracking, with optional Weights & Biases as a cloud alternative. This had several issues:

1. Development required an active AWS connection and valid credentials.
2. No LLM tracing, data lineage, or infrastructure metrics existed.
3. The cost of running cloud-hosted observability tools for a research project was unjustified.
4. Air-gapped or restricted-network environments (common in biomedical settings and defense applications) could not run the pipeline.

The v2 platform adds an LLM agent layer (LangGraph), data lineage requirements (IEC 62304), and infrastructure monitoring (Prometheus/Grafana), multiplying the observability surface area.

## Decision

We adopt a local-first observability architecture where every tool runs via Docker Compose with zero external API tokens required for development:

| Tool | Role | Deployment |
|------|------|------------|
| MLflow 3.10 | Experiment tracking, model registry | Docker Compose (PostgreSQL + MinIO backend) |
| Langfuse v3 | LLM tracing, cost tracking, prompt management | Docker Compose (self-hosted, PostgreSQL backend) |
| OpenLineage / Marquez | Data lineage tracking (IEC 62304) | Docker Compose |
| Prometheus | Infrastructure and application metrics | Docker Compose |
| Grafana | Dashboards for metrics, drift reports, model health | Docker Compose |
| OpenTelemetry Collector | Trace/metric collection and forwarding | Docker Compose |
| DuckDB | In-process SQL analytics over MLflow runs | Library (no server) |
| Evidently AI | Data/model drift detection | Library (reports exported to Grafana) |
| whylogs | Lightweight data profiling | Library (in-process) |

Docker Compose profiles control resource usage:

- `dev` (4 services, ~4 GB RAM): PostgreSQL, MinIO, MLflow, BentoML
- `monitoring` (7 services, ~8 GB RAM): adds Prometheus, Grafana, OpenTelemetry Collector
- `full` (12 services, ~16 GB RAM): adds Langfuse, Marquez, Label Studio, MONAI Label, Ollama

Cloud-hosted alternatives (managed MLflow, Langfuse Cloud, Grafana Cloud) are supported via environment variable overrides in Dynaconf, but are never required.

## Consequences

**Positive:**

- `docker compose --profile dev up` gives a fully functional MLOps stack with zero API tokens, zero cloud accounts, and zero internet dependency.
- The same stack runs in air-gapped environments by pre-pulling Docker images.
- Developers can inspect LLM traces (Langfuse), data lineage (Marquez), experiment history (MLflow), and infrastructure metrics (Grafana) locally.
- DuckDB enables ad-hoc SQL analytics over MLflow experiment data without a separate analytics database.

**Negative:**

- Running the `full` profile requires approximately 16 GB RAM, which may not be available on all development machines.
- Self-hosted Langfuse and Marquez require PostgreSQL, adding database administration overhead.
- Local Docker Compose does not provide high availability, multi-user access control, or automated backups that managed cloud services offer.

**Neutral:**

- The `monitoring` profile is the recommended default for daily development; `full` is used for integration testing and demo sessions.
- Volume persistence ensures data survives container restarts, but developers are responsible for `docker volume prune` to reclaim disk space.
