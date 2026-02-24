# Federated Learning Support (NVIDIA FLARE) — Implementation Plan (Issue #48)

## Current State
- ModelAdapter ABC supports train/predict separation (FL-compatible)
- No federated learning infrastructure
- PRD decision `federated_learning` covers NVIDIA FLARE option

## Architecture

### New Module: `src/minivess/pipeline/federated.py`
- **FLStrategy** — StrEnum: FED_AVG, FED_PROX, SCAFFOLD
- **FLClientConfig** — Dataclass: client_id, data_path, local_epochs, batch_size
- **FLServerConfig** — Dataclass: num_rounds, min_clients, aggregation strategy
- **DPConfig** — Dataclass: epsilon, delta, max_grad_norm (differential privacy)
- **FLRoundResult** — Dataclass: round number, client metrics, aggregated metrics
- **FederatedAveraging** — Federated averaging implementation:
  - aggregate_weights() — weighted average of model state dicts
  - compute_client_weight() — proportional to dataset size
- **FLSimulator** — Simulates multi-site federated training:
  - add_client() — register a client configuration
  - simulate_round() — run one FL round
  - to_markdown() — training report

## Test Plan
- `tests/v2/unit/test_federated.py` (~12 tests)
  - TestFLStrategy: enum values
  - TestFLClientConfig: construction, defaults
  - TestFLServerConfig: construction, defaults
  - TestDPConfig: construction, privacy budget
  - TestFederatedAveraging: aggregate weights, client weights
  - TestFLSimulator: add client, simulate round, markdown
