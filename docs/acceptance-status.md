# Acceptance status

Status as of the local SDK build:

## Locally verified

- Independent/non-affiliation, scope, safety, and claims documentation.
- Installable dependency-free core; wheel builds with `--no-deps`.
- Verbatim 935-line Kaggle pip snapshot with matching SHA-256.
- Versioned schema validation and create-only artifact storage.
- CPU-safe offline tests for schemas, artifacts, fake telemetry, calibration,
  features, grouping/leakage, rank evidence, preflight parsing, and reporting.
- Strict rejection of CPU/one-GPU/non-T4 execution for dual-T4 results.
- Torchrun command uses two workers, NCCL-only worker checks, unique rendezvous
  port, timeout/process-group cleanup, rank event evidence, and rank stream logs.
- Bounded DDP, inference, calibration, compute, host/device, model/checkpoint,
  idle, and optional peer-copy workload implementations.
- CPU-validated eight-family standard plan, explicit stable configuration IDs,
  opt-in bounded variants, pre-execution corpus manifests, finalized completed-
  run allow-lists, and per-family coverage-first summaries.
- Grouped splitting, two-session holdout, baselines, signal ablations,
  per-family/run metrics, bootstrap uncertainty, abstention, held-out adversarial
  families, and duration-cost reporting.
- Notebook is orchestration-only and installs with `--no-deps`.

## Requires Kaggle dual-T4 evidence

- Actual two-T4 inventory and peer/topology artifact.
- Successful two-rank NCCL initialization and distinct UUID bindings.
- Real NVML support status for every field.
- Controlled payload response and calibration gate outcome.
- Real DDP backward/synchronization/optimizer evidence.
- Sampling jitter and end-to-end overhead on T4.
- Smoke and standard artifact corpus.
- Second-session reproduction and cross-session holdout.
- Detector, ablation, adversarial, false-positive/negative, and abstention
  results generated from measured artifacts.

No item in the second section is claimed complete by local tests.
