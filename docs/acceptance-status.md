# Acceptance status

Status as of the local SDK build:

## Locally verified

- Independent/non-affiliation, scope, safety, and claims documentation.
- Installable dependency-free core; wheel builds with `--no-deps`.
- Verbatim 935-line Kaggle pip snapshot with matching SHA-256.
- Versioned schema validation and create-only artifact storage.
- Hash-verified, create-only archive restoration that rejects traversal,
  duplicates, links, special members, and configured size/member overages.
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
- CPU-validated bounded metadata and synchronization semantics for eight
  approval-gated adversarial strategies; real-process segmentation plans;
  per-rank sync/proxy/efficiency/correctness manifest requirements; combined
  designation-aware extraction; and sealed family/session/config holdout logic.
- Grouped splitting, independent-session hierarchy, baselines, signal ablations,
  per-family/run metrics, bootstrap uncertainty, abstention, held-out adversarial
  families, and duration-cost reporting.
- Four deterministic canonical notebooks are orchestration-only, install a
  clean remote-visible detached commit with `--no-deps`, chain exact archive
  hashes, and remain unexecuted in Git.

## Historical Kaggle evidence, below current acceptance

- A saved historical preflight observed two Tesla T4 GPUs with CUDA and NCCL.
- A historical calibration decision was `supported` under its source contract.
- All 18 short benign pilot launches completed, but the merged feature evidence
  contains only DDP and idle five-second rows and includes calibration idle.
- The amended two-run PCIe-only test missed its training run. This is a negative
  DDP-versus-idle result, not broad classifier validation.

Exact hashes and claim boundaries are in [`evidence-index.md`](evidence-index.md).

## Requires Kaggle dual-T4 evidence

- Completion-series two-T4 inventory and peer/topology artifact under the
  current provenance/schema contract.
- Current two-rank NCCL initialization and distinct UUID bindings.
- Current NVML support status for every field.
- Current controlled payload response and calibration gate outcome.
- Current DDP backward/synchronization/optimizer evidence.
- Sampling jitter and end-to-end overhead on T4.
- Duration-valid 24-run standard corpus with all eight required families.
- Second-session reproduction and cross-session holdout.
- Detector, ablation, adversarial, false-positive/negative, and abstention
  results generated from measured artifacts.

No item in the “Requires Kaggle dual-T4 evidence” section is claimed complete
by local tests or historical evidence.
