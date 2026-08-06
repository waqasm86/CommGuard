# Acceptance status

## Kaggle prototype completion boundary

The source workflow is locally CPU-tested and Kaggle-executable. No new full
Kaggle package is scientifically accepted. Smoke mode always records
`development_smoke_only=true` and `scientific_acceptance_eligible=false`.
Calibration result states are `supported`, `partially_supported`,
`inconclusive`, `not_supported`, and `failed`; process exit zero alone cannot
select any scientific state.

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
- Idle-aware calibration policy with five idle and five per-payload
  repetitions, deterministic capture/monotonicity/dynamic-range decisions, and
  explicit legacy compatibility that cannot satisfy the modern gate.
- Hash-bound calibration provenance through final corpus, extraction,
  evaluation, and reporting artifacts; ambiguous filename-order selection is
  rejected.
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
  hashes and artifact paths, distinguish prior/current calibration, and remain
  unexecuted in Git.

## Historical Kaggle evidence, below current acceptance

- A saved historical preflight observed two Tesla T4 GPUs with CUDA and NCCL.
- A historical calibration decision was `supported` under its source contract.
- All 18 short benign pilot launches completed, but the merged feature evidence
  contains only DDP and idle five-second rows and includes calibration idle.
- The amended two-run PCIe-only test missed its training run. This is a negative
  DDP-versus-idle result, not broad classifier validation.

Exact hashes and claim boundaries are in [`evidence-index.md`](evidence-index.md).

## Requires Kaggle dual-T4 evidence

The first modern calibration-v3 attempt does not satisfy these items. Although
it verified two T4 devices, reviewed-source import, CUDA/NCCL, and collective
execution, all idle runs failed on the then-missing `idle` worker dispatch. Its
zero-idle result was correctly `not_supported` and is retained only for
debugging; a clean patched calibration rerun is required.

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

## Current downstream status

Calibration-v3 is not accepted for downstream use. Its clean execution and
partial support do not authorize benign-corpus collection.

The active gate is calibration-v4. Downstream execution is permitted only
when the saved full-run evidence records:

- `scientific_acceptance_eligible: true`
- `accepted: true`
- `result_state: supported`
- `modern_capture_gate_passed: true`
- `source_dirty: false`

Any smoke, unsupported, partially supported, inconclusive, dirty-source, or
failed-capture result remains blocked.
