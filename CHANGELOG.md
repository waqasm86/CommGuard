# Changelog

## Unreleased (0.2.0 candidate)

- Implement the missing two-rank idle worker path used by calibration-v3,
  including measured-interval lifecycle/heartbeat evidence without a measured
  collective, and bound idle PCIe summaries to that interval.
- Create and validate preflight output directories, give every calibration
  payload an accurate workload identity, refuse repeated sweeps in one output
  root, validate the exact 15-run matrix, and make archive export create-only
  and atomic by default.
- Harden the canonical calibration notebook's source/import/hardware/workspace,
  progress, bootstrap-provenance, standard-result validation, and diagnostic
  evidence export flow.
- Require idle-aware three-repetition calibration at 1/4/16/64 MiB for new
  evidence, and bind benign/evaluation artifacts to one hash-verified calibration.
- Make exact authenticated central-ingestion retries idempotent after lost
  acknowledgments while rejecting changed content under a reused message ID.
- Clip simple-rule detector logits before the sigmoid to keep extreme-value
  probabilities finite without overflow warnings.
- Add explicit session/collection/corpus/node provenance, schema-2 legacy
  migration, declared corpus membership, per-plan coverage, timestamp-aligned
  features, duration evidence, and strict 30-second primary gates.
- Add leakage-resistant whole-run/session split planning, train-only
  preprocessing, validation-only selection, communication-only primary results,
  hard-negative diagnostics, and small-sample uncertainty suppression.
- Add a signed, privacy-allow-listed two-agent central-monitoring reference path
  with replay/order/staleness checks and forced incomplete-window abstention.
- Add the eight-family duration-aware benign matrix and eight bounded,
  approval-gated adversarial strategies with a sealed final holdout.
- Add four deterministic unexecuted Kaggle notebooks with clean pushed-commit
  gates, safe hash-verified archive chaining, exact extraction selection, and
  coverage/approval stops.
- Add an immutable evidence index, historical coverage-failure analysis,
  research report template, applicant evidence notes, and complete attribution.
- Add repetition-aware calibration with idle-baseline capture gating.
- Report per-payload median, mean, median absolute deviation, coefficient of
  variation, capture count, capture rate, and reliability.
- Add the `partially_supported` status and explicit reliable/unreliable payload
  groups while retaining the existing correlation and dynamic-range keys.
- Add staged Kaggle calibration, benign-corpus, and grouped detector-evaluation
  notebooks with an experiment roadmap.
- Add calibration repeatability and falsification edge-case tests.

The first modern calibration-v3 Kaggle T4 x2 attempt reached the reviewed
source checkout, CUDA/NCCL, and collective execution, but all idle launches
failed because worker mode `idle` was not implemented. Its zero-idle result was
correctly `not_supported`; it is retained as debugging evidence and is not a
calibration gate or evidence that PCIe telemetry failed. No benign, detector,
adversarial, or physical multi-node follow-on was run.

## 0.1.0 - 2026-07-30

- Initial CommGuard research SDK.
- Strict Kaggle dual-T4 preflight and torchrun/NCCL participation evidence.
- Nine-field NVML collection with explicit unsupported values.
- Fixed-frequency NCCL calibration and falsification gate.
- Bounded DDP, inference, control, and robustness workloads.
- Immutable artifact contracts, deterministic features, grouped evaluation,
  ablations, abstention, uncertainty, reporting, and Kaggle notebook.
- Public import/CLI acceptance tests and explicit validation of per-run split
  metadata.
- Nullable NVML capability values retain support and error metadata.
