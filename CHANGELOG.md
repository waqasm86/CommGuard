# Changelog

## Unreleased

- Add repetition-aware calibration with idle-baseline capture gating.
- Report per-payload median, mean, median absolute deviation, coefficient of
  variation, capture count, capture rate, and reliability.
- Add the `partially_supported` status and explicit reliable/unreliable payload
  groups while retaining the existing correlation and dynamic-range keys.
- Add staged Kaggle calibration, benign-corpus, and grouped detector-evaluation
  notebooks with an experiment roadmap.
- Add calibration repeatability and falsification edge-case tests.

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
