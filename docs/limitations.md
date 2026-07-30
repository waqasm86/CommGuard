# Limitations

Local CPU tests validate schemas, storage behavior, feature extraction, grouped
splits, CLI construction, capability errors, and participation evidence parsing.
They do not validate CUDA execution, NVML field availability, NCCL collectives,
dual-T4 rank binding, sampling overhead, calibration, or detector performance.

CommGuard is a bounded research prototype. It does not establish production
readiness, privacy guarantees, adversarial robustness, or conclusions beyond
the recorded hardware, sessions, and workload families.
