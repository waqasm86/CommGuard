# Architecture

CommGuard separates:

1. `preflight`: hardware/software inventory, topology, and readiness gates.
2. `schemas` and `artifacts`: versioned evidence contracts and create-only I/O.
3. `telemetry`: one host collector, per-GPU NVML field status, timing/jitter.
4. `distributed`: torchrun isolation, rank binding, heartbeats, and cleanup.
5. `calibration`: controlled payloads and PCIe-response falsification gate.
6. `workloads`: DDP training, inference, controls, and bounded variants.
7. `orchestrator`: run lifecycle, matrix estimates, timeouts, partial evidence.
8. `features`: deterministic windows and cross-GPU signal relationships.
9. `evaluation`: grouped splits, baselines, ablations, abstention, leakage audit.
10. `reporting`: reports generated only from validated saved artifacts.

The notebook calls public APIs and CLI commands; it contains no SDK
implementation. Main integrations use a fresh `torchrun` subprocess so notebook
CUDA state cannot contaminate workers. Every rank binds its local device before
tensor allocation, requires NCCL, emits identity/evidence records, and destroys
the process group in `finally`.

## Calibration gate

The main corpus is blocked until controlled collective payload/frequency changes
produce a sufficiently responsive PCIe signal. Nominal tensor bytes, collective
semantics, PyTorch timing, and NVML readings remain separate channels. A
negative or unsupported calibration is preserved as a result and narrows the
research question; it is not bypassed through unrelated telemetry unless the
user explicitly selects negative-calibration analysis mode.
