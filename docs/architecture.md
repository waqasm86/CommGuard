# Architecture

CommGuard separates:

1. `preflight`: hardware/software inventory, topology, and readiness gates.
2. `schemas` and `artifacts`: versioned evidence contracts and create-only I/O.
3. `telemetry`: one host collector, per-GPU NVML field status, timing/jitter.
4. `distributed`: torchrun isolation, rank binding, heartbeats, and cleanup.
5. `calibration`: controlled payloads and PCIe-response falsification gate.
6. `workloads`: DDP training, inference, controls, and bounded variants.
7. `orchestrator`: pre-execution corpus plans, run lifecycle, matrix estimates,
   timeouts, finalized allow-lists, and coverage-first summaries.
8. `features`: deterministic windows and cross-GPU signal relationships.
9. `evaluation`: grouped splits, baselines, ablations, abstention, leakage audit.
10. `central`: signed node batches, offline/optional HTTP transport, ingestion,
    node health, aggregation, and a detector decision interface.
11. `reporting`: reports generated only from validated saved artifacts.

The notebooks call public APIs and CLI commands; they contain no SDK
implementation. Canonical notebooks form an immutable archive/hash chain and
bind exact extraction summaries when benign and adversarial evidence share an
artifact root. Main integrations use a fresh `torchrun` subprocess so notebook
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

## Central monitoring boundary

The central path is a locally testable reference architecture, not evidence of
a physical multi-node deployment. Node agents use a pluggable content-agnostic
backend and HMAC-sign versioned batches. The default test transport writes
create-only request and acknowledgment JSON. The central service validates
identity, protocol, payload size, privacy keys, clock skew, sequence order,
replay state, and batch chaining before retaining samples. It aligns accepted
samples by UTC windows, reports missing/stale nodes, and forces detector
abstention on incomplete windows. See
[`central-monitoring-design.md`](central-monitoring-design.md).

The local central simulation leaves physical assumptions unresolved: hosts need
verified workload identity, independently managed secrets, TLS termination,
durable replay state, bounded queues, synchronized UTC clocks, retention and
access controls, and GPU/NIC attribution appropriate to the real topology.
None of these assumptions is converted into a measured deployment claim.
