# Limitations

Local CPU tests validate schemas, storage behavior, feature extraction, grouped
splits, CLI construction, capability errors, and participation evidence parsing.
They do not validate CUDA execution, NVML field availability, NCCL collectives,
dual-T4 rank binding, sampling overhead, calibration, or detector performance.

Historical Kaggle artifacts do validate one dual-T4 pilot under an older source
contract, but they do not satisfy the current primary evaluation gate. The
effective feature task was three DDP runs versus five idle runs, two of which
were calibration controls; all rows were five-second diagnostics. The primary
PCIe-only test missed the sole training test run. See
[`current-results.md`](current-results.md) and the
[`evidence index`](evidence-index.md).

The local two-agent central-monitoring simulation validates protocol behavior,
not a physical network deployment. It does not measure TLS termination,
cross-host clock synchronization, packet loss, WAN behavior, storage durability,
or two-node GPU interconnect telemetry. Physical multi-node validation remains
pending.

CommGuard is a bounded research prototype. It does not establish production
readiness, privacy guarantees, adversarial robustness, or conclusions beyond
the recorded hardware, sessions, and workload families.

The current standard matrix has one base configuration per family. Three
repetitions support a train/validation/test family-presence gate, but do not by
themselves establish configuration diversity or stable generalization.
