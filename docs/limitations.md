# Limitations

> **CommGuard’s Kaggle workflow is a single-node, dual-NVIDIA-T4 research
> prototype. It validates experimental methodology and software behavior on
> two local GPU ranks. It does not establish generalization to two physical
> 8-GPU nodes, NVLink/NVSwitch fabrics, RoCE or InfiniBand networks, large
> frontier-model workloads, or production treaty-verification deployments.**

NVML PCIe polling can alias short bursts and is not a direct NCCL byte counter.
The optional 0.1-second diagnostic may add overhead and is not the default.

Local CPU tests validate schemas, storage behavior, feature extraction, grouped
splits, CLI construction, capability errors, and participation evidence parsing.
They do not validate CUDA execution, NVML field availability, NCCL collectives,
dual-T4 rank binding, sampling overhead, an empirical calibration outcome, or
detector performance.

The first modern calibration-v3 Kaggle attempt did verify two Tesla T4 GPUs,
the reviewed source import, CUDA/NCCL availability, and collective execution.
It did not produce a usable calibration: the unimplemented worker mode `idle`
made all idle runs fail, so zero idle repetitions were usable and the result was
correctly `not_supported`. The failed archive is debugging evidence only and
cannot support a conclusion about PCIe telemetry or gate a benign collection.

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
readiness, reliable training-versus-inference detection, privacy guarantees,
adversarial robustness, or conclusions beyond the recorded hardware, sessions,
and workload families. No successful modern calibration, new benign corpus,
detector evaluation, adversarial run, or physical multi-node run exists yet.

The current standard matrix has one base configuration per family. Three
repetitions support a train/validation/test family-presence gate, but do not by
themselves establish configuration diversity or stable generalization.
