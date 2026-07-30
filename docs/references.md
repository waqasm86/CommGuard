# Local reference analysis

The implementation is informed by the local source set supplied with the
project:

- Rahman and Tajdari motivate nine 1 Hz NVML signals, temporal features,
  run-grouped evaluation, and iterative adversarial testing. CommGuard does not
  reproduce their broader hardware/workload corpus or their published results.
- Seferis and Fist motivate communication/bandwidth as one signal in detecting
  structured compute, while relying on cloud-provider capabilities not present
  in Kaggle. CommGuard tests only accessible local PCIe readings.
- Xi et al.'s PrismLLM shows that downscaled execution may not preserve
  scale-dependent behavior. Emulation is future work, not evidence that two T4s
  represent large clusters.
- Cankaya's system overview motivates separating evidence capture from
  evaluation and preserving provenance. Its network-tap, secure-hardware,
  physical-security, replay, and treaty architecture are outside this SDK.
- Merzouk et al.'s output-length probing paper and the supplied LessWrong
  welcome page do not define CommGuard's detector or implementation.

The two copies of `How Much is Left?` are byte-identical. The three supplied
Kaggle pip snapshots are byte-identical. The version-2 instruction ZIP matches
the expanded on-disk bundle; the older ZIP is a superseded subset.

Optional NVIDIA references are `nccl-tests`, `nvbandwidth`, `cuda-samples`,
NCCL, and CUDA Python. They are never compiled or downloaded automatically, and
their bandwidth/timing outputs are not relabeled as NVML measurements.
