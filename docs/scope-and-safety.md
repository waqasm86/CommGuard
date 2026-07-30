# Scope, safety, and claims

CommGuard permits standard training, inference, benign controls, and bounded
scientific robustness variants. It does not implement telemetry tampering,
provider-monitor bypass, privilege escalation, covert exfiltration, host
compromise, or attacks on real monitoring systems.

All reporting uses four claim classes:

- **Observed:** directly present in saved artifacts from the stated hardware.
- **Inferred:** supported by those observations and the stated analysis.
- **Hypothesized:** proposed for future experiments, not measured here.
- **Out of scope:** production security, privacy guarantees, frontier clusters,
  NVSwitch, RoCE/InfiniBand, multi-node behavior, determined operators, and
  treaty-grade verification.

Avoiding model content does not by itself establish privacy. PCIe TX/RX is not a
unique or complete measure of inter-GPU traffic. A high test score is not
evidence of adversarial robustness. Two T4 GPUs do not represent two 8-GPU
nodes.

SPAR application free-response text is outside this repository's purpose.
CommGuard documentation and reports describe only technical repository
behavior and measurements; they must not be repurposed as AI-written
application answers.
