# Scope, safety, and claims

CommGuard permits standard training, inference, benign controls, and bounded
scientific robustness variants. It does not implement telemetry tampering,
provider-monitor bypass, privilege escalation, covert exfiltration, host
compromise, or attacks on real monitoring systems.

Adversarial strategies are absent from every normal profile and stop before
preflight or artifact creation unless a human explicitly approves the reviewed
bounded plan. Final family/session/configuration holdouts start sealed. See
[`adversarial-research.md`](adversarial-research.md) for strategy limits and
correctness/claim requirements.

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

Central telemetry accepts only sample identity/timing, GPU identity, and the
nine declared content-agnostic field readings. Prompt, token, dataset, example,
content, model-weight, credential, secret, and API-key fields are rejected.
This allow-list reduces collection scope but is not a proof of privacy.

SPAR application free-response text is outside this repository's purpose.
CommGuard documentation and reports describe only technical repository
behavior and measurements; they must not be repurposed as AI-written
application answers.
