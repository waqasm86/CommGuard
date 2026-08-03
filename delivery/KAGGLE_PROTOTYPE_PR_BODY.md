## Summary

Completes the application-ready CommGuard research workflow for one Kaggle host
with exactly two NVIDIA Tesla T4 GPUs, while preserving the existing negative,
failed, and diagnostic evidence without reinterpretation.

## Implemented

- Organizes exactly four unexecuted canonical notebooks and one separated v4
  sampling diagnostic under notebook policy v3.
- Adds one authoritative single-node dual-T4 scope declaration and structured
  scope fields to environment, provenance, run status, and stage manifests.
- Calibrates idle plus 1/4/16/64/128 MiB AllReduce with five repetitions,
  selectable 1.0/0.5/0.2-second polling, robust idle MAD thresholds,
  duration/cadence/participation gates, and five distinct result states.
- Adds duration-valid resumable corpus identities, exact configuration hashes,
  deterministic order, separate calibration/corpus idle evidence, and the
  stronger eight-family 24-run full matrix.
- Adds complete-run communication, auxiliary, and combined feature sets;
  grouped binary evaluation; explicit control handling and a secondary
  three-class diagnostic; GroupKFold/LeaveOneGroupOut plans; interpretable
  thresholds, logistic regression, a shallow tree, and one secondary forest.
- Adds bounded periodic synchronization training at `k = 1, 2, 4, 8, 16`,
  finite-loss and synchronization evidence, utility/communication tradeoffs,
  and benign-only frozen detector scoring with the final adversarial holdout
  sealed.
- Extends the local central collector to two same-node rank agents with schema
  validation, deduplication, missing/stale-agent detection, bounded windows, and
  incomplete-window abstention.
- Adds deterministic, non-destructive verified archives, sibling SHA-256 files,
  required stage package materializers, artifact verification/review-bundle CLI
  commands, and safe historical-schema compatibility.
- Adds the Kaggle runbook, prototype report skeleton, evidence/status updates,
  completion report, and user handoff.

## Validation performed locally

- Full CPU/unit/optional-analysis suite passes.
- The host's real NVML field-read integration test passes on its available
  non-T4 NVIDIA GPU.
- Ruff lint and format checks pass.
- Wheel and sdist build successfully.
- Canonical notebook generation/check, delivery policy, secret/path scan, and
  immutable historical-evidence hash checks pass.

## Not performed or claimed

- No new Kaggle notebook was executed in this change.
- No new two-T4/NCCL result, calibration acceptance, 24-run corpus, detector
  metric, adversarial outcome, or cross-session validation is claimed.
- No physical two-node, 16-GPU, NVLink/NVSwitch, RoCE/InfiniBand, frontier-model,
  production, or treaty-verification deployment was tested.

## First Kaggle action

Run `notebooks/commguard_calibration_v3.ipynb` in smoke mode, then run full mode
in a fresh workspace. The first expected archive is
`commguard-calibration-prototype-<timestamp>.tar.gz` with a sibling
`.tar.gz.sha256` file.

## Reviewer notes

Historical calibration-v3 failure remains a failed idle-dispatch diagnostic,
not a telemetry-capability conclusion. The v4 notebook remains diagnostic and
is not a canonical gate. Low or negative future results remain valid outputs.
