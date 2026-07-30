# Artifact schemas

All JSON/JSONL records carry `artifact_kind` and `schema_version`. Version `1.0`
is additive-only. A future breaking change must use a new major schema version
and an explicit, tested migration; raw evidence itself is never rewritten.

## Environment report

Records Python/OS/kernel, GPUs, driver, CUDA/PyTorch/NCCL, packages, topology
command, peer capability, RAM/disk, optional network check, source commit,
session fingerprint, and readiness gates.

## Run manifest

Records identity, label/family/designation, seed, model/workload dimensions,
precision, batch/sequence/accumulation, world size, environment fingerprint,
source commit, warmup, timestamps, NCCL environment, rank exit codes,
participation validity, failure category/reason, and exit status.

## Telemetry sample

Each row has run/GPU/sequence identity, UTC and monotonic timestamps, and exactly
nine `FieldReading` objects. A supported field contains a finite value and unit.
An unsupported field contains `value: null`, `supported: false`, and an error.
Zeros remain legitimate measurements and are not used as missing sentinels.

## Other records

Workload events retain rank-local progress and checksums. Feature rows retain
split metadata separately from numeric features. Split assignments map a whole
run to one split. Evaluation and calibration results record their thresholds,
metrics, grouping unit, limitations, and source artifacts.
