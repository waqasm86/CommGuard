# Artifact schemas

All JSON/JSONL records carry `artifact_kind` and `schema_version`. Historical
version `1.0` remains readable. New provenance and corpus manifests use version
`2.0`; its non-destructive legacy view is documented in
[`schema-migrations.md`](schema-migrations.md). Raw evidence is never rewritten.

## Environment report

Records Python/OS/kernel, GPUs, driver, CUDA/PyTorch/NCCL, packages, topology
command, peer capability, RAM/disk, optional network check, source state,
experiment session, node, environment fingerprint, and readiness gates. The
environment fingerprint describes stable capabilities and is not a session ID.

## Run manifest

Records session/corpus/collection/run/node identity, label/family/designation,
seed, model/workload dimensions,
precision, batch/sequence/accumulation, world size, environment fingerprint,
source commit, warmup, timestamps, NCCL environment, rank exit codes,
participation validity, failure category/reason, source dirty state, optional
notebook/input archive provenance, and exit status.

## Corpus manifest

Declares every planned family/label/configuration slot and the exact accepted
run allow-list. A run can fill at most one slot, and designation-aware selection
prevents calibration controls from leaking into benign evaluation.

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
