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
notebook/input archive provenance, and exit status. Completed, participation-
valid version 2 runs also record the common monotonic measurement start, end,
and duration. Validation requires those three values to be ordered and exactly
consistent; they describe the intersection of the two rank-local measured
intervals.

## Corpus manifest

Declares every planned family/label/configuration slot and the exact accepted
run allow-list. A run can fill at most one slot, and designation-aware selection
prevents calibration controls from leaking into benign evaluation.

## Telemetry sample

Each row has run/GPU/sequence identity, UTC and monotonic timestamps, and exactly
nine `FieldReading` objects. A supported field contains a finite value and unit.
An unsupported field contains `value: null`, `supported: false`, and an error.
Zeros remain legitimate measurements and are not used as missing sentinels.

## Feature extraction and coverage

Version 2 extraction is selected only through a declared corpus manifest. It
writes three linked, create-only artifacts under `features/`: feature-row
JSONL, coverage-record JSONL, and an extraction summary JSON. The summary names
the exact two JSONL inputs, records the requested windows, and labels 30 seconds
as the primary policy while 5- and 15-second windows remain diagnostic.

There is exactly one coverage record per planned corpus slot. It records the
plan/run/family/label/session/collection/corpus/node context, inclusion status,
structured reason code and detail, rows per GPU, common monotonic interval,
warmup and usable duration, per-GPU sampling-gap statistics, timestamp-aligned
pair count/tolerance, and emitted counts for every requested window. Missing,
short, invalid, or unaligned evidence is therefore represented explicitly
rather than disappearing from the feature table.

Version 1 feature rows remain loadable and are marked as having ambiguous
legacy grouping when migrated in memory. Direct low-level feature extraction
without corpus provenance returns version 1-compatible diagnostic rows; new
persisted research extraction uses version 2 rows with true grouping IDs.

Version 2 feature rows also carry a stable `workload_config_id`. An explicit
planned ID is used when supplied; otherwise the ID is deterministically derived
from the family and canonical serialized configuration, so repetitions share a
configuration identity without sharing a run identity.

## Split and evaluation artifacts

A split plan records the requested mode, actual strategy, deterministic seed,
diagnostic flag, exact run and session groups, and per-split class/family/config
run counts. Primary plans require non-empty train/validation/test partitions,
both target classes, and every required family in each partition. Family and
configuration holdouts are explicitly diagnostic.

Version 2 evaluation results contain a 30-second
`primary_communication_only` block, the complete split plan and assignments,
train/validation/test sample counts, train-only preprocessing disclosure,
validation-only model/threshold selection metadata, run/window/per-family
metrics, hard-negative false-positive rates, abstention coverage/selective
risk, and warnings. Confidence intervals are null when independent test groups
are insufficient. Non-communication and combined ablations are diagnostics.

## Other records

Workload events retain rank-local progress and checksums. Feature rows retain
split metadata separately from numeric features. Split assignments map a whole
run to one split. Evaluation and calibration results record their thresholds,
metrics, grouping unit, limitations, and source artifacts.
