# Kaggle prototype runbook

> **CommGuard’s Kaggle workflow is a single-node, dual-NVIDIA-T4 research
> prototype. It validates experimental methodology and software behavior on
> two local GPU ranks. It does not establish generalization to two physical
> 8-GPU nodes, NVLink/NVSwitch fabrics, RoCE or InfiniBand networks, large
> frontier-model workloads, or production treaty-verification deployments.**

This runbook begins with unexecuted source notebooks. It does not claim that a
new calibration, corpus, detector, or adversarial result exists. Preserve every
executed notebook and archive, including failures, as a separate immutable
record.

## 1. Create the Kaggle runtime

Create a new Kaggle notebook, select the `GPU T4 x2` accelerator, and use the
Python 3.11 runtime. `/kaggle/input` is read-only; all working data must go under
`/kaggle/working`. Turn Internet on only if installing from the pinned public
Git commit. A wheel or source archive uploaded as a private Kaggle Dataset does
not require Internet after the dataset is attached.

Copy the canonical notebook source into Kaggle without executing it. Set
`EXPECTED_NOTEBOOK_SHA256` to the SHA-256 of that exact source notebook.

## 2. Install CommGuard using the recorded priority

The source cell implements this order:

1. Exactly one `commguard*.whl` under `/kaggle/input`, with
   `EXPECTED_PACKAGE_SHA256` set.
2. Exactly one CommGuard `.tar.gz` or `.zip` source archive, also checksum
   pinned.
3. `PINNED_PUBLIC_COMMIT`, a 40-character public commit fetched from
   `https://github.com/waqasm86/CommGuard.git`.
4. `INSTALL_SOURCE="development"` only with
   `DEVELOPMENT_SMOKE_TEST=True`; this is never accepted GPU evidence.

Do not add a repository token. The cell records the install source, package
hash or commit, package version, Python version, package import path, and
`pip freeze`. Preflight adds PyTorch, CUDA, NCCL when discoverable, NVIDIA
driver, topology, GPU UUIDs, and telemetry capabilities.

## 3. Verify both GPUs and run preflight

Run the notebook from the first cell. Its hardware gate is equivalent to:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
python -c 'import torch; print(torch.cuda.device_count(), torch.distributed.is_nccl_available())'
commguard preflight --strict --output /kaggle/working/commguard-preflight
```

Proceed only with exactly two Tesla T4 devices, two distinct UUIDs, CUDA, and
NCCL. There is no CPU, Gloo, or one-GPU evidence fallback.

## 4. Run calibration smoke mode

Open `notebooks/commguard_calibration_v3.ipynb`, set `RUN_MODE="smoke"`, and run
all cells. This short run is labeled:

```text
development_smoke_only = true
scientific_acceptance_eligible = false
```

Download both
`commguard-calibration-prototype-<timestamp>.tar.gz` and its sibling
`.tar.gz.sha256`. A smoke archive proves only that the software path executed.

## 5. Run full calibration

Start a fresh notebook/workspace, set `RUN_MODE="full"`, and run all cells. The
full plan has five idle repetitions and five AllReduce repetitions at 1, 4, 16,
64, and 128 MiB: 30 runs total. Each run targets 5 seconds warm-up, at least 30
measured seconds, and approximately 5 seconds cooldown. A short bounded idle
collector comparison measures jitter and sampler duty at 1.0, 0.5, and 0.2
seconds; the full collective sweep uses the selected interval, initially 0.5
seconds. Repeat the full sweep at another supported interval only as a separate
attempt. The 0.1 second interval remains diagnostic.

The first expected archive is:

```text
commguard-calibration-prototype-<timestamp>.tar.gz
```

Download it even if the gate fails. Continue only if
`prototype/calibration/calibration_acceptance.json` records an eligible,
supported result. `partially_supported`, `inconclusive`, `not_supported`, and
`failed` remain distinct outcomes.

## 6. Verify and supply calibration to the corpus notebook

On your local machine:

```bash
sha256sum -c commguard-calibration-prototype-<timestamp>.tar.gz.sha256
```

Upload the archive as a private Kaggle Dataset. In
`commguard_benign_corpus_v2.ipynb`, set the exact dataset/archive path,
`EXPECTED_INPUT_SHA256`, exact calibration JSON path, and its SHA-256. The
restored calibration is prior-session evidence; the corpus session creates a
fresh current-session calibration gate and never reuses calibration idle runs
as corpus idle controls.

## 7. Build the benign corpus in resumable batches

First set `RUN_MODE="smoke"`; the output is development-only. For the full
prototype use `RUN_MODE="full"`, which selects the stronger existing matrix:
eight workload families times three repetitions, or 24 planned runs. Every
family is labeled as a real implementation, controlled proxy, or synthetic
control in `docs/benign-workload-matrix.md`.

The plan is deterministically randomized. Rerunning the cell with the same
session and corpus identifiers validates plan/configuration hashes, skips only
one exact accepted match, refuses duplicate accepted identities, and creates a
new attempt directory for incomplete or failed slots. Never change the plan
parameters while resuming. After a Kaggle restart, restore the most recent
archive and retain the original session/corpus identifiers before resuming.

Download and verify:

```text
commguard-benign-corpus-prototype-<timestamp>.tar.gz
commguard-benign-corpus-prototype-<timestamp>.tar.gz.sha256
```

Do not proceed unless `corpus_acceptance.json` and the eight-family completion
matrix pass the full 30-second coverage policy.

## 8. Evaluate detectors

Attach the accepted corpus archive, set its exact SHA-256 and matrix-summary
path in `commguard_detector_evaluation_v2.ipynb`, then run all cells. Evaluation
splits complete runs and sessions, never telemetry rows. The primary task is
training versus inference with `communication_only`; controls are reported
separately and are never silently relabeled as inference. Auxiliary and combined
feature sets are secondary.

Download and verify:

```text
commguard-detector-evaluation-prototype-<timestamp>.tar.gz
commguard-detector-evaluation-prototype-<timestamp>.tar.gz.sha256
```

Metric targets are non-normative. Preserve honest low, uncertain, or abstaining
results.

## 9. Run bounded adversarial evaluation

Attach the detector archive. Review the plan, then set
`RUN_MODE="full"` and `ADVERSARIAL_HUMAN_APPROVAL=True`. The required bounded
local-update study uses
`k = 1, 2, 4, 8, 16`. It targets only the local Kaggle process group. The frozen
benign-only detector is not fitted on adversarial rows, and the separate final
family/session/config holdout remains sealed.

Download and verify:

```text
commguard-adversarial-redteam-prototype-<timestamp>.tar.gz
commguard-adversarial-redteam-prototype-<timestamp>.tar.gz.sha256
```

## 10. Create the review bundle

After restoring the four verified stage archives into a clean staging tree:

```bash
commguard build-review-bundle \
  --input /kaggle/working/commguard-review-staging \
  --output /kaggle/working/commguard-prototype-review-bundle-<timestamp>.tar.gz
```

The command is non-destructive, checks archive membership, and writes a sibling
checksum. Download both files.

## 11. Preserve executed notebooks and run a second session

Download Kaggle's executed `.ipynb` separately from the source and archives.
Never overwrite the unexecuted canonical notebook. For cross-session validation,
start a second fresh T4 x2 session, use new experiment/session IDs, repeat full
calibration and corpus collection, then supply both sessions to grouped detector
evaluation. Until those artifacts exist, cross-session validation is pending.

## Failure guidance

- **Only one GPU visible:** stop, select `GPU T4 x2`, restart the session, and
  rerun preflight. Do not fall back to CPU or one GPU.
- **Idle mode unsupported:** preserve the archive and traceback. Confirm the
  pinned commit contains the idle worker fix; create a new run ID after repair.
- **NCCL initialization failure:** preserve both rank logs, driver/PyTorch/NCCL
  inventory, and topology output. Restart only into a new attempt directory.
- **Rank timeout:** preserve the timed-out attempt; verify both rank logs and
  cleanup evidence before rerunning the slot.
- **Unsupported NVML PCIe counters:** record `not_supported`; do not substitute
  GPU utilization into the primary communication-only claim.
- **Dirty Git state:** accepted export is refused. Use a clean pinned commit or
  explicitly run development smoke mode.
- **Duplicate run identity:** do not delete either run. Inspect plan ID and
  configuration hash; use a fresh session if two accepted matches exist.
- **Short measured duration:** the run is scientifically rejected even if
  `torchrun` returned zero. Rerun that plan slot into a new attempt.
- **Missing calibration:** restore the exact archive and JSON reference; never
  select the lexicographically newest file.
- **Failed acceptance gate:** download the evidence and stop the downstream
  claim path. A failed gate is a useful result.
- **Kaggle interruption:** download the latest non-destructive archive, restore
  it, keep the same plan/session identities, and use resume mode.
- **Archive export failure:** preserve the source artifact tree, choose a new
  output archive path, check free space, and rerun export. Never delete source
  outputs to make an archive succeed.
