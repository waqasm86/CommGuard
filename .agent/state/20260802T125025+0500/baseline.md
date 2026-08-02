# CommGuard authoritative checkout baseline

Captured: `2026-08-02T12:50:25+05:00` (Asia/Karachi)

This inventory records the live checkout before CommGuard completion work. The prompt package audit was used only as a comparison source. No repository file had been changed by Codex when the commands below were run.

## Paths and Git state

- Working directory and Git root: `/media/waqasm86/External1/Waqas-Projects/Project-William-Fowler/Project-Monitoring-GPU-Communication`
- Branch: `main`, tracking `origin/main`
- HEAD: `1e790895ae3bd0919dde0f383e78fb14f767d359`
- Remote: `origin https://github.com/waqasm86/CommGuard.git`
- Commits present:
  - `1e79089` (2026-08-01) Adding first Kaggle Notebook for William Fowler Research using Kaggle's dual Nvidia T4 gpus.
  - `1c57d5e` (2026-08-01) Improve repetition-aware calibration and add Kaggle study notebooks
  - `e4278b4` (2026-07-30) Initial CommGuard SDK release

Command: `git status --short --branch`

```text
## main...origin/main
 D notebooks/commguard_benign_corpus.ipynb
 D notebooks/commguard_detector_evaluation.ipynb
 D notebooks/commguard_dual_t4.ipynb
 D notebooks/commguard_william_fowler_dual_t4_research.ipynb
?? notebooks/commguard-benign-corpus.ipynb
?? notebooks/commguard-detector-evaluation.ipynb
?? notebooks/commguard_dual_t4_research.ipynb
?? tar-files/
```

The four deletions and all untracked files predate Codex work and are user state to preserve. The ignored `Project-Monitoring-GPU-Communication.txt` is a 4,729-line captured recursive directory listing, not a research-result artifact; it is left untouched.

## Authored-file inventory

The checkout contains:

- packaging and policy: `.gitattributes`, `.gitignore`, `.github/workflows/ci.yml`, `pyproject.toml`, README, changelog, license, notice, security, and contributing files;
- 15 documentation files under `docs/` plus `artifacts/README.md`;
- 29 Python modules under `src/commguard/` covering artifacts, calibration, CLI, distributed execution, preflight, evaluation, feature extraction, reporting, telemetry, schemas, provenance, orchestration, and workloads;
- 21 test files under `tests/` (1,115 lines at baseline);
- four live notebook files plus four deleted tracked notebook paths in HEAD;
- five local archive files under `tar-files/`;
- the immutable 935-line `pip-kaggle-list.txt` snapshot and `examples/smoke.json`.

Excluded from line-by-line inventory: `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, compiled files, notebook output payloads, and raw telemetry payload rows. Documentation, source, tests, notebook sources/metadata, and evidence manifests/summaries were inspected.

## Notebook inventory

| Live path | Bytes | SHA-256 | Cells/code | Executed code cells | Output objects | Baseline role |
|---|---:|---|---:|---:|---:|---|
| `notebooks/commguard_calibration_v2.ipynb` | 8,905 | `892adba8b84a0fbfb0726a69546ca6ae823602f62bb6d19ffbaac1af6e99f35a` | 9/4 | 0 | 0 | tracked, canonical, unexecuted |
| `notebooks/commguard_dual_t4_research.ipynb` | 30,446 | `392ed680951dbd935accff42627e32ade9f27eefcff4b704ebb16bb575fcc717` | 38/17 | 0 | 0 | untracked rename of deleted tracked research notebook |
| `notebooks/commguard-benign-corpus.ipynb` | 43,692 | `7e7573fd6e6fd21838a3b037a7e339b3ab6f054927da4ff3ede610f03e71fff1` | 39/23 | 15 | 18 | untracked executed evidence copy with Google Drive ID |
| `notebooks/commguard-detector-evaluation.ipynb` | 58,896 | `2903b2b2dcd8a5b2d7d113add47e3da8b6930ec72936743ea09b100398e014b2` | 20/9 | 9 | 13 | untracked executed evidence copy with Google Drive URL |

Deleted tracked HEAD notebook hashes were recorded without restoring them:

- `commguard_benign_corpus.ipynb`: `4ac36ab0efa22a87aa0f3f8e3c9ac17d9427c29cf5a9710d48adea3b8074972b`
- `commguard_detector_evaluation.ipynb`: `77bd6d6998c54e4f8f9efe0dabab1b4d507711ddcce3bb870b78242f3a8f1f57`
- `commguard_dual_t4.ipynb`: `21e806e120982ed92abedc82590c33d0d045d3626873ef2f6d18d38ff6353d86`
- `commguard_william_fowler_dual_t4_research.ipynb`: `392ed680951dbd935accff42627e32ade9f27eefcff4b704ebb16bb575fcc717` (byte-identical to the live underscore rename)

## Local archives

| Path | Bytes | SHA-256 |
|---|---:|---|
| `tar-files/commguard-benign-corpus.tar.gz` | 221,072 | `13f0122997ac4127a6d45c032d5a2806d57e60f107e85e004f17b5f3e4308cc6` |
| `tar-files/commguard-calibration-pilot.tar.gz` | 83,888 | `2460a44a2100d024bef6176aadd85cf9c57e343e7d3301c2dfc321acd99de48a` |
| `tar-files/commguard-detector-evaluation.tar.gz` | 255,098 | `840fbb6fd8860c3768dc98968e4c5d0c9ad96db4b8b04fe97e228274e34dc07d` |
| `tar-files/commguard-review-bundle.zip` | 357,982 | `946184076a7fdf6c71b2dd1bcd60d98d34dd63a52e020d399ebe044d73c7d660` |
| `tar-files/commguard.tar.gz` | 98,139 | `0d52977b0db40bc06bf29155332ebdd48e581d0de063fd6930db42c8500ed70f` |

Archive member names were validated for path traversal and links before two representative archives were extracted into a fresh `/tmp/commguard-audit.*` directory for read-only manifest inspection. Repository archives were not modified.

## Evidence findings reverified from manifests

- Measured: strict-ready Kaggle environment records show two Tesla T4 GPUs, compute capability 7.5, 15,360 MiB each, driver 580.159.04, PyTorch 2.10.0+cu128, CUDA 12.8, and NCCL 2.27.5.
- Measured: `benign-corpus-summary.json` records 18 planned, 18 completed, zero failed benign runs at source commit `1e790895...`.
- Observed limitation: the merged detector archive has 26 completed manifests, including six calibration runs and five idle runs; calibration idle controls entered the derived set.
- Observed limitation: the 18 benign run wall durations are approximately 6–12 seconds. After the configured two-second warmup, non-DDP/non-idle families have insufficient duration for a five-second primary window.
- Observed limitation: all 26 manifests have distinct `environment_fingerprint` values, showing that this value cannot truthfully represent an independent experiment session.
- Derived: the feature artifact contains 16 rows from eight runs: five rows/three DDP runs and 11 rows/five idle runs, only at five seconds. It contains no inference, compute, or host-transfer feature rows.
- Measured/derived result: the amended deterministic whole-run split tests only two runs and three windows. Combined and non-PCIe diagnostics separate that tiny test; PCIe-only run-level balanced accuracy is 0.5, training recall is 0, and false-negative rate is 1.0. This is insufficient evidence of generalization.
- The older `commguard.tar.gz` records calibration only (`standard_corpus_executed: false`, zero feature rows, no evaluation), while later archives record the bounded pilot above.

## Python and tools

- Project virtualenv: Python 3.11.15, pip 24.0, pytest 8.3.5, Ruff 0.16.0, setuptools 79.0.1.
- System: Python 3.12.3 at `/usr/bin/python3`.
- No `python` executable is on PATH.
- The `build` module and wheel package were not present in the project virtualenv; system Python also lacks `build`.
- Available command-line tools include Git, ripgrep, jq, sha256sum, tar, unzip, and zip.
- `pip-kaggle-list.txt`: 935 lines, SHA-256 `f45bb00d807da387d944f692b7626f91737056c735e4681127415ecdccfa0dce`.

## CPU-safe baseline

Command: `.venv/bin/python -m pytest -m 'not gpu and not multigpu'`

Result: **failed**, 56 passed, 1 failed, 5 deselected. The only failure is `tests/test_repository.py::test_notebook_is_valid_json_and_has_no_sdk_implementation`: missing `notebooks/commguard_dual_t4.ipynb`.

Command: `.venv/bin/python -m ruff check src tests`

Result: **passed** (`All checks passed!`).

Command: `.venv/bin/python -m ruff check .`

Result: **failed** with 22 issues, all in live notebook code cells (imports, one long line, and unused imports). The current CI checks only `src tests`.

Command: `.venv/bin/python -m ruff format --check src tests`

Result: **failed**; 12 files would be reformatted and 38 were already formatted.

Command: `.venv/bin/python -m ruff format --check .`

Result: **failed**; 16 files would be reformatted and 57 were already formatted.

Command: `.venv/bin/python -m build`

Result: **not runnable**: `No module named build`. No dependency was installed during baseline capture.

## Recovery

The baseline user state can be compared at any time with this file, the archive hashes above, and commit `1e790895...`. Do not reset, clean, force-checkout, or delete local archives/notebooks. Canonical notebook repair must preserve the executed notebook hashes and archive evidence.
