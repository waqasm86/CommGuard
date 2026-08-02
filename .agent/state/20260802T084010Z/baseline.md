# CommGuard resumption baseline

Captured: `2026-08-02T13:40:10+05:00` (`2026-08-02T08:40:10Z`)

This state inventory records the authoritative checkout before resuming Phase 02. It supplements, and does not replace, the original pre-change inventory at `.agent/state/20260802T125025+0500/baseline.md`.

## Git state

- Root: `/media/waqasm86/External1/Waqas-Projects/Project-William-Fowler/Project-Monitoring-GPU-Communication`
- Branch: `codex/commguard-research-completion`
- HEAD: `c920a738c1610eaf37e1ab3ae8912d2ac41db287`
- HEAD subject: `Repair repository and canonical notebook policy`
- Remote: `origin https://github.com/waqasm86/CommGuard.git`
- History: `c920a73`, `1e79089`, `1c57d5e`, `e4278b4`

Command: `git status --short --branch`

```text
## codex/commguard-research-completion
 M src/commguard/provenance.py
 M src/commguard/schemas.py
?? src/commguard/corpus.py
```

The three paths are a partial Phase 02 implementation that existed at the resumption boundary. They were inspected before further edits and are preserved as the starting point for Phase 02.

| Path | Boundary SHA-256 | Boundary diff |
|---|---|---:|
| `src/commguard/provenance.py` | `5aa67adefbaa1c108059429c338b1287bf1805e1df2f05465eaf696954e6f2b5` | 89 additions, 7 deletions |
| `src/commguard/schemas.py` | `d6c39dbf6b6477b1fc0c804666d82f351ffd253fb825b6e865ffa39ba819cbb2` | 78 additions, 10 deletions |
| `src/commguard/corpus.py` | `085ae845dfc28d1e0843685ed93be7aad405b05b61f2637eb689b99dcf4f496f` | new, 127 lines |

Ignored local evidence and environment paths remain present and unmodified: `.venv/`, caches, `dist/`, the two hyphen-named executed notebooks, `tar-files/`, and `Project-Monitoring-GPU-Communication.txt`.

## Current authored inventory

- Root policy/packaging: `AGENTS.md`, `.github/workflows/ci.yml`, `.gitignore`, `.gitattributes`, `pyproject.toml`, README, changelog, contributing, security, license, and notice.
- Documentation: `artifacts/README.md` and 16 files under `docs/`.
- SDK: 30 Python files under `src/commguard/`, including the untracked Phase 02 `corpus.py`.
- Tests: 21 Python files, 62 collected tests at this boundary.
- Canonical notebooks: five underscore-named, unexecuted notebooks listed in `notebooks/canonical_notebooks.json`.
- Local executed evidence: two ignored hyphen-named notebooks.
- Local archives: five ignored files under `tar-files/`; wheel/sdist files under ignored `dist/`.

## Notebook state

| Path | Bytes | SHA-256 | Executed code / outputs | Role |
|---|---:|---|---:|---|
| `commguard_calibration_v2.ipynb` | 8,905 | `892adba8b84a0fbfb0726a69546ca6ae823602f62bb6d19ffbaac1af6e99f35a` | 0 / 0 | canonical |
| `commguard_benign_corpus.ipynb` | 8,733 | `4ac36ab0efa22a87aa0f3f8e3c9ac17d9427c29cf5a9710d48adea3b8074972b` | 0 / 0 | canonical |
| `commguard_detector_evaluation.ipynb` | 7,971 | `77bd6d6998c54e4f8f9efe0dabab1b4d507711ddcce3bb870b78242f3a8f1f57` | 0 / 0 | canonical |
| `commguard_dual_t4.ipynb` | 6,245 | `21e806e120982ed92abedc82590c33d0d045d3626873ef2f6d18d38ff6353d86` | 0 / 0 | canonical |
| `commguard_dual_t4_research.ipynb` | 30,446 | `392ed680951dbd935accff42627e32ade9f27eefcff4b704ebb16bb575fcc717` | 0 / 0 | canonical |
| `commguard-benign-corpus.ipynb` | 43,692 | `7e7573fd6e6fd21838a3b037a7e339b3ab6f054927da4ff3ede610f03e71fff1` | 15 / 18 | ignored executed evidence |
| `commguard-detector-evaluation.ipynb` | 58,896 | `2903b2b2dcd8a5b2d7d113add47e3da8b6930ec72936743ea09b100398e014b2` | 9 / 13 | ignored executed evidence |

## Archive state and reverified evidence

Archive hashes are unchanged from the original inventory:

```text
13f0122997ac4127a6d45c032d5a2806d57e60f107e85e004f17b5f3e4308cc6  tar-files/commguard-benign-corpus.tar.gz
2460a44a2100d024bef6176aadd85cf9c57e343e7d3301c2dfc321acd99de48a  tar-files/commguard-calibration-pilot.tar.gz
840fbb6fd8860c3768dc98968e4c5d0c9ad96db4b8b04fe97e228274e34dc07d  tar-files/commguard-detector-evaluation.tar.gz
946184076a7fdf6c71b2dd1bcd60d98d34dd63a52e020d399ebe044d73c7d660  tar-files/commguard-review-bundle.zip
0d52977b0db40bc06bf29155332ebdd48e581d0de063fd6930db42c8500ed70f  tar-files/commguard.tar.gz
```

Read-only archive inspection reconfirmed:

- 18/18 planned benign runs completed, but the merged detector archive has 26 completed v1 manifests, including calibration artifacts.
- All 26 manifests have distinct environment fingerprints; those fingerprints are not true independent session IDs.
- The feature artifact has 16 five-second rows from eight runs: three DDP runs and five idle runs. No inference, compute, or host-transfer run emitted a primary feature window.
- The amended whole-run test has two runs and three windows. PCIe-only balanced accuracy is 0.5 with training recall 0 and false-negative rate 1.0. Combined and non-PCIe results on this tiny DDP-versus-idle diagnostic do not establish generalization.
- No new GPU, Kaggle, adversarial, model-quality, or physical multi-node experiment was run during this resumption audit.

## Python and available tools

```text
Python 3.11.15
pytest 8.3.5
ruff 0.16.0
build 1.5.0
```

Interpreter: `.venv/bin/python`. Unlike the original Phase 00 boundary, the project virtualenv now contains the `build` module. This audit did not install or change dependencies.

## CPU-safe resumption baseline

- `.venv/bin/python -m pytest -m 'not gpu and not multigpu'`: **passed**, 57 passed and 5 deselected.
- `.venv/bin/python -m ruff check .`: **passed**.
- `.venv/bin/python -m ruff format --check .`: **failed only on the three partial Phase 02 paths**; `corpus.py` and `provenance.py` would be reformatted.
- `.venv/bin/python -m build --outdir <fresh temporary directory>`: **passed**, producing a 61,175-byte wheel and 62,205-byte sdist. The ignored repository `dist/` evidence was not used as the build output.
- `git diff --check`: **passed**.

The current tests do not yet cover the partial Phase 02 schema/corpus work. In particular, `RunManifest` v2 requires provenance fields that the current orchestrator does not provide; Phase 02 must repair this before acceptance.

## Recovery boundary

Resume from the three file hashes above and commit `c920a73`. Do not reset, clean, delete ignored evidence, or overwrite raw archives. New state files and derived artifacts must be additive.
