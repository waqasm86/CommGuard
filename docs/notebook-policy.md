# Notebook source and evidence policy

Policy version `commguard-notebook-policy-v3` requires exactly four canonical
notebooks in registered order plus the separated v4 diagnostic. Canonical code
cells are output-free, parameter-tagged, use installed SDK APIs, provide smoke
and full gates where hardware is collected, record final `run_status.json`, and
export non-destructive deterministic archives with sibling SHA-256 files.

CommGuard separates canonical notebook source from executed evidence.

## Canonical source

Canonical notebooks live directly under `notebooks/`, use stable
underscore-separated names, parse as notebook format 4, have null execution
counts and no outputs, and orchestrate public SDK APIs. The machine-readable
inventory is `notebooks/canonical_notebooks.json`. Changes to the inventory and
repository notebook tests must be made together.

Canonical source must not contain copied SDK implementations, credentials,
personal storage links, or claims based on anticipated results. A result
section says `not executed` until a separately preserved executed artifact
exists.

The current canonical sequence is generated deterministically by
`tools/generate_canonical_notebooks.py` and checked with `--check`:

1. `commguard_calibration_v3.ipynb`
2. `commguard_benign_corpus_v2.ipynb`
3. `commguard_detector_evaluation_v2.ipynb`
4. `commguard_adversarial_redteam_v1.ipynb`

Each source refuses a mutable branch install. The operator must provide a
40-character reviewed commit that is present on the configured remote; the
notebook fetches that object, uses detached HEAD, and fails if the checkout is
dirty or not visible from a remote ref. Each downstream notebook also requires
the exact SHA-256 printed by its predecessor and restores the archive through
the SDK's create-only, traversal/link-rejecting loader.

The committed `PINNED_PUBLIC_COMMIT` value remains an explicit placeholder until
the reviewed feature branch is pushed. Full calibration uses five idle
repetitions plus five repetitions at 1, 4, 16, 64, and 128 MiB at the selected
sampling interval; a bounded idle collector comparison covers the three standard
intervals. The benign notebook consumes the exact accepted calibration
archive and binds the corpus to that provenance. Later notebooks consume the
exact matrix, extraction, calibration, and evaluation paths printed by their
predecessor; they never choose a calibration by filename order.

After installation, every canonical notebook invalidates import caches, removes
stale `commguard` modules, reimports the installed package, and records the
resolved import path. Pinned Git installation uses a detached remote-visible
checkout; editable installation is restricted to explicitly dirty development
smoke mode and can never export accepted evidence.

Calibration creates one unique artifact directory per `NOTEBOOK_RUN_ID` before
strict preflight and refuses directory reuse. The SDK then writes create-only
sweep start/completion markers. Re-running the calibration cell against the
same context is an error; partial evidence is retained and a new attempt needs a
new notebook run ID and workspace. Export uses a unique archive name, refuses
overwrite, and preserves a structurally complete but `not_supported` result as
diagnostic evidence before telling the operator not to continue to benign work.

## Executed evidence

Executed notebooks are evidence/release artifacts, not editable source. They
must preserve source commit, input archive hashes, notebook provenance,
environment evidence, and a result-status banner. Local historical copies use
hyphenated names and are ignored by Git; raw archives remain under the ignored
`tar-files/` directory. Ignoring these files does not authorize deletion.

The pre-existing executed copies at the Phase 00 boundary are:

| Local file | SHA-256 | Evidence status |
|---|---|---|
| `commguard-benign-corpus.ipynb` | `7e7573fd6e6fd21838a3b037a7e339b3ab6f054927da4ff3ede610f03e71fff1` | 18/18 benign runs recorded; coverage not established |
| `commguard-detector-evaluation.ipynb` | `2903b2b2dcd8a5b2d7d113add47e3da8b6930ec72936743ea09b100398e014b2` | 16 DDP/idle feature rows; tiny amended whole-run evaluation |

These files contain historical Google Drive locations and outputs. They remain
local-only and must not be used as canonical templates or silently rewritten.
Their corresponding archive hashes are recorded in the timestamped Phase 00
state inventory and the evidence index.

## Naming transitions

The research-completion series supersedes earlier staged sources with the
versioned canonical sequence above. Superseded source notebooks were removed
from the active directory only after confirming their Git-history provenance
and preserved historical executed evidence. They remain recoverable from Git
history but are not result evidence or inputs to the current sequence.

Superseding a canonical source never deletes executed evidence.
