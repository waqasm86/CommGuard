# Notebook source and evidence policy

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

`commguard_dual_t4_research.ipynb` is a byte-identical rename of the deleted
`commguard_william_fowler_dual_t4_research.ipynb`; the shorter name avoids
personal naming in a canonical path. The deleted name remains visible in Git
history and its hash is recorded in the baseline inventory.

The research-completion series will supersede the current staged notebooks
with versioned canonical sources:

1. `commguard_calibration_v3.ipynb`
2. `commguard_benign_corpus_v2.ipynb`
3. `commguard_detector_evaluation_v2.ipynb`
4. `commguard_adversarial_redteam_v1.ipynb`

Superseding a canonical source never deletes executed evidence.
