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

The committed `REVIEWED_COMMIT` value remains an explicit placeholder until the
reviewed feature branch is pushed. Calibration uses three idle repetitions plus
three repetitions at 1, 4, 16, and 64 MiB. The benign notebook records the
restored calibration as prior-session evidence and binds its corpus to a fresh
current-session calibration. Later notebooks consume the exact matrix,
extraction, calibration, and evaluation paths printed by their predecessor;
they never choose a calibration by filename order.

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

The research-completion series supersedes the earlier staged sources with the
versioned canonical sequence above. The older underscore-named notebooks stay
in Git as historical source files but are no longer in the canonical inventory.
They are not result evidence and are not inputs to the current sequence.

Superseding a canonical source never deletes executed evidence.
