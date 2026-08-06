# Evidence index

## Prospective prototype packages

The new canonical workflow will create calibration, benign-corpus, detector,
and adversarial archives named `commguard-*-prototype-<timestamp>.tar.gz` with
sibling checksums. None is indexed as executed or accepted yet. After Kaggle
execution, add exact archive size, SHA-256, source identity, session ID, and
acceptance state here before using a result in README or reports.

This index is the claim boundary for CommGuard's current empirical results. Raw
archives and executed notebooks are ignored local evidence and are not committed
or rewritten. The pre-completion inventory informed this index, but its private
local-path state snapshots are intentionally excluded from the public tree.

## Immutable local evidence

| ID | Local artifact | Bytes | SHA-256 | Source identity | Status and supported claims |
|---|---|---:|---|---|---|
| E1 | `tar-files/commguard.tar.gz` | 98,139 | `0d52977b0db40bc06bf29155332ebdd48e581d0de063fd6930db42c8500ed70f` | source SHA not embedded; associated review bundle records `e4278b42d9a31223dadea72b17fd9bb67d289df7` | Historical calibration-only archive: strict dual-T4 environment and calibration records; standard corpus was not executed, feature rows were empty, and no evaluation exists. |
| E2 | `tar-files/commguard-review-bundle.zip` | 357,982 | `946184076a7fdf6c71b2dd1bcd60d98d34dd63a52e020d399ebe044d73c7d660` | clean `e4278b42d9a31223dadea72b17fd9bb67d289df7` in bundle metadata | Review wrapper for E1. It supports provenance and environment inspection, not a detector result. |
| E3 | `tar-files/commguard-calibration-pilot.tar.gz` | 83,888 | `2460a44a2100d024bef6176aadd85cf9c57e343e7d3301c2dfc321acd99de48a` | `1e790895ae3bd0919dde0f383e78fb14f767d359` in preflight | Historical dual-T4 calibration pilot. The saved decision is `supported` under the historical calibration contract; it does not establish classifier validity. |
| E4 | `tar-files/commguard-benign-corpus.tar.gz` | 221,072 | `13f0122997ac4127a6d45c032d5a2806d57e60f107e85e004f17b5f3e4308cc6` | `1e790895ae3bd0919dde0f383e78fb14f767d359` in `benign-corpus-summary.json` | Historical pilot with 18 planned, 18 completed, and zero failed benign runs. Completion is not feature coverage. |
| E5 | `tar-files/commguard-detector-evaluation.tar.gz` | 255,098 | `840fbb6fd8860c3768dc98968e4c5d0c9ad96db4b8b04fe97e228274e34dc07d` | `1e790895ae3bd0919dde0f383e78fb14f767d359` in detector notebook provenance | Derived historical features and amended whole-run evaluation. Supports only the DDP-versus-idle negative finding described in [`current-results.md`](current-results.md). |
| E6 | `notebooks/commguard-benign-corpus.ipynb` | 43,692 | `7e7573fd6e6fd21838a3b037a7e339b3ab6f054927da4ff3ede610f03e71fff1` | historical executed copy associated with E4 | Executed notebook evidence for E4. It contains historical private-storage locations and is not canonical source. |
| E7 | `notebooks/commguard-detector-evaluation.ipynb` | 58,896 | `2903b2b2dcd8a5b2d7d113add47e3da8b6930ec72936743ea09b100398e014b2` | notebook provenance records `1e790895ae3bd0919dde0f383e78fb14f767d359` | Executed notebook evidence for E5. It records why a deterministic class-stratified whole-run override replaced the invalid historical fingerprint-based split. |
| E8 | `pip-kaggle-list.txt` | 44,561 | `f45bb00d807da387d944f692b7626f91737056c735e4681127415ecdccfa0dce` | supplied environment snapshot; not runtime provenance | Supports only the package snapshot contents and line count. Runtime versions come from preflight artifacts instead. |

The hashes above were rechecked at the Phase 08 boundary. E6 and E7 retained
their baseline hashes. The archives were inspected read-only after rejecting
unsafe member names/links; no raw artifact was migrated or edited.

## Evidence-derived claim map

| Claim | Evidence | Classification |
|---|---|---|
| A Kaggle session exposed exactly two Tesla T4 GPUs and strict CUDA/NCCL readiness under the historical preflight contract. | E3 and E5 environment reports | measured |
| The historical benign orchestrator completed all planned pilot launches. | E4 `benign-corpus-summary.json` | measured |
| The merged feature set contains only five-second DDP and idle rows, and includes calibration-idle runs. | E5 feature JSONL plus run manifests | derived, observed limitation |
| The historical PCIe-only test missed its sole training test run. | E5 amended evaluation JSON | measured/derived negative result |
| Reliable training-versus-inference detection is established. | no supporting artifact | unsupported |
| The new eight-family, 30-second primary coverage gate has passed on dual T4. | no supporting artifact | pending Kaggle execution |
| Adversarial robustness or final-holdout performance is established. | no supporting artifact | pending approval and Kaggle execution |
| Physical multi-node behavior is validated. | no supporting artifact | pending hardware validation |

## Adding evidence

New evidence must be additive. Record archive name, byte size, SHA-256, reviewed
source commit, dirty-state decision, input hashes, environment report, corpus
manifest, result status, and the narrow claims it supports. A corrected analysis
gets a new archive and index row; old archives remain immutable.

## Calibration-v4 pending evidence

The calibration-v3 full result is retained as immutable evidence of partial
support at a 0.5-second interval. It does not unlock downstream notebooks.

The next required evidence item is a full canonical
`commguard_calibration_v4.ipynb` execution using a 0.2-second sampling
interval. Its output archive and sibling SHA-256 must be preserved regardless
of whether the scientific gate passes.
