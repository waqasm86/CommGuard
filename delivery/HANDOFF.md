# Reviewer, push, and Kaggle handoff

## Inspect locally

```bash
git switch codex/commguard-pre-kaggle-audit-fixes
git status --short --branch
git log --oneline origin/main..HEAD
git diff --check origin/main...HEAD
.venv/bin/python tools/verify_delivery.py
```

Expected state before push: the branch is ahead of `origin/main`, the worktree
is clean, delivery checks pass, and no ignored archive/evidence file appears in
the tracked diff.

## Feature-branch publication

The exact command is:

```bash
git push -u origin codex/commguard-pre-kaggle-audit-fixes
```

Never push or merge `main`, and never force-push. After the feature branch is
pushed, record the immutable reviewed commit for Kaggle:

```bash
git rev-parse HEAD
git branch -r --contains HEAD
```

## Kaggle notebook sequence

Upload and run on a fresh Kaggle `GPU T4 x2` accelerator in this order:

1. `notebooks/commguard_calibration_v3.ipynb`
2. `notebooks/commguard_benign_corpus_v2.ipynb`
3. `notebooks/commguard_detector_evaluation_v2.ipynb`
4. `notebooks/commguard_adversarial_redteam_v1.ipynb` only after benign
   acceptance and separate human approval

Set the same pushed 40-character `REVIEWED_COMMIT` in every notebook. For each
downstream notebook, upload the preceding archive unchanged as a private Kaggle
Dataset, set the exact `INPUT_ARCHIVE`, and copy the printed SHA-256 into
`EXPECTED_INPUT_SHA256`.

The calibration notebook is the first hardware gate: it runs three idle
repetitions plus three AllReduce repetitions at 1, 4, 16, and 64 MiB. The
benign notebook labels the restored calibration as prior-session evidence and
runs a fresh calibration as its exact current-session collection gate. Do not
substitute either artifact for the other.

Expected output patterns:

| Notebook | Output archive |
|---|---|
| calibration | `commguard-calibration-v3-<NOTEBOOK_RUN_ID>.tar.gz` |
| benign corpus | `commguard-benign-corpus-v2-<NOTEBOOK_RUN_ID>.tar.gz` |
| detector evaluation | `commguard-detector-evaluation-v2-<NOTEBOOK_RUN_ID>.tar.gz` |
| approved adversarial run | `commguard-adversarial-redteam-v1-<NOTEBOOK_RUN_ID>.tar.gz` |

The benign notebook defaults to a smoke pilot. For completion evidence, set
`RUN_BENIGN_PILOT = False` and `RUN_STANDARD_BENIGN_MATRIX = True`; leave the
expanded matrix off initially. The detector notebook must stop if primary
coverage is incomplete.

Do not enable `RUN_ADVERSARIAL_PILOT` or set its approval variable on Codex's
authority. A human must review the benign acceptance artifact, bounds, plan,
estimated cost, and sealed identities first. Releasing the final holdout needs
a second, separate approval.
