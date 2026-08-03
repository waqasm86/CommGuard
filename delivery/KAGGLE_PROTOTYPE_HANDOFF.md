# CommGuard Kaggle prototype handoff

> **CommGuard’s Kaggle workflow is a single-node, dual-NVIDIA-T4 research
> prototype. It validates experimental methodology and software behavior on
> two local GPU ranks. It does not establish generalization to two physical
> 8-GPU nodes, NVLink/NVSwitch fabrics, RoCE or InfiniBand networks, large
> frontier-model workloads, or production treaty-verification deployments.**

The source and CPU/static validation are complete. New Kaggle measurements and
scientific acceptance are still required.

1. Pull or download branch
   `codex/commguard-kaggle-prototype-completion`.
2. Upload a checksum-pinned CommGuard wheel/source archive to Kaggle, or use the
   branch's final public commit in `PINNED_PUBLIC_COMMIT`.
3. Run `notebooks/commguard_calibration_v3.ipynb` in smoke mode, then in a fresh
   workspace in full mode.
4. Download and verify
   `commguard-calibration-prototype-<timestamp>.tar.gz` and its sibling
   `.sha256`; preserve it even if calibration fails.
5. Supply an accepted calibration archive to
   `commguard_benign_corpus_v2.ipynb`, then run the resumable full 24-run corpus.
6. Supply the accepted corpus archive to
   `commguard_detector_evaluation_v2.ipynb` and preserve all low, uncertain, or
   abstaining results.
7. After reviewing the bounded plan, run
   `commguard_adversarial_redteam_v1.ipynb` with `RUN_MODE="full"` and explicit
   human approval for the `k = 1, 2, 4, 8, 16` study.
8. Return the four stage archives, sibling checksums, and separately downloaded
   executed notebooks for evidence review and report completion.

Exact installation, failure recovery, archive verification, review-bundle, and
second-session steps are in `docs/KAGGLE_PROTOTYPE_RUNBOOK.md`.
