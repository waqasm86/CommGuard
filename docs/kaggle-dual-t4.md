# Kaggle Dual T4

1. Create or open a Kaggle notebook.
2. Enable Internet access.
3. Select the `GPU T4 x2` accelerator.
4. Import the four notebooks listed in
   [`NEXT_KAGGLE_EXPERIMENTS.md`](NEXT_KAGGLE_EXPERIMENTS.md).
5. Start with `commguard_calibration_v3.ipynb` and run from the first cell.

Do not reuse the first failed calibration-v3 archive. That T4 x2 attempt reached
CUDA/NCCL and collectives, but worker mode `idle` was absent, every idle run
failed, and the calibration was correctly `not_supported`. Preserve it as
debugging evidence and use a fresh notebook run ID/directory for the patched
rerun.

Each notebook clones `https://github.com/waqasm86/CommGuard.git` into a fresh
working directory and installs the detached reviewed checkout with:

```bash
python -m pip install --no-build-isolation --no-deps -e /kaggle/working/commguard-source
```

Set `REVIEWED_COMMIT` to a 40-character commit present on an origin remote ref.
The notebook rejects mutable branch-only installation, dirty source, detached
commits that are not contained by a remote ref, archive hash mismatches, and
restore destinations that already exist.

Leave the committed placeholder empty until the reviewed feature branch has
been pushed, then insert that pushed commit's full SHA into all four notebooks.
The calibration notebook runs the bounded idle-aware 3 × (idle + 1/4/16/64 MiB)
matrix exactly once in a unique output directory. Copy its exact calibration
artifact path and SHA-256, not merely the archive name, into the benign notebook
only when its modern status is `supported`.

The benign notebook labels the restored calibration as prior-session input and
runs a fresh current-session calibration as its actual gate. It prints the exact
matrix and extraction-summary paths for the detector notebook. Detector fitting
verifies the corpus-bound current calibration and stops before model work if
the required eight-family 30-second coverage is incomplete.

Strict preflight must observe exactly two T4 devices, CUDA, and NCCL before any
dual-GPU experiment is launched. Installation does not establish GPU readiness.

The smoke worker is launched with two `torchrun` processes. It rejects CPU,
Gloo, one-GPU, and DataParallel execution. Rank 0 creates a participation
summary only after both create-only rank result files validate.

Historical successful dual-T4 artifacts exist locally and are indexed in
[`evidence-index.md`](evidence-index.md), but they do not pass the current broad
coverage contract. No completion-series notebook has been executed.

For the staged research update, use
[`NEXT_KAGGLE_EXPERIMENTS.md`](NEXT_KAGGLE_EXPERIMENTS.md) and run calibration,
benign-corpus collection, then grouped detector evaluation. Download the output
bundle after each session and upload it as a Kaggle Dataset for the next stage.
No training-versus-inference performance claim is supported until the detector
notebook creates a non-empty grouped evaluation artifact after coverage passes.
