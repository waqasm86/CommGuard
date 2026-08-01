# Kaggle Dual T4

1. Create or open a Kaggle notebook.
2. Enable Internet access.
3. Select the `GPU T4 x2` accelerator.
4. Import or download
   [`notebooks/commguard_dual_t4.ipynb`](../notebooks/commguard_dual_t4.ipynb).
5. Run the notebook from the installation cell.

The notebook clones `https://github.com/waqasm86/CommGuard.git` into
`/kaggle/working/CommGuard` and installs that checkout with:

```bash
python -m pip install --no-build-isolation --no-deps -e /kaggle/working/CommGuard
```

For a repeatable run, set `GIT_REF` in the first notebook cell to a reviewed
commit SHA instead of `main`. When Kaggle Internet access is disabled, upload
a repository snapshot as a Kaggle dataset and point `REPO` at that snapshot.

Strict preflight must observe exactly two T4 devices, CUDA, and NCCL before any
dual-GPU experiment is launched. Installation does not establish GPU readiness.

The smoke worker is launched with two `torchrun` processes. It rejects CPU,
Gloo, one-GPU, and DataParallel execution. Rank 0 creates a participation
summary only after both create-only rank result files validate.

No successful Kaggle GPU run is included in this repository.

For the staged research update, use
[`NEXT_KAGGLE_EXPERIMENTS.md`](NEXT_KAGGLE_EXPERIMENTS.md) and run calibration,
benign-corpus collection, then grouped detector evaluation. Download the output
bundle after each session and upload it as a Kaggle Dataset for the next stage.
No training-versus-inference performance claim is supported until the final
evaluation notebook creates a non-empty grouped evaluation artifact.
