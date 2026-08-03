# Reproducibility

Canonical notebooks record installation source, exact wheel/archive SHA-256 or
public Git commit, CommGuard version, Python, PyTorch, CUDA, NCCL when
discoverable, driver/GPU inventory, focused package snapshot, notebook SHA-256,
configuration hash, dirty state, session/corpus IDs, scope metadata, run status,
and stage checksums. Accepted export refuses dirty source; editable source is
development-smoke-only.

1. Use the canonical sequence in
   [`NEXT_KAGGLE_EXPERIMENTS.md`](NEXT_KAGGLE_EXPERIMENTS.md) with Kaggle's
   `GPU T4 x2` accelerator.
2. Set `PINNED_PUBLIC_COMMIT` to a 40-character commit present on an origin remote
   ref. The notebook fetches that object, uses detached HEAD, and refuses dirty
   or remote-unreachable source.
3. Install with `--no-build-isolation --no-deps`; never update Kaggle's
   scientific/CUDA packages in place.
4. Save strict preflight output before any CUDA workload.
5. Run the bounded calibration pilot. Preserve negative or partial decisions.
6. Export the create-only archive, record its SHA-256, and upload it unchanged as
   a private Kaggle Dataset for the next notebook.
7. In each downstream notebook, set the exact `INPUT_ARCHIVE` and
   `EXPECTED_INPUT_SHA256`; restore only into a fresh destination.
8. For primary evaluation, opt into the 24-run standard benign matrix and stop
   unless every required family has three valid 30-second runs.
9. Repeat across independently identified sessions before a session-held-out
   claim. An environment fingerprint is never a session ID.
10. Do not enable adversarial work until benign acceptance exists and a human
    approves the bounded plan. Keep the final holdout sealed separately.

The bundled `pip-kaggle-list.txt` is the supplied environment snapshot, not a
runtime assertion. CommGuard records actual versions. It never upgrades
PyTorch, CUDA, Triton, NumPy, pandas, or RAPIDS.

Random seeds and deterministic flags are recorded, but bitwise CUDA/NCCL
determinism is not claimed. OOM configurations and failed ranks are retained;
automatic scientific-profile batch-size reduction is prohibited.

After every run, add the archive/hash/source/status/claim boundary to
[`evidence-index.md`](evidence-index.md). Never replace an older row or archive.
