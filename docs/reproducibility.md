# Reproducibility

1. Use a Kaggle notebook with the `GPU T4 x2` accelerator.
2. Clone `https://github.com/waqasm86/CommGuard.git` at a recorded commit SHA.
3. Install with
   `python -m pip install --no-build-isolation --no-deps -e /kaggle/working/CommGuard`.
4. Save `commguard preflight --strict` output before any CUDA workload.
5. Run the smoke profile and inspect both rank identity records.
6. Run calibration; do not continue to detector fitting after a negative gate.
7. Run bounded profiles with fixed seeds and randomized run order.
8. Export the complete artifact archive without editing generated data.
9. Repeat in a second Kaggle session for session-held-out evaluation.

The bundled `pip-kaggle-list.txt` is the supplied environment snapshot, not a
runtime assertion. CommGuard records actual versions. It never upgrades
PyTorch, CUDA, Triton, NumPy, pandas, or RAPIDS.

Random seeds and deterministic flags are recorded, but bitwise CUDA/NCCL
determinism is not claimed. OOM configurations and failed ranks are retained;
automatic scientific-profile batch-size reduction is prohibited.
