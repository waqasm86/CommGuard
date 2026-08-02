# CommGuard

CommGuard is an installable research SDK for a controlled, single-host Kaggle
experiment on two NVIDIA T4 GPUs. It collects content-agnostic GPU telemetry,
calibrates accessible PCIe traffic readings against known NCCL workloads, runs
bounded training/inference/control workloads, and evaluates leakage-resistant
workload classifiers.

> This is an independent, unofficial research prototype. It is not affiliated
> with or endorsed by SPAR, Kairos, ERA, UChicago XLab, William Fowler, or the
> authors and institutions cited in the related literature.

## Scope

CommGuard is designed to answer a limited empirical question: whether short
windows of signals available in one dual-T4 Kaggle session distinguish the
included PyTorch DDP training workloads from included benign inference and
control workloads. It does not validate frontier-scale, multi-node, NVLink,
NVSwitch, RoCE, InfiniBand, privacy, security, production, or treaty claims.
NVML PCIe TX/RX values are always described as **PCIe traffic readings**. They
are not treated as NCCL byte counts or a complete measure of GPU communication.

The SDK distinguishes measured observations, evidence-supported inferences,
untested hypotheses, and out-of-scope claims. A failed calibration or a
classifier that does not generalize is a valid research result.

## Current evidence status

**Observed limitation:** the current pilot's feature coverage is DDP versus idle
only. Although 18/18 planned benign runs completed, the saved feature artifact
contains 16 five-second windows from eight DDP/idle runs; inference, compute, and
host-transfer runs were too short after warmup. Calibration-stage idle controls
also entered the merged derived set.

**Measured/derived negative result:** PCIe-only detection missed the held-out
training run (run-level balanced accuracy 0.5, training recall 0, false-negative
rate 1.0) in a test containing only two runs. Combined and non-PCIe diagnostic
features separated that tiny pilot, but current generalization evidence is
insufficient. CommGuard has not established reliable training-versus-inference
detection or adversarial robustness. See the evidence hash inventory and
coverage-failure analysis before interpreting any historical notebook output.

## Kaggle quick start

CommGuard is distributed from GitHub and is not published to PyPI. To install
the latest `main` branch without changing Kaggle's preinstalled dependency
stack:

```bash
python -m pip install --no-build-isolation --no-deps \
  "commguard @ git+https://github.com/waqasm86/CommGuard.git@main"
```

For a reproducible experiment, replace `main` with a reviewed commit SHA.

On Kaggle, enable Internet access, select the `GPU T4 x2` accelerator, download
or import [`notebooks/commguard_dual_t4.ipynb`](notebooks/commguard_dual_t4.ipynb),
and run it from the first cell. The notebook clones this repository into
`/kaggle/working/CommGuard`, installs it with `--no-deps`, runs strict preflight
and a two-rank smoke test, then requires explicit opt-in before expensive
profiles.

If Kaggle Internet access is disabled, upload a snapshot of this repository as
a Kaggle dataset and change the notebook's `REPO` path to that read-only
dataset directory.

For the repetition-aware study, run these notebooks in order:

1. [`commguard_calibration_v2.ipynb`](notebooks/commguard_calibration_v2.ipynb)
2. [`commguard_benign_corpus.ipynb`](notebooks/commguard_benign_corpus.ipynb)
3. [`commguard_detector_evaluation.ipynb`](notebooks/commguard_detector_evaluation.ipynb)

Calibration can be `supported`, `partially_supported`, or `not_supported`.
Between Kaggle sessions, download each notebook's exported evidence archive and
upload it as a Kaggle Dataset for the next notebook. See the
[experiment roadmap](docs/NEXT_KAGGLE_EXPERIMENTS.md) for the artifact flow and
claim boundary.

## Local checkout

```bash
git clone https://github.com/waqasm86/CommGuard.git
cd CommGuard
python -m pip install --no-build-isolation --no-deps -e .
```

Equivalent shell commands:

```bash
python -m pip install --no-build-isolation --no-deps -e .
commguard preflight --strict --output artifacts
commguard run --profile smoke --output artifacts
commguard run --profile standard --repetitions 3 --output artifacts
# Inspect the matrix summary and continue only if primary_coverage_gate.passed is true.
commguard evaluate --input artifacts --output artifacts --minimum-runs-per-family 3
commguard report --input artifacts --output artifacts/report.md
```

The standard and extended experiment profiles are calibration-gated. They stop
when exactly two T4 GPUs, CUDA/NCCL, two distinct rank bindings, or responsive
PCIe readings cannot be demonstrated. There is no CPU, Gloo, or one-GPU
fallback for results labelled dual-GPU.

The standard profile is the bounded eight-family benign pilot. The extended
profile is opt-in and adds configuration variants plus optional peer copy. Both
write a corpus plan before execution and report per-family coverage before any
detector evaluation. See the [benign workload matrix](docs/benign-workload-matrix.md).

Defensive adversarial strategies are not part of those profiles. They require
an explicit human approval flag and a predeclared sealed holdout plan; no
adversarial result is bundled. See
[bounded adversarial research](docs/adversarial-research.md).

## Public API

```python
from commguard import (
    check_environment,
    evaluate_detector,
    extract_features,
    generate_report,
    list_workloads,
    load_artifact,
    run_experiment,
    run_matrix,
)
```

Core schema and artifact tests are offline and CPU-safe:

```bash
pytest -m "not gpu and not multigpu and not network and not slow"
```

See `docs/` for architecture, artifact contracts, methodology, safety, and
reproducibility details:

- [Kaggle dual-T4 instructions](docs/kaggle-dual-t4.md)
- [Next Kaggle experiments](docs/NEXT_KAGGLE_EXPERIMENTS.md)
- [Artifact contracts](docs/artifacts.md)
- [Methodology](docs/methodology.md)
- [Canonical versus executed notebooks](docs/notebook-policy.md)
- [Limitations and untested behavior](docs/limitations.md)
- [Reproducibility](docs/reproducibility.md)

Source, releases, and issues are hosted at
[github.com/waqasm86/CommGuard](https://github.com/waqasm86/CommGuard).
