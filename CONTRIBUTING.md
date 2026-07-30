# Contributing

Changes must preserve offline CPU-safe core tests, explicit unsupported-field
semantics, whole-run split grouping, create-only raw evidence, strict dual-GPU
validation, and limited claims. GPU, multi-GPU, network, and slow tests must use
their pytest markers. Never add a required dependency that replaces Kaggle's
CUDA/PyTorch stack.

Generated quantitative claims must trace to validated artifacts. Do not add
SPAR application responses or prose intended for application forms.

## Development checks

```bash
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install "pytest>=8" "ruff>=0.9"
pytest -m "not gpu and not multigpu and not network and not slow"
ruff check src tests
```

GPU-specific changes require separate evidence from a compatible dual-T4
environment. CPU tests cannot be used to claim CUDA, NCCL, or NVML behavior.
