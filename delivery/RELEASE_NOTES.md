# CommGuard 0.2.0 candidate release notes

CommGuard 0.2.0 is a research-completion SDK release candidate. It does not
ship a validated detector or new GPU results.

## Highlights

- Version-2 provenance, run, corpus, feature, coverage, split, and evaluation
  contracts with backward-compatible, non-destructive legacy loading.
- Create-only artifacts plus hash-verified, traversal/link-rejecting archive
  restoration.
- Duration-aware eight-family benign corpus planning and strict coverage before
  metrics.
- Leakage-resistant whole-run/session evaluation with honest small-sample
  warnings and communication-only primary results.
- Content-agnostic signed central-node protocol and CPU-tested local two-agent
  simulation.
- Eight bounded defensive adversarial strategies, frozen benign-only scoring,
  cost proxies, and a sealed family/session/configuration holdout.
- Four deterministic, output-free Kaggle notebooks pinned to clean
  remote-visible commits and immutable archive hashes.
- Evidence index, negative historical coverage analysis, report template, and
  complete citations/claim boundaries.

## Compatibility

- Core remains dependency-free and importable on CPU-only Python 3.10/3.11.
- Analysis, telemetry, and development dependencies remain optional extras.
- Historical schema-1 artifacts remain readable; raw evidence is never changed
  in place.

## Known limitations

- Completion-series CUDA/NCCL/NVML execution is pending dual-T4 hardware.
- Local analysis integration is skipped when scikit-learn is absent.
- Historical detector evidence is DDP versus idle only and does not pass the
  current 30-second eight-family gate.
- No adversarial or physical multi-node result is included.
