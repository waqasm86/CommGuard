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
- Idle-aware three-repetition calibration at 1/4/16/64 MiB, with explicit
  current-versus-prior session labels and exact downstream artifact binding.
- Idempotent central acknowledgment-loss retries that do not double-ingest an
  identical authenticated batch, while changed content remains rejected.
- Numerically stable simple-rule probabilities for extreme detector logits.
- Evidence index, negative historical coverage analysis, report template, and
  complete citations/claim boundaries.
- A real two-rank idle calibration worker, safe preflight output preparation,
  accurate payload identities, one-sweep enforcement, standard-matrix
  validation, and non-destructive atomic export.

## Compatibility

- Core remains dependency-free and importable on CPU-only Python 3.10/3.11.
- Analysis, telemetry, and development dependencies remain optional extras.
- Historical schema-1 artifacts remain readable; raw evidence is never changed
  in place or reinterpreted as passing the modern calibration gate.

## Known limitations

- Completion-series CUDA/NCCL/NVML execution is pending dual-T4 hardware.
- Local analysis integration is skipped when scikit-learn is absent.
- Historical detector evidence is DDP versus idle only and does not pass the
  current 30-second eight-family gate.
- No adversarial or physical multi-node result is included.
- The first modern calibration-v3 T4 x2 attempt failed all idle launches because
  mode `idle` was missing and was correctly `not_supported`. It is debugging
  evidence only; no successful modern calibration or downstream run is included.
- Canonical notebooks remain unexecuted.
