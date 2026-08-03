# CommGuard Kaggle prototype completion ExecPlan

Living plan started: `2026-08-03` (Asia/Karachi)

## Objective and honest completion boundary

Complete and publish a reproducible prototype workflow for one Kaggle host with
exactly two NVIDIA Tesla T4 GPUs. Local completion covers SDK implementation,
CPU and synthetic validation, deterministic notebook generation, packaging,
artifact verification, documentation, Git publication, and a draft pull
request. New CUDA/NCCL/NVML measurements, scientific acceptance, detector
performance, adversarial outcomes, cross-session evidence, and physical
multi-node validation remain pending actual Kaggle execution.

The shared scope declaration will be authoritative in SDK metadata and reused
verbatim in README, notebooks, reports, and handoff material:

> CommGuard’s Kaggle workflow is a single-node, dual-NVIDIA-T4 research
> prototype. It validates experimental methodology and software behavior on
> two local GPU ranks. It does not establish generalization to two physical
> 8-GPU nodes, NVLink/NVSwitch fabrics, RoCE or InfiniBand networks, large
> frontier-model workloads, or production treaty-verification deployments.

## Initial repository assessment

- Base: `origin/main` at `5af57bc220cddd750a1931a3bd52c4da24f990a0`.
- Work branch: `codex/commguard-kaggle-prototype-completion`.
- The repository is already a substantial `0.2.0` prototype with schema-2
  provenance, strict coverage gates, grouped evaluation, bounded workloads, a
  two-agent local collector, deterministic canonical notebooks, and 169 prior
  CPU-safe passes recorded by the previous completion report.
- The worktree was not clean at branch creation. Preserved in-scope changes:
  removal of superseded `notebooks/commguard_benign_corpus.ipynb` and
  `notebooks/commguard_calibration_v2.ipynb`, an execution-stripped update to
  `notebooks/commguard_calibration_v3.ipynb`, and a new separated diagnostic
  notebook directory. No reset, clean, stash, or evidence rewrite was used.
- Immutable local evidence is ignored by Git under
  `artifacts/executed-notebooks/{historical,diagnostics}` and
  `tar-files/{historical,failed,diagnostics}`. It will be hash-audited but not
  modified or committed.
- The historical claim boundary remains negative/inconclusive: the legacy
  PCIe-only pilot had inadequate workload coverage and missed its held-out
  training run; calibration-v3 failed its idle gate because of the historical
  dispatch defect. Code readiness is not empirical validation.

## Exact completed file set

The completed branch changes these files relative to the recorded base. This
supersedes the narrower provisional list from the first implementation pass.

- Planning, publication, and delivery: this plan, `.gitignore`,
  `CODEX_KAGGLE_PROTOTYPE_COMPLETION_REPORT.md`, `CODEX_COMPLETION_REPORT.md`,
  `delivery/HANDOFF.md`, `delivery/KAGGLE_PROTOTYPE_HANDOFF.md`,
  `delivery/KAGGLE_PROTOTYPE_PR_BODY.md`, `delivery/PR_BODY.md`, and
  `delivery/RELEASE_NOTES.md`.
- Package metadata and overview: `pyproject.toml`, `README.md`, and
  `CHANGELOG.md`.
- SDK: `src/commguard/__init__.py`, `src/commguard/adversarial.py`,
  `src/commguard/calibration.py`, `src/commguard/cli.py`,
  `src/commguard/orchestrator.py`, `src/commguard/provenance.py`,
  `src/commguard/schemas.py`, `src/commguard/scope.py`,
  `src/commguard/workloads.py`, `src/commguard/artifacts/{__init__.py,prototype.py,storage.py}`,
  `src/commguard/central/{agent.py,security.py,server.py}`,
  `src/commguard/distributed/worker.py`,
  `src/commguard/environment/preflight.py`,
  `src/commguard/evaluation/{__init__.py,baselines.py,splits.py}`,
  `src/commguard/features/extraction.py`, and
  `src/commguard/telemetry/{__init__.py,nvml.py,schema.py}`.
- Notebook tooling and registry: `tools/generate_canonical_notebooks.py`,
  `tools/verify_delivery.py`, `notebooks/canonical_notebooks.json`, the four
  registered canonical notebooks, and
  `notebooks/diagnostics/commguard_calibration_v4_sampling_study.ipynb`.
- Superseded active notebook sources removed after provenance review:
  `notebooks/commguard_benign_corpus.ipynb`,
  `notebooks/commguard_calibration_v2.ipynb`,
  `notebooks/commguard_detector_evaluation.ipynb`,
  `notebooks/commguard_dual_t4.ipynb`, and
  `notebooks/commguard_dual_t4_research.ipynb`.
- Tests: `tests/test_artifacts.py`, `tests/test_benign_matrix.py`,
  `tests/test_calibration_provenance.py`,
  `tests/test_calibration_repeatability.py`,
  `tests/test_central_monitoring.py`, `tests/test_cli.py`,
  `tests/test_coverage.py`, `tests/test_evaluation_grouping.py`,
  `tests/test_feature_extraction.py`, `tests/test_features.py`,
  `tests/test_gpu_integration.py`, `tests/test_prototype_packages.py`,
  `tests/test_repository.py`, `tests/test_schemas.py`, `tests/test_scope.py`,
  and `tests/test_telemetry.py`.
- Documentation and report: `docs/KAGGLE_PROTOTYPE_RUNBOOK.md`,
  `docs/NEXT_KAGGLE_EXPERIMENTS.md`, `docs/acceptance-status.md`,
  `docs/adversarial-research.md`, `docs/artifact-schemas.md`,
  `docs/benign-workload-matrix.md`, `docs/central-monitoring-design.md`,
  `docs/current-results.md`, `docs/evidence-index.md`,
  `docs/kaggle-dual-t4.md`, `docs/limitations.md`, `docs/methodology.md`,
  `docs/notebook-policy.md`, `docs/nvidia-calibration.md`,
  `docs/reproducibility.md`, and
  `reports/commguard-kaggle-prototype-report.md`.

## Notebook migration strategy

1. Hash and inspect every historical/diagnostic executed notebook before any
   source migration.
2. Preserve executed hyphen-named notebooks byte-for-byte in ignored evidence
   directories; never derive canonical claims by rewriting them.
3. Keep exactly four generated, underscore-named canonical notebooks directly
   under `notebooks/`; keep the v4 sampling study under `notebooks/diagnostics/`.
4. Verify superseded tracked source roles and Git history before removal. Do
   not mislabel an unexecuted legacy source as executed evidence.
5. Enhance the generator instead of hand-editing large notebook JSON. Generated
   notebooks must be deterministic, output-free, parameterized, smoke/full
   aware, strict about two T4s, dirty state, archive inputs, and final status.
6. Extend `canonical_notebooks.json` to encode policy version, exact order, and
   roles; validate it in tests and delivery scanning.

## SDK implementation work

- Establish one importable scope declaration and structured scope metadata.
- Compare existing behavior with every required calibration, duration,
  telemetry, corpus-resume, feature-set, grouped-evaluation, abstention,
  periodic-sync, collector-window, and archive-integrity criterion.
- Extend existing abstractions only for demonstrated gaps; do not create
  notebook-only or parallel implementations.
- Preserve launch completion separately from scientific acceptance and preserve
  all unsupported/inconclusive states.
- Keep core imports CPU-safe and make hardware tests skip honestly.

## Test strategy

- Start with a baseline CPU-safe suite and gap-specific static audits.
- Add regression tests for every schema/API behavior changed, especially scope
  metadata, notebook inventory/parameters/output stripping, sampling interval
  validation, calibration states, duration/UUID/rank acceptance, resumability,
  feature robustness and sets, grouped control handling, abstention, periodic
  synchronization, central deduplication/incomplete windows, artifact checksums,
  non-destructive export, dirty-tree policy, and real-checkout delivery scans.
- Per commit: `.venv/bin/python -m pytest -m 'not gpu and not multigpu'`,
  `.venv/bin/python -m ruff check .`,
  `.venv/bin/python -m ruff format --check .`, and
  `.venv/bin/python -m build`.
- Final requested aliases also run `python3.11 -m pytest -q`, Ruff, build, and
  `tools/verify_delivery.py`, with skips enumerated from pytest output.

## Kaggle-only validation boundaries

- This Ubuntu host is not assumed to expose two T4s. CPU and synthetic tests do
  not validate CUDA, NCCL, NVML support, detector accuracy, adversarial evasion,
  utility, or empirical acceptance.
- The first required Kaggle action is full/smoke execution of
  `notebooks/commguard_calibration_v3.ipynb` from a reviewed remote-visible
  commit. Downstream notebooks remain gated by exact archive hashes and
  scientific acceptance.
- No physical multi-node, NVLink/NVSwitch, RoCE, InfiniBand, or production claim
  is permitted.

## Documentation updates

- Make SDK scope metadata the single source of truth and quote it prominently.
- Create the full Kaggle runbook and prototype report skeleton with explicit
  implementation/evidence status labels and artifact-bound placeholders.
- Reconcile the named result, acceptance, evidence, limitation, methodology,
  central, adversarial, reproducibility, notebook-policy, README, completion,
  and handoff documents without upgrading historical evidence.

## Commit plan and actual commits

1. `9708d2e` — organize canonical Kaggle notebook workflow and preserve/hash
   evidence.
2. `4f60a08` — complete the calibration, corpus, features, grouped evaluation,
   bounded red-team, collector, artifact, CLI, and notebook SDK workflows.
3. `9401ac4` — add the Kaggle runbook, report, reconciled documentation,
   handoff, and draft PR body.
4. Final report-only commit — record the completion report and final living-plan
   state, rerun the repository gate, push, and verify local/remote equality.

Every implementation/documentation commit received the repository-required
test/lint/format/build gate. No history was rewritten and no force-push was
used.

## Known limitations

- No new Kaggle run is available in this local session.
- Historical feature coverage is too narrow for the current detector claim.
- The failed calibration-v3 archive is debugging evidence, not telemetry
  falsification.
- NVML PCIe counters are capability-dependent and are neither direct NCCL-byte
  counts nor physical network-interface measurements.
- One-host, two-rank local collector behavior does not validate secure or
  clock-synchronized multi-node operation.
- Python 3.10 and optional-analysis availability will be reported from actual
  local/CI capability rather than assumed.

## Final acceptance checklist

- [x] Dirty baseline and all evidence hashes recorded; evidence unchanged.
- [x] Four canonical notebooks plus one diagnostic source are the only active
      workflow notebooks; all canonical cells are unexecuted.
- [x] Notebook manifest has exact roles, order, and policy version.
- [x] Shared scope declaration and structured fields are used throughout.
- [x] Demonstrated SDK gaps are implemented with focused tests.
- [x] CPU/unit suite passes; GPU skips are explained.
- [x] Ruff check and format-check pass.
- [x] Wheel and sdist build pass and contain no forbidden evidence.
- [x] Delivery, secret/path, notebook, archive, and diff audits pass.
- [x] Kaggle runbook, research report, completion report, handoff, and PR body
      are complete and distinguish code readiness from empirical validation.
- [x] Logical implementation commits are pushed to
      `origin/codex/commguard-kaggle-prototype-completion`.
- [x] Draft PR to `main` is open at
      `https://github.com/waqasm86/CommGuard/pull/3`.
- [x] Final report commit is pushed; local and remote branch SHAs match and the
      working tree is clean.

## Progress log

- [x] `2026-08-03` Read the repository instructions, prior living plan, and
  William Fowler research brief; inventoried all requested areas; verified GitHub
  CLI authentication; fetched `origin`; and created the required branch from
  exact `origin/main` while preserving the dirty notebook boundary.
- [x] `2026-08-03` Baseline validation recorded 167 passes, one optional
  scikit-learn skip, six hardware deselections, and two notebook-generator
  mismatches caused by the preserved executed calibration copy. The host has
  one NVIDIA GeForce 940M, not two T4s.
- [x] `2026-08-03` Removed the five superseded tracked notebook sources from
  the active directory after confirming Git-history preservation, registered
  exact canonical order/roles under policy v2, separated and stripped the v4
  diagnostic source, and restored deterministic generation. The checkpoint
  gate passed with 170 CPU-safe tests, one optional-analysis skip, six hardware
  deselections, Ruff check/format, build, delivery scan, and evidence hashes.
- [x] `2026-08-04` Completed the requirement-gap audit and SDK implementation:
  shared scope metadata, current telemetry fields and unit conversion metadata,
  duration/cadence-aware five-state calibration, exact-match corpus resume,
  complete-run feature sets, grouped binary/secondary multiclass evaluation,
  GroupKFold/LeaveOneGroupOut plans, periodic-sync tradeoffs, two same-node
  collector agents, verified deterministic archive export, and focused tests.
- [x] `2026-08-04` Regenerated all four policy-v3 canonical notebooks with
  package-source priority, dirty-source smoke enforcement, smoke/full modes,
  strict two-T4 gates, machine run status, stage materializers, and verified
  prototype archive names. Static notebook compilation and delivery scans pass.
- [x] `2026-08-04` Added the Kaggle runbook, research report skeleton, scope and
  evidence reconciliation, workload labels, handoff, and draft PR text without
  adding empirical claims.
- [x] `2026-08-04` Ran the final requested suite: system Python reported 182
  passed and eight optional/hardware skips; the declared-extra venv reported
  185 passed and five strict dual-T4 skips; Ruff check/format, wheel/sdist
  build, canonical generation check, delivery verification, notebook audit,
  package-content inspection, and immutable-evidence hash audit all passed.
- [x] `2026-08-04` Published three logical commits and opened draft PR #3.
- [x] `2026-08-04` Published completion-report commit `0bd2b9a`, verified its
  local SHA, remote SHA, and draft-PR head were identical, and observed a clean
  worktree. This final plan-only bookkeeping commit is followed by the same
  publication verification in the session handoff.
