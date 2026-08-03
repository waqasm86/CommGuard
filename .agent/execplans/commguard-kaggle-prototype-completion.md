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

> CommGuard's Kaggle workflow is a single-node, dual-NVIDIA-T4 research
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

## Exact files currently expected to change

This list is authoritative for the first implementation pass and will be
updated before staging if the audit identifies a narrower or additional file.

- Planning/delivery: this plan,
  `CODEX_KAGGLE_PROTOTYPE_COMPLETION_REPORT.md`,
  `delivery/KAGGLE_PROTOTYPE_HANDOFF.md`, and
  `delivery/KAGGLE_PROTOTYPE_PR_BODY.md`.
- Scope/provenance/artifacts: `src/commguard/scope.py`,
  `src/commguard/provenance.py`, `src/commguard/schemas.py`,
  `src/commguard/artifacts/models.py`, `src/commguard/artifacts/storage.py`,
  and their focused tests.
- Telemetry/calibration: `src/commguard/telemetry/schema.py`,
  `src/commguard/telemetry/nvml.py`, `src/commguard/calibration.py`,
  `src/commguard/duration.py`, and focused telemetry/calibration tests.
- Corpus/features/evaluation/adversarial/central only where the gap audit
  demonstrates a missing acceptance criterion: existing modules under
  `src/commguard/{corpus.py,features/,evaluation/,adversarial.py,central/}` and
  corresponding tests.
- CLI/delivery: `src/commguard/cli.py`, `tools/verify_delivery.py`,
  `.gitignore`, and focused CLI/repository/artifact tests.
- Notebooks: `tools/generate_canonical_notebooks.py`,
  `notebooks/canonical_notebooks.json`, the four canonical notebooks, and
  `notebooks/diagnostics/commguard_calibration_v4_sampling_study.ipynb`.
  Superseded sources will be removed only after provenance and immutable
  historical equivalents are verified.
- Documentation: `README.md`, `CHANGELOG.md`,
  `docs/KAGGLE_PROTOTYPE_RUNBOOK.md`, `docs/current-results.md`,
  `docs/acceptance-status.md`, `docs/evidence-index.md`,
  `docs/limitations.md`, `docs/methodology.md`, `docs/kaggle-dual-t4.md`,
  `docs/NEXT_KAGGLE_EXPERIMENTS.md`,
  `docs/central-monitoring-design.md`, `docs/adversarial-research.md`,
  `docs/reproducibility.md`, `docs/notebook-policy.md`, and
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

## Commit plan

1. Organize canonical notebook sources and preserve/hash evidence.
2. Add shared prototype scope and close demonstrated SDK/artifact gaps.
3. Harden canonical notebook generation and its policy tests.
4. Add Kaggle runbook, report, and reconciled research documentation.
5. Add final completion/handoff/PR materials and recorded validation.

Commits will be split further only when the actual diff has an independently
reviewable dependency boundary. Every commit receives the repository-required
test/lint/format/build gate. No history rewrite or force-push is allowed.

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

- [ ] Dirty baseline and all evidence hashes recorded; evidence unchanged.
- [ ] Four canonical notebooks plus one diagnostic source are the only active
      workflow notebooks; all canonical cells are unexecuted.
- [ ] Notebook manifest has exact roles, order, and policy version.
- [ ] Shared scope declaration and structured fields are used throughout.
- [ ] Demonstrated SDK gaps are implemented with focused tests.
- [ ] CPU/unit suite passes; GPU skips are explained.
- [ ] Ruff check and format-check pass.
- [ ] Wheel and sdist build pass and contain no forbidden evidence.
- [ ] Delivery, secret/path, notebook, archive, and diff audits pass.
- [ ] Kaggle runbook, research report, completion report, handoff, and PR body
      are complete and distinguish code readiness from empirical validation.
- [ ] Logical commits are pushed to
      `origin/codex/commguard-kaggle-prototype-completion`.
- [ ] Draft PR to `main` is open.
- [ ] Local and remote branch SHAs match and the working tree is clean.

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
- [ ] Complete remaining requirement-gap audit and implementation.
- [ ] Complete implementation and notebook migration.
- [ ] Complete documentation and delivery materials.
- [ ] Complete final validation, commits, push, and draft PR.
