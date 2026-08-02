# CommGuard research completion ExecPlan

Living plan started: `2026-08-02T12:50:25+05:00`

## Purpose and success definition

Advance the authoritative CommGuard checkout into a truthful, CPU-testable research platform for communication-correlated telemetry studies. Completion means that repository state, schemas, coverage diagnostics, grouped evaluation, local central-monitoring simulation, bounded benign/adversarial workload orchestration, canonical Kaggle notebooks, evidence-grounded documentation, and GitHub handoff materials satisfy the prompt-package gates to the extent possible on this Ubuntu host. GPU, Kaggle, and physical multi-node claims remain pending until actual immutable artifacts exist.

Observable success:

- a coherent underscore-named, output-free canonical notebook set and preserved executed evidence hashes;
- dependency-free CPU import, green CPU-safe tests, green Ruff check/format, coherent build/import/CLI checks;
- explicit session/corpus/run/node/environment semantics with legacy ambiguity surfaced;
- one structured coverage record per planned run, timestamp-aligned cross-GPU features, and a strict primary coverage gate;
- leak-resistant split selection with actual strategy metadata and train-only preprocessing/model selection;
- signed/versioned node telemetry batches, replay/staleness controls, offline two-agent integration tests, and optional online reference code without a physical-deployment claim;
- duration-aware benign and bounded adversarial workload definitions with CPU orchestration/correctness tests;
- canonical unexecuted calibration, benign, detector, and adversarial Kaggle notebooks using SDK APIs and immutable provenance gates;
- an evidence index, research/report templates, current-results disclosure, completion report, and exact non-pushing GitHub handoff commands.

## Baseline

The full timestamped baseline is `.agent/state/20260802T125025+0500/baseline.md`.

The authoritative resumption audit is
`.agent/state/20260802T084010Z/baseline.md`. It records the state after the
Phase 01 commit and before continuing the partial Phase 02 work.

- Git: `main` at `1e790895ae3bd0919dde0f383e78fb14f767d359`, tracking `origin/main`.
- Pre-existing worktree: four tracked notebook deletions, three untracked notebook replacements, and five untracked archives.
- CPU tests: 56 passed, one repository notebook-path test failed, five hardware tests deselected.
- Ruff: `ruff check src tests` passes; root check and formatting fail, substantially because live executed notebooks are included and tracked Python predates the current Ruff formatter.
- Build: unavailable because the local environment lacks the `build` module.
- Evidence: real dual-T4 calibration and 18-run benign collection exist, but primary derived coverage is DDP-versus-idle only. PCIe-only detection missed the held-out training run.

Current resumption state:

- Git: `codex/commguard-research-completion` at `c920a738c1610eaf37e1ab3ae8912d2ac41db287`.
- Worktree: modified `provenance.py` and `schemas.py`, plus new `corpus.py`; these pre-existing partial Phase 02 changes are hash-recorded in the resumption inventory.
- CPU tests: 57 passed and five hardware tests deselected.
- Ruff check: passed. Ruff format: only the partial Phase 02 files require formatting.
- Build: passed in a fresh temporary output directory using the now-available project-virtualenv `build 1.5.0`.
- Canonical notebook and archive hashes remain unchanged from the Phase 01 boundary.

## Source-grounded research constraints

The current working tree and local evidence outrank the completion package. The intended project is defensive verification research: implement content-agnostic communication telemetry, test broad benign training/inference/hard negatives, then evaluate bounded adversarial structures. Current evidence is a single-host dual-T4 pilot, not a two-node server-grade validation.

Every result statement will use one of: measured, derived, observed limitation, inference, hypothesis, planned, or pending hardware validation. No text may claim reliable LLM-training detection, adversarial robustness, faithful DiLoCo reproduction, proven privacy, or physical multi-node/server-grade validation. Application answers remain for the applicant to write independently.

Telemetry must not contain prompts, examples, tokens, model weights, credentials, or unrelated user data. NVML PCIe readings remain explicitly distinct from direct NCCL byte counts and from GPU-attributed network traffic.

## Milestones

### Phase 00 — Preflight, state preservation, and ExecPlan

Acceptance: live Git/tool/notebook/archive/evidence state recorded; every uncommitted path accounted for; CPU baseline captured; plan and timestamped inventory exist; safe feature branch created without discarding changes.

### Phase 01 — Repository repair and canonical notebooks

Acceptance: root `AGENTS.md`; documented underscore naming; output-free canonical notebooks; executed originals preserved as local evidence with hashes; `.gitignore`, metadata, CLI/import/build policy, and repository tests coherent; CPU tests green.

### Phase 02 — Provenance, corpus membership, and schema migration

Acceptance: explicit experiment session, corpus/collection, run, node, environment, source, dirty-state, notebook, input hash, and seed semantics; corpus manifest allow-list; calibration leakage prevention; tested legacy loader with ambiguous-grouping flag; schema/migration docs.

### Phase 03 — Feature coverage, time alignment, and duration controls

Acceptance: extraction result contains features plus one coverage record per planned run; all required reason codes; exact duration/row/window diagnostics; deterministic tolerance-based timestamp pairing; strict family/minimum-run primary gate; diagnostic short windows remain separate; too-short and leakage regressions tested; duration-aware stop conditions and measured interval provenance.

### Phase 04 — Grouped splits and honest evaluation

Acceptance: validated session → deterministic stratified whole-run hierarchy plus family/config holdouts; no one-class primary split; whole runs never cross splits; actual strategy/groups persisted; train-only imputation/scaling/selection/calibration; communication-only primary metrics with run/window/per-family counts, hard-negative FPRs, abstention, and small-sample warnings.

### Phase 05 — Node agent and central monitoring reference architecture

Acceptance: versioned batches/heartbeats/acks/decisions/errors; HMAC/integrity, payload limit, replay/order/clock/staleness checks; privacy allow-list; offline deterministic transport; local two-agent central aggregation test; optional online transport remains isolated; physical deployment labeled pending.

### Phase 06 — Broad benign workload matrix

Acceptance: stable family/config IDs for DDP, independent prefill/decode, synchronized inference, compute, host transfer, model/checkpoint load, optional peer copy, and idle; duration-aware plans; CPU orchestration smoke tests; T4-safe pilot and opt-in expansion; coverage gate precedes detector metrics.

### Phase 07 — Adversarial workloads and held-out-family evaluation

Acceptance: bounded metadata/interface and code paths for gradient accumulation, periodic local SGD, DiLoCo-inspired sparse synchronization, segmentation, idle/burst shaping, randomized synchronization, mixed phases, and synthetic decoys; disabled by default; sync/state correctness tests; efficiency/quality proxy schema; untouched final holdout/hardening round definitions.

### Phase 08 — Kaggle notebooks

Acceptance: four canonical unexecuted notebooks parse, contain no SDK implementation duplication, have strict two-T4/environment evidence, immutable source/archive checks, safe extraction, deterministic session/corpus provenance, initial pilot, opt-in expensive matrices, coverage gate, adversarial acceptance gate, final archive hash/summary, and “not executed” result sections.

### Phase 09 — Evidence, reports, and docs

Acceptance: archive/evidence index with hashes and supported claims; exact current results and coverage failure analysis; architecture/methodology/schema/limitations/reproducibility/Kaggle/safety docs updated; central/benign/adversarial plans and research report template; prominent README negative result; no unsupported application answer.

### Phase 10 — GitHub-ready delivery without push

Acceptance: all available CPU, lint, format, build, import, CLI, notebook, secret/path, diff, and size checks recorded; status explained; no archives/caches/secrets/output notebooks staged; completion report, change summary, PR body, release notes/checklist, exact Kaggle/user commands, and push command prepared; no automatic push.

## Progress log

- [x] `2026-08-02` Re-verified clean reviewed commit `0e3b72c8` and created
  `codex/fix-calibration-idle-and-kaggle-workflow` without rewriting history or
  modifying historical evidence.
- [x] `2026-08-02` Reproduced the calibration-v3 failure: `calibration_idle`
  resolved to unsupported worker mode `idle`, so every idle launch raised
  `ValueError: unknown mode 'idle'`.
- [x] `2026-08-02` Implemented and CPU-tested the explicit two-rank idle
  lifecycle, measured-interval heartbeats, participation/cleanup evidence, and
  exclusion of setup/teardown collectives from idle PCIe summaries.
- [x] `2026-08-02` Made preflight create/validate output directories, corrected
  per-payload calibration identities, added create-only one-sweep markers and
  exact 15-run validation, and hardened create-only atomic archive export.
- [x] `2026-08-02` Regenerated/audited all four canonical notebooks, aligned the
  failed-run claim boundary, and passed the full pre-commit gate: 169 CPU-safe
  tests passed, one optional scikit-learn test skipped, six hardware tests were
  deselected, Ruff check/format passed, and the 0.2.0 wheel/sdist built.
- [x] `2026-08-02` Completed fresh-wheel, delivery/path,
  distribution-content, evidence-hash, notebook, and final Git checks. Version
  0.2.0 imported from an isolated environment's `site-packages`; CLI help and
  `pip check` passed; wheel/sdist contents were clean; all eight indexed
  historical byte sizes and SHA-256 hashes matched; canonical notebooks were
  deterministic and output-free; `git diff --check` and strict `git fsck`
  passed (with only pre-existing unreachable blobs reported).
- [ ] `2026-08-02` Push only
  `codex/fix-calibration-idle-and-kaggle-workflow`, confirm its upstream/remote
  URL, and leave merge/Kaggle execution to the human gates.

- [x] `2026-08-02` Re-audited the authoritative checkout at completion commit
  `5ac130c322df7c1c9c359ae2d568d515c153d1b0`, verified the Git remote/history,
  created `codex/commguard-pre-kaggle-audit-fixes`, and preserved all immutable
  historical evidence.
- [x] `2026-08-02` Removed tracked `.agent/state/` captures from the public
  branch, ignored future state captures, and made the delivery scanner reject
  personal home/media paths consistently without a directory exemption.
- [x] `2026-08-02` Added schema-2 idle-aware repeated calibration and exact
  calibration references through corpus extraction, evaluation, and reporting;
  schema-1 evidence remains explicitly legacy and is not upgraded to a modern
  gate pass.
- [x] `2026-08-02` Made exact authenticated central batch retries idempotent,
  rejected altered content under reused message IDs, and stabilized sigmoid
  probabilities for extreme logits.
- [x] `2026-08-02` Regenerated all four canonical notebooks with exact artifact
  chaining and the final pushed-commit placeholder, strengthened public claim
  checks, and retained the adversarial human-authorization gate.
- [x] `2026-08-02` Final pre-Kaggle local validation: 147 CPU-safe tests passed,
  one optional scikit-learn integration skipped, five hardware tests were
  deselected, Ruff check/format passed, deterministic notebook regeneration and
  delivery scans passed, clean wheel/sdist builds passed, and a fresh isolated
  wheel install imported version 0.2.0 and passed CLI/dependency checks. All
  eight indexed historical evidence hashes and byte counts remained unchanged.

- [x] `2026-08-02T12:50:25+05:00` Read the completion package in the required order: master prompt, package `AGENTS.md`, plan rules, seven context files, acceptance YAML, experiment CSV, and phases 00–10.
- [x] `2026-08-02T12:50:25+05:00` Confirmed live path/Git root, branch, HEAD, remotes, history, status, diff, and untracked paths.
- [x] `2026-08-02T12:50:25+05:00` Audited repository configuration, docs, source, tests, notebook sources/metadata, archive member names/hashes, and extracted evidence manifests/summaries without modifying evidence.
- [x] `2026-08-02T12:50:25+05:00` Ran CPU test/lint/format/build baseline; exact outcomes are recorded in the state inventory.
- [x] `2026-08-02T12:50:25+05:00` Created the timestamped state inventory and this living ExecPlan.
- [x] `2026-08-02T12:54:00+05:00` Created and switched to `codex/commguard-research-completion`; the pre-existing notebook/archive worktree state was preserved.
- [x] `2026-08-02T13:05:00+05:00` Phase 01 restored the three missing tracked canonical sources, retained the byte-identical shorter research-notebook rename, added root instructions and notebook policy/inventory, ignored local executed evidence/archives without deleting them, repaired CI/metadata/tests, and formatted tracked Python.
- [x] `2026-08-02T13:18:00+05:00` Phase 01 acceptance: repository tests 5 passed; CPU-safe suite 57 passed/5 deselected; Ruff check and format passed; canonical notebooks are unexecuted; executed hashes remained unchanged; wheel/sdist, import, CLI, and `git diff --check` passed.
- [x] `2026-08-02T13:40:10+05:00` Re-read the completion package in the required order and independently re-audited the authoritative checkout, all authored source/tests/docs/notebook sources, executed-notebook evidence, and archive manifests/summaries before resuming implementation.
- [x] `2026-08-02T13:40:10+05:00` Recorded the current feature-branch/HEAD/tool/check state and the exact three-file partial Phase 02 worktree boundary in `.agent/state/20260802T084010Z/baseline.md`.
- [x] `2026-08-02T13:51:00+05:00` Phase 02 completed: added v2 provenance/corpus contracts, true session/collection/corpus/node propagation through orchestration, stable capability-based environment fingerprints, dirty-source/input/notebook provenance, and non-destructive legacy ambiguity mapping.
- [x] `2026-08-02T13:51:00+05:00` Phase 02 acceptance: 65 CPU-safe tests passed/5 deselected; Ruff check/format, temporary-output wheel/sdist build, import/public API, CLI help, and `git diff --check` passed. New tests cover shared-session/unique-run IDs, separate sessions in one corpus, exact designation-aware allow-lists, actual matrix context reuse, v1 migration, v2 required fields, and stable environment hashing.
- [x] `2026-08-02T13:53:00+05:00` Phase 03 started from clean commit `0e8b505`; implement coverage records/results, corpus-only selection, timestamp alignment, strict gates, and duration-controlled execution before running acceptance checks.
- [x] `2026-08-02T14:11:37+05:00` Phase 03 completed: declared-corpus extraction now emits linked v2 feature, coverage, and summary artifacts with one diagnostic per plan; cross-GPU values use deterministic tolerance-based timestamp pairs; actual rank measurement intersections bound new feature windows; and the 30-second primary gate cannot be silently replaced by diagnostic short windows.
- [x] `2026-08-02T14:11:37+05:00` Phase 03 acceptance: 79 CPU-safe tests passed/5 deselected; Ruff check/format, temporary-output wheel/sdist build, import, CLI help, and `git diff --check` passed. Regressions cover exact 4.4-second post-warmup exclusion, calibration-idle leakage, sampling gaps, missing plans, timestamp skew, incomplete evaluation refusal, primary/diagnostic separation, duration cap failure, and measured-interval schema/participation validation.
- [x] `2026-08-02T14:13:00+05:00` Phase 04 started from clean commit `3163275`; replace environment-fingerprint grouping, validate the split hierarchy, and separate train/validation selection from final test reporting.
- [x] `2026-08-02T14:25:52+05:00` Phase 04 completed CPU-safe implementation: true-session-aware split plans persist actual strategies and exact groups; deterministic fallback preserves whole runs, classes, and required families; family/config holdouts are explicit diagnostics; primary reporting is 30-second communication-only; and model/threshold selection accepts validation data only.
- [x] `2026-08-02T14:25:52+05:00` Phase 04 acceptance: 89 CPU-safe tests passed, one optional analysis integration test skipped, and five hardware tests were deselected; Ruff check/format passed. The project virtualenv lacks the declared `analysis` extra (`pandas`, NumPy, scikit-learn), so no dependency was installed and no detector metrics were fabricated; dependency-free tests cover the split hierarchy, one-class refusal, legacy ambiguity, old per-run environment-fingerprint regression, family/config modes, selection isolation, schema, and small-group interval suppression.
- [x] `2026-08-02T14:28:00+05:00` Phase 05 started from clean commit `ea48109`; implement the signed, privacy-bounded offline reference path before optional online adapters.
- [x] `2026-08-02T14:35:56+05:00` Phase 05 completed: added versioned batch/heartbeat/ack/decision/error messages, canonical HMAC, exact sample allow-lists, payload/clock/sequence/replay/chain validation, bounded node agents with retry buffers, create-only offline logs, cross-node UTC aggregation, node health, forced abstention on incomplete windows, and optional HTTPS client/handler references.
- [x] `2026-08-02T14:35:56+05:00` Phase 05 acceptance: 94 CPU-safe tests passed, one optional analysis test skipped, and five hardware tests were deselected; Ruff check/format, CPU import, and diff checks passed. The five central tests include a local two-agent success path plus replay, stale/gapped order, invalid signature/chain, oversized payload, unknown protocol, privacy rejection, clock skew, buffered retry, node loss, and abstention. No socket/TLS/physical multi-node execution is claimed.
- [x] `2026-08-02T14:49:32+05:00` Phase 06 completed CPU-safe implementation: canonicalized all eight required benign family IDs; added explicit configuration IDs, a bounded 24-run standard pilot and opt-in expanded variants; wrote corpus plans before calibration/execution and separate finalized allow-lists; and made matrix summaries extract and report per-family primary coverage without detector metrics.
- [x] `2026-08-02T14:49:32+05:00` Phase 06 acceptance: 101 CPU-safe tests passed, one optional analysis integration test skipped, and five hardware tests were deselected. CPU planning tests cover every required worker mode, deterministic unique slots, duration/T4 bounds, stable configuration identity, opt-in variation, pre-execution manifest timing, finalized membership, incomplete evidence, and canonical/legacy DDP efficiency baselines. Canonical benign/quickstart notebook cells parse, remain unexecuted, default GPU work off, and use the coverage-aware SDK matrix path. No GPU workload was executed or claimed.
- [x] `2026-08-02T14:49:32+05:00` Phase 06 commit gate: required Ruff check and format-check passed; wheel and sdist built successfully with `.venv/bin/python -m build`; CPU import, CLI estimates, changed-notebook compilation, canonical notebook JSON/output policy, and `git diff --check` passed.
- [x] `2026-08-02T15:20:52+05:00` Phase 07 completed CPU-safe implementation: added a common defensive strategy contract and bounded code paths for gradient accumulation, periodic local SGD, DiLoCo-inspired sparse averaging, separately launched short segments, idle shaping, seeded randomized synchronization, mixed training/inference, and a synthetic communication decoy. Normal profiles remain disjoint and approval is checked before preflight/artifact creation.
- [x] `2026-08-02T15:20:52+05:00` Phase 07 acceptance: 115 CPU-safe tests passed, one optional analysis integration test skipped, and five hardware tests were deselected. Tests cover bounds/identity override rejection, deterministic rank-shared schedules, scalar parameter divergence/re-agreement, real-process segment planning and restart gaps, per-rank manifest sync/proxy/throughput/loss/memory truthfulness, explicit combined designation extraction, benign-only fitting partitions, evasion versus decoy metrics, report wording, and sealed family/session/config identities. No adversarial GPU path, detector score, or final holdout was executed.
- [x] `2026-08-02T15:20:52+05:00` Phase 07 commit gate: required Ruff check and format-check passed; wheel and sdist built successfully with `.venv/bin/python -m build`; CPU import/public API, CLI approval help, normal/adversarial profile separation, and `git diff --check` passed.
- [x] `2026-08-02T15:48:00+05:00` Phase 08 completed: added the four required deterministic, output-free canonical notebooks and generator; each notebook pins a pushed detached commit, rejects dirty source, verifies/restores exact archive hashes safely, records one provenance context, performs strict dual-T4 preflight, and ends with archive/hash handoff instructions. Calibration and benign pilots are bounded defaults; the 24-run benign corpus and expanded work are explicit opt-ins; detector fitting follows a printed strict coverage table; adversarial execution and evaluation remain off pending benign acceptance and human approval.
- [x] `2026-08-02T15:48:00+05:00` Phase 08 acceptance: 124 CPU-safe tests passed, one optional scikit-learn integration test skipped, and five hardware tests were deselected. Ruff check/format, wheel/sdist build, CPU import, CLI help, notebook regeneration/JSON/cell compilation/output/source-policy checks, and `git diff --check` passed. Safe archive regressions cover hash mismatch, create-only restore, traversal, links, and exact extraction-summary selection. Mocked CPU orchestration proves the adversarial corpus plan exists before ten development/hardening launches while the declared DiLoCo-inspired final family/config remains sealed. Historical executed notebook hashes stayed unchanged; no GPU, Kaggle, detector, or adversarial result was produced.
- [x] `2026-08-02T16:24:00+05:00` Phase 09 completed: added an immutable evidence index with archive/notebook hashes, source identities, statuses, and claim boundaries; exact current-results/coverage-failure analysis; a populated research-report template with corpus, split, per-family, and adversarial-cost tables; applicant technical evidence notes that explicitly avoid application prose; and complete research/software attribution. README now leads with the PCIe-only negative result and the four-notebook immutable workflow. Architecture, methodology, schemas, artifacts, central deployment assumptions, Kaggle, reproducibility, safety, limitations, acceptance status, and changelog are aligned.
- [x] `2026-08-02T16:24:00+05:00` Phase 09 acceptance: 128 CPU-safe tests passed, one optional scikit-learn integration test skipped, and five hardware tests were deselected. Ruff check/format and wheel/sdist build passed. Documentation tests verify the five historical archive hashes, negative-result wording, unexecuted report status, lack of mutable `@main` installation, and every local Markdown link. Report regressions cover current communication-only results, coverage/split/per-family tables, legacy PCIe-only archives, and exact input artifact hashes. The ignored detector archive was hash-verified, safely restored to a fresh temporary directory, and successfully rendered without modifying evidence.
- [x] `2026-08-02` Phase 10 completed locally: set the documented `0.2.0` candidate version, added a reproducible delivery-policy scanner and regression, prepared the completion report, PR body, release notes, reviewer/Kaggle handoff, and exact unexecuted push command. No push, GPU workload, adversarial run, or physical multi-node deployment was performed.
- [x] `2026-08-02` Phase 10 acceptance: 129 CPU-safe tests passed, one optional scikit-learn integration test skipped, and five hardware tests were deselected. Ruff check/format and wheel/sdist build passed; a fresh isolated environment installed/imported the wheel and ran its CLI; package metadata, dependency health, canonical notebook regeneration/output policy, Markdown links, diff whitespace, and delivery secret/path/archive/cache/symlink/size policy passed. Local Python is 3.11.15; Python 3.10 remains configured in CI because no 3.10 interpreter is installed on this host.

## Decision log

- `2026-08-02`: New schema-2 calibration support requires at least three idle
  and per-payload repetitions, a capture threshold relative to idle, adequate
  per-payload capture, correlation, and dynamic range. Legacy schema-1 support
  is readable only through explicitly labeled compatibility metadata.
- `2026-08-02`: A restored calibration may be prior-session input evidence,
  but the benign matrix must create and hash a fresh current-session supported
  calibration as its collection gate. Downstream consumers never choose the
  lexicographically latest filename.

- `2026-08-02`: Treat the hyphen-named executed notebooks as immutable local evidence, not canonical source. Their hashes and outputs must be preserved; new canonical notebooks will use underscore names and contain no outputs.
- `2026-08-02`: Treat `environment_fingerprint` as an environment description only. Existing per-run values will be loaded as legacy ambiguous grouping, never silently upgraded into true session IDs.
- `2026-08-02`: Preserve old artifacts byte-for-byte. New schemas/loaders may adapt them in memory and new derived artifacts may reference their hashes.
- `2026-08-02`: Keep required core dependencies empty. Analysis, telemetry, and optional online serving stay in extras or use the standard library where practical.
- `2026-08-02`: Use version `0.2.0` as an unreleased candidate because the implementation, schema, API, notebook, and documentation milestones are complete and CPU-verified. Do not describe it as a validated detector release; GPU evidence remains gated and pending.
- `2026-08-02`: Do not install missing build tooling without explicit approval. First check whether a truthful equivalent build validation is available; otherwise record/request the dependency at the gate.
- `2026-08-02`: Continue the existing partial Phase 02 implementation selectively. Do not replace it wholesale; first make its v2 manifest requirements coherent with orchestration and add regression tests for every new contract.
- `2026-08-02`: Treat a corpus as a deliberate matrix that may span sessions. Each session-specific collection manifest has a unique `collection_id` and true `experiment_session_id`, while deliberately repeated collections may share the same `corpus_id`.
- `2026-08-02`: Keep `session_fingerprint` only as a compatibility alias in new environment reports. It equals `environment_fingerprint`, is explicitly non-semantic for grouping, and live telemetry values are excluded from its stable-capability hash.
- `2026-08-02`: Define 30 seconds as the immutable primary feature window and 5/15 seconds as diagnostics. Benign workload defaults collect 35 measured seconds after warmup so asynchronous sampler edges do not make a nominal 30-second run incapable of producing a 30-second common window.
- `2026-08-02`: Require at least three primary runs per required family for detector evaluation, because a valid train/validation/test plan cannot represent a family with fewer. Derive stable configuration IDs from family plus canonical configuration when the plan does not provide one explicitly.
- `2026-08-02`: Keep central protocol versioning separate from research artifact schemas. Use a standard-library, create-only offline transport as the authoritative test path; expose online HTTPS components only as operator-wrapped references with external TLS/identity/secret requirements.
- `2026-08-02`: Use the acceptance YAML’s `control_model_or_checkpoint_load` as the canonical family ID; treat the experiment CSV’s `control_model_load` as a human-readable variant name only. Keep the standard profile benign-only with all eight required families, and move adversarial workloads to the Phase 07 opt-in workflow.
- `2026-08-02`: Define the default benign pilot as three repetitions of eight base configurations. The expanded profile is opt-in and adds bounded configuration variants plus optional peer copy. A matrix plan is immutable evidence written before calibration, while the separately named final corpus manifest accepts completed runs only.
- `2026-08-02`: Keep adversarial workloads outside every normal profile and require explicit approval before even preflight or artifact directory creation. Bounds and immutable strategy identity apply after user overrides. Runtime manifests must record detector score as null; only later frozen evaluation may create one.
- `2026-08-02`: Treat segmented execution as separate torchrun subprocesses sharing a segment-group ID, not idle gaps inside one process. Preserve each short run and its explicit primary-window exclusion; never concatenate telemetry across restart gaps. Label sparse parameter averaging `DiLoCo-inspired`, never faithful DiLoCo.
- `2026-08-02`: Freeze adversarial robustness fitting to benign primary training rows only. A declared plan separates development and hardening and seals any matching final family, session, or configuration by default; releasing the final round requires a separate explicit argument.
- `2026-08-02`: Generate the four Phase 08 notebooks from a checked-in deterministic script. Require a remote-visible 40-character commit at detached HEAD and a clean checkout, then chain notebooks only through create-only archives whose exact SHA-256 is supplied by the operator.
- `2026-08-02`: Keep benign and adversarial extraction summaries as separate immutable artifacts. Detector evaluation accepts exact root-relative summary paths, applies the primary coverage gate to benign data only, and appends adversarial rows only for frozen-baseline robustness scoring.
- `2026-08-02`: Make `docs/evidence-index.md` the repository claim boundary. Numerical empirical statements in README/reports must resolve to an indexed immutable artifact; generated reports additionally list the exact SHA-256 of every input they summarize.

## Discoveries

- The package audit is still accurate on notebook worktree state and the one failing CPU test.
- The ignored 253 KiB `.txt` file is a captured recursive repository listing, including `.git` filenames; it is not current project-source prose or result evidence.
- The current root has no `AGENTS.md`; only the completion package does.
- The executed benign notebook's extraction checks traversal but does not reject links; the executed detector notebook does reject links but deletes a working restore directory. Both are evidence copies and will not be rewritten. Canonical replacements must use safer no-overwrite extraction.
- The tracked CI runs `ruff check src tests`, while the package requires `ruff check .` and formatting. Canonical notebook policy and Ruff exclusions/formatting must reconcile those gates explicitly.
- The 18 benign workloads completed, but several wall durations were only six to seven seconds; completion is not coverage.
- Every merged manifest has a distinct environment fingerprint. The feature artifact therefore reports eight apparent “sessions” for eight runs, reproducing the invalid grouping defect.
- Calibration-stage idle runs entered the 16-row feature artifact. Primary corpus selection must come from a declared corpus manifest, not designation alone.
- At the Phase 02 resumption boundary, schema validation accepts v2 but the orchestrator still constructs a v2 `RunManifest` without required session/corpus/node/dirty/seed fields. Existing tests do not exercise that path because GPU orchestration is hardware-marked.
- Iteration-only completion was not sufficient evidence of usable telemetry duration. Rank-local measurement events and their common monotonic intersection are now required for completed, participation-valid v2 manifests; physical GPU execution remains pending compatible hardware.
- The Phase 04 host has the development tools but not the optional analysis stack. Split and selection-policy behavior is fully CPU/dependency-free tested; fitting pandas/scikit-learn models remains unexecuted locally and must not be reported as a measured detector result.
- Cross-node aggregation must use validated UTC timestamps because monotonic clocks are node-local. The local simulation sets `physical_multi_node_validated: false`; it cannot establish network, TLS, clock-sync, durability, or GPU-cluster behavior.
- The pre-Phase-06 standard profile mixed benign and adversarial workloads, omitted required idle coverage, and used family names that did not match the acceptance YAML. The canonical notebooks also bypassed corpus planning and one called feature extraction without the now-required corpus argument. The registry, profiles, orchestration, and affected disabled-by-default notebook cells are now coherent; actual dual-T4 execution is still pending.
- The prior adversarial implementation had only gradient accumulation, idle padding, and parameter-efficient DDP variants, mixed them into the standard profile, and recorded no common sync/cost/correctness contract. Its apparent “segmentation” roadmap was not code. Phase 07 replaces that profile boundary and adds bounded CPU-verifiable semantics, but CUDA/NCCL correctness and all outcome claims remain pending hardware.
- The earlier canonical notebook set used mutable staged workflows and could not safely chain multiple feature extractions in one restored artifact tree. Phase 08 makes the exact extraction-summary identity explicit and uses an SDK archive loader that rejects mismatched hashes, existing destinations, traversal, duplicate members, links, special members, and configured size/member overages.
- The historical detector artifact's amended split is whole-run but not a valid current session holdout. It has four train, two validation, and two test runs; the three test windows come from one DDP and one idle run. Its old two-run bootstrap interval is rejected by the current 20-independent-test-run minimum, even though the raw historical result remains immutable.
- Final local validation used Python 3.11.15. Python 3.10 and the optional scikit-learn analysis integration could not be exercised without changing the host dependency environment; the configured Python 3.10/3.11 CI matrix and truthful optional-test skip remain in place.

## Schema migrations

Baseline schema `1.0` has `run_id`, `environment_fingerprint`, and legacy `session_fingerprint` feature metadata, but no explicit true session/corpus/node semantics and no coverage artifact.

Planned schema migration:

| Artifact | Before | After | Compatibility |
|---|---|---|---|
| Environment | `session_fingerprint` conflates grouping/environment | `experiment_session_id`, `node_id`, `environment_fingerprint`, source dirty state | legacy loader retains old field and marks grouping ambiguous |
| Run manifest | run/environment IDs only | session, corpus/collection, run, node, config/family IDs, measured interval, source/notebook/input provenance | additive loader/migration; raw files unchanged |
| Corpus | absent/ad hoc summary | planned run entries plus accepted run allow-list and required families/counts | legacy directory import requires explicit selection or diagnostic mode |
| Features | rows only, legacy session fingerprint | versioned extraction result plus aligned feature rows and coverage records | old feature-row loader remains supported with ambiguity flag |
| Evaluation | strategy prose may disagree with override | actual strategy, groups, selection protocol, counts/warnings and communication-only primary block | old result remains readable, never relabeled |

Exact version identifiers and field mappings will be updated in this section during Phases 02–04.

Phase 02 implemented schema `2.0` for new run, environment, and corpus records.
Telemetry/workload/calibration/summary records remain readable under `1.0`
where their contract did not change. `load_artifact(..., migrate_legacy=True)`
returns a new `2.0` view with `source_schema_version: "1.0"` and
`legacy_grouping_ambiguous: true`; it never changes the source file. Unknown
session/corpus/node fields remain null rather than being inferred. Full field
semantics and compatibility behavior are in `docs/schema-migrations.md`.

Phase 03 extends version `2.0` with linked `feature_row`, `coverage_record`, and
`feature_extraction_result` artifacts. Coverage includes sampling gaps and exact
window counts, while completed v2 run manifests include an ordered, internally
consistent measured interval. Historical v1 feature rows remain readable and
are not assigned invented grouping IDs.

Phase 07 adds backward-compatible extraction-summary designation metadata so a
declared corpus can explicitly select benign and adversarial rows without
relabeling either. Completed new version 2 adversarial run manifests require
both ranks' structured strategy and peak-memory evidence; old raw artifacts are
not rewritten and runtime detector scores remain null.

## Validation matrix

| Area | Check | Status |
|---|---|---|
| CPU | package import | baseline passed in tests |
| CPU | CPU-safe pytest | baseline 56 passed / 1 notebook-path failure / 5 deselected |
| CPU | Ruff lint `src tests` | baseline passed |
| CPU | Ruff lint/format root | baseline failed; pending repair |
| CPU | build wheel/sdist | pending; `build` module unavailable |
| CPU | notebook JSON/output/source policy | pending repair |
| CPU | schema/coverage/split/central/workload tests | pending implementation |
| GPU | local CUDA/NCCL/NVML | not suitable on this host; pending compatible hardware |
| Kaggle | historical dual-T4 evidence | measured old pilot only; no new execution in this plan yet |
| Kaggle | v2 notebooks and full benign coverage | pending user execution after immutable commit/push |
| Multi-node | two-agent CPU simulation | pending implementation |
| Multi-node | physical online deployment/interconnect | pending hardware validation; no claim permitted |

Resumption overrides for current tool state: CPU-safe pytest, Ruff check, and
temporary-output build pass; Ruff format is pending only for the partial Phase
02 files. The original baseline rows above remain historical facts from the
pre-Phase-01 boundary.

Phase 02 validation update: schema/provenance/corpus tests pass locally;
coverage, split, central-monitoring, and expanded workload tests remain pending
their respective phases. Ruff format no longer has a pending Phase 02 failure.

Phase 03 validation update: coverage, timestamp alignment, duration control,
measurement provenance, and strict pre-evaluation coverage gates pass locally.
Grouped split/evaluation semantics beyond the coverage gate were deferred to
Phase 04 at this checkpoint.

Phase 04 validation update: true-session/class/family/config split planning,
legacy grouping refusal, selection isolation, primary-report schema, and
small-independent-group warnings pass locally. The optional pandas/scikit-learn
integration test is skipped because the `analysis` extra is not installed; no
local model metrics are claimed.

Phase 05 validation update: central protocol schemas, HMAC/allow-list/size/order
controls, offline transport, two-agent aggregation, node staleness, retry, and
decision abstention pass locally. Online TLS and physical multi-node behavior
remain pending external deployment and hardware.

Phase 06 validation update: CPU-only matrix planning and mocked lifecycle tests
cover all eight required families, explicit configuration IDs, bounded pilot
and expanded parameters, corpus-plan-before-calibration ordering, final accepted
membership, and coverage-first per-family counts. The standard estimate is 24
runs and 35 GPU-minutes at declared durations. This is a cost estimate, not a
measured GPU result; CUDA/NCCL/NVML and peer behavior remain pending Kaggle.

Phase 07 validation update: bounded strategy/profile/approval contracts,
synchronization schedules, parameter-agreement simulation, segmented subprocess
planning, adversarial manifest requirements, combined extraction selection,
benign-only frozen-evaluation partitions, cost proxies, and sealed holdout
rounds pass locally. The optional analysis stack and all GPU execution remain
unavailable, so no evasion, false-positive, efficiency, quality, or robustness
result is claimed.

Phase 10 validation update: the full CPU-safe suite has 129 passes, one optional
analysis skip, and five hardware deselections. Ruff check/format, the 0.2.0 wheel
and sdist build, clean-environment wheel import/CLI, dependency health, notebook
generation/output checks, documentation links, and delivery policy scans pass.
The branch is prepared for human review but intentionally remains unpushed.

Pre-Kaggle audit validation update: the full CPU-safe suite now has 147 passes,
one optional analysis skip, and five hardware deselections. GitHub CI installs
the optional analysis extra on Python 3.10 and 3.11 so that integration path is
exercised there. Local compile, Ruff, notebook policy/regeneration, delivery and
path scanning, clean package build, isolated-wheel import/CLI/dependency checks,
distribution-content inspection, diff checks, and evidence-hash verification
pass. No GPU, Kaggle, adversarial, or physical multi-node execution occurred.

## Recovery and idempotence

- Never use Git reset, clean, force checkout, history rewrite, or force push.
- Before each phase, record `git status --short --branch` and inspect overlapping user changes.
- All generated evidence writes remain create-only. Schema migrations operate on loaded copies or create new derived artifacts; raw archives stay immutable.
- Tests use fresh pytest temporary directories. Local state snapshots are timestamped and do not overwrite earlier records.
- Canonical notebook generation must be deterministic and clear execution counts/outputs; executed originals remain local-only by hash.
- If interrupted, resume from the first unchecked progress item, re-read the latest decisions/discoveries, and rerun the most recent targeted checks. Phase commits, when created, are additive recovery points.

## Artifacts

- `.agent/execplans/commguard-research-completion.md` — this living plan (hash changes as the plan evolves).
- The former tracked `.agent/state/` baseline captures were removed from the
  public branch because they exposed a personal local path; `.agent/state/` is
  now ignored. Historical evidence hashes remain indexed in
  `docs/evidence-index.md`; raw archives are intentionally not copied or modified.

## Final retrospective

All phases that can be completed truthfully on this CPU-only Ubuntu host are
complete. The repository now has CPU-tested provenance, coverage, evaluation,
central-monitoring, benign/adversarial orchestration, notebook, evidence, and
delivery contracts. Historical executed notebooks and raw archives were
preserved byte-for-byte and remain local-only; their indexed evidence still
supports only the negative DDP-versus-idle PCIe pilot result.

The unresolved risks are empirical: no new CUDA/NCCL/NVML run, valid eight-family
benign corpus, detector-performance result, adversarial outcome, or physical
multi-node validation exists. The highest-value next experiment is the gated
24-run benign matrix on Kaggle T4 x2, starting from a reviewed remote-visible
commit and proceeding to detector evaluation only if the strict primary coverage
artifact passes. Adversarial execution requires separate human approval. Git
push is also a human gate and was not performed.
