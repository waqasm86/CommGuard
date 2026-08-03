# Suggested pull request

> Superseded by `delivery/KAGGLE_PROTOTYPE_PR_BODY.md` for the current branch.

Suggested title:

> Fix calibration-v3 idle dispatch and Kaggle evidence workflow

## Summary

- implement the missing two-rank idle worker path with lifecycle, heartbeat,
  participation, and measured-interval evidence but no measured collective;
- make preflight prepare safe output directories;
- use truthful payload-specific calibration identities, refuse repeated sweeps,
  and validate the exact 30-run standard matrix;
- make archive export non-destructive and atomic by default;
- harden and regenerate the canonical Kaggle notebooks with exact source/import,
  T4, workspace, provenance, progress, validation, and export checks;
- repair repository/notebook policy without deleting historical user evidence;
- remove tracked local state, scan every tracked public text file for personal
  paths, and ignore future `.agent/state/` captures;
- require five-repetition idle-aware calibration at 1/4/16/64/128 MiB and bind
  downstream artifacts to one exact hash-verified calibration;
- add explicit session, collection, corpus, run, node, source, and input provenance;
- require declared corpus membership, one coverage record per planned run, and
  duration-valid 30-second primary windows;
- make evaluation whole-run/session grouped with train-only preprocessing,
  validation-only selection, and communication-only primary reporting;
- add a signed privacy-allow-listed two-agent central-monitoring reference path
  with idempotent exact retries and conflicting-ID rejection;
- add eight benign families and eight bounded approval-gated adversarial
  strategies with a sealed final holdout;
- add four immutable-commit/hash-chained canonical Kaggle notebooks; and
- index historical evidence while making the DDP-versus-idle PCIe negative
  result prominent.

## Validation

- 169 CPU-safe tests passed; one optional analysis integration skipped; six
  GPU/multi-GPU tests deselected.
- Ruff check and format check passed.
- Wheel and sdist built as `commguard 0.2.0`.
- Fresh-wheel install/import/CLI, notebook regeneration/compilation/output
  checks, local Markdown links, secret/path/archive/cache/size scan, and
  `git diff --check` passed.

## Research status

The first calibration-v3 Kaggle T4 x2 attempt is reported only as debugging
evidence: reviewed-source import, CUDA/NCCL, and collectives worked, but the
missing `idle` dispatch made every idle run fail. Its zero-idle calibration was
correctly `not_supported`; it is not evidence that PCIe telemetry failed and
must not gate a benign run. No successful modern calibration, benign corpus,
detector, adversarial, or physical multi-node result is claimed.
Historical evidence has only
five-second DDP/idle feature coverage, includes calibration-idle rows, and the
primary PCIe-only test missed its training run. Full benign coverage,
cross-session validation, adversarial execution, and physical multi-node
deployment remain pending their explicit gates.

The canonical notebooks were regenerated and remain unexecuted. Their
`REVIEWED_COMMIT` placeholder must be set to the final pushed 40-character SHA
before the calibration notebook is run on Kaggle T4 x2.

## Reviewer checklist

- [ ] Inspect `docs/current-results.md` and `docs/evidence-index.md` first.
- [ ] Confirm canonical notebooks contain no outputs or mutable branch install.
- [ ] Confirm normal profiles exclude adversarial workloads.
- [ ] Confirm final holdout remains sealed by default.
- [ ] Confirm no ignored evidence/archive is included in the diff.
- [ ] Approve push/merge separately from any later GPU/adversarial execution.
