# Suggested pull request

Suggested title:

> Fix CommGuard pre-Kaggle calibration, provenance, and delivery blockers

## Summary

- repair repository/notebook policy without deleting historical user evidence;
- remove tracked local state, scan every tracked public text file for personal
  paths, and ignore future `.agent/state/` captures;
- require three-repetition idle-aware calibration at 1/4/16/64 MiB and bind
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

- 147 CPU-safe tests passed; one optional analysis integration skipped; five
  GPU/multi-GPU tests deselected.
- Ruff check and format check passed.
- Wheel and sdist built as `commguard 0.2.0`.
- Fresh-wheel install/import/CLI, notebook regeneration/compilation/output
  checks, local Markdown links, secret/path/archive/cache/size scan, and
  `git diff --check` passed.

## Research status

No new GPU, Kaggle, adversarial, or physical multi-node result is claimed.
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
