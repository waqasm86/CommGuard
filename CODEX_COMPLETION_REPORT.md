# CommGuard research-completion and pre-Kaggle audit report

Completed locally: 2026-08-02 (Asia/Karachi)

Branch: `codex/fix-calibration-idle-and-kaggle-workflow`

## Outcome

The repository is now a CPU-verifiable research SDK and reproducible dual-T4
experiment workflow. It has explicit provenance/corpus/session semantics,
coverage-first features, leakage-resistant evaluation, a local central-monitoring
reference, bounded benign/adversarial workloads, four canonical Kaggle notebooks,
and evidence-grounded reporting. The pre-Kaggle audit additionally requires an
idle-aware repeated calibration gate, binds all downstream artifacts to its
exact provenance, makes central retry acknowledgments idempotent, and stabilizes
extreme detector logits. Package metadata is `0.2.0`.

The first modern calibration-v3 Kaggle attempt verified two T4 GPUs, the
reviewed source import, CUDA/NCCL, and collective execution, but all idle runs
failed because worker mode `idle` was unimplemented. Its zero-idle result was
correctly `not_supported` and is debugging evidence only. It is not evidence
that PCIe telemetry failed, and no benign, detector, adversarial, or physical
multi-node run followed it. The indexed historical result remains negative:
usable coverage is DDP versus idle only, and its primary PCIe-only test missed
the training run. See [`docs/current-results.md`](docs/current-results.md).

## Phase commits

| Phase | Commit | Outcome |
|---|---|---|
| 01 | `c920a73` | repository repair and canonical notebook policy |
| 02 | `0e8b505` | explicit provenance and corpus schemas |
| 03 | `3163275` | strict feature coverage and duration controls |
| 04 | `ea48109` | leakage-resistant grouped evaluation |
| 05 | `5a9860e` | central-monitoring reference architecture |
| 06 | `d2668ed` | coverage-aware benign workload matrix |
| 07 | `04554d7` | bounded adversarial research strategies |
| 08 | `de7932f` | reproducible dual-T4 notebook workflow |
| 09 | `3d940d4` | evidence boundaries and current-results reporting |
| 10 | commit containing this report | final validation, version, and handoff |

The pre-Kaggle audit adds six focused commits on top of the completion branch:
public-state sanitization, calibration/provenance enforcement, central retry
idempotency, numerical stability, canonical notebook/claim updates, and final
delivery validation.

This follow-up branch adds the real idle worker lifecycle, safe preflight output
creation, accurate payload identities, one-sweep protection, exact standard
result validation, non-destructive archive export, and a regenerated
calibration-v3 workflow. The fixed code remains CPU-validated pending a fresh
Kaggle rerun.

The living implementation/audit record is
[`commguard-research-completion.md`](.agent/execplans/commguard-research-completion.md).

## Final local validation

Run with the project virtual environment:

| Check | Result |
|---|---|
| `python -m pytest -m 'not gpu and not multigpu'` | 169 passed, one optional scikit-learn test skipped, six hardware tests deselected |
| `python -m ruff check .` | passed |
| `python -m ruff format --check .` | passed |
| `python -m build` | `commguard-0.2.0` wheel and sdist built |
| package import/version | passed; `commguard.__version__ == '0.2.0'` |
| CLI help and CPU-safe estimates | passed |
| canonical notebook JSON/cell/output/generator policy | passed |
| secret, personal-path/link, archive/cache, symlink, and file-size scan | passed |
| local Markdown links and `git diff --check` | passed |
| install/import/CLI from the built wheel in a fresh virtual environment | passed |

Local Python is 3.11.15. A Python 3.10 interpreter is not installed on this
host; the GitHub Actions matrix retains Python 3.10 and 3.11 checks. The optional
analysis stack is not installed locally, so the scikit-learn integration test
was truthfully skipped rather than changing the environment.

## Evidence and claim boundary

- Historical raw archives and executed notebooks remain ignored and unchanged.
- Their hashes and supported claims are in
  [`docs/evidence-index.md`](docs/evidence-index.md).
- Canonical notebooks have null execution counts and no outputs.
- No archive, wheel, sdist, cache, virtual environment, credential, or raw
  telemetry file is tracked for delivery.
- Generated reports list the SHA-256 of every input artifact they summarize.

## Remaining human/hardware gates

1. Review the pushed `codex/fix-calibration-idle-and-kaggle-workflow` branch.
2. Insert its final remote-visible 40-character commit SHA as `REVIEWED_COMMIT`
   in a fresh copy of `commguard_calibration_v3.ipynb`.
3. Select Kaggle T4 x2, run once from the first cell, and share the executed
   notebook, archive, and `.sha256`; continue only if the modern gate is supported.
4. Run the duration-valid 24-run benign matrix on Kaggle T4 x2.
5. Continue to detector evaluation only if the eight-family primary coverage
   artifact passes.
6. Obtain separate human approval before enabling bounded adversarial execution.
7. Keep the final adversarial family/session/configuration holdout sealed unless
   separately approved for release.
8. Treat physical multi-node deployment as pending new architecture/security
   review and hardware evidence.

Exact push and Kaggle handoff commands are in
[`delivery/HANDOFF.md`](delivery/HANDOFF.md).
