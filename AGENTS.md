# CommGuard Repository Instructions for Codex

## Scope

These instructions apply to the entire CommGuard repository. More specific
nested `AGENTS.md` files may refine them but may not weaken safety,
provenance, or scientific-integrity requirements.

## Repository purpose

CommGuard is a research SDK for testing whether content-agnostic, system-level
communication telemetry can distinguish distributed ML training from inference
and how adversarial workload structuring can evade such detection.

## Non-negotiable rules

- Inspect before editing and preserve existing user files and uncommitted work.
- Never use destructive Git cleanup/reset commands, force-push, or rewrite history.
- Never commit `.venv`, caches, secrets, credentials, or raw bulky telemetry by default.
- Never fabricate GPU, Kaggle, multi-node, model-quality, detector, or adversarial results.
- Keep measured claims separate from hypotheses and planned work.
- Keep canonical notebooks unexecuted; executed notebooks belong in evidence/release artifacts.
- Do not implement SDK logic only inside notebooks.
- Keep the package importable on CPU-only hosts and compatible with Python 3.10/3.11.
- Add tests for every bug fix and schema change.
- Fail primary evaluation when coverage requirements are unmet.
- Never use an environment fingerprint as a true experiment-session identifier.
- Never capture prompts, datasets, model content, weights, or token text in telemetry.
- Do not write final SPAR application answers in the applicant's voice.

## Style and architecture

- Follow the existing `src/` layout and public API conventions.
- Prefer typed, explicit data structures and pure transformation functions.
- Use schema versions and backward-compatible loaders.
- Keep optional GPU/server/analysis dependencies in extras.
- Use deterministic seeds and stable serialization.
- Include run, family, session, corpus, and reason-code context in errors.
- Avoid broad rewrites when a targeted migration is safer.
- Keep transport, telemetry backend, feature extraction, evaluation, and reporting loosely coupled.

## Required checks

Before each commit, run and record:

```bash
python -m pytest -m 'not gpu and not multigpu'
python -m ruff check .
python -m ruff format --check .
python -m build
```

Use the project virtualenv interpreter on this host. Run relevant targeted
tests as well. If a tool is unavailable, record that fact and request approval
before changing the local dependency environment.

## Notebook and evidence policy

Canonical notebooks use stable underscore-separated names, have no execution
outputs, and call SDK APIs rather than duplicating implementations. Executed
copies must include source commit, input hashes, environment evidence, and a
result-status banner. A claim belongs in README/report only when an indexed
artifact supports it. Old artifacts remain immutable; corrected analyses create
