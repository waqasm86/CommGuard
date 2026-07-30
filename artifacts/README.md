# Artifact root

Generated data is written under versioned subdirectories:

- `environment/`: preflight and topology reports
- `runs/<run_id>/`: manifests, raw telemetry, rank events, and logs
- `features/`: deterministic derived feature tables
- `splits/`: saved whole-run/session split assignments
- `results/`: generated model evaluations and calibration decisions
- `figures/`: plots generated from saved evidence

Raw files are create-only. Derivation never modifies source evidence.
