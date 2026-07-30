# Artifacts

CommGuard stores JSON, JSONL, logs, and reports beneath a user-selected artifact
root. `ArtifactStore` uses exclusive creation and rejects an existing target.
Raw run directories are never rewritten by feature extraction, evaluation, or
report generation.

Telemetry readings carry `value`, `unit`, `supported`, and `error`. Unsupported
readings require `value: null`, `supported: false`, and a non-empty error.

Artifact schema version `1.0` is validated before structured records are saved.
