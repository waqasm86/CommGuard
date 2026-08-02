# Artifacts

CommGuard stores JSON, JSONL, logs, and reports beneath a user-selected artifact
root. `ArtifactStore` uses exclusive creation and rejects an existing target.
Raw run directories are never rewritten by feature extraction, evaluation, or
report generation.

Telemetry readings carry `value`, `unit`, `supported`, and `error`. Unsupported
readings require `value: null`, `supported: false`, and a non-empty error.

Historical schema version `1.0` remains readable. New provenance, run, corpus,
feature, coverage, split, and evaluation records use the versioned contracts in
[`artifact-schemas.md`](artifact-schemas.md); raw evidence is never upgraded in
place.

`ArtifactStore.export` writes a new tar.gz archive without following symlinks.
`restore_archive` requires the exact expected SHA-256 and a destination that
does not exist. It rejects absolute/traversal/duplicate paths, links, special
members, and configured member or uncompressed-size overages before extracting.
See [`evidence-index.md`](evidence-index.md) for immutable historical hashes.
