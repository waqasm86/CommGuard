# Schema migrations

CommGuard preserves historical evidence byte-for-byte. Migration means loading
an old record into a new in-memory view or writing a new derived artifact; it
never means editing a raw archive or manifest in place.

## Version 1.0 to 2.0 provenance

Version 1.0 used `environment_fingerprint` in run manifests and
`session_fingerprint` in environment/feature records. Historical artifacts show
that these values changed per run, so they cannot support independent-session
grouping.

Version 2.0 adds these fields to new run manifests:

| Field | Meaning |
|---|---|
| `experiment_session_id` | One independent hardware/runtime session; reused across every run collected in that session. |
| `collection_id` | One execution of a deliberate collection plan, normally within one session. |
| `corpus_id` | The deliberate experiment matrix; it may be shared by session-specific collections. |
| `run_id` | One unique workload execution. |
| `node_id` | Stable node identity within a deployment/session. |
| `environment_fingerprint` | Hash of stable environment capabilities; never a grouping or session identifier. |
| `source_commit` / `source_dirty` | Source revision and whether local changes were present. |
| `notebook_version` | Canonical notebook version when collection is notebook-driven. |
| `input_archive_sha256` | Hash of imported evidence when applicable. |
| `random_seed` | Explicit run seed; it must agree with the legacy `seed` field. |

Environment reports retain `session_fingerprint` as a compatibility alias for
`environment_fingerprint`, while adding the true `experiment_session_id`,
`node_id`, source revision, and dirty-state fields. Live telemetry values are
excluded from environment-fingerprint hashing; capability support remains part
of the fingerprint.

## Legacy loading

`load_artifact(path)` validates and returns the original record unchanged.
`load_artifact(path, migrate_legacy=True)` returns a new dictionary with:

- `schema_version: "2.0"`;
- `source_schema_version: "1.0"`;
- `legacy_grouping_ambiguous: true`;
- the old feature/environment fingerprint exposed as
  `environment_fingerprint` where applicable;
- true session, collection, corpus, and node fields set to `null` when the old
  evidence did not record them.

Missing identifiers are not inferred from filenames, timestamps, hostnames, or
environment hashes. Primary session-held-out evaluation must reject ambiguous
legacy grouping or use a separately declared, evidence-backed mapping.

## Corpus manifest

The version 2.0 `corpus_manifest` declares planned runs and an exact accepted
run allow-list. Each accepted run fills at most one plan slot. Designation is
checked alongside membership, so calibration-stage idle controls cannot enter a
benign corpus merely because their workload family resembles a benign control.

A corpus can span independent sessions by using the same `corpus_id` in
multiple session-specific collection manifests. Each such manifest has its own
`collection_id` and `experiment_session_id`.
