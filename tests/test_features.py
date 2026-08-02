from __future__ import annotations

from datetime import datetime, timedelta, timezone

from commguard.features import extract_run_features, numeric_feature_columns
from commguard.schemas import (
    FIELD_UNITS,
    LEGACY_SCHEMA_VERSION,
    TELEMETRY_FIELDS,
    FieldReading,
    TelemetrySample,
    validate_artifact,
)


def telemetry() -> list[dict]:
    records = []
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for sequence in range(40):
        for gpu in (0, 1):
            fields = {}
            for position, name in enumerate(TELEMETRY_FIELDS):
                value = sequence * (position + 1) + gpu
                fields[name] = FieldReading(value, FIELD_UNITS[name], True)
            records.append(
                TelemetrySample(
                    run_id="run-1",
                    gpu_index=gpu,
                    gpu_uuid=f"GPU-{gpu}",
                    sequence=sequence,
                    wall_time_utc=(origin + timedelta(seconds=sequence)).isoformat(),
                    monotonic_ns=sequence * 1_000_000_000,
                    fields=fields,
                ).to_dict()
            )
    return records


def manifest() -> dict:
    return {
        "run_id": "run-1",
        "environment_fingerprint": "session-a",
        "workload_label": "training",
        "workload_family": "ddp",
        "designation": "benign",
        "warmup_seconds": 0,
    }


def test_features_are_deterministic_and_cross_gpu() -> None:
    first = extract_run_features(manifest(), telemetry(), window_lengths=(5,), stride_fraction=1)
    second = extract_run_features(manifest(), telemetry(), window_lengths=(5,), stride_fraction=1)
    assert first == second
    assert first
    assert first[0]["schema_version"] == LEGACY_SCHEMA_VERSION
    validate_artifact(first[0])
    assert first[0]["cross_gpu__power_draw_w__mean_abs_difference"] == 1
    assert first[0]["gpu0__pcie_tx_bytes_per_s__mean"] is not None


def test_identity_metadata_is_not_numeric_feature() -> None:
    rows = extract_run_features(manifest(), telemetry(), window_lengths=(5,))
    columns = numeric_feature_columns(rows)
    assert columns
    assert "run_id" not in columns
    assert "window_start_monotonic_ns" not in columns
    assert all("label" not in name for name in columns)
