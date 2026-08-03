from __future__ import annotations

from datetime import datetime, timezone

import pytest

from commguard.exceptions import ValidationError
from commguard.schemas import (
    CURRENT_SCHEMA_VERSION,
    FIELD_UNITS,
    LEGACY_SCHEMA_VERSION,
    LEGACY_TELEMETRY_FIELDS,
    TELEMETRY_FIELDS,
    FieldReading,
    TelemetrySample,
    validate_artifact,
)


def sample() -> TelemetrySample:
    return TelemetrySample(
        run_id="run-1",
        gpu_index=0,
        gpu_uuid="GPU-a",
        sequence=0,
        wall_time_utc=datetime.now(timezone.utc).isoformat(),
        monotonic_ns=1,
        fields={
            name: FieldReading(value=1.0, unit=FIELD_UNITS[name], supported=True)
            for name in TELEMETRY_FIELDS
        },
        target_sampling_interval_s=1.0,
        rank=0,
        process_id=123,
        raw_unit_metadata={
            "pcie_nvml_raw_unit": "KB/second (NVML API)",
            "pcie_bytes_multiplier": 1024,
            "pcie_conversion_convention": "1 NVML KB = 1024 bytes",
        },
    )


def test_supported_reading_requires_value() -> None:
    with pytest.raises(ValidationError, match="must be present"):
        FieldReading(None, "watts", True).validate()


def test_unsupported_reading_is_null_with_reason() -> None:
    reading = FieldReading(None, "watts", False, "not supported")
    reading.validate()
    with pytest.raises(ValidationError, match="unsupported values must be null"):
        FieldReading(0, "watts", False, "not supported").validate()


def test_telemetry_round_trip() -> None:
    original = sample()
    restored = TelemetrySample.from_dict(original.to_dict())
    assert restored == original


def test_current_telemetry_requires_complete_field_set() -> None:
    data = sample().to_dict()
    data["fields"].pop("pcie_rx_bytes_per_s")
    with pytest.raises(ValidationError, match="schema-compatible"):
        TelemetrySample.from_dict(data)


def test_legacy_nine_field_telemetry_remains_readable() -> None:
    data = sample().to_dict()
    data["schema_version"] = LEGACY_SCHEMA_VERSION
    data["fields"] = {name: data["fields"][name] for name in LEGACY_TELEMETRY_FIELDS}
    for name in (
        "target_sampling_interval_s",
        "actual_sampling_interval_s",
        "sampling_jitter_s",
        "rank",
        "process_id",
        "raw_unit_metadata",
    ):
        data.pop(name)
    assert TelemetrySample.from_dict(data).schema_version == LEGACY_SCHEMA_VERSION


def test_split_assignment_contract_is_actionable() -> None:
    with pytest.raises(ValidationError, match="train, validation, or test"):
        validate_artifact(
            {
                "artifact_kind": "split_assignment",
                "schema_version": "1.0",
                "run_id": "run-1",
                "split": "both",
            }
        )


def test_primary_evaluation_schema_requires_non_diagnostic_split() -> None:
    payload = {
        "artifact_kind": "evaluation_result",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_assignments": {"run-1": "test"},
        "leakage_audit": {"passed": True},
        "ablations": {},
        "primary_communication_only": {"window_seconds": 30.0},
        "actual_split_strategy": "deterministic_class_stratified_whole_run",
        "split_plan": {"diagnostic_only": False},
        "warnings": [],
    }
    validate_artifact(payload)

    payload["split_plan"]["diagnostic_only"] = True
    with pytest.raises(ValidationError, match="diagnostic split"):
        validate_artifact(payload)
