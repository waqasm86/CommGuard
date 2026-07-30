from __future__ import annotations

from commguard.features.extraction import extract_run_features
from commguard.telemetry.schema import (
    FIELD_UNITS,
    TELEMETRY_FIELDS,
    FieldReading,
    TelemetrySample,
)


def _samples() -> list[dict]:
    samples = []
    for sequence in range(6):
        for gpu_index in (0, 1):
            fields = {
                name: (
                    FieldReading(None, FIELD_UNITS[name], False, "not supported")
                    if name == "pcie_rx_bytes_per_s"
                    else FieldReading(sequence + gpu_index, FIELD_UNITS[name], True)
                )
                for name in TELEMETRY_FIELDS
            }
            samples.append(
                TelemetrySample(
                    run_id="run-1",
                    gpu_index=gpu_index,
                    gpu_uuid=f"GPU-{gpu_index}",
                    sequence=sequence,
                    wall_time_utc=f"2026-01-01T00:00:0{sequence}+00:00",
                    monotonic_ns=sequence * 1_000_000_000,
                    fields=fields,
                ).to_dict()
            )
    return samples


def test_feature_extraction_preserves_missing_telemetry() -> None:
    manifest = {
        "run_id": "run-1",
        "environment_fingerprint": "session-1",
        "workload_label": "inference",
        "workload_family": "inference",
        "designation": "benign",
        "warmup_seconds": 0,
    }
    rows = extract_run_features(
        manifest,
        _samples(),
        window_lengths=(5,),
        stride_fraction=1,
        exclude_startup=False,
    )
    assert rows
    assert rows[0]["gpu0__pcie_rx_bytes_per_s__mean"] is None
    assert rows[0]["gpu0__pcie_rx_bytes_per_s__missing_fraction"] == 1.0
