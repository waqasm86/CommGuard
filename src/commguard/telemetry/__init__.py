"""Optional NVML telemetry collection and telemetry schemas."""

from commguard.telemetry.nvml import (
    CollectorDiagnostics,
    NvmlBackend,
    TelemetryCollector,
    compare_sampling_intervals,
    compare_sampling_rates,
    measure_runtime_overhead,
)
from commguard.telemetry.schema import (
    FIELD_UNITS,
    LEGACY_TELEMETRY_FIELDS,
    TELEMETRY_FIELDS,
    FieldReading,
    TelemetrySample,
)

__all__ = [
    "CollectorDiagnostics",
    "FIELD_UNITS",
    "LEGACY_TELEMETRY_FIELDS",
    "FieldReading",
    "NvmlBackend",
    "TELEMETRY_FIELDS",
    "TelemetryCollector",
    "TelemetrySample",
    "compare_sampling_intervals",
    "compare_sampling_rates",
    "measure_runtime_overhead",
]
