from __future__ import annotations

import pytest

from commguard.environment.preflight import check_environment
from commguard.schemas import TELEMETRY_FIELDS
from commguard.telemetry import NvmlBackend


@pytest.mark.gpu
def test_real_nvml_attempts_all_fields() -> None:
    report = check_environment(strict=False)
    if not report["gpus"]:
        pytest.skip("no NVIDIA GPU")
    if not report["telemetry_capabilities"]["available"]:
        pytest.skip(report["telemetry_capabilities"]["error"])
    backend = NvmlBackend()
    backend.initialize()
    try:
        assert backend.device_count() >= 1
        readings = backend.read_fields(0)
        assert set(readings) == set(TELEMETRY_FIELDS)
        assert all(
            reading.value is not None if reading.supported else reading.error is not None
            for reading in readings.values()
        )
    finally:
        backend.shutdown()
