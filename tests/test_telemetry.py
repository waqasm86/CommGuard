from __future__ import annotations

import time

from commguard.schemas import FIELD_UNITS, TELEMETRY_FIELDS, FieldReading
from commguard.telemetry import TelemetryCollector


class FakeBackend:
    def __init__(self, unsupported: str | None = None) -> None:
        self.unsupported = unsupported
        self.closed = False

    def initialize(self) -> None:
        pass

    def shutdown(self) -> None:
        self.closed = True

    def device_count(self) -> int:
        return 2

    def device_uuid(self, index: int) -> str:
        return f"GPU-{index}"

    def read_fields(self, index: int):
        output = {}
        for position, name in enumerate(TELEMETRY_FIELDS):
            if name == self.unsupported:
                output[name] = FieldReading(None, FIELD_UNITS[name], False, "unsupported")
            else:
                output[name] = FieldReading(index + position, FIELD_UNITS[name], True)
        return output


def test_fake_collector_samples_both_gpus_and_reports_jitter() -> None:
    backend = FakeBackend()
    collector = TelemetryCollector("run", interval_s=0.02, backend=backend)
    collector.start()
    time.sleep(0.075)
    diagnostics = collector.stop()
    assert backend.closed
    assert diagnostics.sample_cycles >= 3
    assert {sample.gpu_index for sample in collector.samples} == {0, 1}
    assert {sample.rank for sample in collector.samples} == {0, 1}
    assert all(sample.process_id for sample in collector.samples)
    assert all(sample.target_sampling_interval_s == 0.02 for sample in collector.samples)
    assert collector.samples[0].raw_unit_metadata["pcie_bytes_multiplier"] == 1024
    assert "memory_total_bytes" in collector.samples[0].fields
    assert diagnostics.mean_interval_s is not None
    assert diagnostics.field_missing_fraction["power_draw_w"] == 0


def test_unsupported_field_is_not_zero() -> None:
    collector = TelemetryCollector(
        "run",
        interval_s=0.01,
        backend=FakeBackend("pcie_tx_bytes_per_s"),
    )
    collector.collect_for(0.025)
    readings = [sample.fields["pcie_tx_bytes_per_s"] for sample in collector.samples]
    assert readings
    assert all(not reading.supported and reading.value is None for reading in readings)
    assert collector.diagnostics().field_missing_fraction["pcie_tx_bytes_per_s"] == 1
