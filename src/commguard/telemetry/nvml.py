"""Host-local NVML telemetry with explicit field support semantics."""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from commguard.schemas import (
    FIELD_UNITS,
    TELEMETRY_FIELDS,
    FieldReading,
    TelemetrySample,
)


class TelemetryBackend(Protocol):
    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def device_count(self) -> int: ...

    def device_uuid(self, index: int) -> str: ...

    def read_fields(self, index: int) -> dict[str, FieldReading]: ...


class NvmlBackend:
    """Thin adapter over nvidia-ml-py; imports it only when used."""

    def __init__(self) -> None:
        self.nvml: Any = None
        self.handles: list[Any] = []

    def initialize(self) -> None:
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())
        ]

    def shutdown(self) -> None:
        if self.nvml is not None:
            self.nvml.nvmlShutdown()

    def device_count(self) -> int:
        return len(self.handles)

    def device_uuid(self, index: int) -> str:
        value = self.nvml.nvmlDeviceGetUUID(self.handles[index])
        return value.decode() if isinstance(value, bytes) else str(value)

    def _read(self, name: str, call: Any, transform: Any = None) -> FieldReading:
        try:
            value = call()
            if transform is not None:
                value = transform(value)
            return FieldReading(value=value, unit=FIELD_UNITS[name], supported=True)
        except Exception as exc:
            return FieldReading(
                value=None,
                unit=FIELD_UNITS[name],
                supported=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def read_fields(self, index: int) -> dict[str, FieldReading]:
        handle = self.handles[index]
        nvml = self.nvml
        utilization: Any = None
        utilization_error: Exception | None = None
        try:
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
        except Exception as exc:
            utilization_error = exc

        def utilization_field(name: str, attribute: str) -> FieldReading:
            if utilization_error is not None:
                return FieldReading(
                    None,
                    FIELD_UNITS[name],
                    False,
                    f"{type(utilization_error).__name__}: {utilization_error}",
                )
            return FieldReading(int(getattr(utilization, attribute)), FIELD_UNITS[name], True)

        fields = {
            "gpu_utilization_pct": utilization_field("gpu_utilization_pct", "gpu"),
            "memory_utilization_pct": utilization_field("memory_utilization_pct", "memory"),
            "memory_used_bytes": self._read(
                "memory_used_bytes", lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used
            ),
            "power_draw_w": self._read(
                "power_draw_w",
                lambda: nvml.nvmlDeviceGetPowerUsage(handle),
                lambda value: float(value) / 1000.0,
            ),
            "temperature_c": self._read(
                "temperature_c",
                lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
            ),
            "sm_clock_mhz": self._read(
                "sm_clock_mhz",
                lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
            ),
            "memory_clock_mhz": self._read(
                "memory_clock_mhz",
                lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM),
            ),
            # NVML returns KiB/s for these calls; preserve that conversion explicitly.
            "pcie_tx_bytes_per_s": self._read(
                "pcie_tx_bytes_per_s",
                lambda: nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES),
                lambda value: int(value) * 1024,
            ),
            "pcie_rx_bytes_per_s": self._read(
                "pcie_rx_bytes_per_s",
                lambda: nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES),
                lambda value: int(value) * 1024,
            ),
        }
        assert set(fields) == set(TELEMETRY_FIELDS)
        return fields


@dataclass(frozen=True)
class CollectorDiagnostics:
    requested_interval_s: float
    sample_cycles: int
    mean_interval_s: float | None
    interval_std_s: float | None
    max_abs_jitter_s: float | None
    overrun_count: int
    mean_cycle_overhead_s: float | None
    collector_wall_duty_fraction: float | None
    field_missing_fraction: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


class TelemetryCollector:
    """One sampler thread covering every visible GPU."""

    def __init__(
        self,
        run_id: str,
        interval_s: float = 1.0,
        backend: TelemetryBackend | None = None,
        expected_gpus: int = 2,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.run_id = run_id
        self.interval_s = interval_s
        self.backend = backend or NvmlBackend()
        self.expected_gpus = expected_gpus
        self.samples: list[TelemetrySample] = []
        self._cycle_starts: list[float] = []
        self._cycle_costs: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("collector already started")
        self._thread = threading.Thread(target=self._run, name="commguard-nvml", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> CollectorDiagnostics:
        self._stop.set()
        if self._thread is None:
            raise RuntimeError("collector was not started")
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("telemetry collector did not stop")
        if self.error is not None:
            raise RuntimeError("telemetry collector failed") from self.error
        return self.diagnostics()

    def collect_for(self, duration_s: float) -> CollectorDiagnostics:
        self.start()
        self._stop.wait(duration_s)
        return self.stop()

    def _run(self) -> None:
        try:
            self.backend.initialize()
            count = self.backend.device_count()
            if count != self.expected_gpus:
                raise RuntimeError(
                    f"expected {self.expected_gpus} GPUs, telemetry backend found {count}"
                )
            uuids = [self.backend.device_uuid(index) for index in range(count)]
            next_sample = time.monotonic()
            sequence = 0
            while not self._stop.is_set():
                started = time.monotonic()
                self._cycle_starts.append(started)
                monotonic_ns = time.monotonic_ns()
                wall_time = datetime.now(timezone.utc).isoformat()
                for index in range(count):
                    sample = TelemetrySample(
                        run_id=self.run_id,
                        gpu_index=index,
                        gpu_uuid=uuids[index],
                        sequence=sequence,
                        wall_time_utc=wall_time,
                        monotonic_ns=monotonic_ns,
                        fields=self.backend.read_fields(index),
                    )
                    sample.validate()
                    self.samples.append(sample)
                self._cycle_costs.append(time.monotonic() - started)
                sequence += 1
                next_sample += self.interval_s
                delay = next_sample - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
        except BaseException as exc:
            self.error = exc
        finally:
            try:
                self.backend.shutdown()
            except BaseException as exc:
                if self.error is None:
                    self.error = exc

    def diagnostics(self) -> CollectorDiagnostics:
        intervals = [
            current - previous
            for previous, current in zip(self._cycle_starts, self._cycle_starts[1:], strict=False)
        ]
        missing = {name: 0 for name in TELEMETRY_FIELDS}
        for sample in self.samples:
            for name, reading in sample.fields.items():
                if not reading.supported:
                    missing[name] += 1
        denominator = max(len(self.samples), 1)
        costs = self._cycle_costs
        return CollectorDiagnostics(
            requested_interval_s=self.interval_s,
            sample_cycles=len(self._cycle_starts),
            mean_interval_s=statistics.fmean(intervals) if intervals else None,
            interval_std_s=statistics.pstdev(intervals) if len(intervals) > 1 else None,
            max_abs_jitter_s=(
                max(abs(value - self.interval_s) for value in intervals) if intervals else None
            ),
            overrun_count=sum(cost > self.interval_s for cost in costs),
            mean_cycle_overhead_s=statistics.fmean(costs) if costs else None,
            collector_wall_duty_fraction=(
                min(1.0, statistics.fmean(costs) / self.interval_s) if costs else None
            ),
            field_missing_fraction={name: count / denominator for name, count in missing.items()},
        )


def compare_sampling_rates(
    run_id: str,
    rates_hz: tuple[float, ...] = (1.0, 2.0, 10.0),
    duration_s: float = 5.0,
    backend_factory: Any = NvmlBackend,
) -> list[dict[str, Any]]:
    """Measure jitter/overhead at several rates on a bounded control interval."""
    output = []
    for rate in rates_hz:
        collector = TelemetryCollector(
            f"{run_id}-{rate:g}hz", interval_s=1.0 / rate, backend=backend_factory()
        )
        output.append({"rate_hz": rate, **collector.collect_for(duration_s).to_dict()})
    return output


def measure_runtime_overhead(
    workload: Any,
    collector_factory: Any,
    repeats: int = 3,
) -> dict[str, Any]:
    """Compare the same bounded callable with and without collection.

    The caller is responsible for making the workload deterministic and for
    ensuring it is safe to repeat. This measures end-to-end wall and process
    time, unlike the collector's internal wall-duty diagnostic.
    """
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    baseline_wall: list[float] = []
    baseline_cpu: list[float] = []
    collected_wall: list[float] = []
    collected_cpu: list[float] = []
    for with_collector in (False, True):
        for _ in range(repeats):
            collector = collector_factory() if with_collector else None
            if collector is not None:
                collector.start()
            wall_started = time.perf_counter()
            cpu_started = time.process_time()
            workload()
            wall_elapsed = time.perf_counter() - wall_started
            cpu_elapsed = time.process_time() - cpu_started
            if collector is not None:
                collector.stop()
                collected_wall.append(wall_elapsed)
                collected_cpu.append(cpu_elapsed)
            else:
                baseline_wall.append(wall_elapsed)
                baseline_cpu.append(cpu_elapsed)
    baseline_wall_median = statistics.median(baseline_wall)
    baseline_cpu_median = statistics.median(baseline_cpu)
    collected_wall_median = statistics.median(collected_wall)
    collected_cpu_median = statistics.median(collected_cpu)
    return {
        "repeats": repeats,
        "baseline_wall_median_s": baseline_wall_median,
        "collected_wall_median_s": collected_wall_median,
        "wall_slowdown_fraction": (
            collected_wall_median / baseline_wall_median - 1 if baseline_wall_median else None
        ),
        "baseline_process_cpu_median_s": baseline_cpu_median,
        "collected_process_cpu_median_s": collected_cpu_median,
        "process_cpu_increase_fraction": (
            collected_cpu_median / baseline_cpu_median - 1 if baseline_cpu_median else None
        ),
        "method": ("median repeated runtime comparison; callable order baseline then collected"),
    }
