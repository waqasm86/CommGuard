"""Deterministic measurement-duration stop conditions for bounded workloads."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from commguard.exceptions import DurationRequirementError


@dataclass(frozen=True)
class DurationPolicy:
    warmup_seconds: float
    minimum_measured_seconds: float | None
    iteration_cap: int | None
    legacy_iterations: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DurationPolicy:
        minimum = config.get("min_measured_seconds")
        policy = cls(
            warmup_seconds=float(config.get("warmup_seconds", 0.0)),
            minimum_measured_seconds=float(minimum) if minimum is not None else None,
            iteration_cap=(
                int(config["iteration_cap"]) if config.get("iteration_cap") is not None else None
            ),
            legacy_iterations=int(config.get("iterations", 1)),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.warmup_seconds < 0:
            raise ValueError("warmup_seconds cannot be negative")
        if self.minimum_measured_seconds is not None and self.minimum_measured_seconds <= 0:
            raise ValueError("min_measured_seconds must be positive")
        if self.iteration_cap is not None and self.iteration_cap < 1:
            raise ValueError("iteration_cap must be positive when provided")
        if self.legacy_iterations < 1:
            raise ValueError("iterations must be positive")


@dataclass(frozen=True)
class MeasurementInterval:
    measurement_start_monotonic_ns: int
    measurement_end_monotonic_ns: int
    measured_duration_seconds: float
    configured_warmup_seconds: float
    required_measured_seconds: float | None
    iteration_cap: int | None
    iterations_completed: int
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DurationController:
    """Choose the next iteration from elapsed time, with an optional hard cap."""

    def __init__(
        self,
        policy: DurationPolicy,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        policy.validate()
        self.policy = policy
        self.clock_ns = clock_ns
        self.started_ns = clock_ns()
        self.measurement_start_ns = self.started_ns + int(policy.warmup_seconds * 1e9)
        self.iterations_completed = 0

    def measured_seconds(self, now_ns: int | None = None) -> float:
        current = self.clock_ns() if now_ns is None else now_ns
        return max(0.0, (current - self.measurement_start_ns) / 1e9)

    def should_continue(self) -> bool:
        if self.policy.minimum_measured_seconds is None:
            return self.iterations_completed < self.policy.legacy_iterations
        measured = self.measured_seconds()
        if measured >= self.policy.minimum_measured_seconds and self.iterations_completed > 0:
            return False
        if (
            self.policy.iteration_cap is not None
            and self.iterations_completed >= self.policy.iteration_cap
        ):
            raise DurationRequirementError(
                "measurement duration unmet: "
                f"required={self.policy.minimum_measured_seconds:.6f}s "
                f"actual={measured:.6f}s iterations={self.iterations_completed} "
                f"cap={self.policy.iteration_cap}"
            )
        return True

    def complete_iteration(self) -> None:
        self.iterations_completed += 1

    def finish(self) -> MeasurementInterval:
        ended_ns = self.clock_ns()
        measured = self.measured_seconds(ended_ns)
        if (
            self.policy.minimum_measured_seconds is not None
            and measured < self.policy.minimum_measured_seconds
        ):
            raise DurationRequirementError(
                "measurement duration unmet at finish: "
                f"required={self.policy.minimum_measured_seconds:.6f}s actual={measured:.6f}s"
            )
        return MeasurementInterval(
            measurement_start_monotonic_ns=self.measurement_start_ns,
            measurement_end_monotonic_ns=ended_ns,
            measured_duration_seconds=measured,
            configured_warmup_seconds=self.policy.warmup_seconds,
            required_measured_seconds=self.policy.minimum_measured_seconds,
            iteration_cap=self.policy.iteration_cap,
            iterations_completed=self.iterations_completed,
            stop_reason=(
                "minimum_measured_duration_reached"
                if self.policy.minimum_measured_seconds is not None
                else "legacy_iteration_count_reached"
            ),
        )
