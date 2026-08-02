from __future__ import annotations

import pytest

from commguard.duration import DurationController, DurationPolicy
from commguard.exceptions import DurationRequirementError
from commguard.workloads import list_workloads


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1e9)


def test_duration_controller_runs_warmup_plus_required_interval() -> None:
    clock = FakeClock()
    controller = DurationController(
        DurationPolicy(2.0, 5.0, None, 1),
        clock_ns=clock,
    )

    while controller.should_continue():
        clock.advance(1.0)
        controller.complete_iteration()
    interval = controller.finish()

    assert interval.iterations_completed == 7
    assert interval.measured_duration_seconds == 5.0
    assert interval.measurement_start_monotonic_ns == 2_000_000_000
    assert interval.stop_reason == "minimum_measured_duration_reached"


def test_iteration_cap_fails_instead_of_shortening_measurement() -> None:
    clock = FakeClock()
    controller = DurationController(
        DurationPolicy(2.0, 5.0, 3, 1),
        clock_ns=clock,
    )
    for _ in range(3):
        assert controller.should_continue()
        clock.advance(1.0)
        controller.complete_iteration()

    with pytest.raises(DurationRequirementError, match="required=5.000000s"):
        controller.should_continue()


def test_legacy_iteration_mode_remains_bounded() -> None:
    clock = FakeClock()
    controller = DurationController(
        DurationPolicy(0.0, None, None, 3),
        clock_ns=clock,
    )
    while controller.should_continue():
        clock.advance(0.1)
        controller.complete_iteration()

    assert controller.finish().iterations_completed == 3


def test_benign_workloads_declare_primary_measurement_duration() -> None:
    workloads = list_workloads()
    benign = [
        config
        for config in workloads.values()
        if config["designation"] == "benign" and not config.get("optional")
    ]

    assert benign
    assert all(config["warmup_seconds"] >= 0 for config in benign)
    assert all(config["min_measured_seconds"] >= 30 for config in benign)
