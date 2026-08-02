from __future__ import annotations

import pytest

from commguard.distributed import worker
from commguard.distributed.participation import validate_rank_results
from commguard.distributed.smoke import main
from commguard.distributed.worker import SUPPORTED_WORKER_MODES, WORKER_MODE_HANDLERS
from commguard.exceptions import ValidationError
from commguard.workloads import WORKLOADS, get_workload


def test_smoke_without_torchrun_fails_explicitly(tmp_path, capsys) -> None:
    assert main(["--output", str(tmp_path)]) == 2
    error = capsys.readouterr().err
    assert "ReadinessError" in error
    assert "torchrun" in error or "PyTorch is required" in error


def test_rank_results_require_distinct_gpu_uuids() -> None:
    results = [
        {
            "rank": rank,
            "local_rank": rank,
            "world_size": 2,
            "backend": "nccl",
            "gpu_uuid": f"GPU-{rank}",
            "all_reduce_value": 3,
            "completed": True,
        }
        for rank in (0, 1)
    ]
    assert validate_rank_results(results)["participation_valid"]
    results[1]["gpu_uuid"] = "GPU-0"
    with pytest.raises(ValidationError, match="distinct GPU UUID"):
        validate_rank_results(results)


def test_calibration_idle_resolves_to_a_supported_worker_mode() -> None:
    assert get_workload("calibration_idle")["mode"] in SUPPORTED_WORKER_MODES


def test_every_supported_mode_and_workload_has_exact_dispatch() -> None:
    assert frozenset(WORKER_MODE_HANDLERS) == SUPPORTED_WORKER_MODES
    assert all(callable(handler) for handler in WORKER_MODE_HANDLERS.values())
    assert {str(config["mode"]) for config in WORKLOADS.values()} <= SUPPORTED_WORKER_MODES


def test_idle_handler_emits_lifecycle_without_collective() -> None:
    class Cuda:
        synchronizations: list[int] = []

        @classmethod
        def synchronize(cls, local_rank: int) -> None:
            cls.synchronizations.append(local_rank)

    class Torch:
        cuda = Cuda()

    class Dist:
        barriers = 0

        @classmethod
        def barrier(cls) -> None:
            cls.barriers += 1

    class Writer:
        events: list[tuple[str, dict]] = []

        @classmethod
        def emit(cls, event: str, **details) -> None:
            cls.events.append((event, details))

    worker._run_idle(
        Torch(),
        Dist(),
        Writer(),
        {
            "mode": "idle",
            "warmup_seconds": 0,
            "min_measured_seconds": 0.01,
            "iteration_cap": None,
            "iterations": 1,
            "idle_interval_s": 0.002,
        },
        1,
    )

    names = [name for name, _ in Writer.events]
    assert names[0] == "measurement_start"
    assert "heartbeat" in names
    assert "measurement_interval" in names
    assert names[-1] == "measurement_end"
    assert not any("collective" in name for name in names)
    assert Cuda.synchronizations == [1, 1]
    assert Dist.barriers == 1


@pytest.mark.parametrize("idle_interval_s", [0, -0.1, 5.1])
def test_idle_handler_rejects_invalid_poll_interval(idle_interval_s) -> None:
    with pytest.raises(ValueError, match="idle_interval_s"):
        worker._run_idle(
            object(),
            object(),
            object(),
            {"idle_interval_s": idle_interval_s},
            0,
        )
