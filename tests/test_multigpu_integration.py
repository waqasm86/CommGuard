from __future__ import annotations

import json

import pytest

from commguard.environment.preflight import check_environment
from commguard.exceptions import ReadinessError
from commguard.orchestrator import run_experiment

pytestmark = [pytest.mark.gpu, pytest.mark.multigpu, pytest.mark.slow]


def require_dual_t4() -> None:
    try:
        check_environment(strict=True)
    except ReadinessError as exc:
        pytest.skip(str(exc))


def test_two_rank_nccl_smoke(tmp_path) -> None:
    require_dual_t4()
    outcome = run_experiment(
        "collective_all_reduce_1mib",
        output=tmp_path,
        overrides={"iterations": 2, "burst_iterations": 2},
        timeout_s=120,
    )
    assert outcome["manifest"]["participation_valid"]
    assert outcome["manifest"]["rank_exit_codes"] == {"0": 0, "1": 0}


def test_two_rank_idle_calibration_has_no_measured_collective(tmp_path) -> None:
    require_dual_t4()
    outcome = run_experiment(
        "calibration_idle",
        output=tmp_path,
        overrides={"min_measured_seconds": 2.0, "idle_interval_s": 0.25},
        timeout_s=60,
    )
    assert outcome["manifest"]["participation_valid"]
    assert outcome["manifest"]["rank_exit_codes"] == {"0": 0, "1": 0}
    for rank in (0, 1):
        event_path = tmp_path / "runs" / outcome["run_id"] / f"rank-{rank}.events.jsonl"
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        interval = next(event for event in events if event["event"] == "measurement_interval")
        start = interval["details"]["measurement_start_monotonic_ns"]
        end = interval["details"]["measurement_end_monotonic_ns"]
        measured = [event for event in events if start <= event["monotonic_ns"] <= end]
        assert not any("collective" in event["event"] for event in measured)


def test_injected_rank_crash_preserves_failure_artifacts(tmp_path) -> None:
    require_dual_t4()
    outcome = run_experiment(
        "collective_all_reduce_1mib",
        output=tmp_path,
        overrides={"inject_rank_crash": 1},
        timeout_s=60,
        raise_on_failure=False,
    )
    manifest_path = tmp_path / "runs" / outcome["run_id"] / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["exit_status"] == "failed"
    assert not manifest["participation_valid"]


def test_malformed_collective_is_rejected_before_launch(tmp_path) -> None:
    require_dual_t4()
    with pytest.raises(ValueError, match="unsupported calibration collective"):
        run_experiment(
            "collective_all_reduce_1mib",
            output=tmp_path,
            overrides={"collective": "not_a_collective"},
            timeout_s=60,
            raise_on_failure=False,
        )


def test_timeout_terminates_process_tree(tmp_path) -> None:
    require_dual_t4()
    outcome = run_experiment(
        "collective_all_reduce_1mib",
        output=tmp_path,
        overrides={"inject_timeout_rank": 0},
        timeout_s=15,
        raise_on_failure=False,
    )
    assert outcome["manifest"]["exit_status"] in {"timeout", "failed"}
    launch = json.loads((tmp_path / "runs" / outcome["run_id"] / "launch.json").read_text())
    assert launch["cleanup_complete"]
