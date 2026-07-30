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


def test_malformed_collective_is_recorded(tmp_path) -> None:
    require_dual_t4()
    outcome = run_experiment(
        "collective_all_reduce_1mib",
        output=tmp_path,
        overrides={"collective": "not_a_collective"},
        timeout_s=60,
        raise_on_failure=False,
    )
    assert outcome["manifest"]["exit_status"] == "failed"
    assert outcome["manifest"]["failure_reason"]


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
    launch = json.loads(
        (tmp_path / "runs" / outcome["run_id"] / "launch.json").read_text()
    )
    assert launch["cleanup_complete"]
