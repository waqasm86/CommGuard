from __future__ import annotations

from collections import defaultdict

from commguard.evaluation.splits import grouped_split


def test_every_window_from_each_run_stays_in_one_split() -> None:
    rows = [
        {
            "run_id": f"run-{run_id}",
            "experiment_session_id": "session-1",
            "session_fingerprint": f"environment-{run_id}",
            "legacy_grouping_ambiguous": False,
            "target_label": "training" if run_id % 2 else "inference",
            "workload_family": "ddp_training" if run_id % 2 else "control_idle",
            "workload_config_id": f"config-{run_id % 4}",
            "window_seconds": 5,
            "window_index": window_index,
        }
        for run_id in range(12)
        for window_index in range(4)
    ]
    assignments = grouped_split(rows)
    observed = defaultdict(set)
    for row in rows:
        observed[row["run_id"]].add(assignments[row["run_id"]])
    assert set(assignments) == {f"run-{run_id}" for run_id in range(12)}
    assert all(len(splits) == 1 for splits in observed.values())
