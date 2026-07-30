from __future__ import annotations

import pytest

from commguard.evaluation import audit_leakage, evaluate_detector, grouped_split
from commguard.exceptions import CalibrationError


def rows(sessions: int = 1) -> list[dict]:
    output = []
    for run in range(12):
        for window in range(3):
            output.append(
                {
                    "run_id": f"run-{run}",
                    "session_fingerprint": f"session-{run % sessions}",
                    "target_label": "training" if run % 2 else "inference",
                    "window_seconds": 5,
                    "feature": run + window,
                }
            )
    return output


def test_grouped_split_keeps_all_windows_together() -> None:
    assignments = grouped_split(rows())
    assert set(assignments.values()) >= {"train", "test"}
    assert len(assignments) == 12
    audit = audit_leakage(rows(), ["feature"], assignments)
    assert audit["passed"]


def test_grouped_split_rejects_inconsistent_run_metadata() -> None:
    records = rows()
    records[1] = {**records[1], "session_fingerprint": "different-session"}
    with pytest.raises(ValueError, match="spans multiple session"):
        grouped_split(records)

    records = rows()
    records[1] = {**records[1], "target_label": "different-label"}
    with pytest.raises(ValueError, match="spans multiple target"):
        grouped_split(records)


def test_grouped_split_rejects_empty_input_and_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="empty"):
        grouped_split([])
    with pytest.raises(ValueError, match="sum"):
        grouped_split(rows(), test_fraction=0.6, validation_fraction=0.4)


def test_three_sessions_use_whole_session_holdout() -> None:
    records = rows(sessions=3)
    assignments = grouped_split(records)
    session_splits = {}
    for record in records:
        session_splits.setdefault(record["session_fingerprint"], set()).add(
            assignments[record["run_id"]]
        )
    assert all(len(splits) == 1 for splits in session_splits.values())
    assert {next(iter(splits)) for splits in session_splits.values()} == {
        "train",
        "validation",
        "test",
    }


def test_two_sessions_reserve_complete_test_session() -> None:
    records = rows(sessions=2)
    assignments = grouped_split(records)
    test_sessions = {
        record["session_fingerprint"]
        for record in records
        if assignments[record["run_id"]] == "test"
    }
    assert len(test_sessions) == 1
    heldout = next(iter(test_sessions))
    assert all(
        assignments[record["run_id"]] == "test"
        for record in records
        if record["session_fingerprint"] == heldout
    )


def test_leakage_audit_rejects_identity_feature() -> None:
    records = rows()
    audit = audit_leakage(records, ["gpu0__mean", "source_path"], grouped_split(records))
    assert not audit["passed"]
    assert audit["forbidden_feature_columns"] == ["source_path"]


def test_detector_fitting_requires_calibration(tmp_path) -> None:
    with pytest.raises(CalibrationError, match="calibration is missing"):
        evaluate_detector(tmp_path)
