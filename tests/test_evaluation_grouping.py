from __future__ import annotations

import warnings

import pytest

from commguard.evaluation import (
    audit_leakage,
    evaluate_detector,
    grouped_split,
    make_split_plan,
)
from commguard.evaluation.baselines import (
    _evaluate_ablation,
    _select_model_and_threshold,
    _stable_sigmoid,
)
from commguard.exceptions import CoverageError


def rows(sessions: int = 1) -> list[dict]:
    output = []
    for run in range(12):
        for window in range(3):
            output.append(
                {
                    "run_id": f"run-{run}",
                    "experiment_session_id": f"session-{(run // 2) % sessions}",
                    "session_fingerprint": f"environment-{run}",
                    "legacy_grouping_ambiguous": False,
                    "target_label": "training" if run % 2 else "inference",
                    "workload_family": "ddp_training" if run % 2 else "control_idle",
                    "workload_config_id": f"config-{run % 4}",
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
    records[1] = {**records[1], "experiment_session_id": "different-session"}
    with pytest.raises(ValueError, match="spans multiple experiment_session_id"):
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


def test_primary_split_rejects_one_target_class() -> None:
    records = [
        {
            **record,
            "target_label": "training",
            "workload_family": "ddp_training",
        }
        for record in rows()
    ]

    with pytest.raises(ValueError, match="at least two target classes"):
        make_split_plan(records)


def test_three_sessions_use_whole_session_holdout() -> None:
    records = rows(sessions=3)
    assignments = grouped_split(records)
    session_splits = {}
    for record in records:
        session_splits.setdefault(record["experiment_session_id"], set()).add(
            assignments[record["run_id"]]
        )
    assert all(len(splits) == 1 for splits in session_splits.values())
    assert {next(iter(splits)) for splits in session_splits.values()} == {
        "train",
        "validation",
        "test",
    }


def test_two_sessions_fall_back_to_stratified_whole_runs() -> None:
    records = rows(sessions=2)
    plan = make_split_plan(records)

    assert plan.actual_strategy == "deterministic_class_stratified_whole_run"
    assert all(
        set(plan.class_run_counts_by_split[split]) == {"inference", "training"}
        for split in ("train", "validation", "test")
    )


def test_per_run_environment_fingerprints_are_never_treated_as_sessions() -> None:
    records = rows(sessions=1)
    assert len({record["session_fingerprint"] for record in records}) == 12

    plan = make_split_plan(records)

    assert plan.actual_strategy == "deterministic_class_stratified_whole_run"
    assert set().union(*map(set, plan.session_ids_by_split.values())) == {"session-0"}


def test_ambiguous_legacy_grouping_is_rejected() -> None:
    records = rows()
    records[0] = {**records[0], "legacy_grouping_ambiguous": True}

    with pytest.raises(ValueError, match="ambiguous legacy grouping"):
        grouped_split(records)


def test_explicit_session_holdout_rejects_missing_family_coverage() -> None:
    records = rows(sessions=3)
    for record in records:
        if record["experiment_session_id"] == "session-2":
            record["workload_family"] = "control_idle"

    with pytest.raises(ValueError, match="no valid independent-session holdout"):
        make_split_plan(
            records,
            mode="session_holdout",
            required_families=("ddp_training", "control_idle"),
        )


def test_stratified_fallback_preserves_required_families_in_every_split() -> None:
    plan = make_split_plan(
        rows(sessions=1),
        required_families=("ddp_training", "control_idle"),
    )

    assert plan.actual_strategy == "deterministic_class_stratified_whole_run"
    assert all(
        set(plan.family_run_counts_by_split[split]) == {"ddp_training", "control_idle"}
        for split in ("train", "validation", "test")
    )


def test_primary_split_fails_when_a_required_family_has_too_few_runs() -> None:
    records = [record for record in rows() if record["workload_family"] != "ddp_training"]
    records.extend(
        [
            {
                **rows()[index * 3],
                "run_id": f"rare-training-{index}",
                "target_label": "training",
                "workload_family": "ddp_training",
            }
            for index in range(2)
        ]
    )

    with pytest.raises(ValueError, match="at least three"):
        make_split_plan(
            records,
            required_families=("ddp_training", "control_idle"),
        )


def test_family_and_configuration_holdouts_are_explicit_diagnostics() -> None:
    records = rows()
    family = make_split_plan(
        records,
        mode="family_holdout",
        holdout_value="ddp_training",
    )
    configuration = make_split_plan(
        records,
        mode="configuration_holdout",
        holdout_value="config-0",
    )

    assert family.diagnostic_only is True
    assert all(
        family.assignments[record["run_id"]] == "test"
        for record in records
        if record["workload_family"] == "ddp_training"
    )
    assert configuration.actual_strategy == "configuration_holdout"


def test_model_and_threshold_selection_do_not_consult_test_rows() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    records = []
    for run in range(18):
        target = run % 2
        records.append(
            {
                "run_id": f"run-{run}",
                "experiment_session_id": "session-a",
                "session_fingerprint": f"environment-{run}",
                "legacy_grouping_ambiguous": False,
                "target_label": "training" if target else "inference",
                "workload_family": "ddp_training" if target else "control_idle",
                "workload_config_id": f"config-{run % 6}",
                "window_seconds": 30.0,
                "gpu0__pcie_tx_bytes_per_s__mean": float(run + target * 20),
                "gpu1__pcie_rx_bytes_per_s__mean": float(run * 0.5 + target * 10),
                "_target": target,
            }
        )
    plan = make_split_plan(records)
    frame = pd.DataFrame(records)
    columns = [
        "gpu0__pcie_tx_bytes_per_s__mean",
        "gpu1__pcie_rx_bytes_per_s__mean",
    ]
    first = _evaluate_ablation(frame, plan.assignments, columns)
    changed = frame.copy()
    test_mask = changed["run_id"].map(plan.assignments) == "test"
    changed.loc[test_mask, columns] = changed.loc[test_mask, columns] * -1000
    changed.loc[test_mask, "_target"] = 1 - changed.loc[test_mask, "_target"]
    second = _evaluate_ablation(changed, plan.assignments, columns)

    assert first["selected_model"] == second["selected_model"]
    assert first["selected_probability_threshold"] == second["selected_probability_threshold"]
    assert first["selection_protocol"]["test_used_for_selection"] is False
    assert first["feature_preprocessing"]["fit_partition"] == "train"
    assert first["run_level"]["balanced_accuracy_bootstrap"]["ci95"] is None
    assert (
        "insufficient independent groups"
        in first["run_level"]["balanced_accuracy_bootstrap"]["warning"]
    )


def test_selection_api_accepts_validation_data_only() -> None:
    selected, threshold, scores = _select_model_and_threshold(
        [0, 0, 1, 1],
        {
            "bad": [0.8, 0.7, 0.2, 0.1],
            "good": [0.1, 0.2, 0.8, 0.9],
        },
    )

    assert selected == "good"
    assert threshold == 0.5
    assert scores == {"bad": 0.0, "good": 1.0}


def test_leakage_audit_rejects_identity_feature() -> None:
    records = rows()
    audit = audit_leakage(records, ["gpu0__mean", "source_path"], grouped_split(records))
    assert not audit["passed"]
    assert audit["forbidden_feature_columns"] == ["source_path"]


def test_detector_fitting_requires_calibration(tmp_path) -> None:
    with pytest.raises(CoverageError, match="no feature extraction coverage summary"):
        evaluate_detector(tmp_path)


def test_stable_sigmoid_handles_extreme_logits_without_warnings() -> None:
    np = pytest.importorskip("numpy")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        probabilities = _stable_sigmoid(np.array([-1e300, -30.0, 0.0, 30.0, 1e300]))

    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities[0] < probabilities[1] < probabilities[2]
    assert probabilities[2] < probabilities[3] <= probabilities[4]
    assert probabilities[2] == 0.5
