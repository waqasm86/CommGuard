from __future__ import annotations

import json
from copy import deepcopy

import pytest

from commguard import orchestrator
from commguard.adversarial import (
    ADVERSARIAL_STRATEGIES,
    AdversarialHoldoutPlan,
    should_synchronize,
    simulate_parameter_averaging,
    strategy_for_config,
    synchronization_steps,
)
from commguard.artifacts import ArtifactStore
from commguard.corpus import CorpusManifest
from commguard.distributed.worker import SUPPORTED_WORKER_MODES
from commguard.exceptions import ApprovalRequiredError
from commguard.schemas import LEGACY_SCHEMA_VERSION
from commguard.workloads import (
    WORKLOADS,
    adversarial_profile_workloads,
    get_workload,
    profile_workloads,
)

EXPECTED_FAMILIES = {
    "gradient_accumulation",
    "periodic_local_sgd",
    "diloco_inspired",
    "segmented_runs",
    "idle_padding",
    "randomized_synchronization",
    "mixed_training_inference",
    "synthetic_communication_decoy",
}


def test_every_strategy_is_bounded_disabled_and_mapped_to_one_workload() -> None:
    names = adversarial_profile_workloads()
    configs = [get_workload(name) for name in names]

    assert set(ADVERSARIAL_STRATEGIES) == EXPECTED_FAMILIES
    assert {config["family"] for config in configs} == EXPECTED_FAMILIES
    assert {config["mode"] for config in configs} <= SUPPORTED_WORKER_MODES
    assert len(configs) == len(EXPECTED_FAMILIES)
    assert all(config["enabled_by_default"] is False for config in configs)
    assert all(config["requires_human_approval"] is True for config in configs)
    assert all(
        config["min_measured_seconds"] >= 35
        or (
            config["strategy_id"] == "segmented_runs"
            and config["segment_count"] * config["segment_seconds"] >= 35
        )
        for config in configs
    )
    assert not set(names) & set(profile_workloads("smoke"))
    assert not set(names) & set(profile_workloads("standard"))
    assert not set(names) & set(profile_workloads("extended"))
    for config in configs:
        strategy = strategy_for_config(config)
        assert strategy.research_purpose
        assert strategy.synchronization_semantics
        assert strategy.claim_boundary
    assert "DiLoCo-inspired" in ADVERSARIAL_STRATEGIES["diloco_inspired"].claim_boundary


def test_strategy_parameter_bounds_fail_before_gpu_execution() -> None:
    config = deepcopy(WORKLOADS["adversarial_periodic_local_sgd"])
    config["local_steps"] = 101

    with pytest.raises(ValueError, match="outside"):
        strategy_for_config(config)


def test_adversarial_run_requires_approval_before_artifact_or_preflight(
    tmp_path, monkeypatch
) -> None:
    preflight_called = False

    def preflight(*args, **kwargs):
        nonlocal preflight_called
        preflight_called = True
        raise AssertionError("preflight must not run before the approval gate")

    monkeypatch.setattr(orchestrator, "check_environment", preflight)

    with pytest.raises(ApprovalRequiredError, match="bounded defensive red-team"):
        orchestrator.run_experiment("adversarial_idle_padding", output=tmp_path)

    assert preflight_called is False
    assert not list(tmp_path.iterdir())

    with pytest.raises(ValueError, match="hidden_size"):
        orchestrator.run_experiment(
            "adversarial_periodic_local_sgd",
            output=tmp_path,
            overrides={"hidden_size": 1024},
            adversarial_approval=True,
        )
    assert preflight_called is False
    assert not list(tmp_path.iterdir())

    with pytest.raises(ApprovalRequiredError, match="explicit approval"):
        orchestrator.run_segmented_series(output=tmp_path)
    assert not list(tmp_path.iterdir())

    with pytest.raises(ValueError, match="outside"):
        orchestrator.run_experiment(
            "adversarial_periodic_local_sgd",
            output=tmp_path,
            overrides={"local_steps": 101},
            adversarial_approval=True,
        )
    assert preflight_called is False
    assert not list(tmp_path.iterdir())

    with pytest.raises(ValueError, match="identity fields"):
        orchestrator.run_experiment(
            "adversarial_idle_padding",
            output=tmp_path,
            overrides={"designation": "benign"},
            adversarial_approval=True,
        )
    assert not list(tmp_path.iterdir())


def _holdout_plan() -> AdversarialHoldoutPlan:
    return AdversarialHoldoutPlan(
        development_families=(
            "gradient_accumulation",
            "periodic_local_sgd",
            "segmented_runs",
            "idle_padding",
        ),
        hardening_families=(
            "randomized_synchronization",
            "mixed_training_inference",
            "synthetic_communication_decoy",
        ),
        final_family="diloco_inspired",
        final_session_ids=("session-reserved-final",),
        final_config_ids=("diloco-inspired-inner10-v1",),
    )


def test_adversarial_matrix_requires_both_approval_gates_before_writes(tmp_path) -> None:
    with pytest.raises(ApprovalRequiredError, match="explicit approval"):
        orchestrator.run_adversarial_matrix(
            output=tmp_path,
            holdout_plan=_holdout_plan(),
        )
    assert not list(tmp_path.iterdir())

    with pytest.raises(ApprovalRequiredError, match="separate explicit approval"):
        orchestrator.run_adversarial_matrix(
            output=tmp_path,
            holdout_plan=_holdout_plan(),
            adversarial_approval=True,
            release_final_adversarial_holdout=True,
        )
    assert not list(tmp_path.iterdir())


def test_adversarial_matrix_plans_before_execution_and_keeps_final_sealed(
    tmp_path, monkeypatch
) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize()
    store.write_json(
        "results/evaluation-accepted.json",
        {
            "artifact_kind": "evaluation_result",
            "schema_version": LEGACY_SCHEMA_VERSION,
            "coverage_gate": {"passed": True},
            "primary_communication_only": {"metrics": {}},
        },
        validate=False,
    )
    calls = []

    def experiment(workload, *args, **kwargs):
        plan_paths = list((tmp_path / "corpora").glob("*-plan.json"))
        assert len(plan_paths) == 1
        plan = CorpusManifest.from_dict(json.loads(plan_paths[0].read_text()))
        assert len(plan.planned_runs) == 11
        assert not plan.accepted_run_ids
        calls.append((workload, kwargs))
        return {
            "run_id": f"run-adversarial-{len(calls)}",
            "manifest": {"exit_status": "completed"},
        }

    gaps = []
    monkeypatch.setattr(orchestrator, "run_experiment", experiment)
    monkeypatch.setattr(orchestrator.time, "sleep", gaps.append)

    summary = orchestrator.run_adversarial_matrix(
        output=tmp_path,
        holdout_plan=_holdout_plan(),
        adversarial_approval=True,
    )

    assert summary["planned"] == 11
    assert summary["executed"] == 10
    assert summary["completed"] == 10
    assert summary["failed"] == 0
    assert summary["sealed"] == 1
    assert summary["detector_metrics_computed"] is False
    assert len(calls) == 10
    assert all(call[1]["adversarial_approval"] is True for call in calls)
    assert len({call[1]["provenance"].experiment_session_id for call in calls}) == 1
    assert gaps == [1.0, 1.0, 1.0]
    assert summary["family_counts"]["diloco_inspired"]["sealed"] == 1
    assert summary["family_counts"]["diloco_inspired"]["failed"] == 0
    saved = json.loads((tmp_path / summary["summary_artifact"]).read_text())
    assert saved["summary_artifact"] == summary["summary_artifact"]
    assert saved["executed"] == summary["executed"]
    assert saved["holdout_plan"]["final_family"] == "diloco_inspired"
    final = CorpusManifest.from_dict(
        json.loads((tmp_path / summary["final_corpus_manifest"]).read_text())
    )
    assert len(final.accepted_run_ids) == 10
    assert all(plan.designation == "adversarial" for plan in final.planned_runs)


def test_segmented_series_plans_real_process_segments_before_launch(tmp_path, monkeypatch) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize()
    store.write_json(
        "results/calibration-supported.json",
        {
            "artifact_kind": "calibration_result",
            "schema_version": LEGACY_SCHEMA_VERSION,
            "status": "supported",
            "observations": [],
            "falsification_reasons": [],
        },
    )
    observed = []

    def experiment(*args, **kwargs):
        plan_files = list((tmp_path / "corpora").glob("*-plan.json"))
        assert len(plan_files) == 1
        plan = CorpusManifest.from_dict(json.loads(plan_files[0].read_text()))
        assert not plan.accepted_run_ids
        observed.append(kwargs)
        return {
            "run_id": f"run-segment-{len(observed)}",
            "manifest": {"exit_status": "completed"},
        }

    gaps = []
    monkeypatch.setattr(orchestrator, "run_experiment", experiment)
    monkeypatch.setattr(orchestrator.time, "sleep", gaps.append)
    monkeypatch.setattr(orchestrator, "verify_calibration_reference", lambda *args, **kwargs: None)

    summary = orchestrator.run_segmented_series(
        output=tmp_path,
        adversarial_approval=True,
        calibration_reference={},
    )

    assert len(observed) == 4
    assert all(call["adversarial_approval"] is True for call in observed)
    assert all(call["overrides"]["min_measured_seconds"] == 10 for call in observed)
    assert len({call["provenance"].experiment_session_id for call in observed}) == 1
    assert gaps == [1.0, 1.0, 1.0]
    assert summary["completed"] == 4
    assert summary["failed"] == 0
    assert summary["primary_feature_coverage_expected"] is False
    final = CorpusManifest.from_dict(
        json.loads((tmp_path / summary["final_corpus_manifest"]).read_text())
    )
    assert final.accepted_run_ids == tuple(f"run-segment-{index}" for index in range(1, 5))


def test_sparse_sync_schedules_are_rank_independent_and_end_in_agreement() -> None:
    periodic = synchronization_steps("periodic_local_sgd", 12, {"local_steps": 5})
    diloco = synchronization_steps("diloco_inspired", 23, {"inner_steps": 10})
    randomized_first = synchronization_steps(
        "randomized_synchronization", 20, {"sync_probability": 0.25, "seed": 7}
    )
    randomized_second = synchronization_steps(
        "randomized_synchronization", 20, {"sync_probability": 0.25, "seed": 7}
    )

    assert periodic == (5, 10, 12)
    assert diloco == (10, 20, 23)
    assert randomized_first == randomized_second
    assert randomized_first[-1] == 20
    assert should_synchronize("periodic_local_sgd", 5, {"local_steps": 5})

    history = simulate_parameter_averaging(
        (0.0, 0.0),
        ((1.0, 3.0), (1.0, -1.0), (2.0, 4.0)),
        (1, 3),
    )
    assert history[0] == (2.0, 2.0)
    assert history[1][0] != history[1][1]
    assert history[2][0] == history[2][1]


def test_final_family_session_and_config_holdout_stays_sealed() -> None:
    plan = AdversarialHoldoutPlan(
        development_families=(
            "gradient_accumulation",
            "periodic_local_sgd",
            "segmented_runs",
            "idle_padding",
        ),
        hardening_families=(
            "randomized_synchronization",
            "mixed_training_inference",
            "synthetic_communication_decoy",
        ),
        final_family="diloco_inspired",
        final_session_ids=("session-final",),
        final_config_ids=("config-final",),
    )

    plan.validate()
    assert plan.is_final_identity(
        family="ddp_training",
        session_id="session-final",
        config_id="benign-config",
    )
    assert (
        plan.round_for(family="diloco_inspired", session_id="session-dev", config_id="config-dev")
        == "sealed_final_holdout"
    )
    assert (
        plan.round_for(
            family="gradient_accumulation",
            session_id="session-final",
            config_id="config-dev",
        )
        == "sealed_final_holdout"
    )
    assert (
        plan.round_for(
            family="gradient_accumulation",
            session_id="session-dev",
            config_id="config-final",
        )
        == "sealed_final_holdout"
    )
    assert (
        plan.round_for(
            family="gradient_accumulation", session_id="session-dev", config_id="config-dev"
        )
        == "development"
    )
    assert (
        plan.round_for(
            family="diloco_inspired",
            session_id="session-final",
            config_id="config-final",
            release_final=True,
        )
        == "final_test"
    )
