from __future__ import annotations

from collections import defaultdict

import pytest

from commguard.features import PRIMARY_BENIGN_FAMILIES
from commguard.orchestrator import estimate_matrix, plan_matrix
from commguard.workloads import (
    BENIGN_REQUIRED_FAMILIES,
    get_workload,
    list_workloads,
    profile_workloads,
    validate_workload_registry,
)

EXPECTED_EXECUTION_MODES = {
    "ddp_training": "ddp_train",
    "inference_prefill_independent": "inference_independent",
    "inference_decode_independent": "inference_independent",
    "inference_synchronized": "inference_synchronized",
    "control_compute": "control_compute",
    "control_host_transfer": "control_host_transfer",
    "control_model_or_checkpoint_load": "control_model_load",
    "control_idle": "control_idle",
}


def test_required_benign_families_have_cpu_inspectable_execution_plans() -> None:
    validate_workload_registry()
    plans = plan_matrix("standard", repetitions=1)

    assert PRIMARY_BENIGN_FAMILIES == BENIGN_REQUIRED_FAMILIES
    assert {plan.workload_family for plan in plans} == set(BENIGN_REQUIRED_FAMILIES)
    assert len(plans) == len(BENIGN_REQUIRED_FAMILIES)
    for plan in plans:
        assert plan.config["mode"] == EXPECTED_EXECUTION_MODES[plan.workload_family]
        assert plan.config["min_measured_seconds"] >= 35
        assert plan.config["estimated_seconds"] >= (
            plan.config["warmup_seconds"] + plan.config["min_measured_seconds"]
        )
        assert plan.resolved_config_id() == plan.config["config_id"]


def test_matrix_plan_is_deterministic_but_repetition_slots_are_unique() -> None:
    first = plan_matrix("standard", repetitions=3)
    second = plan_matrix("standard", repetitions=3)

    assert first == second
    assert len(first) == 3 * len(BENIGN_REQUIRED_FAMILIES)
    assert len({plan.plan_id for plan in first}) == len(first)
    by_family: dict[str, set[str]] = defaultdict(set)
    for plan in first:
        by_family[plan.workload_family].add(plan.resolved_config_id())
    assert all(len(config_ids) == 1 for config_ids in by_family.values())


def test_extended_profile_is_opt_in_and_varies_bounded_parameters() -> None:
    standard = profile_workloads("standard")
    extended = profile_workloads("extended")
    configs = [get_workload(name) for name in extended]

    assert extended[: len(standard)] == standard
    assert "control_peer_copy" not in standard
    assert get_workload("control_peer_copy")["optional"] is True
    assert any(config.get("sequence_length") == 512 for config in configs)
    assert {config.get("batch_size") for config in configs} >= {1, 2, 4}
    assert {config.get("precision") for config in configs} >= {"float16", "float32"}
    assert any(config.get("hidden_size") == 192 for config in configs)
    assert {config.get("barrier_every") for config in configs} >= {1, 4}
    assert {config.get("payload_mib") for config in configs} >= {16, 64}
    assert estimate_matrix("extended", repetitions=1)["run_count"] == len(extended)


def test_registry_config_ids_are_explicit_and_unique() -> None:
    workloads = list_workloads()
    config_ids = [str(config["config_id"]) for config in workloads.values()]

    assert len(config_ids) == len(set(config_ids))
    assert all(config_id.rsplit("-v", 1)[-1].isdigit() for config_id in config_ids)


def test_plan_rejects_non_positive_repetition_count() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plan_matrix("standard", repetitions=0)
