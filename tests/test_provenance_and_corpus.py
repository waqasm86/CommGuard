from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from commguard import orchestrator
from commguard.corpus import CorpusManifest, PlannedRun
from commguard.environment.preflight import _environment_fingerprint
from commguard.exceptions import ValidationError
from commguard.provenance import ProvenanceContext, new_run_id
from commguard.schemas import (
    CURRENT_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSION,
    RunManifest,
    load_artifact,
    validate_artifact,
)


def legacy_manifest() -> RunManifest:
    timestamp = datetime(2026, 8, 1, tzinfo=timezone.utc).isoformat()
    return RunManifest(
        run_id="legacy-run",
        workload_name="ddp_train",
        workload_label="training",
        workload_family="ddp_full_parameter",
        designation="benign",
        seed=1337,
        world_size=2,
        config={},
        environment={},
        environment_fingerprint="historical-per-run-fingerprint",
        source_commit="1e790895",
        started_at_utc=timestamp,
        ended_at_utc=timestamp,
        warmup_seconds=2.0,
        exit_status="completed",
        failure_category=None,
        failure_reason=None,
        rank_exit_codes={"0": 0, "1": 0},
        nccl_environment={},
        participation_valid=True,
        schema_version=LEGACY_SCHEMA_VERSION,
    )


def test_context_is_shared_across_runs_but_run_ids_are_unique(tmp_path) -> None:
    context = ProvenanceContext.create(
        corpus_id="corpus-benign-v2",
        experiment_session_id="session-kaggle-a",
        collection_id="collection-kaggle-a",
        node_id="node-0",
        repository_root=tmp_path,
    )

    first = new_run_id("ddp train", context.experiment_session_id)
    second = new_run_id("ddp train", context.experiment_session_id)

    assert first != second
    assert first.startswith("run-kaggle-a-ddp-train-")
    assert context.run_fields()["experiment_session_id"] == "session-kaggle-a"
    assert context.source_dirty is True


def test_separate_sessions_can_share_one_deliberate_corpus(tmp_path) -> None:
    first = ProvenanceContext.create(
        corpus_id="corpus-benign-v2",
        experiment_session_id="session-a",
        repository_root=tmp_path,
    )
    second = ProvenanceContext.create(
        corpus_id="corpus-benign-v2",
        experiment_session_id="session-b",
        repository_root=tmp_path,
    )

    assert first.corpus_id == second.corpus_id
    assert first.experiment_session_id != second.experiment_session_id
    assert first.collection_id != second.collection_id


def test_corpus_allow_list_prevents_calibration_idle_leakage() -> None:
    manifest = CorpusManifest(
        corpus_id="corpus-benign-v2",
        collection_id="collection-a",
        experiment_session_id="session-a",
        node_id="node-0",
        planned_runs=(
            PlannedRun(
                plan_id="idle-benign-0",
                workload_family="control_idle",
                target_label="control",
                config={},
                accepted_run_id="run-benign-idle",
            ),
            PlannedRun(
                plan_id="idle-calibration-0",
                workload_family="calibration_idle",
                target_label="calibration",
                config={},
                designation="calibration",
                accepted_run_id="run-calibration-idle",
            ),
        ),
        accepted_run_ids=("run-benign-idle", "run-calibration-idle"),
        source_commit="commit-a",
        source_dirty=False,
        notebook_version="benign-v2",
        input_archive_sha256=None,
        random_seed=1337,
    )

    payload = manifest.to_dict()
    validate_artifact(payload)
    assert manifest.allows("run-benign-idle", "benign")
    assert not manifest.allows("run-calibration-idle", "benign")
    assert manifest.allows("run-calibration-idle", "calibration")


def test_workload_configuration_identity_is_stable_across_repetitions() -> None:
    first = PlannedRun("repeat-0", "ddp_training", "training", {"batch_size": 4})
    second = PlannedRun("repeat-1", "ddp_training", "training", {"batch_size": 4})
    changed = PlannedRun("repeat-2", "ddp_training", "training", {"batch_size": 8})

    assert first.resolved_config_id() == second.resolved_config_id()
    assert first.resolved_config_id() != changed.resolved_config_id()


def test_legacy_manifest_loads_with_explicit_ambiguous_grouping(tmp_path) -> None:
    source = tmp_path / "manifest.json"
    source.write_text(__import__("json").dumps(legacy_manifest().to_dict()), encoding="utf-8")

    original = load_artifact(source)
    migrated = load_artifact(source, migrate_legacy=True)

    assert isinstance(original, dict) and original["schema_version"] == LEGACY_SCHEMA_VERSION
    assert isinstance(migrated, dict) and migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert migrated["source_schema_version"] == LEGACY_SCHEMA_VERSION
    assert migrated["legacy_grouping_ambiguous"] is True
    assert migrated["experiment_session_id"] is None
    validate_artifact(migrated)


def test_new_manifest_requires_true_grouping_fields() -> None:
    payload = legacy_manifest().to_dict()
    payload["schema_version"] = CURRENT_SCHEMA_VERSION

    with pytest.raises(ValidationError, match="experiment_session_id"):
        validate_artifact(payload)


def test_new_manifest_validates_measured_interval_provenance() -> None:
    payload = legacy_manifest().to_dict()
    payload.update(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "experiment_session_id": "session-a",
            "collection_id": "collection-a",
            "corpus_id": "corpus-a",
            "node_id": "node-0",
            "source_dirty": False,
            "random_seed": 1337,
            "measurement_start_monotonic_ns": 1_000_000_000,
            "measurement_end_monotonic_ns": 31_000_000_000,
            "measured_duration_seconds": 30.0,
        }
    )
    validate_artifact(payload)

    payload["measured_duration_seconds"] = 29.0
    with pytest.raises(ValidationError, match="must equal"):
        validate_artifact(payload)


def test_environment_fingerprint_excludes_live_measurement_values() -> None:
    base = {
        "gpus": [{"uuid": "GPU-0"}, {"uuid": "GPU-1"}],
        "platform": "test",
        "kernel": "test",
        "torch": {"version": "test"},
        "packages": {"torch": "test"},
        "hostname": "node-0",
        "kaggle_session": {},
        "telemetry_capabilities": {
            "available": True,
            "devices": [
                {
                    "gpu_index": 0,
                    "gpu_uuid": "GPU-0",
                    "fields": {
                        "power_draw_w": {
                            "value": 10.0,
                            "supported": True,
                            "unit": "watts",
                            "error": None,
                        }
                    },
                }
            ],
            "error": None,
        },
    }
    changed_value = __import__("copy").deepcopy(base)
    changed_value["telemetry_capabilities"]["devices"][0]["fields"]["power_draw_w"]["value"] = 200.0
    changed_support = __import__("copy").deepcopy(base)
    changed_support["telemetry_capabilities"]["devices"][0]["fields"]["power_draw_w"].update(
        {"value": None, "supported": False, "error": "unsupported"}
    )

    assert _environment_fingerprint(base) == _environment_fingerprint(changed_value)
    assert _environment_fingerprint(base) != _environment_fingerprint(changed_support)


def test_matrix_reuses_one_context_for_calibration_and_runs(tmp_path, monkeypatch) -> None:
    observed: list[ProvenanceContext] = []
    run_number = 0

    def calibration(**kwargs):
        observed.append(kwargs["provenance"])
        plan_files = list((tmp_path / "corpora").glob("*-plan.json"))
        assert len(plan_files) == 1
        planned = CorpusManifest.from_dict(json.loads(plan_files[0].read_text()))
        assert planned.accepted_run_ids == ()
        assert all(plan.accepted_run_id is None for plan in planned.planned_runs)
        context = kwargs["provenance"]
        return {
            "status": "supported",
            "path": "calibration.json",
            "reference": {
                "calibration_artifact_path": "results/calibration.json",
                "calibration_sha256": "a" * 64,
                "calibration_experiment_session_id": context.experiment_session_id,
                "calibration_collection_id": context.collection_id,
                "calibration_environment_fingerprint": "environment-test",
                "calibration_source_commit": context.source_commit,
                "calibration_schema_version": CURRENT_SCHEMA_VERSION,
                "calibration_status": "supported",
                "calibration_relationship": "current_session",
                "calibration_created_in_current_session": True,
            },
        }

    def experiment(*args, **kwargs):
        nonlocal run_number
        observed.append(kwargs["provenance"])
        run_number += 1
        return {
            "run_id": f"run-{args[0]}-{run_number}",
            "manifest": {"exit_status": "completed"},
        }

    monkeypatch.setattr(orchestrator, "run_calibration_sweep", calibration)
    monkeypatch.setattr(orchestrator, "run_experiment", experiment)
    monkeypatch.setattr(orchestrator, "profile_workloads", lambda profile: ["ddp_train"])
    monkeypatch.setattr(orchestrator, "verify_calibration_reference", lambda *args, **kwargs: None)

    summary = orchestrator.run_matrix("smoke", output=tmp_path, repetitions=2)

    assert len(observed) == 3
    assert len({item.experiment_session_id for item in observed}) == 1
    assert len({item.collection_id for item in observed}) == 1
    assert len({item.corpus_id for item in observed}) == 1
    assert summary["experiment_session_id"] == observed[0].experiment_session_id
    assert summary["completed"] == 2
    assert summary["family_counts"]["ddp_training"] == {
        "planned": 2,
        "completed": 2,
        "failed": 0,
        "feature_valid": 0,
        "feature_excluded": 2,
        "coverage_reason_counts": {"incomplete_run": 2},
        "workload_config_ids": ["ddp-training-amp-b4-s128-v1"],
    }
    assert summary["primary_coverage_gate"]["passed"] is False
    assert summary["detector_metrics_computed"] is False
    final = CorpusManifest.from_dict(
        json.loads((tmp_path / summary["final_corpus_manifest"]).read_text())
    )
    assert final.accepted_run_ids == ("run-ddp_train-1", "run-ddp_train-2")
    assert final.calibration_reference == summary["calibration_reference"]
    assert summary["feature_extraction"]["calibration_reference"] == final.calibration_reference
