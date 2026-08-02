from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from commguard.artifacts import ArtifactStore
from commguard.corpus import CorpusManifest, PlannedRun
from commguard.evaluation import evaluate_detector
from commguard.exceptions import CoverageError, ValidationError
from commguard.features import (
    REQUIRED_REASON_CODES,
    CoverageReason,
    align_samples_by_timestamp,
    extract_feature_result,
    load_extraction_result,
    require_primary_coverage,
)
from commguard.schemas import (
    FIELD_UNITS,
    LEGACY_SCHEMA_VERSION,
    TELEMETRY_FIELDS,
    FieldReading,
    RunManifest,
    TelemetrySample,
    validate_artifact,
)


def corpus(*plans: PlannedRun) -> CorpusManifest:
    accepted = tuple(plan.accepted_run_id for plan in plans if plan.accepted_run_id)
    return CorpusManifest(
        corpus_id="corpus-test",
        collection_id="collection-test",
        experiment_session_id="session-test",
        node_id="node-0",
        planned_runs=plans,
        accepted_run_ids=accepted,
        source_commit="commit-test",
        source_dirty=False,
        notebook_version="test",
        input_archive_sha256=None,
        random_seed=1337,
    )


def manifest(run_id: str, family: str, label: str, *, warmup: float) -> RunManifest:
    timestamp = datetime(2026, 8, 1, tzinfo=timezone.utc).isoformat()
    return RunManifest(
        run_id=run_id,
        workload_name=family,
        workload_label=label,
        workload_family=family,
        designation="benign",
        seed=1337,
        world_size=2,
        config={},
        environment={},
        environment_fingerprint="environment-test",
        source_commit="commit-test",
        started_at_utc=timestamp,
        ended_at_utc=timestamp,
        warmup_seconds=warmup,
        exit_status="completed",
        failure_category=None,
        failure_reason=None,
        rank_exit_codes={"0": 0, "1": 0},
        nccl_environment={},
        participation_valid=True,
        schema_version=LEGACY_SCHEMA_VERSION,
    )


def telemetry(run_id: str, duration_seconds: float, interval_seconds: float = 0.2) -> list[dict]:
    records = []
    origin = datetime(2026, 8, 1, tzinfo=timezone.utc)
    sequence = 0
    while sequence * interval_seconds <= duration_seconds:
        timestamp = sequence * interval_seconds
        for gpu in (0, 1):
            fields = {
                name: FieldReading(
                    value=sequence + gpu + position,
                    unit=FIELD_UNITS[name],
                    supported=True,
                )
                for position, name in enumerate(TELEMETRY_FIELDS)
            }
            records.append(
                TelemetrySample(
                    run_id=run_id,
                    gpu_index=gpu,
                    gpu_uuid=f"GPU-{gpu}",
                    sequence=sequence,
                    wall_time_utc=(origin + timedelta(seconds=timestamp)).isoformat(),
                    monotonic_ns=int(timestamp * 1e9),
                    fields=fields,
                ).to_dict()
            )
        sequence += 1
    return records


def write_run(
    root,
    run_id: str,
    family: str,
    label: str,
    *,
    warmup: float,
    duration: float,
) -> None:
    store = ArtifactStore(root)
    store.initialize()
    run_manifest = manifest(run_id, family, label, warmup=warmup)
    store.write_json(f"runs/{run_id}/manifest.json", run_manifest.to_dict())
    store.write_jsonl(f"runs/{run_id}/telemetry.jsonl", telemetry(run_id, duration))


def test_every_planned_run_gets_one_coverage_record(tmp_path) -> None:
    write_run(tmp_path, "run-good", "ddp_training", "training", warmup=0, duration=6)
    plans = (
        PlannedRun("good", "ddp_training", "training", {}, accepted_run_id="run-good"),
        PlannedRun("missing", "control_idle", "control", {}),
        PlannedRun(
            "calibration-idle",
            "calibration_idle",
            "calibration",
            {},
            designation="calibration",
            accepted_run_id="run-calibration-idle",
        ),
    )

    result = extract_feature_result(
        tmp_path,
        corpus(*plans),
        output=tmp_path,
        window_lengths=(5.0,),
    )

    assert [record.plan_id for record in result.coverage] == [
        "good",
        "missing",
        "calibration-idle",
    ]
    assert result.coverage[0].status == "included"
    assert result.coverage[1].reason_code == CoverageReason.INCOMPLETE_RUN.value
    assert result.coverage[2].reason_code == CoverageReason.CALIBRATION_RUN_EXCLUDED.value
    assert {row["run_id"] for row in result.features} == {"run-good"}
    coverage_artifact = next((tmp_path / "features").glob("coverage-*.jsonl"))
    assert len(coverage_artifact.read_text(encoding="utf-8").splitlines()) == len(plans)
    assert result.coverage[0].sampling_gap_seconds_by_gpu["0"]["median"] == 0.2
    restored = load_extraction_result(tmp_path)
    assert restored == result
    assert result.summary()["window_policy"] == {
        "primary_seconds": [30.0],
        "diagnostic_short_seconds": [5.0, 15.0],
        "requested_seconds": [5.0],
        "primary_window_is_never_implicitly_reduced": True,
    }

    malformed = result.coverage[0].to_dict()
    malformed["rows_per_gpu"].pop("1")
    with pytest.raises(ValidationError, match="rows_per_gpu"):
        validate_artifact(malformed)


def test_short_historical_run_reports_exact_post_warmup_duration(tmp_path) -> None:
    write_run(
        tmp_path,
        "run-short",
        "inference_prefill_independent",
        "inference",
        warmup=2.0,
        duration=6.4,
    )
    plan = PlannedRun(
        "short",
        "inference_prefill_independent",
        "inference",
        {},
        accepted_run_id="run-short",
    )

    result = extract_feature_result(tmp_path, corpus(plan), window_lengths=(5.0, 15.0, 30.0))
    record = result.coverage[0]

    assert record.reason_code == CoverageReason.POST_WARMUP_INTERVAL_TOO_SHORT.value
    assert record.common_overlap_seconds == pytest.approx(6.4)
    assert record.usable_duration_seconds == pytest.approx(4.4)
    assert "usable=4.400000s" in record.detail
    assert record.emitted_windows == {"5": 0, "15": 0, "30": 0}


def test_timestamp_alignment_never_correlates_independent_arrays_by_index() -> None:
    left = [{"monotonic_ns": value} for value in (0, 1_000_000_000, 2_000_000_000)]
    right = [{"monotonic_ns": value} for value in (100_000_000, 2_100_000_000)]

    pairs = align_samples_by_timestamp(left, right, tolerance_seconds=0.15)

    assert [(a["monotonic_ns"], b["monotonic_ns"]) for a, b in pairs] == [
        (0, 100_000_000),
        (2_000_000_000, 2_100_000_000),
    ]


def test_primary_gate_fails_incomplete_family_and_passes_complete_family(tmp_path) -> None:
    write_run(tmp_path, "run-ddp", "ddp_training", "training", warmup=0, duration=31)
    plans = (
        PlannedRun("ddp", "ddp_training", "training", {}, accepted_run_id="run-ddp"),
        PlannedRun("idle", "control_idle", "control", {}),
    )
    result = extract_feature_result(tmp_path, corpus(*plans), window_lengths=(5.0, 30.0))

    with pytest.raises(CoverageError, match="control_idle"):
        require_primary_coverage(
            result,
            required_families=("ddp_training", "control_idle"),
        )
    passed = require_primary_coverage(result, required_families=("ddp_training",))
    assert passed["passed"] is True


def test_primary_gate_never_substitutes_a_diagnostic_short_window(tmp_path) -> None:
    write_run(tmp_path, "run-short", "ddp_training", "training", warmup=0, duration=6)
    plan = PlannedRun(
        "ddp-short",
        "ddp_training",
        "training",
        {},
        accepted_run_id="run-short",
    )
    result = extract_feature_result(tmp_path, corpus(plan), window_lengths=(5.0,))

    with pytest.raises(CoverageError, match="no_common_window_length"):
        require_primary_coverage(result, required_families=("ddp_training",))
    diagnostic = require_primary_coverage(
        result,
        required_families=("ddp_training",),
        required_window_seconds=(5.0,),
    )
    assert diagnostic["passed"] is True


def test_detector_refuses_incomplete_coverage_before_fitting(tmp_path) -> None:
    write_run(tmp_path, "run-ddp", "ddp_training", "training", warmup=0, duration=31)
    plans = (
        PlannedRun("ddp", "ddp_training", "training", {}, accepted_run_id="run-ddp"),
        PlannedRun("idle", "control_idle", "control", {}),
    )
    extract_feature_result(
        tmp_path,
        corpus(*plans),
        output=tmp_path,
        window_lengths=(30.0,),
    )
    store = ArtifactStore(tmp_path)
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

    with pytest.raises(CoverageError, match="control_idle"):
        evaluate_detector(
            tmp_path,
            required_families=("ddp_training", "control_idle"),
        )


def test_required_reason_code_contract_is_complete() -> None:
    assert {
        "not_in_selected_corpus",
        "invalid_manifest",
        "incomplete_run",
        "participation_invalid",
        "calibration_run_excluded",
        "missing_telemetry",
        "missing_required_gpu",
        "no_common_interval",
        "post_warmup_interval_too_short",
        "insufficient_samples",
        "unsupported_telemetry_schema",
        "no_common_window_length",
        "feature_error",
    } == REQUIRED_REASON_CODES
