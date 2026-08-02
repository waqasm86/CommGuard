from __future__ import annotations

from copy import deepcopy

import pytest

from commguard.exceptions import ValidationError
from commguard.schemas import RunManifest


def _strategy_summary() -> dict[str, object]:
    return {
        "strategy_id": "periodic_local_sgd",
        "expected_sync_rounds": 3,
        "actual_sync_rounds": 3,
        "communication_bytes_proxy": 1024,
        "communication_proxy_definition": ("model_parameter_bytes_per_parameter_average_round"),
        "optimizer_steps": 12,
        "inference_steps": 0,
        "processed_tokens": 4096,
        "throughput_tokens_per_s": 100.0,
        "final_loss_proxy": 1.5,
        "wall_time_s": 41.0,
        "parameter_state_agreement": True,
        "detector_score": None,
        "detector_model": None,
        "detector_score_status": "pending_frozen_evaluation",
    }


def _manifest() -> RunManifest:
    summary = _strategy_summary()
    rank_evidence = {
        rank: {
            "strategy_summary": deepcopy(summary),
            "memory_peak": {"allocated_bytes": 100, "reserved_bytes": 200},
        }
        for rank in ("0", "1")
    }
    return RunManifest(
        run_id="run-adversarial",
        workload_name="adversarial_periodic_local_sgd",
        workload_label="training",
        workload_family="periodic_local_sgd",
        designation="adversarial",
        seed=1337,
        world_size=2,
        config={
            "strategy_id": "periodic_local_sgd",
            "rank_runtime_evidence": rank_evidence,
        },
        environment={},
        environment_fingerprint="environment-a",
        source_commit="commit-a",
        started_at_utc="2026-08-02T00:00:00+00:00",
        ended_at_utc="2026-08-02T00:00:41+00:00",
        warmup_seconds=5.0,
        exit_status="completed",
        failure_category=None,
        failure_reason=None,
        rank_exit_codes={"0": 0, "1": 0},
        nccl_environment={},
        participation_valid=True,
        experiment_session_id="session-a",
        collection_id="collection-a",
        corpus_id="corpus-a",
        node_id="node-a",
        source_dirty=False,
        random_seed=1337,
        measurement_start_monotonic_ns=1_000_000_000,
        measurement_end_monotonic_ns=36_000_000_000,
        measured_duration_seconds=35.0,
    )


def test_completed_adversarial_manifest_requires_truthful_strategy_metrics() -> None:
    manifest = _manifest()

    manifest.validate()
    assert (
        manifest.to_dict()["config"]["rank_runtime_evidence"]["0"]["strategy_summary"][
            "detector_score"
        ]
        is None
    )


def test_adversarial_manifest_rejects_sync_count_disagreement() -> None:
    payload = _manifest().to_dict()
    payload["config"]["rank_runtime_evidence"]["1"]["strategy_summary"]["actual_sync_rounds"] = 2

    with pytest.raises(ValidationError, match="expected and actual sync counts differ"):
        RunManifest.from_dict(payload)


def test_adversarial_manifest_rejects_invented_run_time_detector_score() -> None:
    payload = _manifest().to_dict()
    payload["config"]["rank_runtime_evidence"]["0"]["strategy_summary"].update(
        {"detector_score": 0.9, "detector_model": "unknown"}
    )

    with pytest.raises(ValidationError, match="cannot invent a detector score"):
        RunManifest.from_dict(payload)
