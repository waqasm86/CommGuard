from __future__ import annotations

import json

from commguard.distributed.launcher import validate_participation


def write_events(root, rank: int, events: list[str], same_uuid: bool = False) -> None:
    path = root / f"rank-{rank}.events.jsonl"
    records = []
    for event in events:
        details = {}
        if event == "startup":
            details = {
                "rank": rank,
                "local_rank": rank,
                "gpu_index": rank,
                "gpu_uuid": "same" if same_uuid else f"GPU-{rank}",
                "world_size": 2,
                "backend": "nccl",
            }
        elif event == "measurement_interval":
            details = {
                "measurement_start_monotonic_ns": 1_000_000_000,
                "measurement_end_monotonic_ns": 31_000_000_000,
                "measured_duration_seconds": 30.0,
            }
        records.append({"event": event, "details": details})
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_valid_smoke_participation(tmp_path) -> None:
    required = ["startup", "cuda_operation_complete", "heartbeat", "completion"]
    write_events(tmp_path, 0, required)
    write_events(tmp_path, 1, required)
    valid, problems, codes = validate_participation(tmp_path, "smoke")
    assert valid
    assert not problems
    assert codes == {"0": 0, "1": 0}


def test_duplicate_gpu_binding_is_rejected(tmp_path) -> None:
    required = ["startup", "cuda_operation_complete", "heartbeat", "completion"]
    write_events(tmp_path, 0, required, same_uuid=True)
    write_events(tmp_path, 1, required, same_uuid=True)
    valid, problems, _ = validate_participation(tmp_path, "smoke")
    assert not valid
    assert "not distinct" in " ".join(problems)


def test_training_requires_backward_and_optimizer_evidence(tmp_path) -> None:
    required = ["startup", "cuda_operation_complete", "heartbeat", "completion"]
    write_events(tmp_path, 0, required)
    write_events(tmp_path, 1, required)
    valid, problems, _ = validate_participation(tmp_path, "ddp_train")
    assert not valid
    assert "backward_complete" in " ".join(problems)


def test_non_smoke_participation_requires_a_valid_measured_interval(tmp_path) -> None:
    required = [
        "startup",
        "cuda_operation_complete",
        "heartbeat",
        "measurement_interval",
        "completion",
    ]
    write_events(tmp_path, 0, required)
    write_events(tmp_path, 1, required)

    valid, problems, _ = validate_participation(tmp_path, "inference_independent")

    assert valid
    assert not problems


def test_inconsistent_reported_measurement_duration_is_rejected(tmp_path) -> None:
    required = [
        "startup",
        "cuda_operation_complete",
        "heartbeat",
        "measurement_interval",
        "completion",
    ]
    write_events(tmp_path, 0, required)
    write_events(tmp_path, 1, required)
    path = tmp_path / "rank-1.events.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    next(record for record in records if record["event"] == "measurement_interval")["details"][
        "measured_duration_seconds"
    ] = 29.0
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    valid, problems, _ = validate_participation(tmp_path, "inference_independent")

    assert not valid
    assert "invalid measurement interval" in " ".join(problems)


def test_adversarial_participation_requires_strategy_summary(tmp_path) -> None:
    required = [
        "startup",
        "cuda_operation_complete",
        "heartbeat",
        "measurement_interval",
        "completion",
    ]
    write_events(tmp_path, 0, required)
    write_events(tmp_path, 1, required)

    valid, problems, _ = validate_participation(
        tmp_path, "inference_independent", designation="adversarial"
    )

    assert not valid
    assert "strategy_summary" in " ".join(problems)
