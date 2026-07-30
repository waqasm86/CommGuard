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
