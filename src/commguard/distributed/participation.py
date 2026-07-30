"""Validation for two-rank NCCL smoke-test evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from commguard.exceptions import ValidationError


def validate_rank_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate complete evidence from exactly two NCCL ranks."""
    records = [dict(result) for result in results]
    if len(records) != 2:
        raise ValidationError(f"expected two rank results, found {len(records)}")
    if {record.get("rank") for record in records} != {0, 1}:
        raise ValidationError("rank results must contain ranks 0 and 1 exactly once")
    if {record.get("local_rank") for record in records} != {0, 1}:
        raise ValidationError("local rank results must contain 0 and 1 exactly once")
    if any(record.get("world_size") != 2 for record in records):
        raise ValidationError("every rank must report WORLD_SIZE=2")
    if any(record.get("backend") != "nccl" for record in records):
        raise ValidationError("every rank must report the NCCL backend")
    if any(record.get("all_reduce_value") != 3 for record in records):
        raise ValidationError("every rank must report the expected all_reduce value 3")
    if any(record.get("completed") is not True for record in records):
        raise ValidationError("every rank must report successful barrier completion")
    uuids = {str(record.get("gpu_uuid", "")).strip() for record in records}
    if "" in uuids or len(uuids) != 2:
        raise ValidationError("rank results must contain two distinct GPU UUIDs")
    return {
        "world_size": 2,
        "backend": "nccl",
        "ranks": [0, 1],
        "gpu_uuids": sorted(uuids),
        "participation_valid": True,
    }


def load_rank_results(output: str | Path) -> list[dict[str, Any]]:
    """Load rank 0 and rank 1 result files from one smoke output directory."""
    root = Path(output)
    results = []
    for rank in (0, 1):
        path = root / f"rank-{rank}.json"
        if not path.is_file():
            raise ValidationError(f"missing rank result: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValidationError(f"rank result is not an object: {path}")
        results.append(data)
    return results


__all__ = ["load_rank_results", "validate_rank_results"]
