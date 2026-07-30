from __future__ import annotations

import pytest

from commguard.distributed.participation import validate_rank_results
from commguard.distributed.smoke import main
from commguard.exceptions import ValidationError


def test_smoke_without_torchrun_fails_explicitly(tmp_path, capsys) -> None:
    assert main(["--output", str(tmp_path)]) == 2
    error = capsys.readouterr().err
    assert "ReadinessError" in error
    assert "torchrun" in error or "PyTorch is required" in error


def test_rank_results_require_distinct_gpu_uuids() -> None:
    results = [
        {
            "rank": rank,
            "local_rank": rank,
            "world_size": 2,
            "backend": "nccl",
            "gpu_uuid": f"GPU-{rank}",
            "all_reduce_value": 3,
            "completed": True,
        }
        for rank in (0, 1)
    ]
    assert validate_rank_results(results)["participation_valid"]
    results[1]["gpu_uuid"] = "GPU-0"
    with pytest.raises(ValidationError, match="distinct GPU UUID"):
        validate_rank_results(results)
