"""Strict torchrun-compatible two-rank CUDA/NCCL participation smoke test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from commguard.distributed.participation import load_rank_results, validate_rank_results
from commguard.exceptions import ArtifactExistsError, ReadinessError


def _write_json_create_only(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ArtifactExistsError(f"refusing to overwrite smoke artifact: {path}") from exc


def _gpu_uuid(local_rank: int) -> str:
    command = [
        "nvidia-smi",
        f"--id={local_rank}",
        "--query-gpu=uuid",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ReadinessError(f"cannot query GPU UUID for rank {local_rank}: {exc}") from exc
    uuid = result.stdout.strip()
    if result.returncode != 0 or not uuid:
        message = result.stderr.strip() or "nvidia-smi returned no UUID"
        raise ReadinessError(f"cannot query GPU UUID for rank {local_rank}: {message}")
    return uuid


def run_smoke(output: str | Path, timeout_s: float = 120.0) -> dict[str, Any] | None:
    """Run inside ``torchrun --nproc-per-node=2`` without any fallback backend."""
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ReadinessError("PyTorch is required for the CUDA/NCCL smoke test") from exc

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ReadinessError(
            "smoke test must be launched by torchrun; missing " + ", ".join(missing)
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ReadinessError(f"CommGuard requires WORLD_SIZE=2, got {world_size}")
    if rank not in (0, 1) or local_rank not in (0, 1):
        raise ReadinessError(f"invalid rank mapping rank={rank}, LOCAL_RANK={local_rank}")
    if not torch.cuda.is_available():
        raise ReadinessError("CUDA is unavailable; CPU and Gloo fallbacks are prohibited")
    if torch.cuda.device_count() != 2:
        raise ReadinessError(
            f"CommGuard requires exactly two CUDA devices, found {torch.cuda.device_count()}"
        )
    if not dist.is_available() or not dist.is_nccl_available():
        raise ReadinessError("PyTorch distributed NCCL support is unavailable")

    output_path = Path(output).resolve()
    torch.cuda.set_device(local_rank)
    initialized = False
    try:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_s))
        initialized = True
        backend = str(dist.get_backend())
        if backend != "nccl":
            raise ReadinessError(f"backend fallback is prohibited, got {backend!r}")
        dist.barrier()
        reduced = torch.tensor([rank + 1], device=f"cuda:{local_rank}", dtype=torch.int64)
        dist.all_reduce(reduced)
        reduced_value = int(reduced.item())
        if reduced_value != 3:
            raise RuntimeError(f"unexpected all_reduce result {reduced_value}, expected 3")
        dist.barrier()
        rank_result = {
            "schema_version": "1.0",
            "artifact_kind": "distributed_smoke_rank",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "backend": backend,
            "gpu_uuid": _gpu_uuid(local_rank),
            "all_reduce_value": reduced_value,
            "completed": True,
        }
        _write_json_create_only(output_path / f"rank-{rank}.json", rank_result)
        dist.barrier()
        summary = None
        if rank == 0:
            summary = validate_rank_results(load_rank_results(output_path))
            summary.update(
                {
                    "schema_version": "1.0",
                    "artifact_kind": "distributed_smoke_summary",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json_create_only(output_path / "participation-summary.json", summary)
        dist.barrier()
        return summary
    finally:
        if initialized:
            dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_smoke(args.output, timeout_s=args.timeout)
    except (ReadinessError, ArtifactExistsError, RuntimeError, ValueError) as exc:
        print(f"commguard smoke: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
