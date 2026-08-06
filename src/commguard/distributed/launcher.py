"""Subprocess torchrun launch, timeout cleanup, and participation validation."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate a worker process safely on POSIX and Windows."""
    if process.poll() is not None:
        return

    killpg = getattr(os, "killpg", None)

    if callable(killpg) and os.name == "posix":
        killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()


def _force_kill_process_group(process: subprocess.Popen[str]) -> None:
    """Force-kill a worker process safely on POSIX and Windows."""
    if process.poll() is not None:
        return

    killpg = getattr(os, "killpg", None)
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)

    if callable(killpg) and os.name == "posix":
        killpg(process.pid, sigkill)
    else:
        process.kill()


def _terminate_tree(process: subprocess.Popen[str], grace_s: float = 5.0) -> bool:
    cleaned = False
    if process.poll() is not None:
        return True

    try:
        _terminate_process_group(process)
        process.wait(timeout=grace_s)
        cleaned = True
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            with suppress(ProcessLookupError):
                _force_kill_process_group(process)
            process.wait(timeout=grace_s)
            cleaned = True
    return cleaned


def _load_rank_events(output: Path, rank: int) -> list[dict[str, Any]]:
    path = output / f"rank-{rank}.events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _stale_worker_pids() -> list[int]:
    marker = "commguard.distributed.worker"
    pids: list[int] = []
    proc = Path("/proc")
    if not proc.exists():
        return pids
    for command_path in proc.glob("[0-9]*/cmdline"):
        try:
            command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, UnicodeError):
            continue
        if marker in command:
            pid = int(command_path.parent.name)
            if pid != os.getpid():
                pids.append(pid)
    return sorted(pids)


def validate_participation(
    output: str | Path,
    mode: str,
    designation: str | None = None,
) -> tuple[bool, list[str], dict[str, int]]:
    root = Path(output)
    problems: list[str] = []
    startups: list[dict[str, Any]] = []
    rank_exit_codes: dict[str, int] = {}
    required = {"startup", "cuda_operation_complete", "heartbeat", "completion"}
    if mode != "smoke":
        required.add("measurement_interval")
    if mode == "ddp_train":
        required |= {
            "model_ready",
            "forward_complete",
            "backward_complete",
            "gradient_sync_complete",
            "optimizer_step_complete",
            "memory_peak",
        }
    if mode == "calibration":
        required.add("collective_complete")
    if mode == "idle":
        required |= {"measurement_start", "measurement_end", "process_group_destroyed"}
    if mode == "sparse_sync_training":
        required |= {
            "model_ready",
            "forward_complete",
            "backward_complete",
            "parameter_average_complete",
            "memory_peak",
        }
    if mode == "synthetic_communication_decoy":
        required |= {"decoy_burst_complete", "memory_peak"}
    if designation == "adversarial":
        required.add("strategy_summary")
    for rank in (0, 1):
        events = _load_rank_events(root, rank)
        names = {event.get("event") for event in events}
        missing = required - names
        if missing:
            problems.append(f"rank {rank} missing events: {sorted(missing)}")
        startup = next((event for event in events if event.get("event") == "startup"), None)
        if startup:
            startups.append(startup["details"])
        else:
            problems.append(f"rank {rank} has no startup identity")
        failed = "failure" in names or "completion" not in names
        measurement = next(
            (event for event in events if event.get("event") == "measurement_interval"),
            None,
        )
        if measurement is not None:
            details = measurement.get("details", {})
            start_ns = details.get("measurement_start_monotonic_ns")
            end_ns = details.get("measurement_end_monotonic_ns")
            duration = details.get("measured_duration_seconds")
            if (
                not isinstance(start_ns, int)
                or not isinstance(end_ns, int)
                or end_ns < start_ns
                or not isinstance(duration, (int, float))
                or duration < 0
                or abs(float(duration) - ((end_ns - start_ns) / 1e9)) > 1e-6
            ):
                problems.append(f"rank {rank} has invalid measurement interval details")
        rank_exit_codes[str(rank)] = 1 if failed else 0
    if len(startups) == 2:
        if {item.get("rank") for item in startups} != {0, 1}:
            problems.append("rank identities are not exactly {0, 1}")
        if {item.get("local_rank") for item in startups} != {0, 1}:
            problems.append("local ranks are not exactly {0, 1}")
        if {item.get("gpu_index") for item in startups} != {0, 1}:
            problems.append("GPU bindings are not exactly {0, 1}")
        if len({item.get("gpu_uuid") for item in startups}) != 2:
            problems.append("rank GPU UUIDs are not distinct")
        if any(item.get("world_size") != 2 for item in startups):
            problems.append("startup evidence does not report world size 2")
        if any(item.get("backend") != "nccl" for item in startups):
            problems.append("non-NCCL backend detected")
    return not problems, problems, rank_exit_codes


@dataclass(frozen=True)
class LaunchResult:
    command: list[str]
    return_code: int
    timed_out: bool
    duration_s: float
    cleanup_complete: bool
    rendezvous_port: int
    participation_valid: bool
    participation_problems: list[str]
    rank_exit_codes: dict[str, int]
    stdout: str
    stderr: str


def launch_torchrun(
    config_path: str | Path,
    output: str | Path,
    timeout_s: float = 180.0,
    python_executable: str | None = None,
) -> LaunchResult:
    """Launch exactly two workers without importing or initializing CUDA in the parent."""
    config_file = Path(config_path).resolve()
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    stale_workers = _stale_worker_pids()
    if stale_workers:
        raise RuntimeError(
            f"refusing to launch while stale CommGuard workers exist: {stale_workers}"
        )
    port = _free_port()
    executable = python_executable or sys.executable
    command = [
        executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=127.0.0.1:{port}",
        f"--rdzv-id={config['run_id']}",
        f"--log-dir={output_path / 'torchrun-rank-logs'}",
        "--redirects=3",
        "-m",
        "commguard.distributed.worker",
        "--config",
        str(config_file),
        "--output",
        str(output_path),
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    timed_out = False
    cleanup_complete = True
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        cleanup_complete = _terminate_tree(process)
        stdout, stderr = process.communicate()
    duration = time.monotonic() - started
    valid, problems, rank_codes = validate_participation(
        output_path,
        str(config["mode"]),
        str(config.get("designation", "")),
    )
    if timed_out:
        problems.append("torchrun exceeded the hard timeout")
        valid = False
        for rank in ("0", "1"):
            if rank_codes.get(rank, 1) == 0:
                rank_codes[rank] = 124
    return LaunchResult(
        command=command,
        return_code=int(process.returncode or 0),
        timed_out=timed_out,
        duration_s=duration,
        cleanup_complete=cleanup_complete,
        rendezvous_port=port,
        participation_valid=valid,
        participation_problems=problems,
        rank_exit_codes=rank_codes,
        stdout=stdout,
        stderr=stderr,
    )