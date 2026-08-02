"""Experiment lifecycle, artifact preservation, calibration gates, and profiles."""

from __future__ import annotations

import json
import os
import random
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from commguard.artifacts import ArtifactStore
from commguard.calibration import analyze_calibration
from commguard.distributed.launcher import LaunchResult, launch_torchrun
from commguard.environment.preflight import check_environment
from commguard.exceptions import CalibrationError, WorkloadError
from commguard.provenance import source_identifier
from commguard.schemas import SCHEMA_VERSION, TELEMETRY_FIELDS, RunManifest
from commguard.telemetry import TelemetryCollector
from commguard.workloads import get_workload, profile_workloads


def _new_run_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{name}-{uuid.uuid4().hex[:8]}"


def _nccl_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith("NCCL_") or key.startswith("TORCH_DISTRIBUTED_")
    }


def _failure_category(result: LaunchResult) -> tuple[str | None, str | None]:
    if result.timed_out:
        return "nccl_timeout_or_collective_mismatch", "hard timeout exceeded"
    combined = (result.stderr + "\n" + result.stdout).lower()
    if "out of memory" in combined:
        return "cuda_out_of_memory", "CUDA out of memory; configuration was not auto-tuned"
    if "rendezvous" in combined or "address already in use" in combined:
        return "rendezvous_or_port_failure", "distributed rendezvous failed"
    if "nccl" in combined:
        return "process_group_or_nccl_failure", "NCCL or process-group failure"
    if not result.participation_valid:
        return "rank_crash_or_asymmetric_exit", "; ".join(result.participation_problems)
    if result.return_code != 0:
        return "workload_failure", f"torchrun exited {result.return_code}"
    return None, None


def _pcie_observation(samples: list[Any], run_directory: Path) -> dict[str, Any]:
    phase_start: int | None = None
    phase_end: int | None = None
    for rank in (0, 1):
        event_path = run_directory / f"rank-{rank}.events.jsonl"
        if not event_path.exists():
            continue
        for line in event_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "collective_start":
                timestamp = int(event["monotonic_ns"])
                phase_start = timestamp if phase_start is None else min(phase_start, timestamp)
            elif event.get("event") == "collective_complete":
                timestamp = int(event["monotonic_ns"])
                phase_end = timestamp if phase_end is None else max(phase_end, timestamp)
    measured_samples = (
        [sample for sample in samples if phase_start <= sample.monotonic_ns <= phase_end]
        if phase_start is not None and phase_end is not None
        else samples
    )
    readings: list[float] = []
    supported = True
    for sample in measured_samples:
        tx = sample.fields["pcie_tx_bytes_per_s"]
        rx = sample.fields["pcie_rx_bytes_per_s"]
        if not tx.supported or not rx.supported:
            supported = False
            continue
        readings.append(float(tx.value) + float(rx.value))
    return {
        "pcie_supported": supported and bool(readings),
        "pcie_total_mean_bytes_per_s": statistics.fmean(readings) if readings else None,
        "pcie_total_median_bytes_per_s": statistics.median(readings) if readings else None,
        "pcie_sample_count": len(readings),
        "measurement_phase_bounded": phase_start is not None and phase_end is not None,
    }


def _write_run_evidence(
    store: ArtifactStore,
    run_id: str,
    config: dict[str, Any],
    result: LaunchResult,
    collector: TelemetryCollector,
    diagnostics: dict[str, Any],
    preflight: dict[str, Any],
    started: str,
    ended: str,
) -> dict[str, Any]:
    run_prefix = Path("runs") / run_id
    rank_runtime_evidence: dict[str, dict[str, Any]] = {}
    for rank in (0, 1):
        event_path = store.resolve(run_prefix / f"rank-{rank}.events.jsonl")
        evidence: dict[str, Any] = {}
        if event_path.exists():
            for line in event_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("event") in {"startup", "model_ready", "memory_peak"}:
                    evidence[str(event["event"])] = event.get("details", {})
        rank_runtime_evidence[str(rank)] = evidence
    manifested_config = {**config, "rank_runtime_evidence": rank_runtime_evidence}
    if collector.samples:
        store.write_jsonl(
            run_prefix / "telemetry.jsonl",
            [sample.to_dict() for sample in collector.samples],
        )
    else:
        store.write_text(run_prefix / "telemetry-unavailable.txt", "No samples were collected.\n")
    store.write_json(
        run_prefix / "telemetry-diagnostics.json",
        {
            "artifact_kind": "experiment_summary",
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "summary_type": "telemetry_diagnostics",
            **diagnostics,
        },
    )
    store.write_text(run_prefix / "torchrun.stdout.log", result.stdout)
    store.write_text(run_prefix / "torchrun.stderr.log", result.stderr)
    store.write_json(
        run_prefix / "launch.json",
        {
            "artifact_kind": "experiment_summary",
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "summary_type": "launch",
            "command": result.command,
            "return_code": result.return_code,
            "timed_out": result.timed_out,
            "duration_s": result.duration_s,
            "cleanup_complete": result.cleanup_complete,
            "rendezvous_port": result.rendezvous_port,
            "participation_problems": result.participation_problems,
        },
    )
    category, reason = _failure_category(result)
    if diagnostics.get("collector_error"):
        category = "telemetry_sampler_failure"
        reason = str(diagnostics["collector_error"])
    exit_status = (
        "timeout"
        if result.timed_out
        else (
            "completed"
            if result.return_code == 0
            and result.participation_valid
            and not diagnostics.get("collector_error")
            else "failed"
        )
    )
    manifest = RunManifest(
        run_id=run_id,
        workload_name=str(config["workload_name"]),
        workload_label=str(config["label"]),
        workload_family=str(config["family"]),
        designation=str(config["designation"]),
        seed=int(config["seed"]),
        world_size=2,
        config=manifested_config,
        environment={
            "gpus": preflight["gpus"],
            "platform": preflight["platform"],
            "torch": preflight["torch"],
            "nvcc": preflight["nvcc"],
            "topology": preflight["topology"],
            "packages": preflight["packages"],
            "telemetry_capabilities": preflight["telemetry_capabilities"],
        },
        environment_fingerprint=str(preflight["session_fingerprint"]),
        source_commit=source_identifier(),
        started_at_utc=started,
        ended_at_utc=ended,
        warmup_seconds=float(config.get("warmup_seconds", 2.0)),
        exit_status=exit_status,
        failure_category=category,
        failure_reason=reason,
        rank_exit_codes=result.rank_exit_codes,
        nccl_environment=_nccl_environment(),
        participation_valid=result.participation_valid,
    )
    store.write_json(run_prefix / "manifest.json", manifest.to_dict())
    return {
        "run_id": run_id,
        "manifest": manifest.to_dict(),
        "launch": result,
        "telemetry_diagnostics": diagnostics,
        **_pcie_observation(collector.samples, store.resolve(run_prefix)),
    }


def run_experiment(
    workload: str,
    output: str | Path = "artifacts",
    overrides: dict[str, Any] | None = None,
    timeout_s: float = 180.0,
    strict_preflight: bool = True,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Run one isolated two-rank experiment and preserve success or failure evidence."""
    store = ArtifactStore(output)
    store.initialize()
    preflight = check_environment(strict=strict_preflight)
    config = get_workload(workload)
    if overrides:
        config.update(overrides)
    run_id = _new_run_id(workload)
    config.update(
        {
            "run_id": run_id,
            "workload_name": workload,
            "seed": int(config.get("seed", 1337)),
            "process_group_timeout_s": min(float(timeout_s) * 0.8, 120.0),
            "warmup_seconds": float(config.get("warmup_seconds", 2.0)),
            "deterministic": bool(config.get("deterministic", False)),
        }
    )
    run_directory = store.resolve(Path("runs") / run_id)
    run_directory.mkdir(parents=True, exist_ok=False)
    config_path = store.write_json(Path("runs") / run_id / "config.json", config, validate=False)
    collector = TelemetryCollector(
        run_id,
        interval_s=float(config.get("sampling_interval_s", 1.0)),
        expected_gpus=2,
    )
    started = datetime.now(timezone.utc).isoformat()
    collector.start()
    result = launch_torchrun(config_path, run_directory, timeout_s=timeout_s)
    telemetry_error: str | None = None
    try:
        collector_diagnostics = collector.stop().to_dict()
    except BaseException as exc:
        telemetry_error = f"{type(exc).__name__}: {exc}"
        collector_diagnostics = {
            "collector_error": telemetry_error,
            "sample_cycles": len(collector._cycle_starts),
            "field_missing_fraction": {name: 1.0 for name in TELEMETRY_FIELDS},
        }
    ended = datetime.now(timezone.utc).isoformat()
    outcome = _write_run_evidence(
        store,
        run_id,
        config,
        result,
        collector,
        collector_diagnostics,
        preflight,
        started,
        ended,
    )
    if telemetry_error is not None:
        outcome["telemetry_error"] = telemetry_error
    if raise_on_failure and outcome["manifest"]["exit_status"] != "completed":
        raise WorkloadError(
            f"run {run_id} failed; artifacts preserved at {run_directory}: "
            f"{outcome['manifest']['failure_reason']}"
        )
    return outcome


def run_calibration_sweep(
    output: str | Path = "artifacts",
    payload_mib: tuple[int, ...] = (1, 4, 16, 64),
    collective: str = "all_reduce",
    repetitions: int = 1,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for payload in payload_mib:
            outcome = run_experiment(
                "collective_all_reduce_1mib",
                output=output,
                overrides={
                    "payload_mib": payload,
                    "collective": collective,
                    "repetition": repetition,
                },
                timeout_s=timeout_s,
                strict_preflight=True,
                raise_on_failure=False,
            )
            observations.append(
                {
                    "run_id": outcome["run_id"],
                    "payload_mib": payload,
                    "collective": collective,
                    "repetition": repetition,
                    "participation_valid": outcome["manifest"]["participation_valid"],
                    "exit_status": outcome["manifest"]["exit_status"],
                    "pcie_supported": outcome["pcie_supported"],
                    "pcie_total_mean_bytes_per_s": outcome["pcie_total_mean_bytes_per_s"],
                    "pcie_total_median_bytes_per_s": outcome["pcie_total_median_bytes_per_s"],
                    "pcie_sample_count": outcome["pcie_sample_count"],
                }
            )
    result = analyze_calibration(observations)
    result["session_fingerprint"] = check_environment(strict=True)["session_fingerprint"]
    store = ArtifactStore(output)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = store.write_json(f"results/calibration-{timestamp}.json", result)
    result["path"] = str(path)
    return result


def estimate_matrix(profile: str, repetitions: int = 1) -> dict[str, Any]:
    names = profile_workloads(profile)
    seconds = sum(float(get_workload(name).get("estimated_seconds", 60)) for name in names)
    seconds *= repetitions
    return {
        "profile": profile,
        "run_count": len(names) * repetitions,
        "estimated_gpu_minutes": seconds * 2 / 60,
        "estimated_artifact_mib": max(5.0, seconds * 0.02),
        "estimate_only": True,
    }


def run_matrix(
    profile: str = "smoke",
    output: str | Path = "artifacts",
    repetitions: int | None = None,
    negative_calibration_mode: bool = False,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run a randomized profile after a session-specific calibration gate."""
    if repetitions is None:
        repetitions = 1 if profile == "smoke" else 3
    estimate = estimate_matrix(profile, repetitions)
    calibration = run_calibration_sweep(output=output, timeout_s=timeout_s)
    if (
        calibration["status"] != "supported"
        and profile != "smoke"
        and not negative_calibration_mode
    ):
        raise CalibrationError(
            f"standard/extended matrix blocked by negative calibration; see {calibration['path']}"
        )
    names = profile_workloads(profile)
    names = [name for name in names if not name.startswith("collective_")]
    schedule = names * repetitions
    random.Random(20260730).shuffle(schedule)
    outcomes = [
        run_experiment(
            name,
            output=output,
            timeout_s=timeout_s,
            strict_preflight=True,
            raise_on_failure=False,
        )
        for name in schedule
    ]
    summary = {
        "artifact_kind": "experiment_summary",
        "schema_version": SCHEMA_VERSION,
        "summary_type": "matrix",
        "profile": profile,
        "estimate": estimate,
        "calibration_status": calibration["status"],
        "negative_calibration_mode": negative_calibration_mode,
        "schedule": schedule,
        "run_ids": [outcome["run_id"] for outcome in outcomes],
        "completed": sum(outcome["manifest"]["exit_status"] == "completed" for outcome in outcomes),
        "failed": sum(outcome["manifest"]["exit_status"] != "completed" for outcome in outcomes),
    }
    store = ArtifactStore(output)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    store.write_json(f"results/matrix-{profile}-{timestamp}.json", summary)
    return summary
