"""Experiment lifecycle, artifact preservation, calibration gates, and profiles."""

from __future__ import annotations

import json
import os
import random
import statistics
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from commguard.artifacts import ArtifactStore
from commguard.calibration import analyze_calibration
from commguard.corpus import CorpusManifest, PlannedRun
from commguard.distributed.launcher import LaunchResult, launch_torchrun
from commguard.environment.preflight import check_environment
from commguard.exceptions import CalibrationError, CoverageError, WorkloadError
from commguard.features import (
    PRIMARY_BENIGN_FAMILIES,
    PRIMARY_WINDOW_SECONDS,
    extract_feature_result,
    require_primary_coverage,
)
from commguard.provenance import ProvenanceContext, new_corpus_id, new_run_id
from commguard.schemas import CURRENT_SCHEMA_VERSION, SCHEMA_VERSION, TELEMETRY_FIELDS, RunManifest
from commguard.telemetry import TelemetryCollector
from commguard.workloads import get_workload, profile_workloads


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
    provenance: ProvenanceContext,
    started: str,
    ended: str,
) -> dict[str, Any]:
    run_prefix = Path("runs") / run_id
    rank_runtime_evidence: dict[str, dict[str, Any]] = {}
    measurement_intervals: list[dict[str, Any]] = []
    for rank in (0, 1):
        event_path = store.resolve(run_prefix / f"rank-{rank}.events.jsonl")
        evidence: dict[str, Any] = {}
        if event_path.exists():
            for line in event_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("event") in {
                    "startup",
                    "model_ready",
                    "memory_peak",
                    "measurement_interval",
                }:
                    evidence[str(event["event"])] = event.get("details", {})
                if event.get("event") == "measurement_interval":
                    measurement_intervals.append(dict(event.get("details", {})))
        rank_runtime_evidence[str(rank)] = evidence
    measurement_start_ns = (
        max(int(item["measurement_start_monotonic_ns"]) for item in measurement_intervals)
        if len(measurement_intervals) == 2
        else None
    )
    measurement_end_ns = (
        min(int(item["measurement_end_monotonic_ns"]) for item in measurement_intervals)
        if len(measurement_intervals) == 2
        else None
    )
    measured_duration_seconds = (
        max(0.0, (measurement_end_ns - measurement_start_ns) / 1e9)
        if measurement_start_ns is not None and measurement_end_ns is not None
        else None
    )
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
        environment_fingerprint=str(preflight["environment_fingerprint"]),
        source_commit=provenance.source_commit,
        started_at_utc=started,
        ended_at_utc=ended,
        warmup_seconds=float(config.get("warmup_seconds", 2.0)),
        exit_status=exit_status,
        failure_category=category,
        failure_reason=reason,
        rank_exit_codes=result.rank_exit_codes,
        nccl_environment=_nccl_environment(),
        participation_valid=result.participation_valid,
        experiment_session_id=provenance.experiment_session_id,
        collection_id=provenance.collection_id,
        corpus_id=provenance.corpus_id,
        node_id=provenance.node_id,
        source_dirty=provenance.source_dirty,
        input_archive_sha256=provenance.input_archive_sha256,
        notebook_version=provenance.notebook_version,
        random_seed=int(config["seed"]),
        measurement_start_monotonic_ns=measurement_start_ns,
        measurement_end_monotonic_ns=measurement_end_ns,
        measured_duration_seconds=measured_duration_seconds,
        schema_version=CURRENT_SCHEMA_VERSION,
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
    provenance: ProvenanceContext | None = None,
) -> dict[str, Any]:
    """Run one isolated two-rank experiment and preserve success or failure evidence."""
    store = ArtifactStore(output)
    store.initialize()
    context = provenance or ProvenanceContext.create(corpus_id=new_corpus_id(workload))
    preflight = check_environment(strict=strict_preflight, provenance=context)
    config = get_workload(workload)
    if overrides:
        config.update(overrides)
    run_id = new_run_id(workload, context.experiment_session_id)
    config.update(
        {
            "run_id": run_id,
            "workload_name": workload,
            "seed": int(config.get("seed", 1337)),
            "process_group_timeout_s": min(float(timeout_s) * 0.8, 120.0),
            "warmup_seconds": float(config.get("warmup_seconds", 2.0)),
            "deterministic": bool(config.get("deterministic", False)),
            **context.run_fields(random_seed=int(config.get("seed", 1337))),
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
        context,
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
    provenance: ProvenanceContext | None = None,
) -> dict[str, Any]:
    context = provenance or ProvenanceContext.create(corpus_id=new_corpus_id("calibration"))
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
                provenance=context,
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
    environment = check_environment(strict=True, provenance=context)
    result.update(
        {
            "experiment_session_id": context.experiment_session_id,
            "collection_id": context.collection_id,
            "corpus_id": context.corpus_id,
            "node_id": context.node_id,
            "environment_fingerprint": environment["environment_fingerprint"],
            "session_fingerprint": environment["environment_fingerprint"],
            "source_commit": context.source_commit,
            "source_dirty": context.source_dirty,
        }
    )
    store = ArtifactStore(output)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = store.write_json(f"results/calibration-{timestamp}.json", result)
    result["path"] = str(path)
    return result


def estimate_matrix(profile: str, repetitions: int = 1) -> dict[str, Any]:
    plans = plan_matrix(profile, repetitions)
    seconds = sum(float(plan.config.get("estimated_seconds", 60)) for plan in plans)
    return {
        "profile": profile,
        "run_count": len(plans),
        "estimated_gpu_minutes": seconds * 2 / 60,
        "estimated_artifact_mib": max(5.0, seconds * 0.02),
        "estimate_only": True,
    }


def plan_matrix(profile: str, repetitions: int = 1) -> tuple[PlannedRun, ...]:
    """Build a deterministic, CPU-only matrix plan with stable configuration IDs."""
    if repetitions < 1:
        raise ValueError("matrix repetitions must be at least one")
    names = [name for name in profile_workloads(profile) if not name.startswith("collective_")]
    schedule = names * repetitions
    random.Random(20260730).shuffle(schedule)
    occurrences: Counter[str] = Counter()
    plans: list[PlannedRun] = []
    for index, name in enumerate(schedule):
        config = get_workload(name)
        occurrence = occurrences[name]
        occurrences[name] += 1
        config["workload_name"] = name
        plans.append(
            PlannedRun(
                plan_id=f"{profile}-{index:04d}-{name}-{occurrence:03d}",
                workload_family=str(config["family"]),
                target_label=str(config["label"]),
                config=config,
                designation=str(config["designation"]),
                random_seed=int(config.get("seed", 1337)),
                workload_config_id=str(config["config_id"]),
            )
        )
    return tuple(plans)


def _corpus_manifest(
    context: ProvenanceContext,
    planned_runs: tuple[PlannedRun, ...],
) -> CorpusManifest:
    return CorpusManifest(
        corpus_id=context.corpus_id,
        collection_id=context.collection_id,
        experiment_session_id=context.experiment_session_id,
        node_id=context.node_id,
        planned_runs=planned_runs,
        accepted_run_ids=tuple(
            plan.accepted_run_id for plan in planned_runs if plan.accepted_run_id is not None
        ),
        source_commit=context.source_commit,
        source_dirty=context.source_dirty,
        notebook_version=context.notebook_version,
        input_archive_sha256=context.input_archive_sha256,
        random_seed=20260730,
    )


def _family_summary(
    plans: tuple[PlannedRun, ...],
    coverage: tuple[Any, ...],
) -> dict[str, dict[str, Any]]:
    by_plan = {record.plan_id: record for record in coverage}
    result: dict[str, dict[str, Any]] = {}
    for family in sorted({plan.workload_family for plan in plans}):
        family_plans = [plan for plan in plans if plan.workload_family == family]
        family_coverage = [by_plan[plan.plan_id] for plan in family_plans]
        reasons = Counter(
            record.reason_code for record in family_coverage if record.reason_code is not None
        )
        result[family] = {
            "planned": len(family_plans),
            "completed": sum(plan.accepted_run_id is not None for plan in family_plans),
            "failed": sum(plan.accepted_run_id is None for plan in family_plans),
            "feature_valid": sum(
                record.status == "included"
                and all(
                    record.emitted_windows.get(f"{window:g}", 0) > 0
                    for window in PRIMARY_WINDOW_SECONDS
                )
                for record in family_coverage
            ),
            "feature_excluded": sum(record.status == "excluded" for record in family_coverage),
            "coverage_reason_counts": dict(sorted(reasons.items())),
            "workload_config_ids": sorted({plan.resolved_config_id() for plan in family_plans}),
        }
    return result


def run_matrix(
    profile: str = "smoke",
    output: str | Path = "artifacts",
    repetitions: int | None = None,
    negative_calibration_mode: bool = False,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run a planned benign profile after a session-specific calibration gate."""
    if repetitions is None:
        repetitions = 1 if profile == "smoke" else 3
    estimate = estimate_matrix(profile, repetitions)
    context = ProvenanceContext.create(corpus_id=new_corpus_id(f"{profile}-matrix"))
    plans = plan_matrix(profile, repetitions)
    store = ArtifactStore(output)
    store.initialize()
    plan_manifest = _corpus_manifest(context, plans)
    plan_path = store.write_json(
        f"corpora/{context.corpus_id}-plan.json",
        plan_manifest.to_dict(),
    )
    calibration = run_calibration_sweep(
        output=output,
        timeout_s=timeout_s,
        provenance=context,
    )
    if (
        calibration["status"] != "supported"
        and profile != "smoke"
        and not negative_calibration_mode
    ):
        raise CalibrationError(
            f"standard/extended matrix blocked by negative calibration; see {calibration['path']}"
        )
    outcomes = []
    finalized_plans: list[PlannedRun] = []
    for plan in plans:
        name = str(plan.config["workload_name"])
        outcome = run_experiment(
            name,
            output=output,
            timeout_s=timeout_s,
            strict_preflight=True,
            raise_on_failure=False,
            provenance=context,
        )
        outcomes.append(outcome)
        accepted_run_id = (
            str(outcome["run_id"]) if outcome["manifest"]["exit_status"] == "completed" else None
        )
        finalized_plans.append(replace(plan, accepted_run_id=accepted_run_id))
    final_manifest = _corpus_manifest(context, tuple(finalized_plans))
    final_path = store.write_json(
        f"corpora/{context.corpus_id}-final.json",
        final_manifest.to_dict(),
    )
    extraction = extract_feature_result(output, final_manifest, output=output)
    try:
        coverage_gate = require_primary_coverage(
            extraction,
            required_families=PRIMARY_BENIGN_FAMILIES,
            minimum_runs_per_family=3,
        )
    except CoverageError as exc:
        coverage_gate = {
            "passed": False,
            "required_families": list(PRIMARY_BENIGN_FAMILIES),
            "minimum_runs_per_family": 3,
            "required_window_seconds": list(PRIMARY_WINDOW_SECONDS),
            "reason": str(exc),
        }
    family_summary = _family_summary(tuple(finalized_plans), extraction.coverage)
    schedule = [str(plan.config["workload_name"]) for plan in plans]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    summary_relative = f"results/matrix-{profile}-{timestamp}.json"
    summary = {
        "artifact_kind": "experiment_summary",
        "schema_version": SCHEMA_VERSION,
        "summary_type": "matrix",
        "profile": profile,
        "estimate": estimate,
        "calibration_status": calibration["status"],
        "negative_calibration_mode": negative_calibration_mode,
        "schedule": schedule,
        "planned_corpus_manifest": str(plan_path.relative_to(store.root)),
        "final_corpus_manifest": str(final_path.relative_to(store.root)),
        "run_ids": [outcome["run_id"] for outcome in outcomes],
        "experiment_session_id": context.experiment_session_id,
        "collection_id": context.collection_id,
        "corpus_id": context.corpus_id,
        "node_id": context.node_id,
        "source_commit": context.source_commit,
        "source_dirty": context.source_dirty,
        "completed": sum(outcome["manifest"]["exit_status"] == "completed" for outcome in outcomes),
        "failed": sum(outcome["manifest"]["exit_status"] != "completed" for outcome in outcomes),
        "feature_valid": sum(values["feature_valid"] for values in family_summary.values()),
        "family_counts": family_summary,
        "feature_extraction": extraction.summary(),
        "primary_coverage_gate": coverage_gate,
        "detector_metrics_computed": False,
        "summary_artifact": summary_relative,
    }
    store.write_json(summary_relative, summary)
    return summary
