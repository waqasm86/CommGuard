"""Experiment lifecycle, artifact preservation, calibration gates, and profiles."""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from commguard.adversarial import AdversarialHoldoutPlan, strategy_for_config
from commguard.artifacts import ArtifactStore
from commguard.calibration import (
    STANDARD_CALIBRATION_PAYLOAD_MIB,
    STANDARD_CALIBRATION_REPETITIONS,
    analyze_calibration,
    build_calibration_reference,
    validate_standard_calibration_result,
    verify_calibration_reference,
)
from commguard.corpus import CorpusManifest, PlannedRun
from commguard.distributed.launcher import LaunchResult, launch_torchrun
from commguard.environment.preflight import check_environment
from commguard.exceptions import (
    ApprovalRequiredError,
    ArtifactExistsError,
    CalibrationError,
    CoverageError,
    WorkloadError,
)
from commguard.features import (
    PRIMARY_BENIGN_FAMILIES,
    PRIMARY_WINDOW_SECONDS,
    extract_feature_result,
    require_primary_coverage,
)
from commguard.provenance import ProvenanceContext, new_corpus_id, new_run_id
from commguard.schemas import CURRENT_SCHEMA_VERSION, SCHEMA_VERSION, TELEMETRY_FIELDS, RunManifest
from commguard.telemetry import TelemetryCollector
from commguard.workloads import (
    adversarial_profile_workloads,
    calibration_workload_name,
    get_workload,
    profile_workloads,
)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load and validate a JSON object from a file path."""
    value: Any = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return value


def load_optional_json_object(path: Path) -> dict[str, Any] | None:
    """Load and validate a JSON object from a file path, returning None if not found."""
    if not path.exists():
        return None

    value: Any = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return value


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
    idle_intervals: list[tuple[int, int]] = []
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
            elif event.get("event") == "measurement_interval":
                details = event.get("details", {})
                start = int(details["measurement_start_monotonic_ns"])
                end = int(details["measurement_end_monotonic_ns"])
                if start <= end:
                    idle_intervals.append((start, end))
    if phase_start is None and phase_end is None and idle_intervals:
        # Use the interval common to both ranks. This excludes setup/teardown
        # synchronization traffic from an idle baseline observation.
        idle_start = max(interval[0] for interval in idle_intervals)
        idle_end = min(interval[1] for interval in idle_intervals)
        if idle_start <= idle_end:
            phase_start, phase_end = idle_start, idle_end
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
    preflight_data: dict[str, Any],
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
                    "strategy_summary",
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
            "gpus": preflight_data["gpus"],
            "platform": preflight_data["platform"],
            "torch": preflight_data["torch"],
            "nvcc": preflight_data["nvcc"],
            "topology": preflight_data["topology"],
            "packages": preflight_data["packages"],
            "telemetry_capabilities": preflight_data["telemetry_capabilities"],
        },
        environment_fingerprint=str(preflight_data["environment_fingerprint"]),
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
    adversarial_approval: bool = False,
) -> dict[str, Any]:
    """Run one isolated two-rank experiment and preserve success or failure evidence."""
    config = get_workload(workload)
    if overrides:
        identity_fields = {
            "config_id",
            "strategy_id",
            "mode",
            "label",
            "family",
            "designation",
            "enabled_by_default",
            "requires_human_approval",
        }
        changed_identity = sorted(
            key for key in identity_fields & set(overrides) if overrides[key] != config.get(key)
        )
        if changed_identity:
            raise ValueError(f"workload identity fields cannot be overridden: {changed_identity}")
        config.update(overrides)
    if config["mode"] == "calibration":
        expected_workload = calibration_workload_name(
            str(config.get("collective", "all_reduce")),
            int(config.get("payload_mib", 0)),
        )
        if workload != expected_workload:
            raise ValueError(
                "calibration workload identity conflicts with its configured payload: "
                f"workload={workload!r} expected={expected_workload!r}"
            )
    if config["designation"] == "adversarial" and not adversarial_approval:
        raise ApprovalRequiredError(
            f"workload={workload} family={config['family']} is a bounded defensive red-team "
            "strategy; pass explicit adversarial approval only after reviewing its plan"
        )
    if config["designation"] == "adversarial":
        strategy_for_config(config)
    store = ArtifactStore(output)
    store.initialize()
    context = provenance or ProvenanceContext.create(corpus_id=new_corpus_id(workload))
    preflight_data = check_environment(strict=strict_preflight, provenance=context)
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
        preflight_data,
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
    payload_mib: tuple[int, ...] = STANDARD_CALIBRATION_PAYLOAD_MIB,
    collective: str = "all_reduce",
    repetitions: int = STANDARD_CALIBRATION_REPETITIONS,
    timeout_s: float = 180.0,
    sampling_interval_s: float = 0.5,
    provenance: ProvenanceContext | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    allow_prior_sweep_evidence: bool = False,
    development_smoke_only: bool = False,
) -> dict[str, Any]:
    from commguard.telemetry.nvml import validate_sampling_interval

    sampling_interval_s = validate_sampling_interval(sampling_interval_s)
    if repetitions < 1:
        raise ValueError("calibration repetitions must be at least one")
    if not payload_mib or any(payload <= 0 for payload in payload_mib):
        raise ValueError("calibration payloads must be positive")
    if len(payload_mib) != len(set(payload_mib)):
        raise ValueError("calibration payloads must be unique")
    context = provenance or ProvenanceContext.create(corpus_id=new_corpus_id("calibration"))
    store = ArtifactStore(output)
    store.initialize()
    sweep_id = f"{context.experiment_session_id}-{context.collection_id}"
    marker_prefix = Path("results/calibration-sweeps") / sweep_id
    started_marker = store.resolve(marker_prefix / "started.json")
    current_markers = [started_marker] if started_marker.exists() else []
    existing_markers = sorted(store.resolve("results/calibration-sweeps").glob("*/started.json"))
    if current_markers or (existing_markers and not allow_prior_sweep_evidence):
        marker = current_markers[0] if current_markers else existing_markers[0]
        completed = marker.with_name("completed.json").is_file()
        state = "completed" if completed else "partial or in-progress"
        raise ArtifactExistsError(
            f"refusing to repeat a {state} calibration sweep in {store.root}; "
            "use a new notebook run ID and artifact directory"
        )
    expected_matrix = [
        {"observation_type": "idle_baseline", "repetition": repetition, "payload_mib": 0}
        for repetition in range(repetitions)
    ] + [
        {
            "observation_type": "collective",
            "repetition": repetition,
            "payload_mib": payload,
            "collective": collective,
            "workload_name": calibration_workload_name(collective, payload),
        }
        for repetition in range(repetitions)
        for payload in payload_mib
    ]
    store.write_json(
        marker_prefix / "started.json",
        {
            "artifact_kind": "experiment_summary",
            "schema_version": SCHEMA_VERSION,
            "summary_type": "calibration_sweep_started",
            "sweep_id": sweep_id,
            "experiment_session_id": context.experiment_session_id,
            "collection_id": context.collection_id,
            "notebook_version": context.notebook_version,
            "source_commit": context.source_commit,
            "planned_run_count": len(expected_matrix),
            "expected_matrix": expected_matrix,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    observations: list[dict[str, Any]] = []

    def progress(stage: str, **details: Any) -> None:
        if progress_callback is not None:
            progress_callback({"stage": stage, "sweep_id": sweep_id, **details})

    progress("sweep_started", planned_run_count=len(expected_matrix))

    duration_overrides = (
        {"warmup_seconds": 1.0, "min_measured_seconds": 5.0, "cooldown_seconds": 1.0}
        if development_smoke_only
        else {}
    )

    def observe(outcome: dict[str, Any], **identity: Any) -> dict[str, Any]:
        return {
            "run_id": outcome["run_id"],
            "workload_name": outcome["manifest"]["workload_name"],
            "worker_mode": outcome["manifest"]["config"]["mode"],
            "run_directory": f"runs/{outcome['run_id']}",
            **identity,
            "participation_valid": outcome["manifest"]["participation_valid"],
            "exit_status": outcome["manifest"]["exit_status"],
            "planned_measured_duration_seconds": outcome["manifest"]["config"].get(
                "min_measured_seconds"
            ),
            "measured_duration_seconds": outcome["manifest"].get("measured_duration_seconds"),
            "gpu_uuids": sorted(
                str(gpu.get("uuid"))
                for gpu in outcome["manifest"].get("environment", {}).get("gpus", [])
                if gpu.get("uuid")
            ),
            "pcie_supported": outcome["pcie_supported"],
            "pcie_total_mean_bytes_per_s": outcome["pcie_total_mean_bytes_per_s"],
            "pcie_total_median_bytes_per_s": outcome["pcie_total_median_bytes_per_s"],
            "pcie_sample_count": outcome["pcie_sample_count"],
            "collector_diagnostics": outcome.get("telemetry_diagnostics"),
        }

    for repetition in range(repetitions):
        progress(
            "run_started",
            observation_type="idle_baseline",
            workload_name="calibration_idle",
            repetition=repetition,
        )
        idle_outcome = run_experiment(
            "calibration_idle",
            output=output,
            overrides={
                "repetition": repetition,
                "sampling_interval_s": sampling_interval_s,
                **duration_overrides,
            },
            timeout_s=timeout_s,
            strict_preflight=True,
            raise_on_failure=False,
            provenance=context,
        )
        observations.append(
            observe(
                idle_outcome,
                observation_type="idle_baseline",
                is_idle=True,
                payload_mib=0,
                collective=None,
                repetition=repetition,
            )
        )
        progress(
            "run_completed",
            observation_type="idle_baseline",
            workload_name="calibration_idle",
            repetition=repetition,
            run_id=idle_outcome["run_id"],
            exit_status=idle_outcome["manifest"]["exit_status"],
        )
        for payload in payload_mib:
            workload_name = calibration_workload_name(collective, payload)
            progress(
                "run_started",
                observation_type="collective",
                workload_name=workload_name,
                payload_mib=payload,
                repetition=repetition,
            )
            outcome = run_experiment(
                workload_name,
                output=output,
                overrides={
                    "payload_mib": payload,
                    "collective": collective,
                    "repetition": repetition,
                    "sampling_interval_s": sampling_interval_s,
                    **duration_overrides,
                },
                timeout_s=timeout_s,
                strict_preflight=True,
                raise_on_failure=False,
                provenance=context,
            )
            observations.append(
                observe(
                    outcome,
                    observation_type="collective",
                    is_idle=False,
                    payload_mib=payload,
                    collective=collective,
                    repetition=repetition,
                )
            )
            progress(
                "run_completed",
                observation_type="collective",
                workload_name=workload_name,
                payload_mib=payload,
                repetition=repetition,
                run_id=outcome["run_id"],
                exit_status=outcome["manifest"]["exit_status"],
            )
    result = analyze_calibration(
        observations,
        minimum_sizes=len(payload_mib),
        minimum_repetitions=(
            repetitions if development_smoke_only else STANDARD_CALIBRATION_REPETITIONS
        ),
        minimum_measured_duration_s=5.0 if development_smoke_only else 30.0,
        maximum_interval_relative_error=0.2,
    )
    environment = check_environment(strict=True, provenance=context)
    result.update(
        {
            "sweep_id": sweep_id,
            "experiment_session_id": context.experiment_session_id,
            "collection_id": context.collection_id,
            "corpus_id": context.corpus_id,
            "node_id": context.node_id,
            "environment_fingerprint": environment["environment_fingerprint"],
            "session_fingerprint": environment["environment_fingerprint"],
            "source_commit": context.source_commit,
            "source_dirty": context.source_dirty,
            "target_sampling_interval_s": sampling_interval_s,
            "development_smoke_only": development_smoke_only,
            "scientific_acceptance_eligible": not development_smoke_only,
        }
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = store.write_json(f"results/calibration-{timestamp}.json", result)
    result["path"] = str(path)
    result["reference"] = build_calibration_reference(
        store.root,
        path,
        current_experiment_session_id=context.experiment_session_id,
    )
    validation = (
        {
            "clean_standard_calibration": False,
            "development_smoke_only": True,
            "scientific_acceptance_eligible": False,
            "planned_run_count": len(expected_matrix),
        }
        if development_smoke_only
        else validate_standard_calibration_result(store.root, result)
    )
    result["standard_sweep_validation"] = validation
    store.write_json(
        marker_prefix / "completed.json",
        {
            "artifact_kind": "experiment_summary",
            "schema_version": SCHEMA_VERSION,
            "summary_type": "calibration_sweep_completed",
            "sweep_id": sweep_id,
            "experiment_session_id": context.experiment_session_id,
            "collection_id": context.collection_id,
            "result_artifact": result["reference"]["calibration_artifact_path"],
            "result_sha256": result["reference"]["calibration_sha256"],
            "calibration_status": result["status"],
            "clean_standard_calibration": validation["clean_standard_calibration"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    progress(
        "sweep_completed",
        calibration_status=result["status"],
        clean_standard_calibration=validation["clean_standard_calibration"],
    )
    return result


def run_segmented_series(
    output: str | Path = "artifacts",
    *,
    adversarial_approval: bool = False,
    calibration_reference: dict[str, Any] | None = None,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run separately launched short DDP segments after an explicit human gate."""
    workload = "adversarial_segmented_runs"
    config = get_workload(workload)
    if not adversarial_approval:
        raise ApprovalRequiredError(
            "segmented_runs is a bounded defensive red-team series; explicit approval is required"
        )
    root = Path(output)
    if calibration_reference is None:
        raise CalibrationError(
            "segmented adversarial series requires an exact calibration reference"
        )
    verify_calibration_reference(
        root,
        calibration_reference,
        require_current_session=False,
    )
    context = ProvenanceContext.create(corpus_id=new_corpus_id("segmented-runs"))
    segment_count = int(config["segment_count"])
    segment_seconds = float(config["segment_seconds"])
    restart_gap_seconds = float(config["restart_gap_seconds"])
    segment_group_id = f"segments-{context.experiment_session_id}"
    plans = tuple(
        PlannedRun(
            plan_id=f"segmented-{index:03d}",
            workload_family=str(config["family"]),
            target_label=str(config["label"]),
            config={
                **config,
                "workload_name": workload,
                "segment_index": index,
                "segment_group_id": segment_group_id,
            },
            designation="adversarial",
            random_seed=int(config.get("seed", 1337)),
            workload_config_id=str(config["config_id"]),
        )
        for index in range(segment_count)
    )
    store = ArtifactStore(output)
    store.initialize()
    plan_manifest = _corpus_manifest(context, plans)
    plan_path = store.write_json(
        f"corpora/{context.corpus_id}-plan.json",
        plan_manifest.to_dict(),
    )
    outcomes: list[dict[str, Any]] = []
    finalized: list[PlannedRun] = []
    for index, plan in enumerate(plans):
        outcome = run_experiment(
            workload,
            output=output,
            overrides={
                "segment_index": index,
                "segment_group_id": segment_group_id,
                "warmup_seconds": float(config["warmup_seconds"]),
                "min_measured_seconds": segment_seconds,
            },
            timeout_s=timeout_s,
            strict_preflight=True,
            raise_on_failure=False,
            provenance=context,
            adversarial_approval=True,
        )
        outcomes.append(outcome)
        accepted = (
            str(outcome["run_id"]) if outcome["manifest"]["exit_status"] == "completed" else None
        )
        finalized.append(replace(plan, accepted_run_id=accepted))
        if index + 1 < segment_count:
            time.sleep(restart_gap_seconds)
    final_manifest = _corpus_manifest(context, tuple(finalized))
    final_path = store.write_json(
        f"corpora/{context.corpus_id}-final.json",
        final_manifest.to_dict(),
    )
    extraction = extract_feature_result(
        output,
        final_manifest,
        output=output,
        selected_designation="adversarial",
    )
    family_summary = _family_summary(tuple(finalized), extraction.coverage)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    summary = {
        "artifact_kind": "experiment_summary",
        "schema_version": SCHEMA_VERSION,
        "summary_type": "segmented_adversarial_series",
        "strategy_id": "segmented_runs",
        "segment_group_id": segment_group_id,
        "segment_count": segment_count,
        "segment_seconds": segment_seconds,
        "restart_gap_seconds": restart_gap_seconds,
        "planned_corpus_manifest": str(plan_path.relative_to(store.root)),
        "final_corpus_manifest": str(final_path.relative_to(store.root)),
        "run_ids": [outcome["run_id"] for outcome in outcomes],
        "completed": sum(plan.accepted_run_id is not None for plan in finalized),
        "failed": sum(plan.accepted_run_id is None for plan in finalized),
        "family_counts": family_summary,
        "feature_extraction": extraction.summary(),
        "primary_feature_coverage_expected": False,
        "coverage_note": (
            "Segments are intentionally shorter than the 30-second primary window and must not "
            "be concatenated across restart gaps."
        ),
        "detector_metrics_computed": False,
    }
    store.write_json(f"results/segmented-series-{timestamp}.json", summary)
    return summary


def run_adversarial_matrix(
    output: str | Path = "artifacts",
    *,
    holdout_plan: AdversarialHoldoutPlan,
    adversarial_approval: bool = False,
    release_final_adversarial_holdout: bool = False,
    final_holdout_approval: bool = False,
    repetitions: int = 1,
    timeout_s: float = 180.0,
    provenance: ProvenanceContext | None = None,
    workload_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collect a bounded adversarial corpus only after benign acceptance and approval."""
    if not adversarial_approval:
        raise ApprovalRequiredError(
            "the bounded adversarial matrix requires explicit approval before artifact creation"
        )
    if release_final_adversarial_holdout and not final_holdout_approval:
        raise ApprovalRequiredError(
            "releasing the sealed final adversarial holdout requires separate explicit approval"
        )
    if repetitions < 1:
        raise ValueError("adversarial repetitions must be at least one")
    holdout_plan.validate()
    root = Path(output)
    evaluation_paths = sorted((root / "results").glob("evaluation-*.json"))
    if not evaluation_paths:
        raise CoverageError("adversarial collection requires a saved benign acceptance evaluation")
    benign_acceptance = json.loads(evaluation_paths[-1].read_text(encoding="utf-8"))
    if not benign_acceptance.get("coverage_gate", {}).get("passed"):
        raise CoverageError("saved benign evaluation does not contain a passing coverage gate")
    if not benign_acceptance.get("primary_communication_only"):
        raise CoverageError("saved benign evaluation has no primary communication-only result")

    context = provenance or ProvenanceContext.create(corpus_id=new_corpus_id("adversarial-matrix"))
    segment_group_id = f"segments-{context.experiment_session_id}"
    plans: list[PlannedRun] = []

    # Convert Sequence to list for indexing
    if workload_names is None:
        selected_workloads: Sequence[str] = tuple(adversarial_profile_workloads())
    else:
        selected_workloads = list(workload_names)

    if len(selected_workloads) == 0 or len(set(selected_workloads)) != len(selected_workloads):
        raise ValueError("adversarial workload_names must be non-empty and unique")

    # Now safely iterate over the sequence
    workload_list = list(selected_workloads)  # Convert to list for safe iteration
    for workload in workload_list:
        config = get_workload(workload)
        round_name = holdout_plan.round_for(
            family=str(config["family"]),
            session_id=context.experiment_session_id,
            config_id=str(config["config_id"]),
            release_final=release_final_adversarial_holdout,
        )
        slot_count = (
            int(config["segment_count"])
            if config["strategy_id"] == "segmented_runs" and round_name != "sealed_final_holdout"
            else 1
        )
        for repetition in range(repetitions):
            for slot in range(slot_count):
                planned_config = {
                    **config,
                    "workload_name": workload,
                    "adversarial_round": round_name,
                    "repetition": repetition,
                }
                if config["strategy_id"] == "segmented_runs":
                    planned_config.update(
                        {
                            "segment_index": slot,
                            "segment_group_id": segment_group_id,
                        }
                    )
                plans.append(
                    PlannedRun(
                        plan_id=(
                            f"adversarial-{config['strategy_id']}-r{repetition:03d}-s{slot:03d}"
                        ),
                        workload_family=str(config["family"]),
                        target_label=str(config["label"]),
                        config=planned_config,
                        designation="adversarial",
                        random_seed=int(config.get("seed", 1337)),
                        workload_config_id=str(config["config_id"]),
                    )
                )
    store = ArtifactStore(output)
    store.initialize()
    plan_manifest = _corpus_manifest(context, tuple(plans))
    plan_path = store.write_json(
        f"corpora/{context.corpus_id}-plan.json",
        plan_manifest.to_dict(),
    )
    outcomes: list[dict[str, Any]] = []
    finalized: list[PlannedRun] = []
    for index, plan in enumerate(plans):
        if plan.config["adversarial_round"] == "sealed_final_holdout":
            finalized.append(plan)
            continue
        workload = str(plan.config["workload_name"])
        overrides = {
            "repetition": int(plan.config["repetition"]),
            "adversarial_round": str(plan.config["adversarial_round"]),
        }
        if plan.config["strategy_id"] == "segmented_runs":
            overrides.update(
                {
                    "segment_index": int(plan.config["segment_index"]),
                    "segment_group_id": str(plan.config["segment_group_id"]),
                    "min_measured_seconds": float(plan.config["segment_seconds"]),
                }
            )
        outcome = run_experiment(
            workload,
            output=output,
            overrides=overrides,
            timeout_s=timeout_s,
            strict_preflight=True,
            raise_on_failure=False,
            provenance=context,
            adversarial_approval=True,
        )
        outcomes.append(outcome)
        accepted = (
            str(outcome["run_id"]) if outcome["manifest"]["exit_status"] == "completed" else None
        )
        finalized.append(replace(plan, accepted_run_id=accepted))
        if plan.config["strategy_id"] == "segmented_runs" and index + 1 < len(plans):
            next_plan = plans[index + 1]
            if (
                next_plan.config["strategy_id"] == "segmented_runs"
                and next_plan.config["repetition"] == plan.config["repetition"]
            ):
                time.sleep(float(plan.config["restart_gap_seconds"]))
    final_manifest = _corpus_manifest(context, tuple(finalized))
    final_path = store.write_json(
        f"corpora/{context.corpus_id}-final.json",
        final_manifest.to_dict(),
    )
    prior_extractions = set((store.root / "features").glob("extraction-*.json"))
    extraction = extract_feature_result(
        output,
        final_manifest,
        output=output,
        selected_designation="adversarial",
    )
    new_extractions = set((store.root / "features").glob("extraction-*.json")) - prior_extractions
    if len(new_extractions) != 1:
        raise RuntimeError("adversarial matrix did not create exactly one extraction summary")
    extraction_path = new_extractions.pop()
    family_summary = _family_summary(tuple(finalized), extraction.coverage)
    for family, counts in family_summary.items():
        sealed = sum(
            plan.workload_family == family
            and plan.config["adversarial_round"] == "sealed_final_holdout"
            for plan in finalized
        )
        counts["sealed"] = sealed
        counts["failed"] -= sealed
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    summary_relative = f"results/adversarial-matrix-{timestamp}.json"
    summary = {
        "artifact_kind": "experiment_summary",
        "schema_version": SCHEMA_VERSION,
        "summary_type": "adversarial_matrix",
        "experiment_session_id": context.experiment_session_id,
        "collection_id": context.collection_id,
        "corpus_id": context.corpus_id,
        "source_commit": context.source_commit,
        "source_dirty": context.source_dirty,
        "input_archive_sha256": context.input_archive_sha256,
        "holdout_plan": holdout_plan.to_dict(),
        "final_holdout_released": release_final_adversarial_holdout,
        "planned_corpus_manifest": str(plan_path.relative_to(store.root)),
        "final_corpus_manifest": str(final_path.relative_to(store.root)),
        "feature_extraction_summary": str(extraction_path.relative_to(store.root)),
        "planned": len(plans),
        "executed": len(outcomes),
        "sealed": sum(
            plan.config["adversarial_round"] == "sealed_final_holdout" for plan in finalized
        ),
        "completed": sum(plan.accepted_run_id is not None for plan in finalized),
        "failed": sum(
            plan.accepted_run_id is None
            and plan.config["adversarial_round"] != "sealed_final_holdout"
            for plan in finalized
        ),
        "run_ids": [outcome["run_id"] for outcome in outcomes],
        "family_counts": family_summary,
        "feature_extraction": extraction.summary(),
        "detector_metrics_computed": False,
        "summary_artifact": summary_relative,
    }
    store.write_json(summary_relative, summary)
    return summary


def run_periodic_synchronization_study(
    output: str | Path = "artifacts",
    *,
    holdout_plan: AdversarialHoldoutPlan,
    adversarial_approval: bool = False,
    synchronization_intervals: tuple[int, ...] = (1, 2, 4, 8, 16),
    repetitions: int = 1,
    timeout_s: float = 180.0,
    provenance: ProvenanceContext | None = None,
) -> dict[str, Any]:
    """Run the bounded primary red-team study against an already frozen detector."""
    if synchronization_intervals != (1, 2, 4, 8, 16):
        raise ValueError("the canonical periodic study requires k=(1, 2, 4, 8, 16)")
    result = run_adversarial_matrix(
        output,
        holdout_plan=holdout_plan,
        adversarial_approval=adversarial_approval,
        repetitions=repetitions,
        timeout_s=timeout_s,
        provenance=provenance,
        workload_names=tuple(
            f"adversarial_periodic_local_sgd_k{interval}" for interval in synchronization_intervals
        ),
    )
    result["study_type"] = "periodic_synchronization_local_update_training"
    result["synchronization_intervals"] = list(synchronization_intervals)
    result["bounded_local_research_only"] = True
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
        config.update({"workload_name": name, "repetition": occurrence})
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
    calibration_reference: dict[str, Any] | None = None,
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
        calibration_reference=calibration_reference,
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
    provenance: ProvenanceContext | None = None,
    prior_calibration_reference: dict[str, Any] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run a planned benign profile after a session-specific calibration gate."""
    if repetitions is None:
        repetitions = 1 if profile == "smoke" else 3
    development_smoke_only = profile == "smoke"
    estimate = estimate_matrix(profile, repetitions)
    context = provenance or ProvenanceContext.create(corpus_id=new_corpus_id(f"{profile}-matrix"))
    plans = plan_matrix(profile, repetitions)
    store = ArtifactStore(output)
    store.initialize()
    existing_summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(store.resolve("results").glob(f"matrix-{profile}-*.json"))
    ]
    matching_summaries = [
        item
        for item in existing_summaries
        if item.get("experiment_session_id") == context.experiment_session_id
        and item.get("corpus_id") == context.corpus_id
    ]
    if matching_summaries:
        if not resume or len(matching_summaries) != 1:
            raise ArtifactExistsError("an exact matrix summary already exists")
        return cast(dict[str, Any], matching_summaries[0])
    plan_manifest = _corpus_manifest(context, plans)
    plan_relative = Path(f"corpora/{context.corpus_id}-plan.json")
    plan_path = store.resolve(plan_relative)
    if plan_path.exists():
        if not resume:
            raise ArtifactExistsError(f"corpus plan already exists: {plan_path}")
        saved_plan = CorpusManifest.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
        saved_plan.validate()
        if saved_plan.to_dict() != plan_manifest.to_dict():
            raise ArtifactExistsError("existing corpus plan configuration does not match")
    else:
        plan_path = store.write_json(plan_relative, plan_manifest.to_dict())
    sweep_id = f"{context.experiment_session_id}-{context.collection_id}"
    completed_marker = store.resolve(
        Path("results/calibration-sweeps") / sweep_id / "completed.json"
    )
    if resume and completed_marker.is_file():
        marker = json.loads(completed_marker.read_text(encoding="utf-8"))
        calibration_path = store.resolve(str(marker["result_artifact"]))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration.update(
            {
                "path": str(calibration_path),
                "reference": build_calibration_reference(
                    store.root,
                    calibration_path,
                    current_experiment_session_id=context.experiment_session_id,
                ),
            }
        )
    else:
        calibration = run_calibration_sweep(
            output=output,
            payload_mib=((16,) if development_smoke_only else STANDARD_CALIBRATION_PAYLOAD_MIB),
            repetitions=(1 if development_smoke_only else STANDARD_CALIBRATION_REPETITIONS),
            timeout_s=timeout_s,
            provenance=context,
            allow_prior_sweep_evidence=prior_calibration_reference is not None,
            development_smoke_only=development_smoke_only,
        )
    calibration_reference = dict(calibration["reference"])
    verify_calibration_reference(
        store.root,
        calibration_reference,
        require_supported=not development_smoke_only,
        expected_experiment_session_ids={context.experiment_session_id},
        expected_environment_fingerprints={
            str(calibration_reference["calibration_environment_fingerprint"])
        },
        expected_source_commits={context.source_commit},
    )
    if prior_calibration_reference is not None:
        verify_calibration_reference(
            store.root,
            prior_calibration_reference,
            require_current_session=False,
            require_supported=False,
        )
        if prior_calibration_reference["calibration_relationship"] != "prior_session":
            raise CalibrationError("supplied prior calibration must be labeled prior_session")
    if (
        calibration["status"] != "supported"
        and not negative_calibration_mode
        and not development_smoke_only
    ):
        raise CalibrationError(
            f"benign matrix blocked by negative calibration; see {calibration['path']}"
        )
    outcomes: list[dict[str, Any]] = []
    finalized_plans: list[PlannedRun] = []
    for plan in plans:
        name = str(plan.config["workload_name"])
        planned_hash = hashlib.sha256(
            json.dumps(plan.config, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        accepted_matches: list[dict[str, Any]] = []
        if resume:
            for manifest_path in sorted(store.resolve("runs").glob("*/manifest.json")):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                config = dict(manifest.get("config", {}))
                if config.get("plan_id") != plan.plan_id:
                    continue
                if config.get("planned_configuration_hash") != planned_hash:
                    raise ArtifactExistsError(
                        f"plan {plan.plan_id} has an existing mismatched attempt"
                    )
                if (
                    manifest.get("exit_status") == "completed"
                    and manifest.get("participation_valid") is True
                    and float(manifest.get("measured_duration_seconds") or 0) >= 30.0
                ):
                    accepted_matches.append(manifest)
        if len(accepted_matches) > 1:
            raise ArtifactExistsError(f"plan {plan.plan_id} has duplicate accepted runs")
        if accepted_matches:
            accepted_id = str(accepted_matches[0]["run_id"])
            outcomes.append({"run_id": accepted_id, "manifest": accepted_matches[0]})
            finalized_plans.append(replace(plan, accepted_run_id=accepted_id))
            continue
        outcome = run_experiment(
            name,
            output=output,
            overrides={
                "communication_threshold_bytes_per_s": calibration.get(
                    "capture_threshold_bytes_per_s"
                ),
                "plan_id": plan.plan_id,
                "repetition": int(plan.config["repetition"]),
                "planned_configuration_hash": planned_hash,
            },
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
    final_manifest = _corpus_manifest(
        context,
        tuple(finalized_plans),
        calibration_reference=calibration_reference,
    )
    final_relative = Path(f"corpora/{context.corpus_id}-final.json")
    final_path = store.resolve(final_relative)
    if final_path.exists():
        if json.loads(final_path.read_text(encoding="utf-8")) != final_manifest.to_dict():
            raise ArtifactExistsError("existing final corpus manifest does not match resumed runs")
    else:
        final_path = store.write_json(final_relative, final_manifest.to_dict())
    prior_extractions = set((store.root / "features").glob("extraction-*.json"))
    extraction = extract_feature_result(output, final_manifest, output=output)
    new_extractions = set((store.root / "features").glob("extraction-*.json")) - prior_extractions
    if len(new_extractions) != 1:
        raise RuntimeError("benign matrix did not create exactly one extraction summary")
    extraction_path = new_extractions.pop()
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
        "calibration_reference": calibration_reference,
        "prior_calibration_reference": prior_calibration_reference,
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
        "feature_extraction_summary": str(extraction_path.relative_to(store.root)),
        "primary_coverage_gate": coverage_gate,
        "detector_metrics_computed": False,
        "development_smoke_only": development_smoke_only,
        "scientific_acceptance_eligible": not development_smoke_only,
        "summary_artifact": summary_relative,
    }
    store.write_json(summary_relative, summary)
    return summary
