"""Deterministic temporal features with explicit corpus coverage diagnostics."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from commguard.artifacts import ArtifactStore
from commguard.corpus import CorpusManifest, PlannedRun
from commguard.exceptions import ValidationError
from commguard.features.coverage import (
    DEFAULT_WINDOW_SECONDS,
    CoverageReason,
    CoverageRecord,
    ExtractionResult,
)
from commguard.schemas import (
    CURRENT_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSION,
    TELEMETRY_FIELDS,
    load_artifact,
)

METADATA_COLUMNS = {
    "artifact_kind",
    "schema_version",
    "run_id",
    "plan_id",
    "experiment_session_id",
    "collection_id",
    "corpus_id",
    "node_id",
    "source_commit",
    "environment_fingerprint",
    "session_fingerprint",
    "legacy_grouping_ambiguous",
    "target_label",
    "workload_family",
    "workload_config_id",
    "designation",
    "window_seconds",
    "window_index",
    "window_start_monotonic_ns",
    "window_end_monotonic_ns",
    "startup_excluded",
    "sample_count_gpu0",
    "sample_count_gpu1",
    "aligned_sample_pairs",
    "alignment_max_delta_seconds",
}


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    if denominator == 0:
        return None
    return (
        sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
        / denominator
    )


def _slope(values: list[float], times: list[float]) -> float | None:
    if len(values) < 2:
        return None
    x_mean = statistics.fmean(times)
    y_mean = statistics.fmean(values)
    denominator = sum((value - x_mean) ** 2 for value in times)
    if denominator == 0:
        return None
    return (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(times, values, strict=True)) / denominator
    )


def _stats(values: list[float], times: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            key: None
            for key in (
                "mean",
                "std",
                "min",
                "max",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "mad",
                "iqr",
                "range",
                "cv",
                "slope",
                "autocorr_lag1",
            )
        }
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    p25 = _quantile(values, 0.25)
    p75 = _quantile(values, 0.75)
    median = _quantile(values, 0.5)
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "p25": p25,
        "p50": median,
        "p75": p75,
        "p90": _quantile(values, 0.9),
        "p95": _quantile(values, 0.95),
        "mad": statistics.median(abs(value - median) for value in values),
        "iqr": p75 - p25,
        "range": max(values) - min(values),
        "cv": std / abs(mean) if mean else None,
        "slope": _slope(values, times),
        "autocorr_lag1": _correlation(values[:-1], values[1:]) if len(values) > 2 else None,
    }


def _communication_features(
    values: list[float],
    times: list[float],
    *,
    threshold: float | None,
    sample_count: int,
) -> dict[str, float | int | None]:
    """Return robust, finite communication features for one complete window."""
    stats = _stats(values, times)
    result: dict[str, float | int | None] = {
        "mean": stats["mean"],
        "median": stats["p50"],
        "maximum": stats["max"],
        "p90": stats["p90"],
        "p95": stats["p95"],
        "std": stats["std"],
        "mad": stats["mad"],
        "coefficient_of_variation": stats["cv"],
        "periodicity_lag1_autocorrelation": stats["autocorr_lag1"],
        "sampling_validity_fraction": len(values) / sample_count if sample_count else 0.0,
    }
    if not values or threshold is None:
        result.update(
            {
                "fraction_above_idle_threshold": None,
                "communication_duty_cycle": None,
                "burst_count": None,
                "average_burst_duration_s": None,
                "maximum_burst_duration_s": None,
                "longest_consecutive_burst": None,
                "near_idle_sample_fraction": None,
            }
        )
        return result
    active = [value > threshold for value in values]
    runs: list[tuple[int, float]] = []
    start: int | None = None
    for index, is_active in enumerate([*active, False]):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            end = index - 1
            interval = (
                statistics.median(
                    [right - left for left, right in zip(times, times[1:], strict=False)]
                )
                if len(times) > 1
                else 0.0
            )
            runs.append((end - start + 1, max(0.0, times[end] - times[start] + interval)))
            start = None
    durations = [duration for _, duration in runs]
    fraction = sum(active) / len(active)
    result.update(
        {
            "fraction_above_idle_threshold": fraction,
            "communication_duty_cycle": fraction,
            "burst_count": len(runs),
            "average_burst_duration_s": statistics.fmean(durations) if durations else 0.0,
            "maximum_burst_duration_s": max(durations, default=0.0),
            "longest_consecutive_burst": max((count for count, _ in runs), default=0),
            "near_idle_sample_fraction": 1.0 - fraction,
        }
    )
    return result


def _valid_values(samples: list[dict[str, Any]], field: str) -> tuple[list[float], list[float]]:
    values: list[float] = []
    times: list[float] = []
    origin = int(samples[0]["monotonic_ns"]) if samples else 0
    for sample in samples:
        reading = sample["fields"][field]
        if reading["supported"] and reading["value"] is not None:
            values.append(float(reading["value"]))
            times.append((int(sample["monotonic_ns"]) - origin) / 1e9)
    return values, times


def _paired_fields(
    samples: list[dict[str, Any]], left_field: str, right_field: str
) -> tuple[list[float], list[float]]:
    left: list[float] = []
    right: list[float] = []
    for sample in samples:
        first = sample["fields"][left_field]
        second = sample["fields"][right_field]
        if (
            first["supported"]
            and second["supported"]
            and first["value"] is not None
            and second["value"] is not None
        ):
            left.append(float(first["value"]))
            right.append(float(second["value"]))
    return left, right


def align_samples_by_timestamp(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    tolerance_seconds: float = 0.25,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair timestamp-ordered samples without reuse inside a fixed tolerance."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds cannot be negative")
    tolerance_ns = int(tolerance_seconds * 1e9)
    left_rows = sorted(left, key=lambda item: int(item["monotonic_ns"]))
    right_rows = sorted(right, key=lambda item: int(item["monotonic_ns"]))
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left_rows) and right_index < len(right_rows):
        left_ns = int(left_rows[left_index]["monotonic_ns"])
        right_ns = int(right_rows[right_index]["monotonic_ns"])
        delta = left_ns - right_ns
        if abs(delta) <= tolerance_ns:
            pairs.append((left_rows[left_index], right_rows[right_index]))
            left_index += 1
            right_index += 1
        elif delta < 0:
            left_index += 1
        else:
            right_index += 1
    return pairs


def _cross_field_values(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], field: str
) -> tuple[list[float], list[float]]:
    left_values: list[float] = []
    right_values: list[float] = []
    for left, right in pairs:
        left_reading = left["fields"][field]
        right_reading = right["fields"][field]
        if (
            left_reading["supported"]
            and right_reading["supported"]
            and left_reading["value"] is not None
            and right_reading["value"] is not None
        ):
            left_values.append(float(left_reading["value"]))
            right_values.append(float(right_reading["value"]))
    return left_values, right_values


def _feature_window(
    manifest: Mapping[str, Any],
    samples_by_gpu: dict[int, list[dict[str, Any]]],
    window_seconds: float,
    window_index: int,
    window_start_ns: int,
    window_end_ns: int,
    startup_excluded: bool,
    alignment_tolerance_seconds: float,
    plan: PlannedRun | None = None,
    corpus: CorpusManifest | None = None,
) -> dict[str, Any] | None:
    selected = {
        gpu: [
            sample
            for sample in samples
            if window_start_ns <= int(sample["monotonic_ns"]) < window_end_ns
        ]
        for gpu, samples in samples_by_gpu.items()
    }
    if set(selected) != {0, 1} or any(len(selected[gpu]) < 2 for gpu in (0, 1)):
        return None
    aligned = align_samples_by_timestamp(
        selected[0], selected[1], tolerance_seconds=alignment_tolerance_seconds
    )
    if len(aligned) < 2:
        return None
    alignment_deltas = [
        abs(int(left["monotonic_ns"]) - int(right["monotonic_ns"])) / 1e9 for left, right in aligned
    ]
    environment_fingerprint = str(manifest.get("environment_fingerprint", ""))
    has_explicit_provenance = plan is not None and corpus is not None
    row: dict[str, Any] = {
        "artifact_kind": "feature_row",
        "schema_version": (
            CURRENT_SCHEMA_VERSION if has_explicit_provenance else LEGACY_SCHEMA_VERSION
        ),
        "run_id": manifest["run_id"],
        "plan_id": plan.plan_id if plan is not None else None,
        "experiment_session_id": (
            manifest.get("experiment_session_id")
            or (corpus.experiment_session_id if corpus is not None else None)
        ),
        "collection_id": manifest.get("collection_id")
        or (corpus.collection_id if corpus is not None else None),
        "corpus_id": manifest.get("corpus_id")
        or (corpus.corpus_id if corpus is not None else None),
        "node_id": manifest.get("node_id") or (corpus.node_id if corpus is not None else None),
        "source_commit": manifest.get("source_commit")
        or (corpus.source_commit if corpus is not None else None),
        "environment_fingerprint": environment_fingerprint,
        "session_fingerprint": environment_fingerprint,
        "legacy_grouping_ambiguous": bool(
            manifest.get("legacy_grouping_ambiguous", not has_explicit_provenance)
        ),
        "target_label": plan.target_label if plan is not None else manifest["workload_label"],
        "workload_family": (
            plan.workload_family if plan is not None else manifest["workload_family"]
        ),
        "workload_config_id": (
            plan.resolved_config_id() if plan is not None else manifest.get("workload_config_id")
        ),
        "designation": plan.designation if plan is not None else manifest["designation"],
        "window_seconds": window_seconds,
        "window_index": window_index,
        "window_start_monotonic_ns": window_start_ns,
        "window_end_monotonic_ns": window_end_ns,
        "startup_excluded": startup_excluded,
        "sample_count_gpu0": len(selected[0]),
        "sample_count_gpu1": len(selected[1]),
        "aligned_sample_pairs": len(aligned),
        "alignment_max_delta_seconds": max(alignment_deltas),
    }
    values_by_gpu: dict[tuple[int, str], list[float]] = {}
    communication_by_gpu: dict[int, list[float]] = {}
    communication_threshold_value = dict(manifest.get("config", {})).get(
        "communication_threshold_bytes_per_s"
    )
    communication_threshold = (
        float(communication_threshold_value)
        if isinstance(communication_threshold_value, (int, float))
        and not isinstance(communication_threshold_value, bool)
        else None
    )
    row["communication_idle_threshold_bytes_per_s"] = communication_threshold
    for gpu in (0, 1):
        for field in TELEMETRY_FIELDS:
            values, times = _valid_values(selected[gpu], field)
            values_by_gpu[(gpu, field)] = values
            for statistic_name, value in _stats(values, times).items():
                row[f"gpu{gpu}__{field}__{statistic_name}"] = value
            row[f"gpu{gpu}__{field}__missing_fraction"] = 1.0 - (len(values) / len(selected[gpu]))
        utilization = values_by_gpu[(gpu, "gpu_utilization_pct")]
        row[f"gpu{gpu}__utilization_idle_fraction"] = (
            sum(value <= 5 for value in utilization) / len(utilization) if utilization else None
        )
        row[f"gpu{gpu}__utilization_duty_cycle"] = (
            sum(value > 5 for value in utilization) / len(utilization) if utilization else None
        )
        tx, rx = _paired_fields(selected[gpu], "pcie_tx_bytes_per_s", "pcie_rx_bytes_per_s")
        if tx:
            totals = [first + second for first, second in zip(tx, rx, strict=True)]
            communication_by_gpu[gpu] = totals
            origin_ns = int(selected[gpu][0]["monotonic_ns"])
            total_times = [
                (int(sample["monotonic_ns"]) - origin_ns) / 1e9
                for sample in selected[gpu]
                if sample["fields"]["pcie_tx_bytes_per_s"]["supported"]
                and sample["fields"]["pcie_rx_bytes_per_s"]["supported"]
                and sample["fields"]["pcie_tx_bytes_per_s"]["value"] is not None
                and sample["fields"]["pcie_rx_bytes_per_s"]["value"] is not None
            ]
            for feature_name, value in _communication_features(
                totals,
                total_times,
                threshold=communication_threshold,
                sample_count=len(selected[gpu]),
            ).items():
                row[f"gpu{gpu}__pcie_total__{feature_name}"] = value
            row[f"gpu{gpu}__pcie_total_mean_bytes_per_s"] = statistics.fmean(totals)
            rx_mean = statistics.fmean(rx)
            row[f"gpu{gpu}__pcie_tx_rx_ratio"] = statistics.fmean(tx) / rx_mean if rx_mean else None
        else:
            communication_by_gpu[gpu] = []
            row[f"gpu{gpu}__pcie_total_mean_bytes_per_s"] = None
            row[f"gpu{gpu}__pcie_tx_rx_ratio"] = None
    for field in TELEMETRY_FIELDS:
        left, right = _cross_field_values(aligned, field)
        if left:
            difference = statistics.fmean(
                abs(first - second) for first, second in zip(left, right, strict=True)
            )
            row[f"cross_gpu__{field}__mean_abs_difference"] = difference
            scale = statistics.fmean(abs(value) for value in left + right)
            row[f"cross_gpu__{field}__normalized_divergence"] = (
                difference / scale if scale else None
            )
            row[f"cross_gpu__{field}__correlation"] = _correlation(left, right)
            row[f"cross_gpu__{field}__lag1_correlation"] = (
                _correlation(left[:-1], right[1:]) if len(left) > 2 else None
            )
        else:
            row[f"cross_gpu__{field}__mean_abs_difference"] = None
            row[f"cross_gpu__{field}__normalized_divergence"] = None
            row[f"cross_gpu__{field}__correlation"] = None
            row[f"cross_gpu__{field}__lag1_correlation"] = None
    total_left = communication_by_gpu[0]
    total_right = communication_by_gpu[1]
    paired_count = min(len(total_left), len(total_right))
    if paired_count:
        left_values = total_left[:paired_count]
        right_values = total_right[:paired_count]
        row["cross_gpu__pcie_total__correlation"] = _correlation(left_values, right_values)
        difference = statistics.fmean(
            abs(left - right) for left, right in zip(left_values, right_values, strict=True)
        )
        scale = statistics.fmean(abs(value) for value in left_values + right_values)
        row["cross_gpu__pcie_total__asymmetry"] = difference / scale if scale else 0.0
    else:
        row["cross_gpu__pcie_total__correlation"] = None
        row["cross_gpu__pcie_total__asymmetry"] = None
    return row


def extract_run_features(
    manifest: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    window_lengths: Sequence[float] = DEFAULT_WINDOW_SECONDS,
    stride_fraction: float = 0.5,
    exclude_startup: bool = True,
    alignment_tolerance_seconds: float = 0.25,
    *,
    plan: PlannedRun | None = None,
    corpus: CorpusManifest | None = None,
) -> list[dict[str, Any]]:
    if not 0 < stride_fraction <= 1:
        raise ValueError("stride_fraction must be in (0, 1]")
    if not window_lengths or any(float(value) <= 0 for value in window_lengths):
        raise ValueError("window lengths must be positive")
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_gpu[int(sample["gpu_index"])].append(sample)
    if set(by_gpu) != {0, 1}:
        return []
    for records in by_gpu.values():
        records.sort(key=lambda item: int(item["monotonic_ns"]))
    common_start_ns = max(int(by_gpu[gpu][0]["monotonic_ns"]) for gpu in (0, 1))
    common_end_ns = min(int(by_gpu[gpu][-1]["monotonic_ns"]) for gpu in (0, 1)) + 1
    start_ns = common_start_ns
    end_ns = common_end_ns
    if exclude_startup:
        measured_start = manifest.get("measurement_start_monotonic_ns")
        measured_end = manifest.get("measurement_end_monotonic_ns")
        if measured_start is not None:
            start_ns = max(start_ns, int(measured_start))
        else:
            start_ns += int(float(manifest.get("warmup_seconds", 0.0)) * 1e9)
        if measured_end is not None:
            end_ns = min(end_ns, int(measured_end) + 1)
    rows: list[dict[str, Any]] = []
    for window_seconds in window_lengths:
        window_ns = int(float(window_seconds) * 1e9)
        stride_ns = max(1, int(window_ns * stride_fraction))
        window_start = start_ns
        index = 0
        while window_start + window_ns <= end_ns:
            row = _feature_window(
                manifest,
                by_gpu,
                float(window_seconds),
                index,
                window_start,
                window_start + window_ns,
                exclude_startup,
                alignment_tolerance_seconds,
                plan,
                corpus,
            )
            if row is not None:
                rows.append(row)
            window_start += stride_ns
            index += 1
    return rows


def _load_corpus_manifest(
    manifest: CorpusManifest | Mapping[str, Any] | str | Path,
) -> CorpusManifest:
    if isinstance(manifest, CorpusManifest):
        manifest.validate()
        return manifest
    if isinstance(manifest, Mapping):
        result = CorpusManifest.from_dict(manifest)
        result.validate()
        return result
    loaded = load_artifact(manifest)
    if not isinstance(loaded, dict):
        raise ValidationError("corpus manifest must be a JSON object")
    return CorpusManifest.from_dict(loaded)


def _excluded_record(
    corpus: CorpusManifest,
    plan: PlannedRun,
    windows: tuple[float, ...],
    tolerance: float,
    reason: CoverageReason,
    detail: str,
    *,
    manifest: Mapping[str, Any] | None = None,
    rows_per_gpu: dict[str, int] | None = None,
    common_start_ns: int | None = None,
    common_end_ns: int | None = None,
    overlap_seconds: float = 0.0,
    usable_seconds: float = 0.0,
    aligned_pairs: int = 0,
    sampling_gaps: dict[str, dict[str, float | int | None]] | None = None,
) -> CoverageRecord:
    return CoverageRecord(
        plan_id=plan.plan_id,
        run_id=plan.accepted_run_id,
        workload_family=plan.workload_family,
        target_label=plan.target_label,
        designation=plan.designation,
        experiment_session_id=corpus.experiment_session_id,
        collection_id=corpus.collection_id,
        corpus_id=corpus.corpus_id,
        node_id=corpus.node_id,
        status="excluded",
        reason_code=reason.value,
        detail=detail,
        started_at_utc=str(manifest.get("started_at_utc")) if manifest else None,
        ended_at_utc=str(manifest.get("ended_at_utc")) if manifest else None,
        rows_per_gpu=rows_per_gpu or {"0": 0, "1": 0},
        common_start_monotonic_ns=common_start_ns,
        common_end_monotonic_ns=common_end_ns,
        common_overlap_seconds=overlap_seconds,
        configured_warmup_seconds=float(manifest.get("warmup_seconds", 0.0)) if manifest else 0.0,
        usable_duration_seconds=usable_seconds,
        requested_window_seconds=windows,
        emitted_windows={f"{window:g}": 0 for window in windows},
        sampling_gap_seconds_by_gpu=sampling_gaps or _empty_sampling_gap_stats(),
        alignment_tolerance_seconds=tolerance,
        aligned_sample_pairs=aligned_pairs,
        legacy_grouping_ambiguous=bool(
            manifest.get("legacy_grouping_ambiguous", False) if manifest else False
        ),
    )


def _empty_sampling_gap_stats() -> dict[str, dict[str, float | int | None]]:
    return {
        str(gpu): {
            "interval_count": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
        for gpu in (0, 1)
    }


def _sampling_gap_stats(
    samples_by_gpu: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, float | int | None]]:
    result = _empty_sampling_gap_stats()
    for gpu in (0, 1):
        timestamps = sorted(int(sample["monotonic_ns"]) for sample in samples_by_gpu.get(gpu, ()))
        gaps = [
            (right - left) / 1e9 for left, right in zip(timestamps, timestamps[1:], strict=False)
        ]
        if gaps:
            result[str(gpu)] = {
                "interval_count": len(gaps),
                "minimum": min(gaps),
                "median": statistics.median(gaps),
                "p95": _quantile(gaps, 0.95),
                "maximum": max(gaps),
            }
    return result


def _extract_planned_run(
    root: Path,
    corpus: CorpusManifest,
    plan: PlannedRun,
    windows: tuple[float, ...],
    stride_fraction: float,
    exclude_startup: bool,
    tolerance: float,
    selected_designations: tuple[str, ...],
) -> tuple[list[dict[str, Any]], CoverageRecord]:
    if plan.accepted_run_id is None:
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.INCOMPLETE_RUN,
            f"corpus={corpus.corpus_id} plan={plan.plan_id} has no accepted run ID",
        )
    if plan.designation not in selected_designations:
        reason = (
            CoverageReason.CALIBRATION_RUN_EXCLUDED
            if plan.designation == "calibration"
            else CoverageReason.NOT_IN_SELECTED_CORPUS
        )
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            reason,
            f"run={plan.accepted_run_id} designation={plan.designation} is not "
            f"selected designations={list(selected_designations)}",
        )
    if not corpus.allows(plan.accepted_run_id, plan.designation):
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.NOT_IN_SELECTED_CORPUS,
            f"run={plan.accepted_run_id} is not in the exact corpus/designation allow-list",
        )

    manifest_path = root / "runs" / plan.accepted_run_id / "manifest.json"
    if not manifest_path.is_file():
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.INCOMPLETE_RUN,
            f"run={plan.accepted_run_id} manifest is missing at {manifest_path}",
        )
    try:
        loaded_manifest = load_artifact(manifest_path, migrate_legacy=True)
        assert isinstance(loaded_manifest, dict)
        manifest = loaded_manifest
    except Exception as exc:
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.INVALID_MANIFEST,
            f"run={plan.accepted_run_id} manifest error: {type(exc).__name__}: {exc}",
        )
    if manifest.get("corpus_id") not in {None, corpus.corpus_id}:
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.NOT_IN_SELECTED_CORPUS,
            f"run={plan.accepted_run_id} manifest corpus={manifest.get('corpus_id')} does not "
            f"match selected corpus={corpus.corpus_id}",
            manifest=manifest,
        )
    if manifest.get("exit_status") != "completed":
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.INCOMPLETE_RUN,
            f"run={plan.accepted_run_id} exit_status={manifest.get('exit_status')}",
            manifest=manifest,
        )
    if not manifest.get("participation_valid"):
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.PARTICIPATION_INVALID,
            f"run={plan.accepted_run_id} failed two-rank participation validation",
            manifest=manifest,
        )
    if (
        str(manifest.get("workload_family")) != plan.workload_family
        or str(manifest.get("workload_label")) != plan.target_label
    ):
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.INVALID_MANIFEST,
            f"run={plan.accepted_run_id} family/label does not match plan={plan.plan_id}",
            manifest=manifest,
        )

    telemetry_path = manifest_path.parent / "telemetry.jsonl"
    if not telemetry_path.is_file():
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.MISSING_TELEMETRY,
            f"run={plan.accepted_run_id} telemetry is missing",
            manifest=manifest,
        )
    try:
        loaded_samples = load_artifact(telemetry_path, migrate_legacy=True)
        assert isinstance(loaded_samples, list)
        samples = loaded_samples
    except ValidationError as exc:
        reason = (
            CoverageReason.UNSUPPORTED_TELEMETRY_SCHEMA
            if "unsupported version" in str(exc)
            else CoverageReason.FEATURE_ERROR
        )
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            reason,
            f"run={plan.accepted_run_id} telemetry error: {exc}",
            manifest=manifest,
        )

    samples_by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_gpu[int(sample["gpu_index"])].append(sample)
    rows_per_gpu = {str(gpu): len(samples_by_gpu.get(gpu, [])) for gpu in (0, 1)}
    for values in samples_by_gpu.values():
        values.sort(key=lambda item: int(item["monotonic_ns"]))
    sampling_gaps = _sampling_gap_stats(samples_by_gpu)
    if set(samples_by_gpu) != {0, 1} or any(not samples_by_gpu[gpu] for gpu in (0, 1)):
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.MISSING_REQUIRED_GPU,
            f"run={plan.accepted_run_id} telemetry GPU indices={sorted(samples_by_gpu)}; "
            "required=[0, 1]",
            manifest=manifest,
            rows_per_gpu=rows_per_gpu,
            sampling_gaps=sampling_gaps,
        )
    common_start_ns = max(int(samples_by_gpu[gpu][0]["monotonic_ns"]) for gpu in (0, 1))
    common_end_ns = min(int(samples_by_gpu[gpu][-1]["monotonic_ns"]) for gpu in (0, 1))
    if common_end_ns <= common_start_ns:
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.NO_COMMON_INTERVAL,
            f"run={plan.accepted_run_id} has no common GPU telemetry interval",
            manifest=manifest,
            rows_per_gpu=rows_per_gpu,
            common_start_ns=common_start_ns,
            common_end_ns=common_end_ns,
            sampling_gaps=sampling_gaps,
        )
    overlap_seconds = (common_end_ns - common_start_ns) / 1e9
    warmup = float(manifest.get("warmup_seconds", 0.0)) if exclude_startup else 0.0
    usable_start_ns = common_start_ns
    usable_end_ns = common_end_ns
    if exclude_startup:
        measured_start = manifest.get("measurement_start_monotonic_ns")
        measured_end = manifest.get("measurement_end_monotonic_ns")
        if measured_start is not None:
            usable_start_ns = max(usable_start_ns, int(measured_start))
        else:
            usable_start_ns += int(warmup * 1e9)
        if measured_end is not None:
            usable_end_ns = min(usable_end_ns, int(measured_end))
    usable_seconds = max(0.0, (usable_end_ns - usable_start_ns) / 1e9)
    if usable_seconds < min(windows):
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.POST_WARMUP_INTERVAL_TOO_SHORT,
            f"run={plan.accepted_run_id} family={plan.workload_family} "
            f"overlap={overlap_seconds:.6f}s "
            f"warmup={warmup:.6f}s usable={usable_seconds:.6f}s minimum_window={min(windows):g}s",
            manifest=manifest,
            rows_per_gpu=rows_per_gpu,
            common_start_ns=common_start_ns,
            common_end_ns=common_end_ns,
            overlap_seconds=overlap_seconds,
            usable_seconds=usable_seconds,
            sampling_gaps=sampling_gaps,
        )
    usable_samples_by_gpu = {
        gpu: [
            sample
            for sample in samples_by_gpu[gpu]
            if usable_start_ns <= int(sample["monotonic_ns"]) <= usable_end_ns
        ]
        for gpu in (0, 1)
    }
    aligned_pairs = len(
        align_samples_by_timestamp(
            usable_samples_by_gpu[0],
            usable_samples_by_gpu[1],
            tolerance_seconds=tolerance,
        )
    )
    if aligned_pairs < 2:
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.INSUFFICIENT_SAMPLES,
            f"run={plan.accepted_run_id} aligned_sample_pairs={aligned_pairs}; required>=2",
            manifest=manifest,
            rows_per_gpu=rows_per_gpu,
            common_start_ns=common_start_ns,
            common_end_ns=common_end_ns,
            overlap_seconds=overlap_seconds,
            usable_seconds=usable_seconds,
            aligned_pairs=aligned_pairs,
            sampling_gaps=sampling_gaps,
        )
    try:
        features = extract_run_features(
            manifest,
            samples,
            window_lengths=windows,
            stride_fraction=stride_fraction,
            exclude_startup=exclude_startup,
            alignment_tolerance_seconds=tolerance,
            plan=plan,
            corpus=corpus,
        )
    except Exception as exc:
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.FEATURE_ERROR,
            f"run={plan.accepted_run_id} feature error: {type(exc).__name__}: {exc}",
            manifest=manifest,
            rows_per_gpu=rows_per_gpu,
            common_start_ns=common_start_ns,
            common_end_ns=common_end_ns,
            overlap_seconds=overlap_seconds,
            usable_seconds=usable_seconds,
            aligned_pairs=aligned_pairs,
            sampling_gaps=sampling_gaps,
        )
    if not features:
        return [], _excluded_record(
            corpus,
            plan,
            windows,
            tolerance,
            CoverageReason.INSUFFICIENT_SAMPLES,
            f"run={plan.accepted_run_id} emitted no complete, timestamp-aligned windows",
            manifest=manifest,
            rows_per_gpu=rows_per_gpu,
            common_start_ns=common_start_ns,
            common_end_ns=common_end_ns,
            overlap_seconds=overlap_seconds,
            usable_seconds=usable_seconds,
            aligned_pairs=aligned_pairs,
            sampling_gaps=sampling_gaps,
        )
    emitted = Counter(f"{float(row['window_seconds']):g}" for row in features)
    coverage = CoverageRecord(
        plan_id=plan.plan_id,
        run_id=plan.accepted_run_id,
        workload_family=plan.workload_family,
        target_label=plan.target_label,
        designation=plan.designation,
        experiment_session_id=corpus.experiment_session_id,
        collection_id=corpus.collection_id,
        corpus_id=corpus.corpus_id,
        node_id=corpus.node_id,
        status="included",
        reason_code=None,
        detail=f"run={plan.accepted_run_id} emitted {len(features)} feature windows",
        started_at_utc=str(manifest.get("started_at_utc")),
        ended_at_utc=str(manifest.get("ended_at_utc")),
        rows_per_gpu=rows_per_gpu,
        common_start_monotonic_ns=common_start_ns,
        common_end_monotonic_ns=common_end_ns,
        common_overlap_seconds=overlap_seconds,
        configured_warmup_seconds=warmup,
        usable_duration_seconds=usable_seconds,
        requested_window_seconds=windows,
        emitted_windows={f"{window:g}": emitted[f"{window:g}"] for window in windows},
        sampling_gap_seconds_by_gpu=sampling_gaps,
        alignment_tolerance_seconds=tolerance,
        aligned_sample_pairs=aligned_pairs,
        legacy_grouping_ambiguous=bool(manifest.get("legacy_grouping_ambiguous", False)),
    )
    return features, coverage


def extract_feature_result(
    input_root: str | Path,
    corpus_manifest: CorpusManifest | Mapping[str, Any] | str | Path,
    output: str | Path | None = None,
    window_lengths: Sequence[float] = DEFAULT_WINDOW_SECONDS,
    stride_fraction: float = 0.5,
    exclude_startup: bool = True,
    alignment_tolerance_seconds: float = 0.25,
    selected_designation: str | Sequence[str] = "benign",
) -> ExtractionResult:
    """Extract declared-corpus features and one coverage record per planned run."""
    corpus = _load_corpus_manifest(corpus_manifest)
    windows = tuple(float(value) for value in window_lengths)
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("window lengths must be positive")
    if len(windows) != len(set(windows)):
        raise ValueError("window lengths must be unique")
    if alignment_tolerance_seconds < 0:
        raise ValueError("alignment_tolerance_seconds cannot be negative")
    selected_designations = (
        (selected_designation,)
        if isinstance(selected_designation, str)
        else tuple(str(value) for value in selected_designation)
    )
    if not selected_designations or len(selected_designations) != len(set(selected_designations)):
        raise ValueError("selected designations must be non-empty and unique")
    unknown_designations = set(selected_designations) - {
        "benign",
        "adversarial",
        "calibration",
    }
    if unknown_designations:
        raise ValueError(f"unknown selected designations: {sorted(unknown_designations)}")
    root = Path(input_root)
    features: list[dict[str, Any]] = []
    coverage: list[CoverageRecord] = []
    for plan in corpus.planned_runs:
        run_features, record = _extract_planned_run(
            root,
            corpus,
            plan,
            windows,
            stride_fraction,
            exclude_startup,
            alignment_tolerance_seconds,
            selected_designations,
        )
        features.extend(run_features)
        coverage.append(record)
    result = ExtractionResult(
        tuple(features),
        tuple(coverage),
        corpus.corpus_id,
        windows,
        selected_designations=selected_designations,
        calibration_reference=(
            dict(corpus.calibration_reference) if corpus.calibration_reference is not None else None
        ),
    )
    if output is not None:
        store = ArtifactStore(output)
        store.initialize()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        feature_path = store.write_jsonl(f"features/features-{stamp}.jsonl", result.features)
        coverage_path = store.write_jsonl(
            f"features/coverage-{stamp}.jsonl",
            [record.to_dict() for record in result.coverage],
        )
        summary = {
            **result.summary(),
            "feature_artifact": str(feature_path.relative_to(store.root)),
            "coverage_artifact": str(coverage_path.relative_to(store.root)),
        }
        store.write_json(f"features/extraction-{stamp}.json", summary)
    return result


def extract_features(
    input_root: str | Path = "artifacts",
    output: str | Path | None = None,
    window_lengths: Sequence[float] = DEFAULT_WINDOW_SECONDS,
    stride_fraction: float = 0.5,
    exclude_startup: bool = True,
    *,
    corpus_manifest: CorpusManifest | Mapping[str, Any] | str | Path | None = None,
    alignment_tolerance_seconds: float = 0.25,
    selected_designation: str | Sequence[str] = "benign",
) -> list[dict[str, Any]]:
    """Backward-shaped row return with mandatory explicit corpus selection."""
    if corpus_manifest is None:
        raise ValueError(
            "corpus_manifest is required; implicit directory-wide feature selection is prohibited"
        )
    result = extract_feature_result(
        input_root,
        corpus_manifest,
        output=output,
        window_lengths=window_lengths,
        stride_fraction=stride_fraction,
        exclude_startup=exclude_startup,
        alignment_tolerance_seconds=alignment_tolerance_seconds,
        selected_designation=selected_designation,
    )
    return list(result.features)


def numeric_feature_columns(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Return model-eligible columns; identity and labels are always excluded."""
    records = list(rows)
    if not records:
        return []
    columns = set().union(*(record.keys() for record in records)) - METADATA_COLUMNS
    return sorted(
        column
        for column in columns
        if any(
            isinstance(record.get(column), (int, float))
            and not isinstance(record.get(column), bool)
            for record in records
        )
    )
