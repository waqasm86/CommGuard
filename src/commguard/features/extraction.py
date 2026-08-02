"""Deterministic temporal and cross-GPU feature extraction."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from commguard.artifacts import ArtifactStore
from commguard.schemas import SCHEMA_VERSION, TELEMETRY_FIELDS, load_artifact

METADATA_COLUMNS = {
    "artifact_kind",
    "schema_version",
    "run_id",
    "session_fingerprint",
    "target_label",
    "workload_family",
    "designation",
    "window_seconds",
    "window_index",
    "window_start_monotonic_ns",
    "window_end_monotonic_ns",
    "startup_excluded",
    "sample_count_gpu0",
    "sample_count_gpu1",
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
    count = min(len(left), len(right))
    if count < 2:
        return None
    left = left[:count]
    right = right[:count]
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
                "p95",
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
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "p25": p25,
        "p50": _quantile(values, 0.5),
        "p75": p75,
        "p95": _quantile(values, 0.95),
        "iqr": p75 - p25,
        "range": max(values) - min(values),
        "cv": std / abs(mean) if mean else None,
        "slope": _slope(values, times),
        "autocorr_lag1": _correlation(values[:-1], values[1:]) if len(values) > 2 else None,
    }


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


def _feature_window(
    manifest: dict[str, Any],
    samples_by_gpu: dict[int, list[dict[str, Any]]],
    window_seconds: float,
    window_index: int,
    window_start_ns: int,
    window_end_ns: int,
    startup_excluded: bool,
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
    row: dict[str, Any] = {
        "artifact_kind": "feature_row",
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "session_fingerprint": manifest["environment_fingerprint"],
        "target_label": manifest["workload_label"],
        "workload_family": manifest["workload_family"],
        "designation": manifest["designation"],
        "window_seconds": window_seconds,
        "window_index": window_index,
        "window_start_monotonic_ns": window_start_ns,
        "window_end_monotonic_ns": window_end_ns,
        "startup_excluded": startup_excluded,
        "sample_count_gpu0": len(selected[0]),
        "sample_count_gpu1": len(selected[1]),
    }
    values_by_gpu: dict[tuple[int, str], list[float]] = {}
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
        tx = values_by_gpu[(gpu, "pcie_tx_bytes_per_s")]
        rx = values_by_gpu[(gpu, "pcie_rx_bytes_per_s")]
        if tx and rx:
            count = min(len(tx), len(rx))
            row[f"gpu{gpu}__pcie_total_mean_bytes_per_s"] = (
                statistics.fmean(tx[:count] + rx[:count]) * 2
            )
            rx_mean = statistics.fmean(rx[:count])
            row[f"gpu{gpu}__pcie_tx_rx_ratio"] = (
                statistics.fmean(tx[:count]) / rx_mean if rx_mean else None
            )
        else:
            row[f"gpu{gpu}__pcie_total_mean_bytes_per_s"] = None
            row[f"gpu{gpu}__pcie_tx_rx_ratio"] = None
    for field in TELEMETRY_FIELDS:
        left = values_by_gpu[(0, field)]
        right = values_by_gpu[(1, field)]
        count = min(len(left), len(right))
        if count:
            left = left[:count]
            right = right[:count]
            row[f"cross_gpu__{field}__mean_abs_difference"] = statistics.fmean(
                abs(a - b) for a, b in zip(left, right, strict=True)
            )
            scale = statistics.fmean([abs(value) for value in left + right])
            row[f"cross_gpu__{field}__normalized_divergence"] = (
                row[f"cross_gpu__{field}__mean_abs_difference"] / scale if scale else None
            )
            row[f"cross_gpu__{field}__correlation"] = _correlation(left, right)
            row[f"cross_gpu__{field}__lag1_correlation"] = (
                _correlation(left[:-1], right[1:]) if count > 2 else None
            )
        else:
            row[f"cross_gpu__{field}__mean_abs_difference"] = None
            row[f"cross_gpu__{field}__normalized_divergence"] = None
            row[f"cross_gpu__{field}__correlation"] = None
            row[f"cross_gpu__{field}__lag1_correlation"] = None
    return row


def extract_run_features(
    manifest: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    window_lengths: Sequence[float] = (5.0, 15.0, 30.0),
    stride_fraction: float = 0.5,
    exclude_startup: bool = True,
) -> list[dict[str, Any]]:
    if not 0 < stride_fraction <= 1:
        raise ValueError("stride_fraction must be in (0, 1]")
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_gpu[int(sample["gpu_index"])].append(sample)
    if set(by_gpu) != {0, 1}:
        return []
    for records in by_gpu.values():
        records.sort(key=lambda item: int(item["monotonic_ns"]))
    start_ns = max(int(by_gpu[gpu][0]["monotonic_ns"]) for gpu in (0, 1))
    end_ns = min(int(by_gpu[gpu][-1]["monotonic_ns"]) for gpu in (0, 1)) + 1
    if exclude_startup:
        start_ns += int(float(manifest.get("warmup_seconds", 0.0)) * 1e9)
    rows: list[dict[str, Any]] = []
    for window_seconds in window_lengths:
        window_ns = int(window_seconds * 1e9)
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
            )
            if row is not None:
                rows.append(row)
            window_start += stride_ns
            index += 1
    return rows


def extract_features(
    input_root: str | Path = "artifacts",
    output: str | Path | None = None,
    window_lengths: Sequence[float] = (5.0, 15.0, 30.0),
    stride_fraction: float = 0.5,
    exclude_startup: bool = True,
    include_calibration: bool = False,
) -> list[dict[str, Any]]:
    """Load completed runs, derive features, and optionally save a create-only table."""
    root = Path(input_root)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "runs").glob("*/manifest.json")):
        manifest = load_artifact(manifest_path)
        assert isinstance(manifest, dict)
        if manifest["exit_status"] != "completed" or not manifest["participation_valid"]:
            continue
        if manifest["designation"] == "calibration" and not include_calibration:
            continue
        telemetry_path = manifest_path.parent / "telemetry.jsonl"
        if not telemetry_path.exists():
            continue
        samples = load_artifact(telemetry_path)
        assert isinstance(samples, list)
        rows.extend(
            extract_run_features(
                manifest,
                samples,
                window_lengths=window_lengths,
                stride_fraction=stride_fraction,
                exclude_startup=exclude_startup,
            )
        )
    labels = {str(row["target_label"]) for row in rows}
    lengths_by_label = {
        label: {float(row["window_seconds"]) for row in rows if str(row["target_label"]) == label}
        for label in labels
    }
    common_lengths = set.intersection(*lengths_by_label.values()) if lengths_by_label else set()
    rows = [row for row in rows if float(row["window_seconds"]) in common_lengths]
    if output is not None:
        store = ArtifactStore(output)
        store.initialize()
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        store.write_jsonl(f"features/features-{stamp}.jsonl", rows)
    return rows


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
