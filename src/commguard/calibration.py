"""Communication-response calibration with repetition-aware falsification gates."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from commguard.schemas import SCHEMA_VERSION


def _ranks(values: list[float]) -> list[float]:
    """Return zero-based average ranks, including deterministic tie handling."""
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    output = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in range(start, end):
            output[ordered[index][0]] = rank
        start = end
    return output


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=False))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _median_absolute_deviation(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def _nonnegative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _observation_sort_key(row: dict[str, Any]) -> tuple[int, float, str, float, str]:
    payload = _nonnegative_number(row.get("payload_mib"))
    signal = _nonnegative_number(row.get("pcie_total_mean_bytes_per_s"))
    return (
        0 if row.get("is_idle") else 1,
        payload if payload is not None else math.inf,
        str(row.get("run_id", "")),
        signal if signal is not None else math.inf,
        str(row.get("repetition", "")),
    )


def _validate_thresholds(
    minimum_sizes: int,
    minimum_rank_correlation: float,
    minimum_dynamic_range: float,
    minimum_repetitions: int,
    minimum_capture_rate: float,
    baseline_multiplier: float,
    baseline_floor_bytes_per_s: float,
) -> None:
    if minimum_sizes < 1:
        raise ValueError("minimum_sizes must be at least 1")
    if not -1 <= minimum_rank_correlation <= 1:
        raise ValueError("minimum_rank_correlation must be between -1 and 1")
    if minimum_dynamic_range <= 0:
        raise ValueError("minimum_dynamic_range must be positive")
    if minimum_repetitions < 1:
        raise ValueError("minimum_repetitions must be at least 1")
    if not 0 <= minimum_capture_rate <= 1:
        raise ValueError("minimum_capture_rate must be between 0 and 1")
    if baseline_multiplier < 0:
        raise ValueError("baseline_multiplier cannot be negative")
    if baseline_floor_bytes_per_s < 0:
        raise ValueError("baseline_floor_bytes_per_s cannot be negative")


def analyze_calibration(
    observations: Iterable[dict[str, Any]],
    minimum_sizes: int = 3,
    minimum_rank_correlation: float = 0.7,
    minimum_dynamic_range: float = 1.2,
    minimum_repetitions: int = 1,
    minimum_capture_rate: float = 0.8,
    baseline_multiplier: float = 3.0,
    baseline_floor_bytes_per_s: float = 1_000_000.0,
) -> dict[str, Any]:
    """Assess response monotonicity and per-payload capture repeatability.

    Rows with ``is_idle`` establish a capture threshold. For backward
    compatibility, legacy inputs without any idle rows do not apply the capture
    gate: every otherwise usable repetition counts as captured. The returned
    artifact records that compatibility mode explicitly.
    """
    _validate_thresholds(
        minimum_sizes,
        minimum_rank_correlation,
        minimum_dynamic_range,
        minimum_repetitions,
        minimum_capture_rate,
        baseline_multiplier,
        baseline_floor_bytes_per_s,
    )

    rows = list(observations)
    idle_rows = [row for row in rows if row.get("is_idle")]
    idle_signals = [
        signal
        for row in idle_rows
        if row.get("pcie_supported")
        and (signal := _nonnegative_number(row.get("pcie_total_mean_bytes_per_s"))) is not None
    ]
    idle_median = statistics.median(idle_signals) if idle_signals else None
    capture_gate_applied = idle_median is not None
    capture_threshold = (
        max(
            idle_median * baseline_multiplier,
            idle_median + baseline_floor_bytes_per_s,
        )
        if idle_median is not None
        else None
    )

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    invalid_payload_count = 0
    for row in rows:
        if row.get("is_idle"):
            continue
        payload = _nonnegative_number(row.get("payload_mib"))
        if payload is None:
            invalid_payload_count += 1
            continue
        grouped[payload].append(row)

    invalid_participation_count = 0
    invalid_signal_count = 0
    payload_summaries: list[dict[str, Any]] = []
    for payload, repetitions in sorted(grouped.items()):
        values: list[float] = []
        for row in repetitions:
            if not row.get("pcie_supported"):
                continue
            if not row.get("participation_valid"):
                invalid_participation_count += 1
                continue
            signal = _nonnegative_number(row.get("pcie_total_mean_bytes_per_s"))
            if signal is None:
                invalid_signal_count += 1
                continue
            values.append(signal)

        median = statistics.median(values) if values else None
        mean = statistics.fmean(values) if values else None
        if capture_threshold is None:
            capture_success_count = len(values)
        else:
            capture_success_count = sum(value > capture_threshold for value in values)
        capture_rate = capture_success_count / len(values) if values else 0.0
        enough_repetitions = len(values) >= minimum_repetitions
        passes_capture_gate = not capture_gate_applied or capture_rate >= minimum_capture_rate

        if not values:
            coefficient_of_variation = None
        elif len(values) == 1 or mean == 0:
            coefficient_of_variation = 0.0
        else:
            assert mean is not None
            coefficient_of_variation = statistics.pstdev(values) / abs(mean)

        payload_summaries.append(
            {
                "payload_mib": payload,
                "repetitions": len(repetitions),
                "usable_repetitions": len(values),
                "median_bytes_per_s": median,
                "mean_bytes_per_s": mean,
                "mad_bytes_per_s": (_median_absolute_deviation(values) if values else None),
                "coefficient_of_variation": coefficient_of_variation,
                "capture_success_count": capture_success_count,
                "capture_rate": capture_rate,
                "reliably_observed": enough_repetitions and passes_capture_gate,
            }
        )

    usable_summaries = [
        summary for summary in payload_summaries if summary["median_bytes_per_s"] is not None
    ]
    payloads = [float(summary["payload_mib"]) for summary in usable_summaries]
    median_signals = [float(summary["median_bytes_per_s"]) for summary in usable_summaries]
    rank_correlation = (
        _correlation(_ranks(payloads), _ranks(median_signals))
        if len(usable_summaries) >= 2
        else None
    )
    positive_signals = [value for value in median_signals if value > 0]
    dynamic_range = (
        max(positive_signals) / min(positive_signals) if len(positive_signals) >= 2 else None
    )

    reliable_payloads = [
        float(summary["payload_mib"])
        for summary in payload_summaries
        if summary["reliably_observed"]
    ]
    unreliable_payloads = [
        float(summary["payload_mib"])
        for summary in payload_summaries
        if not summary["reliably_observed"]
    ]
    unsupported_count = sum(not bool(row.get("pcie_supported")) for row in rows)
    invalid_idle_count = len(idle_rows) - len(idle_signals)

    reasons: list[str] = []
    if not rows:
        reasons.append("no calibration observations were provided")
    if invalid_payload_count:
        reasons.append(f"{invalid_payload_count} non-idle observations had an invalid payload size")
    if unsupported_count:
        reasons.append(
            f"PCIe TX/RX was unsupported in {unsupported_count} calibration observations"
        )
    if idle_rows and not idle_signals:
        reasons.append("idle baseline rows were present but none had a usable PCIe reading")
    elif invalid_idle_count:
        reasons.append(
            f"{invalid_idle_count} idle baseline observations had no usable PCIe reading"
        )
    if invalid_participation_count:
        reasons.append(f"{invalid_participation_count} observations had invalid rank participation")
    if invalid_signal_count:
        reasons.append(f"{invalid_signal_count} supported observations had no usable PCIe reading")
    if len(usable_summaries) < minimum_sizes:
        reasons.append(
            f"only {len(usable_summaries)} usable payload sizes; "
            f"at least {minimum_sizes} are required"
        )
    if rank_correlation is None or rank_correlation < minimum_rank_correlation:
        reasons.append(f"rank correlation {rank_correlation!r} is below {minimum_rank_correlation}")
    if dynamic_range is None or dynamic_range < minimum_dynamic_range:
        reasons.append(f"dynamic range {dynamic_range!r} is below {minimum_dynamic_range}")
    if not reliable_payloads:
        reasons.append("no payload group passed the repetition-aware capture gate")
    if unreliable_payloads:
        reasons.append(f"repetition-aware capture gate failed for payloads: {unreliable_payloads}")

    aggregate_gate_failed = (
        len(usable_summaries) < minimum_sizes
        or rank_correlation is None
        or rank_correlation < minimum_rank_correlation
        or dynamic_range is None
        or dynamic_range < minimum_dynamic_range
        or not reliable_payloads
    )
    hard_falsification = (
        not rows
        or bool(unsupported_count)
        or bool(idle_rows and not idle_signals)
        or aggregate_gate_failed
    )
    if hard_falsification:
        status = "not_supported"
    elif reasons:
        status = "partially_supported"
    else:
        status = "supported"

    limitations = [
        "Nominal payload bytes are not observed PCIe bytes.",
        "PyTorch timing and NVML PCIe readings are distinct evidence channels.",
        "This decision does not transfer beyond the recorded dual-T4 session.",
    ]
    if not idle_rows:
        limitations.append(
            "No idle baseline was supplied; capture counts use legacy compatibility mode."
        )

    return {
        "artifact_kind": "calibration_result",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "signal_name": "NVML PCIe traffic readings",
        "observations": sorted(rows, key=_observation_sort_key),
        "payload_summaries": payload_summaries,
        "idle_baseline_median_bytes_per_s": idle_median,
        "capture_threshold_bytes_per_s": capture_threshold,
        "capture_gate_applied": capture_gate_applied,
        "capture_gate_note": (
            "Capture requires exceeding the idle-derived threshold."
            if capture_gate_applied
            else "No idle rows were supplied; usable repetitions count as captured for "
            "backward compatibility."
        ),
        "supported_payload_range_mib": reliable_payloads,
        "unreliable_payload_range_mib": unreliable_payloads,
        "usable_observation_count": sum(
            int(summary["usable_repetitions"]) for summary in payload_summaries
        ),
        "usable_payload_size_count": len(usable_summaries),
        "spearman_rank_correlation": rank_correlation,
        "dynamic_range_ratio": dynamic_range,
        "thresholds": {
            "minimum_sizes": minimum_sizes,
            "minimum_rank_correlation": minimum_rank_correlation,
            "minimum_dynamic_range": minimum_dynamic_range,
            "minimum_repetitions": minimum_repetitions,
            "minimum_capture_rate": minimum_capture_rate,
            "baseline_multiplier": baseline_multiplier,
            "baseline_floor_bytes_per_s": baseline_floor_bytes_per_s,
        },
        "falsification_reasons": reasons,
        "claim": (
            "Calibration supports proceeding within this exact session."
            if status == "supported"
            else "Calibration supports only the reported reliable payload groups."
            if status == "partially_supported"
            else "Calibration did not support communication-size detector escalation."
        ),
        "limitations": limitations,
    }
