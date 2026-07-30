"""Communication-response calibration and explicit falsification decisions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

from commguard.schemas import SCHEMA_VERSION


def _ranks(values: list[float]) -> list[float]:
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
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=False)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def analyze_calibration(
    observations: Iterable[dict[str, Any]],
    minimum_sizes: int = 3,
    minimum_rank_correlation: float = 0.7,
    minimum_dynamic_range: float = 1.2,
) -> dict[str, Any]:
    """Decide whether PCIe readings respond to controlled nominal payload changes."""
    rows = sorted(list(observations), key=lambda row: float(row["payload_mib"]))
    usable = [
        row
        for row in rows
        if row.get("pcie_supported")
        and row.get("pcie_total_mean_bytes_per_s") is not None
        and float(row["pcie_total_mean_bytes_per_s"]) >= 0
        and row.get("participation_valid")
    ]
    reasons: list[str] = []
    if len(usable) < minimum_sizes:
        reasons.append(
            f"only {len(usable)} usable payload sizes; at least {minimum_sizes} are required"
        )
    payloads = [float(row["payload_mib"]) for row in usable]
    signals = [float(row["pcie_total_mean_bytes_per_s"]) for row in usable]
    rank_correlation = (
        _correlation(_ranks(payloads), _ranks(signals)) if len(usable) >= 2 else None
    )
    positive = [value for value in signals if value > 0]
    dynamic_range = max(positive) / min(positive) if len(positive) >= 2 else None
    if rank_correlation is None or rank_correlation < minimum_rank_correlation:
        reasons.append(
            f"rank correlation {rank_correlation!r} is below {minimum_rank_correlation}"
        )
    if dynamic_range is None or dynamic_range < minimum_dynamic_range:
        reasons.append(f"dynamic range {dynamic_range!r} is below {minimum_dynamic_range}")
    unsupported = any(not row.get("pcie_supported") for row in rows)
    if unsupported:
        reasons.append("PCIe TX/RX was unsupported in at least one calibration run")
    status = "supported" if not reasons else "not_supported"
    return {
        "artifact_kind": "calibration_result",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "signal_name": "NVML PCIe traffic readings",
        "observations": rows,
        "usable_observation_count": len(usable),
        "spearman_rank_correlation": rank_correlation,
        "dynamic_range_ratio": dynamic_range,
        "thresholds": {
            "minimum_sizes": minimum_sizes,
            "minimum_rank_correlation": minimum_rank_correlation,
            "minimum_dynamic_range": minimum_dynamic_range,
        },
        "falsification_reasons": reasons,
        "claim": (
            "Calibration supports proceeding within this exact session."
            if status == "supported"
            else "Calibration did not support communication-size detector escalation."
        ),
        "limitations": [
            "Nominal payload bytes are not observed PCIe bytes.",
            "PyTorch timing and NVML PCIe readings are distinct evidence channels.",
            "This decision does not transfer beyond the recorded dual-T4 session.",
        ],
    }
