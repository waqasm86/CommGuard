"""Communication-response calibration with repetition-aware falsification gates."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from commguard.artifacts import sha256_file
from commguard.exceptions import CalibrationError
from commguard.schemas import (
    CURRENT_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    validate_artifact,
)

STANDARD_CALIBRATION_PAYLOAD_MIB = (1, 4, 16, 64, 128)
STANDARD_CALIBRATION_REPETITIONS = 5
CALIBRATION_REFERENCE_FIELDS = (
    "calibration_artifact_path",
    "calibration_sha256",
    "calibration_experiment_session_id",
    "calibration_collection_id",
    "calibration_environment_fingerprint",
    "calibration_source_commit",
    "calibration_schema_version",
    "calibration_status",
    "calibration_relationship",
    "calibration_created_in_current_session",
)


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
    minimum_repetitions: int | None = None,
    minimum_capture_rate: float = 0.8,
    baseline_multiplier: float = 3.0,
    baseline_floor_bytes_per_s: float = 1_000_000.0,
    threshold_method: str = "median_plus_mad",
    minimum_measured_duration_s: float | None = None,
    maximum_interval_relative_error: float | None = None,
    *,
    legacy_compatibility: bool = False,
) -> dict[str, Any]:
    """Assess response monotonicity and per-payload capture repeatability.

    New schema-2 evidence must include repeated ``idle_baseline`` and
    ``collective`` observations. ``legacy_compatibility`` is narrowly scoped to
    re-analysis/reporting of old schema-1 evidence and can never establish that
    the modern capture gate passed.
    """
    if minimum_repetitions is None:
        minimum_repetitions = 1 if legacy_compatibility else 3
    if threshold_method not in {"median_plus_mad", "median_multiplier"}:
        raise ValueError("threshold_method must be median_plus_mad or median_multiplier")
    if minimum_measured_duration_s is not None and minimum_measured_duration_s <= 0:
        raise ValueError("minimum_measured_duration_s must be positive when supplied")
    if maximum_interval_relative_error is not None and not 0 <= maximum_interval_relative_error < 1:
        raise ValueError("maximum_interval_relative_error must be in [0, 1)")
    _validate_thresholds(
        minimum_sizes,
        minimum_rank_correlation,
        minimum_dynamic_range,
        minimum_repetitions,
        minimum_capture_rate,
        baseline_multiplier,
        baseline_floor_bytes_per_s,
    )

    rows: list[dict[str, Any]] = []
    invalid_observation_type_count = 0
    for supplied in observations:
        row = dict(supplied)
        observation_type = row.get("observation_type")
        if observation_type is None:
            observation_type = "idle_baseline" if row.get("is_idle") else "collective"
        if observation_type not in {"idle_baseline", "collective"}:
            invalid_observation_type_count += 1
        row["observation_type"] = observation_type
        row["is_idle"] = observation_type == "idle_baseline"
        rows.append(row)

    def scientifically_usable(row: dict[str, Any]) -> bool:
        if not row.get("participation_valid"):
            return False
        if row.get("exit_status") not in {None, "completed"}:
            return False
        if minimum_measured_duration_s is not None:
            duration = _nonnegative_number(row.get("measured_duration_seconds"))
            if duration is None or duration < minimum_measured_duration_s:
                return False
        if maximum_interval_relative_error is not None:
            diagnostics = row.get("collector_diagnostics")
            if not isinstance(diagnostics, dict):
                return False
            target = _nonnegative_number(diagnostics.get("requested_interval_s"))
            actual = _nonnegative_number(diagnostics.get("mean_interval_s"))
            if target is None or target == 0 or actual is None:
                return False
            if abs(actual - target) / target > maximum_interval_relative_error:
                return False
        return True

    idle_rows = [row for row in rows if row.get("observation_type") == "idle_baseline"]
    idle_signals = [
        signal
        for row in idle_rows
        if row.get("pcie_supported")
        and scientifically_usable(row)
        and (signal := _nonnegative_number(row.get("pcie_total_mean_bytes_per_s"))) is not None
    ]
    idle_median = statistics.median(idle_signals) if idle_signals else None
    idle_mad = _median_absolute_deviation(idle_signals) if idle_signals else None
    idle_baseline_complete = len(idle_signals) >= minimum_repetitions
    capture_gate_applied = idle_median is not None and not legacy_compatibility
    capture_threshold = None
    if idle_median is not None:
        threshold_delta = (
            baseline_multiplier * float(idle_mad or 0.0)
            if threshold_method == "median_plus_mad"
            else idle_median * max(0.0, baseline_multiplier - 1.0)
        )
        capture_threshold = idle_median + max(threshold_delta, baseline_floor_bytes_per_s)

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    invalid_payload_count = 0
    for row in rows:
        if row.get("observation_type") == "idle_baseline":
            continue
        if row.get("observation_type") != "collective":
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
            if not scientifically_usable(row):
                invalid_participation_count += 1
                continue
            signal = _nonnegative_number(row.get("pcie_total_mean_bytes_per_s"))
            if signal is None:
                invalid_signal_count += 1
                continue
            values.append(signal)

        median = statistics.median(values) if values else None
        mean = statistics.fmean(values) if values else None
        if legacy_compatibility and capture_threshold is None:
            capture_success_count = len(values)
        elif capture_threshold is None:
            capture_success_count = 0
        else:
            capture_success_count = sum(value > capture_threshold for value in values)
        capture_rate = capture_success_count / len(values) if values else 0.0
        enough_repetitions = len(values) >= minimum_repetitions
        passes_capture_gate = legacy_compatibility or (
            capture_gate_applied and capture_rate >= minimum_capture_rate
        )

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
    short_duration_count = 0
    invalid_interval_count = 0
    if minimum_measured_duration_s is not None:
        short_duration_count = sum(
            (_nonnegative_number(row.get("measured_duration_seconds")) or 0)
            < minimum_measured_duration_s
            for row in rows
        )
    if maximum_interval_relative_error is not None:
        for row in rows:
            diagnostics = row.get("collector_diagnostics")
            if not isinstance(diagnostics, dict):
                invalid_interval_count += 1
                continue
            target = _nonnegative_number(diagnostics.get("requested_interval_s"))
            actual = _nonnegative_number(diagnostics.get("mean_interval_s"))
            if (
                target is None
                or target == 0
                or actual is None
                or abs(actual - target) / target > maximum_interval_relative_error
            ):
                invalid_interval_count += 1

    reasons: list[str] = []
    if not rows:
        reasons.append("no calibration observations were provided")
    if invalid_observation_type_count:
        reasons.append(
            f"{invalid_observation_type_count} observations had an invalid observation type"
        )
    if not legacy_compatibility and not idle_rows:
        reasons.append("modern calibration requires explicit idle baseline observations")
    if not legacy_compatibility and len(idle_signals) < minimum_repetitions:
        reasons.append(
            f"only {len(idle_signals)} usable idle baseline repetitions; "
            f"at least {minimum_repetitions} are required"
        )
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
    if short_duration_count:
        reasons.append(
            f"{short_duration_count} observations were shorter than "
            f"{minimum_measured_duration_s:g} measured seconds"
        )
    if invalid_interval_count:
        reasons.append(
            f"{invalid_interval_count} observations exceeded the configured collector interval "
            "tolerance or lacked interval evidence"
        )
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
        or bool(invalid_observation_type_count)
        or bool(unsupported_count)
        or bool(idle_rows and not idle_signals)
        or (not legacy_compatibility and not idle_baseline_complete)
        or aggregate_gate_failed
    )
    if hard_falsification:
        status = "not_supported"
    elif reasons:
        status = "partially_supported"
    else:
        status = "supported"
    unexpected_worker_failures = sum(
        row.get("exit_status") in {"failed", "timeout"} or row.get("participation_valid") is False
        for row in rows
    )
    if status in {"supported", "partially_supported"}:
        result_state = status
    elif not rows or invalid_observation_type_count or unexpected_worker_failures:
        result_state = "failed"
    elif unsupported_count and not usable_summaries:
        result_state = "not_supported"
    elif short_duration_count or invalid_interval_count:
        result_state = "inconclusive"
    else:
        result_state = "inconclusive"

    limitations = [
        "Nominal payload bytes are not observed PCIe bytes.",
        "PyTorch timing and NVML PCIe readings are distinct evidence channels.",
        "This decision does not transfer beyond the recorded dual-T4 session.",
    ]
    if legacy_compatibility:
        limitations.append(
            "Legacy compatibility does not establish that the modern idle-baseline capture "
            "gate passed."
        )

    return {
        "artifact_kind": "calibration_result",
        "schema_version": (
            LEGACY_SCHEMA_VERSION if legacy_compatibility else CURRENT_SCHEMA_VERSION
        ),
        "calibration_contract_version": (
            "legacy-1.0-compatibility" if legacy_compatibility else "idle-aware-repeated-v2"
        ),
        "legacy_compatibility_applied": legacy_compatibility,
        "modern_capture_gate_passed": (
            not legacy_compatibility
            and status == "supported"
            and capture_gate_applied
            and idle_baseline_complete
        ),
        "decision_state": (
            "passed"
            if status == "supported" and not legacy_compatibility
            else "legacy_compatible"
            if status == "supported"
            else "inconclusive"
            if status == "partially_supported"
            else "failed"
        ),
        "status": status,
        "result_state": result_state,
        "signal_name": "NVML PCIe traffic readings",
        "observations": sorted(rows, key=_observation_sort_key),
        "payload_summaries": payload_summaries,
        "idle_baseline_median_bytes_per_s": idle_median,
        "idle_baseline_mad_bytes_per_s": idle_mad,
        "idle_baseline_repetitions": len(idle_rows),
        "idle_baseline_usable_repetitions": len(idle_signals),
        "idle_baseline_complete": idle_baseline_complete,
        "capture_threshold_bytes_per_s": capture_threshold,
        "capture_gate_applied": capture_gate_applied,
        "capture_gate_note": (
            "Capture requires exceeding the idle-derived threshold for each payload."
            if capture_gate_applied
            else "Legacy compatibility mode does not apply the modern idle-derived capture gate."
            if legacy_compatibility
            else "The modern idle-derived capture gate could not be applied."
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
            "threshold_method": threshold_method,
            "minimum_measured_duration_s": minimum_measured_duration_s,
            "maximum_interval_relative_error": maximum_interval_relative_error,
        },
        "falsification_reasons": reasons,
        "claim": (
            "Calibration supports proceeding within this exact session."
            if status == "supported" and not legacy_compatibility
            else "Legacy evidence is readable under its historical contract; the modern "
            "capture gate is not satisfied."
            if status == "supported"
            else "Calibration supports only the reported reliable payload groups."
            if status == "partially_supported"
            else "Calibration did not support communication-size detector escalation."
        ),
        "limitations": limitations,
    }


def build_calibration_reference(
    artifact_root: str | Path,
    calibration_path: str | Path,
    *,
    current_experiment_session_id: str,
) -> dict[str, Any]:
    """Build a stable exact-file reference without mutating calibration evidence."""
    root = Path(artifact_root).resolve()
    candidate = Path(calibration_path)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CalibrationError(f"calibration artifact is outside artifact root: {path}") from exc
    if not path.is_file():
        raise CalibrationError(f"referenced calibration artifact is missing: {relative}")
    calibration = json.loads(path.read_text(encoding="utf-8"))
    validate_artifact(calibration)
    session_id = str(calibration.get("experiment_session_id", ""))
    relationship = (
        "current_session" if session_id == current_experiment_session_id else "prior_session"
    )
    return {
        "calibration_artifact_path": relative.as_posix(),
        "calibration_sha256": sha256_file(path),
        "calibration_experiment_session_id": session_id or None,
        "calibration_collection_id": calibration.get("collection_id"),
        "calibration_environment_fingerprint": calibration.get("environment_fingerprint"),
        "calibration_source_commit": calibration.get("source_commit"),
        "calibration_schema_version": calibration.get("schema_version"),
        "calibration_status": calibration.get("status"),
        "calibration_relationship": relationship,
        "calibration_created_in_current_session": relationship == "current_session",
    }


def verify_calibration_reference(
    artifact_root: str | Path,
    reference: dict[str, Any],
    *,
    expected_experiment_session_ids: set[str] | None = None,
    expected_environment_fingerprints: set[str] | None = None,
    expected_source_commits: set[str] | None = None,
    require_current_session: bool = True,
    allow_legacy: bool = False,
    require_supported: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Load one exact calibration and verify its authenticated provenance fields."""
    missing = [name for name in CALIBRATION_REFERENCE_FIELDS if name not in reference]
    if missing:
        raise CalibrationError(f"calibration reference is missing fields: {missing}")
    root = Path(artifact_root).resolve()
    relative = Path(str(reference["calibration_artifact_path"]))
    if relative.is_absolute():
        raise CalibrationError("calibration_artifact_path must be artifact-root-relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CalibrationError("calibration artifact path escapes the artifact root") from exc
    if not path.is_file():
        raise CalibrationError(f"referenced calibration artifact is missing: {relative}")
    actual_hash = sha256_file(path)
    if actual_hash != reference["calibration_sha256"]:
        raise CalibrationError(
            "calibration SHA-256 mismatch: "
            f"expected={reference['calibration_sha256']} actual={actual_hash}"
        )
    calibration = json.loads(path.read_text(encoding="utf-8"))
    validate_artifact(calibration)
    comparisons = {
        "experiment_session_id": "calibration_experiment_session_id",
        "collection_id": "calibration_collection_id",
        "environment_fingerprint": "calibration_environment_fingerprint",
        "source_commit": "calibration_source_commit",
        "schema_version": "calibration_schema_version",
        "status": "calibration_status",
    }
    for artifact_field, reference_field in comparisons.items():
        if calibration.get(artifact_field) != reference.get(reference_field):
            raise CalibrationError(
                f"calibration reference mismatch for {artifact_field}: "
                f"reference={reference.get(reference_field)!r} "
                f"artifact={calibration.get(artifact_field)!r}"
            )
    schema_version = str(calibration.get("schema_version"))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise CalibrationError(f"incompatible calibration schema: {schema_version}")
    if not allow_legacy and schema_version != CURRENT_SCHEMA_VERSION:
        raise CalibrationError(
            f"modern evaluation requires calibration schema {CURRENT_SCHEMA_VERSION}; "
            f"observed={schema_version}"
        )
    if require_supported and calibration.get("status") != "supported":
        raise CalibrationError(
            f"referenced calibration is not supported: {calibration.get('status')}"
        )
    if (
        require_supported
        and schema_version == CURRENT_SCHEMA_VERSION
        and not calibration.get("modern_capture_gate_passed")
    ):
        raise CalibrationError("modern calibration did not pass the idle-aware capture gate")
    relationship = str(reference["calibration_relationship"])
    created_current = bool(reference["calibration_created_in_current_session"])
    if created_current != (relationship == "current_session"):
        raise CalibrationError("calibration current/prior-session labels conflict")
    if require_current_session and relationship != "current_session":
        raise CalibrationError("current collection requires a current-session calibration")
    checks = (
        (
            "experiment session",
            expected_experiment_session_ids,
            calibration.get("experiment_session_id"),
        ),
        (
            "environment fingerprint",
            expected_environment_fingerprints,
            calibration.get("environment_fingerprint"),
        ),
        ("source commit", expected_source_commits, calibration.get("source_commit")),
    )
    for label, expected, observed in checks:
        if expected is not None and (len(expected) != 1 or observed not in expected):
            raise CalibrationError(
                f"calibration {label} mismatch: expected={sorted(expected)} observed={observed!r}"
            )
    return path, calibration


def validate_standard_calibration_result(
    artifact_root: str | Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Verify one complete 5 × (idle + five-payload) calibration evidence matrix."""
    from commguard.workloads import calibration_workload_name

    root = Path(artifact_root).resolve()
    errors: list[str] = []
    observations = list(result.get("observations", []))
    expected_payloads = STANDARD_CALIBRATION_PAYLOAD_MIB
    expected_count = STANDARD_CALIBRATION_REPETITIONS * (1 + len(expected_payloads))
    if len(observations) != expected_count:
        errors.append(f"expected exactly {expected_count} observations, found {len(observations)}")
    run_ids = [str(row.get("run_id", "")) for row in observations]
    if any(not run_id for run_id in run_ids) or len(run_ids) != len(set(run_ids)):
        errors.append("calibration run IDs must be present and unique")
    idle_rows = [row for row in observations if row.get("observation_type") == "idle_baseline"]
    if len(idle_rows) != STANDARD_CALIBRATION_REPETITIONS:
        errors.append(
            f"expected {STANDARD_CALIBRATION_REPETITIONS} idle observations, found {len(idle_rows)}"
        )
    for payload in expected_payloads:
        payload_rows = [
            row
            for row in observations
            if row.get("observation_type") == "collective" and row.get("payload_mib") == payload
        ]
        if len(payload_rows) != STANDARD_CALIBRATION_REPETITIONS:
            errors.append(
                f"expected {STANDARD_CALIBRATION_REPETITIONS} observations for "
                f"{payload} MiB, found {len(payload_rows)}"
            )
    unexpected_payloads = {
        row.get("payload_mib")
        for row in observations
        if row.get("observation_type") == "collective"
    } - set(expected_payloads)
    if unexpected_payloads:
        errors.append(f"unexpected collective payloads: {sorted(unexpected_payloads, key=str)}")

    for row in observations:
        run_id = str(row.get("run_id", ""))
        is_idle = row.get("observation_type") == "idle_baseline"
        expected_name = (
            "calibration_idle"
            if is_idle
            else calibration_workload_name(str(row.get("collective")), int(row["payload_mib"]))
        )
        if row.get("workload_name") != expected_name:
            errors.append(
                f"run {run_id or '<missing>'} workload label mismatch: "
                f"expected={expected_name!r} observed={row.get('workload_name')!r}"
            )
        expected_mode = "idle" if is_idle else "calibration"
        if row.get("worker_mode") != expected_mode:
            errors.append(
                f"run {run_id or '<missing>'} worker mode mismatch: "
                f"expected={expected_mode!r} observed={row.get('worker_mode')!r}"
            )
        if row.get("exit_status") != "completed" or not row.get("participation_valid"):
            continue
        run_directory_value = str(row.get("run_directory", ""))
        run_directory = (root / run_directory_value).resolve()
        try:
            run_directory.relative_to(root)
        except ValueError:
            errors.append(f"run {run_id} directory escapes the artifact root")
            continue
        if not run_directory.is_dir():
            errors.append(f"successful run {run_id} has no run directory: {run_directory_value}")
            continue
        manifest_path = run_directory / "manifest.json"
        if not manifest_path.is_file():
            errors.append(f"successful run {run_id} has no manifest.json")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = dict(manifest.get("config", {}))
        if manifest.get("workload_name") != expected_name:
            errors.append(f"run {run_id} manifest workload name disagrees with its observation")
        if config.get("mode") != expected_mode:
            errors.append(f"run {run_id} manifest worker mode disagrees with its observation")
        if not is_idle:
            if config.get("collective") != row.get("collective"):
                errors.append(f"run {run_id} collective disagrees with its observation")
            if config.get("payload_mib") != row.get("payload_mib"):
                errors.append(f"run {run_id} payload disagrees with its observation")
            continue
        for rank in (0, 1):
            event_path = run_directory / f"rank-{rank}.events.jsonl"
            if not event_path.is_file():
                errors.append(f"successful idle run {run_id} is missing rank {rank} events")
                continue
            events = [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            interval = next(
                (event for event in events if event.get("event") == "measurement_interval"),
                None,
            )
            if interval is None:
                errors.append(
                    f"successful idle run {run_id} rank {rank} lacks measurement interval"
                )
                continue
            details = dict(interval.get("details", {}))
            start = details.get("measurement_start_monotonic_ns")
            end = details.get("measurement_end_monotonic_ns")
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"successful idle run {run_id} rank {rank} has invalid interval")
                continue
            measured_events = [
                event
                for event in events
                if isinstance(event.get("monotonic_ns"), int)
                and start <= event["monotonic_ns"] <= end
            ]
            collective_events = [
                str(event.get("event"))
                for event in measured_events
                if str(event.get("event", "")).startswith("collective_")
                or event.get("event")
                in {"gradient_sync_complete", "parameter_average_complete", "decoy_burst_complete"}
            ]
            if collective_events:
                errors.append(
                    f"idle run {run_id} rank {rank} contains measured collective events: "
                    f"{collective_events}"
                )

    sweep_id = str(result.get("sweep_id", ""))
    started = root / "results" / "calibration-sweeps" / sweep_id / "started.json"
    if not sweep_id or not started.is_file():
        errors.append("exactly one matching calibration sweep marker is required")
    reference = result.get("reference")
    if not isinstance(reference, dict):
        errors.append("calibration result requires an exact artifact reference")
    else:
        relative = Path(str(reference.get("calibration_artifact_path", "")))
        calibration_path = (root / relative).resolve()
        try:
            calibration_path.relative_to(root)
        except ValueError:
            errors.append("calibration reference escapes the artifact root")
        else:
            if not calibration_path.is_file():
                errors.append("calibration reference does not point to an existing JSON artifact")
            else:
                expected_hash = reference.get("calibration_sha256")
                if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                    errors.append("calibration artifact SHA-256 is missing")
                elif sha256_file(calibration_path) != expected_hash:
                    errors.append("calibration artifact SHA-256 does not match its reference")

    if errors:
        raise CalibrationError("standard calibration validation failed: " + "; ".join(errors))
    failed_run_count = sum(
        row.get("exit_status") != "completed" or not row.get("participation_valid")
        for row in observations
    )
    return {
        "clean_standard_calibration": True,
        "planned_run_count": expected_count,
        "observation_count": len(observations),
        "idle_observation_count": len(idle_rows),
        "payload_observation_counts": {
            str(payload): sum(
                row.get("observation_type") == "collective" and row.get("payload_mib") == payload
                for row in observations
            )
            for payload in expected_payloads
        },
        "failed_run_count": failed_run_count,
        "failed_runs_preserved": True,
    }
