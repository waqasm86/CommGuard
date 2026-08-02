from __future__ import annotations

from typing import Any

from commguard.calibration import analyze_calibration
from commguard.schemas import validate_artifact


def row(
    payload: float,
    signal: float | None,
    *,
    idle: bool = False,
    supported: bool = True,
    participation_valid: bool = True,
) -> dict[str, Any]:
    return {
        "payload_mib": payload,
        "pcie_total_mean_bytes_per_s": signal,
        "pcie_supported": supported,
        "participation_valid": participation_valid,
        "is_idle": idle,
    }


def repeated(payload: float, signals: list[float]) -> list[dict[str, Any]]:
    return [row(payload, signal) for signal in signals]


def test_repetition_aware_calibration_is_fully_supported() -> None:
    rows = [row(0, 100_000, idle=True), row(0, 120_000, idle=True)]
    rows += repeated(1, [2_000_000, 2_100_000, 1_900_000])
    rows += repeated(16, [4_000_000, 4_100_000, 3_900_000])
    rows += repeated(64, [8_000_000, 8_100_000, 7_900_000])

    result = analyze_calibration(rows, minimum_repetitions=3)

    assert result["status"] == "supported"
    assert result["capture_gate_applied"] is True
    assert result["supported_payload_range_mib"] == [1.0, 16.0, 64.0]
    assert result["unreliable_payload_range_mib"] == []
    assert result["spearman_rank_correlation"] == 1
    assert result["dynamic_range_ratio"] == 4
    assert not result["falsification_reasons"]


def test_partial_when_small_payloads_are_missed() -> None:
    rows = [row(0, 1_000, idle=True), row(0, 1_200, idle=True)]
    rows += repeated(1, [1_100, 1_050, 4_000_000])
    rows += repeated(64, [8_000_000, 8_100_000, 7_900_000])

    result = analyze_calibration(
        rows,
        minimum_sizes=2,
        minimum_repetitions=3,
        minimum_capture_rate=0.8,
    )

    assert result["status"] == "partially_supported"
    assert result["supported_payload_range_mib"] == [64.0]
    assert result["unreliable_payload_range_mib"] == [1.0]
    assert result["payload_summaries"][0]["capture_success_count"] == 1
    assert result["payload_summaries"][0]["capture_rate"] == 1 / 3
    validate_artifact(result)


def test_no_idle_rows_use_explicit_compatibility_mode() -> None:
    result = analyze_calibration(
        [row(1, 100), row(4, 200), row(16, 500)],
        minimum_repetitions=1,
    )

    assert result["status"] == "supported"
    assert result["capture_gate_applied"] is False
    assert result["capture_threshold_bytes_per_s"] is None
    assert "backward compatibility" in result["capture_gate_note"]
    assert [item["capture_success_count"] for item in result["payload_summaries"]] == [
        1,
        1,
        1,
    ]
    assert "legacy compatibility mode" in result["limitations"][-1]


def test_invalid_participation_is_excluded_from_usable_repetitions() -> None:
    rows = [row(0, 10, idle=True)]
    rows += [row(1, 2_000_000), row(1, 9_000_000, participation_valid=False)]
    rows += repeated(4, [4_000_000, 4_100_000])
    rows += repeated(16, [8_000_000, 8_100_000])

    result = analyze_calibration(rows, minimum_repetitions=2)

    first_summary = result["payload_summaries"][0]
    assert first_summary["repetitions"] == 2
    assert first_summary["usable_repetitions"] == 1
    assert result["status"] == "partially_supported"
    assert "invalid rank participation" in " ".join(result["falsification_reasons"])


def test_unsupported_pcie_is_a_hard_falsification() -> None:
    rows = [row(0, 10, idle=True)]
    rows += repeated(1, [2_000_000, 2_100_000])
    rows += repeated(4, [4_000_000, 4_100_000])
    rows += repeated(16, [8_000_000, 8_100_000])
    rows.append(row(16, None, supported=False))

    result = analyze_calibration(rows, minimum_repetitions=2)

    assert result["status"] == "not_supported"
    assert "unsupported" in " ".join(result["falsification_reasons"]).lower()


def test_missing_values_do_not_crash_and_are_reported() -> None:
    rows = [row(0, 10, idle=True)]
    rows += repeated(1, [2_000_000, 2_100_000])
    rows += repeated(4, [4_000_000, 4_100_000])
    rows += repeated(16, [8_000_000, 8_100_000])
    rows.append(row(16, None))

    result = analyze_calibration(rows, minimum_repetitions=2)

    assert result["status"] == "partially_supported"
    assert "no usable PCIe reading" in " ".join(result["falsification_reasons"])


def test_empty_input_returns_not_supported_artifact() -> None:
    result = analyze_calibration([])

    assert result["artifact_kind"] == "calibration_result"
    assert result["status"] == "not_supported"
    assert result["payload_summaries"] == []
    assert result["spearman_rank_correlation"] is None
    assert result["dynamic_range_ratio"] is None
    assert "no calibration observations" in result["falsification_reasons"][0]


def test_duplicate_payloads_are_grouped_and_sorted_numerically() -> None:
    rows = [row(16, 8_100_000), row(1, 2_000_000), row(16, 7_900_000)]
    rows += [row(4, 4_000_000), row(1, 2_100_000), row(4, 4_100_000)]

    result = analyze_calibration(rows, minimum_repetitions=2)

    assert [item["payload_mib"] for item in result["payload_summaries"]] == [
        1.0,
        4.0,
        16.0,
    ]
    assert [item["usable_repetitions"] for item in result["payload_summaries"]] == [
        2,
        2,
        2,
    ]


def test_low_correlation_is_not_supported() -> None:
    result = analyze_calibration([row(1, 3_000_000), row(4, 1_000_000), row(16, 2_000_000)])

    assert result["status"] == "not_supported"
    assert "rank correlation" in " ".join(result["falsification_reasons"])


def test_small_dynamic_range_is_not_supported() -> None:
    result = analyze_calibration([row(1, 2_000_000), row(4, 2_050_000), row(16, 2_100_000)])

    assert result["status"] == "not_supported"
    assert "dynamic range" in " ".join(result["falsification_reasons"])


def test_no_reliably_observed_payload_group_is_not_supported() -> None:
    rows = [row(0, 100_000, idle=True)]
    rows += repeated(1, [200_000, 210_000])
    rows += repeated(4, [300_000, 310_000])
    rows += repeated(16, [400_000, 410_000])

    result = analyze_calibration(rows, minimum_repetitions=2)

    assert result["status"] == "not_supported"
    assert result["supported_payload_range_mib"] == []
    assert "no payload group" in " ".join(result["falsification_reasons"])


def test_unusable_idle_rows_do_not_silently_disable_capture_gate() -> None:
    rows = [row(0, None, idle=True)]
    rows += [row(1, 2_000_000), row(4, 4_000_000), row(16, 8_000_000)]

    result = analyze_calibration(rows)

    assert result["status"] == "not_supported"
    assert "idle baseline rows" in " ".join(result["falsification_reasons"])
