from __future__ import annotations

from commguard.calibration import analyze_calibration


def row(payload: int, signal: float | None, supported: bool = True) -> dict:
    return {
        "payload_mib": payload,
        "pcie_total_mean_bytes_per_s": signal,
        "pcie_supported": supported,
        "participation_valid": True,
    }


def test_monotonic_responsive_signal_supports_calibration() -> None:
    result = analyze_calibration([row(1, 100), row(4, 200), row(16, 500), row(64, 1000)])
    assert result["status"] == "supported"
    assert result["spearman_rank_correlation"] == 1
    assert not result["falsification_reasons"]


def test_constant_signal_falsifies_calibration() -> None:
    result = analyze_calibration([row(1, 100), row(4, 100), row(16, 100)])
    assert result["status"] == "not_supported"
    assert result["falsification_reasons"]


def test_unsupported_signal_falsifies_calibration() -> None:
    result = analyze_calibration(
        [row(1, None, False), row(4, None, False), row(16, None, False)]
    )
    assert result["status"] == "not_supported"
    assert "unsupported" in " ".join(result["falsification_reasons"]).lower()
