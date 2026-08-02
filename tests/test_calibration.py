from __future__ import annotations

from commguard.calibration import analyze_calibration


def row(
    payload: int,
    signal: float | None,
    supported: bool = True,
    *,
    idle: bool = False,
) -> dict:
    return {
        "payload_mib": payload,
        "pcie_total_mean_bytes_per_s": signal,
        "pcie_supported": supported,
        "participation_valid": True,
        "observation_type": "idle_baseline" if idle else "collective",
    }


def test_monotonic_responsive_signal_supports_calibration() -> None:
    rows = [row(0, 100_000, idle=True) for _ in range(3)]
    signals = ((1, 2_000_000), (4, 4_000_000), (16, 8_000_000), (64, 16_000_000))
    rows += [row(payload, signal) for payload, signal in signals for _ in range(3)]
    result = analyze_calibration(rows)
    assert result["status"] == "supported"
    assert result["spearman_rank_correlation"] == 1
    assert result["modern_capture_gate_passed"] is True
    assert not result["falsification_reasons"]


def test_constant_signal_falsifies_calibration() -> None:
    rows = [row(0, 100_000, idle=True) for _ in range(3)]
    rows += [row(payload, 2_000_000) for payload in (1, 4, 16) for _ in range(3)]
    result = analyze_calibration(rows)
    assert result["status"] == "not_supported"
    assert result["falsification_reasons"]


def test_unsupported_signal_falsifies_calibration() -> None:
    rows = [row(0, 100_000, idle=True) for _ in range(3)]
    rows += [row(payload, None, False) for payload in (1, 4, 16) for _ in range(3)]
    result = analyze_calibration(rows)
    assert result["status"] == "not_supported"
    assert "unsupported" in " ".join(result["falsification_reasons"]).lower()
